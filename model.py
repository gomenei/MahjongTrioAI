"""三人麻将动作模型及候选网络注册表。"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


OBS_CHANNELS = 6
OBS_TILES = 30
ACTION_SIZE = 177
TILE_ACTION_TYPES = 6
ACTION_TILES = 29


def apply_action_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """保证非法动作不会被模型选中。"""
    mask = mask.to(device=logits.device)
    return logits.masked_fill(mask <= 0, torch.finfo(logits.dtype).min)


class CNN(nn.Module):
    """原项目 baseline，参数名称保持不变以兼容已有权重。"""

    def __init__(self, obs_channels: int = OBS_CHANNELS):
        super().__init__()
        self.obs_channels = obs_channels
        self.conv1 = nn.Conv1d(in_channels=obs_channels, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 64, 3, 1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 15, 177)

    def forward(self, input_dict):
        x = input_dict["observation"].float()
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        logits = self.fc(x)
        return apply_action_mask(logits, input_dict["action_mask"])


class StructuredActionHead(nn.Module):
    """分别预测六类牌相关动作和三个全局动作。"""

    def __init__(self, feature_size: int, dropout: float = 0.15):
        super().__init__()
        self.tile_head = nn.Sequential(
            nn.LayerNorm(feature_size),
            nn.Linear(feature_size, feature_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_size, TILE_ACTION_TYPES),
        )
        self.global_head = nn.Sequential(
            nn.LayerNorm(feature_size * 2),
            nn.Linear(feature_size * 2, feature_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_size, 3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        tile_features = features[:, :ACTION_TILES]
        tile_logits = self.tile_head(tile_features)
        # (batch, 29, 6) -> action-major (batch, 6 * 29)
        tile_logits = tile_logits.transpose(1, 2).contiguous().flatten(1)
        pooled = torch.cat((features.mean(dim=1), features.amax(dim=1)), dim=1)
        global_logits = self.global_head(pooled)
        return torch.cat((tile_logits, global_logits), dim=1)


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.conv1(F.gelu(self.norm1(x)))
        x = self.dropout(x)
        x = self.conv2(F.gelu(self.norm2(x)))
        return x + residual


class ResidualCNN(nn.Module):
    """多尺度膨胀残差卷积，适合识别顺子、对子和局部牌形。"""

    def __init__(self, channels: int = 128, blocks: int = 6, dropout: float = 0.12, obs_channels: int = OBS_CHANNELS):
        super().__init__()
        self.obs_channels = obs_channels
        self.stem = nn.Conv1d(obs_channels, channels, kernel_size=3, padding=1)
        dilations = (1, 2, 4, 1, 2, 4)
        self.blocks = nn.Sequential(
            *[
                ResidualBlock1D(channels, dilations[index % len(dilations)], dropout)
                for index in range(blocks)
            ]
        )
        self.final_norm = nn.GroupNorm(8, channels)
        self.head = StructuredActionHead(channels, dropout)

    def forward(self, input_dict):
        x = self.stem(input_dict["observation"].float())
        x = F.gelu(self.final_norm(self.blocks(x))).transpose(1, 2)
        logits = self.head(x)
        return apply_action_mask(logits, input_dict["action_mask"])


class TileTransformer(nn.Module):
    """把 30 个牌位看作 token，显式学习远距离牌之间的关系。"""

    def __init__(
        self,
        width: int = 128,
        layers: int = 4,
        heads: int = 8,
        dropout: float = 0.12,
        obs_channels: int = OBS_CHANNELS,
    ):
        super().__init__()
        self.obs_channels = obs_channels
        self.input_projection = nn.Linear(obs_channels, width)
        self.position = nn.Parameter(torch.zeros(1, OBS_TILES, width))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.final_norm = nn.LayerNorm(width)
        self.head = StructuredActionHead(width, dropout)

    def forward(self, input_dict):
        x = input_dict["observation"].float().transpose(1, 2)
        x = self.input_projection(x) + self.position
        x = self.final_norm(self.encoder(x))
        logits = self.head(x)
        return apply_action_mask(logits, input_dict["action_mask"])


class ConvAttentionHybrid(nn.Module):
    """并行局部卷积 + 全局自注意力，作为精度优先候选。"""

    def __init__(
        self,
        width: int = 160,
        layers: int = 3,
        heads: int = 8,
        dropout: float = 0.15,
        obs_channels: int = OBS_CHANNELS,
    ):
        super().__init__()
        self.obs_channels = obs_channels
        branch = width // 4
        self.conv1 = nn.Conv1d(obs_channels, branch, kernel_size=1)
        self.conv3 = nn.Conv1d(obs_channels, branch, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(obs_channels, branch, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(obs_channels, branch, kernel_size=7, padding=3)
        self.fuse = nn.Sequential(
            nn.Conv1d(branch * 4, width, kernel_size=1),
            nn.GroupNorm(8, width),
            nn.GELU(),
        )
        self.local_blocks = nn.Sequential(
            ResidualBlock1D(width, 1, dropout),
            ResidualBlock1D(width, 2, dropout),
        )
        self.position = nn.Parameter(torch.zeros(1, OBS_TILES, width))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.final_norm = nn.LayerNorm(width)
        self.head = StructuredActionHead(width, dropout)

    def forward(self, input_dict):
        observation = input_dict["observation"].float()
        branches = (
            self.conv1(observation),
            self.conv3(observation),
            self.conv5(observation),
            self.conv7(observation),
        )
        x = self.local_blocks(self.fuse(torch.cat(branches, dim=1)))
        x = x.transpose(1, 2) + self.position
        x = self.final_norm(self.encoder(x))
        logits = self.head(x)
        return apply_action_mask(logits, input_dict["action_mask"])


MODEL_REGISTRY = {
    "baseline_cnn": CNN,
    "residual_cnn": ResidualCNN,
    "tile_transformer": TileTransformer,
    "conv_attention": ConvAttentionHybrid,
}


def create_model(name: str, obs_channels: int = OBS_CHANNELS) -> nn.Module:
    try:
        return MODEL_REGISTRY[name](obs_channels=obs_channels)
    except KeyError as exc:
        choices = ", ".join(MODEL_REGISTRY)
        raise ValueError(f"未知模型 {name!r}，可选：{choices}") from exc


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def load_checkpoint_model(path, map_location="cpu"):
    """同时兼容旧 raw state_dict 和新训练脚本生成的 checkpoint。"""
    path = Path(path)
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=map_location)

    if isinstance(payload, dict) and "model_state_dict" in payload:
        model_name = payload.get("model_name", "baseline_cnn")
        obs_channels = int(payload.get("obs_channels", OBS_CHANNELS))
        model = create_model(model_name, obs_channels=obs_channels)
        model.load_state_dict(payload["model_state_dict"])
        return model, payload

    model = CNN()
    model.load_state_dict(payload)
    return model, {"model_name": "baseline_cnn", "legacy": True}
