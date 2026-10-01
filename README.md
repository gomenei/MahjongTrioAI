# MahjongTrioAI

三人日本麻将研究项目：环境、Pygame GUI、监督学习、PPO，以及 Mahjong Copilot 本地模型适配。

公开仓库只提供代码和脱敏对战汇总。模型、牌谱、数据集、训练日志、爬取代码和账号配置均不包含在内。

## 目录

```text
├── main.py                 三麻 GUI
├── copilot.py              Copilot 安装与启动
├── mahjong_env/             三麻规则、特征、计分与完整场次
├── gui/                     牌桌与本地回放
├── training/                网络、监督学习、PPO 与并行采样
├── evaluation/              模型对战和 Copilot-Mortal3P 对战
├── integrations/            Copilot 源码与协议适配
├── tests/                   核心规则、训练和适配回归测试
├── results/                 16 组实验合并后的公开汇总
├── assets/                  牌面图片
├── requirements/            训练与 Copilot 的附加依赖
├── scripts/                 发布检查与文件清单
└── docs/                    技术说明与第三方来源
```

![三人日麻 GUI，无模型演示](docs/images/gui.png)

## GUI

在仓库根目录执行，运行示例使用 Python 3.11：

```powershell
python -m pip install -r requirements.txt
python main.py
```

没有模型时，两个 AI 使用随机合法动作。加载自备兼容权重：

```powershell
python main.py --model model/model.pt
```

点击手牌出牌，在右侧选择立直、和牌、碰、杠或拔北；`Esc` 取消选择或退出，牌局结束后按 `R` 重开。本地回放使用 `python main.py --replay-dir replays`，需要自行提供符合格式的雀魂 JSON 牌谱。

## 训练

训练统一使用 `python -m training`，保留三种核心流程：

```powershell
python -m pip install -r requirements/training.txt

# 从自备牌谱生成 101 层特征并监督训练
python -m training bc --replay-dir replays --from-scratch --epochs 20

# 加入场况，训练 121 层完整场次策略
python -m training match --replay-dir replays --base-model training_runs/rich_tile_transformer/best.pt

# 在已有场况策略上进行完整南风场 PPO
python -m training ppo --base-model training_runs/match_tile_transformer/best.pt --game-mode south --updates 100 --output-dir ppo_runs/local
```

候选规则与一致性采样通过同一入口启用：

```powershell
python -m training ppo --rules candidate --consistent-forward --base-model model/model.pt --game-mode south --updates 100
```

使用 `python -m training bc --help`、`match --help` 或 `ppo --help` 查看参数。数据格式、观测和断点要求见 [技术说明](docs/GUIDE.md)。

## 对战测试

评测统一使用 `python -m evaluation`：

```powershell
# 两个自备模型，完整南风场、座位轮换
python -m evaluation models --models A=model/a.pt B=model/b.pt --game-mode south --max-steps 3000 --output-dir artifacts/battle

# 与自备 Mortal3P 权重比较，需另行准备匹配的 libriichi3p
python -m evaluation copilot --our-model model/model.pt --mortal-model integrations/MahjongCopilot/models/mortal_3p.pth --output-dir artifacts/copilot_battle
```

候选规则模型评测使用 `python -m evaluation models --rules candidate ... --game-mode south`。Copilot 对战沿用现有 MJAI 桥接规则。

公开结果见 [结果说明](results/README.md)：16 组实验合并为一份 JSON 和两份 CSV，每个模型及每项配对比较的牌山组数均至少为 1000。完成标志、置信区间与规则限制保留；不同模式、阵容和样本数的实验不能直接横向排名。

## Mahjong Copilot

```powershell
python copilot.py --setup
.\.venv-copilot\Scripts\python.exe copilot.py
```

Linux/macOS 启动命令为 `./.venv-copilot/bin/python copilot.py`。安装会创建独立环境并下载 Playwright Chromium。

放入自备 `model/model.pt`，在设置中选择 `MahjongTrioAI`。也可复制 `integrations/MahjongCopilot/settings.example.json` 为同目录的 `settings.json`。本项目适配支持三麻；Mortal 的权重、原生库和进程注入二进制需自行准备，均不随仓库提供。

本地现存 Mortal3P 文件与 Akagi v2 的公开默认三麻权重相同；该默认模型在 Akagi v2 文档中被描述为弱模型。相关对战成绩应按所用公开默认模型及模拟器条件解读。[Akagi v2 说明](https://github.com/shinkuan/Akagi/blob/v2/README.md)

## 检查

```powershell
python -m pytest -q
python scripts/check_public_release.py
```

发布检查会核对 `scripts/public_manifest.json` 中的文件与 SHA-256，并检查敏感文件和 1000 组的最低门槛。原项目 Git 历史没有复制进发布包；后续训练和评测输出请保留在已忽略的目录内。

Copilot 来自 [latorc/MahjongCopilot](https://github.com/latorc/MahjongCopilot)，其许可证与版权声明保留在原目录。其他来源及素材授权状态见 [第三方说明](docs/THIRD_PARTY_NOTICES.md)。本次整理未替项目原创代码选择新许可证。
