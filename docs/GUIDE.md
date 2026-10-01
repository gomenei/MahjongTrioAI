# 核心接口与训练说明

## 环境

`mahjong_env.game.ThreePlayerMahjong` 提供单局，`mahjong_env.match.SouthMatch` 管理完整东南风场。每次只给当前返回观测的玩家提交合法动作：

```python
import numpy as np
from mahjong_env.game import ThreePlayerMahjong

env = ThreePlayerMahjong()
observations = env.reset(current_player=0)
done = False
while not done:
    actions = {
        player: int(np.random.choice(np.flatnonzero(obs["action_mask"])))
        for player, obs in observations.items()
    }
    observations, rewards, done = env.step(actions)
```

| 观测键 | 形状 | 内容 |
|---|---|---|
| `observation` | 6 × 30 | 场风、自风和手牌 |
| `rich_observation` | 101 × 30 | 加入牌河、副露、拔北、宝牌、立直及剩余牌数 |
| `match_observation` | 121 × 30 | 加入局数、本场、供托、分数、点差与比赛进度 |
| `action_mask` | 177 | 当前合法动作 |

| 动作编号 | 动作 |
|---|---|
| 0–28 | 弃牌 |
| 29–57 | 碰 |
| 58–86 | 明杠 |
| 87–115 | 暗杠 |
| 116–144 | 加杠 |
| 145–173 | 立直并弃指定牌 |
| 174 / 175 / 176 | 拔北 / 和牌 / 跳过 |

基础环境与 `training/rules_candidate_*.py` 是两套独立规则实现。候选版包含多家荣和、途中流局和立直杠等修正，使用统一命令的 `--rules candidate` 显式启用，目前要求 `--game-mode south`。GUI 与 Copilot 桥接使用基础环境。历史汇总中的 `limitations` 描述当时环境边界。

## 监督学习

`python -m training bc` 从自备雀魂 JSON 重放生成特征，再训练 Tile Transformer。输入包含 `data.data.actions`，具体事件结构由 `mahjong_env/deal.py` 读取；代码不负责获取牌谱。

`--from-scratch` 从随机权重开始训练，无需旧模型；省略时须用 `--old-model` 提供兼容的 6 层 Transformer 进行初始化。数据按整份牌谱分割训练、验证和测试集，避免相邻状态泄漏。`--generate-only` 只生成数据；`--data`、`--output-dir` 和 `--max-files` 控制位置与规模。默认数据在 `data/rich/`，最佳策略在 `training_runs/rich_tile_transformer/best.pt`。

`python -m training match` 在 101 层策略上扩展 121 层场况输入，数据在 `data/match/`，默认模型在 `training_runs/match_tile_transformer/best.pt`。训练使用的 pickle 清单和连续数组均是本地文件，不随仓库发布。

网络定义集中在 `training/models.py`，保留历史 checkpoint 加载所需的 CNN、残差卷积、Transformer 与卷积注意力结构。

## PPO

`python -m training ppo` 从自备 checkpoint 热启动，用冻结策略或历史对手池采样，支持 GAE、KL 约束、并行环境和唯一合法动作自动推进。单局训练使用 `--game-mode hand`，完整场次使用 `south` 及匹配的场况策略。

`--consistent-forward` 启用采样与梯度更新的数值一致性设置。候选规则断点会核对规则版本与观测标识，不可与旧规则或不同数值配置混用。命令参数、断点恢复选项见 `python -m training ppo --help`。

## 评测

`python -m evaluation models` 用相同牌山与座位轮换比较模型；两模型比较使用镜像阵容。小局模式报告净点等指标，南风场另报告最终顺位。参赛次数与独立牌山组数应区分。

`python -m evaluation copilot` 把环境状态转成 MJAI 事件，与 Mortal3P 对战。需要自备三麻权重和平台/Python 版本匹配的 `libriichi3p`。使用 `--check-only` 可先检查依赖；桥接后备动作与规则覆盖差异会影响比较。

所有新报告默认写入 `artifacts/`。原始报告可能包含模型路径、逐局状态和错误明细，请保留在本地。公开 `results/` 只保存脱敏聚合结果，最低牌山门槛为 1000。

## 发布与验证

`scripts/check_public_release.py` 只依赖标准库，核对公开文件清单、SHA-256、禁止文件类型、高置信凭据模式和样本门槛。检查 ZIP：

```powershell
python scripts/check_public_release.py --archive ../MahjongTrioAI-github-public.zip
```

`.gitignore` 排除了权重、牌谱、数据、日志、证书、浏览器目录和实际设置。它不会自动排除已被 Git 跟踪的敏感文件，因此提交前应查看暂存清单。

2026-10-01，本次整理完成以下验证：

- 核心回归测试：164 项通过，1 条 PyTorch Transformer 提示。
- 11 个命令入口与帮助检查通过。
- 用临时合成数据实际完成 101 层从零监督训练、121 层场况训练，均保存了 checkpoint。
- 用临时随机策略实际完成一次 PPO 更新及小规模模型对战。
- GUI 无模型渲染、Copilot 模块与本地适配器导入通过。

上述验证的临时数据、权重与日志已清理，没有放入发布包。未重新训练历史模型、重跑历史对战或连接游戏账号进行线上验证。
