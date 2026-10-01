# 三人日麻环境

主环境为 `mahjong_env.game.ThreePlayerMahjong`，完整东南风场包装器为 `mahjong_env.match.SouthMatch`。`reset()` 返回当前需决策玩家的字典；`step({player_name: action})` 返回 `(observations, rewards, done)`。玩家名为 `player_0`、`player_1`、`player_2`。

## 观测

| 键 | 形状 | 内容 |
|---|---|---|
| `observation` | `(6, 30)` | 场风、自风与手牌 unary one-hot |
| `rich_observation` | `(101, 30)` | 基础观测、牌河、副露、拔北、宝牌、立直与剩余牌数 |
| `match_observation` | `(121, 30)` | 公开信息加上局数、本场、供托、分数、相对点差与场次进度 |
| `action_mask` | `(177,)` | 当前合法动作，正值表示可执行 |

玩家信息以当前观察者为参照。三麻移除二至八万，赤五按动作编码与普通五区分。观测的 30 个牌位和牌相关动作的 29 个牌位来自历史模型约定，详见 `FeatureAgent.STRING_TO_INDEX` 和 `FeatureAgent.OFFSET_ACT`。

## 动作

| 编号 | 动作 |
|---|---|
| 0–28 | 弃牌 `Play` |
| 29–57 | 碰 `Peng` |
| 58–86 | 明杠 `Kang` |
| 87–115 | 暗杠 `AnKang` |
| 116–144 | 加杠 `BuKang` |
| 145–173 | 立直并弃指定牌 `Riichi` |
| 174 | 拔北 `Pei` |
| 175 | 和牌 `Hu` |
| 176 | 跳过 `Pass` |

用 `response2action()` / `action2response()` 转换编号与文本，并始终依照当前 `action_mask` 选动作。

## 规则与实现版本

环境实现赤五、立直、振听、碰杠、拔北和流局；计分按项目实现的三麻自摸损支付处理。`SouthMatch` 管理连庄、本场、供托、流局罚符和比赛终止。规则测试保留在 `tests/`。

基础环境与 `training/rules_candidate_*.py` 是两条独立实现。候选版包含多家荣和、途中流局、立直杠等规则修正，经 `ppo_train_rules.py`、`battle_models_rules.py` 显式启用。`ppo_train_consistent_rules.py` 还启用训练采样与更新的数值一致性设置。两套实现不可混用旧断点，候选规则代码会校验规则与观测标识。

历史对战报告保留其当时写出的 `limitations`。基础实现中多响、途中流局及部分 Mortal 动作的边界，不代表候选版或当前上游客户端的完整规则。这个环境适合本项目研究与回归验证，规则覆盖请以所选实现、测试和报告为准。
