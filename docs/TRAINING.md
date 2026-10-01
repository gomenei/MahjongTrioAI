# 训练与评测

公开包提供网络与训练代码，训练集、真实牌谱、初始 checkpoint 和运行日志需由用户在本地准备。以下命令均在仓库根目录执行。

## 监督学习

`train_all.py` 读取 `data/data.pkl`。基础 pickle 数据包含 `state`（`observation`、`action_mask`）与 `label`，缓存和字段校验由 `training/data.py` 实现。支持 CNN、残差卷积、Tile Transformer 和卷积注意力网络：

```powershell
python train_all.py --data data/data.pkl --models tile_transformer --epochs 30
python compare_models.py --help
```

`train_rich.py` 从本地雀魂 JSON 记录重放生成 101 层特征。输入需含 `data.data.actions`，其事件结构见 `mahjong_env/deal.py`。使用自己的、符合该结构的文件：

```powershell
python train_rich.py --replay-dir replays --generate-only
python train_rich.py --replay-dir replays --from-scratch --epochs 20
```

`--from-scratch` 表示不加载旧模型。生成的数据是 pickle 清单与连续内存映射数组，位于 `data/rich/`。按整份牌谱划分训练、验证和测试集，避免相邻状态泄漏。

121 层场况监督训练需要先准备 101 层策略：

```powershell
python train_match.py --replay-dir replays --base-model training_runs/rich_tile_transformer/best.pt --epochs 20
```

具体输出目录与调参选项可用各入口的 `--help` 查看。验证决策准确率是选模依据之一；它不能代替实际对战强度测试。

## PPO

PPO 使用已有策略热启动，并用冻结 baseline 或历史策略池作为对手。支持并行环境、自动推进唯一合法动作、GAE、KL 约束与 checkpoint 恢复。

```powershell
python ppo_train.py --base-model training_runs/rich_tile_transformer/best.pt --game-mode hand --updates 100 --output-dir ppo_runs/local_hand
python ppo_train.py --base-model training_runs/match_tile_transformer/best.pt --game-mode south --updates 100 --output-dir ppo_runs/local_south
```

第二条命令使用 `train_match.py` 默认输出目录；自定义训练目录时相应修改路径。完整场况策略使用 `match_observation`；checkpoint 元数据负责声明网络与观测键。

候选规则和一致性训练入口：

```powershell
python ppo_train_rules.py --help
python ppo_train_consistent_rules.py --help
python battle_models_rules.py --help
```

`run_autodl_*.py`、`run_day1_3_experiments.py` 和 `run_ppo_8h_experiment.py` 保留批量实验编排能力；部分默认路径来自历史实验，使用前需检查并替换为自己的模型和输出目录。SSH 连接脚本、服务器配置与凭据未公开。部署说明见 [AUTODL_TRAINING.md](../AUTODL_TRAINING.md)。

## 评测

```powershell
python battle_models.py --models A=model/a.pt B=model/b.pt --game-mode hand --output-dir artifacts/hand_battle
python battle_models.py --models A=model/a.pt B=model/b.pt --game-mode south --max-steps 3000 --output-dir artifacts/south_battle
```

评测采用配对牌山与座位轮换，报告平均点差、顺位、胜率及置信区间。比较两模型时会构造两种镜像阵容；每个模型的参赛次数因此不是独立牌山数。跨实验比较时应同时核对模式、规则、阵容、动作选择和有效样本数。

Mortal 对战需要自备三麻权重和与平台、Python 版本匹配的 `libriichi3p`：

```powershell
python compare_copilot_model.py --our-model model/model.pt --mortal-model integrations/MahjongCopilot/models/mortal_3p.pth --check-only
python compare_copilot_model.py --our-model model/model.pt --mortal-model integrations/MahjongCopilot/models/mortal_3p.pth --game-mode south --output-dir artifacts/copilot_battle
```

不要把新生成的 `progress.json`、错误明细、模型路径或逐局记录加入公开结果。历史公开包仅保留经过字段白名单过滤的汇总。
