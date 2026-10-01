# 对战结果（公开汇总）

这里只保留本地模拟对战的聚合统计，不包含权重、训练日志、真实牌谱、逐局状态、绝对路径或错误消息。历史报告的规则限制和完成标志保留。

## 阅读方法

- `hand` 为独立小局；`south` 为完整东南风场。南风场的 `hands` 字段是该模型参赛场次数，实际小局数另见 `played_hands`。
- `seed_groups` 为每个模型参与的配对牌山组数，不能把座位轮换的参赛次数当成独立样本。
- 公开门槛：每个模型的 `seed_groups` 与每项比较的 `paired_seed_groups` 均至少为 1000；小样本和 smoke 结果已移除。
- `completed: false` 是未完成实验的中间统计。
- JSON 保留完整的汇总字段与 95% 区间；CSV 可直接打开查看。所有数值取自原汇总，未重跑历史实验。
- Copilot 桥接后备动作只保留总计，去除了含单步状态的原因明细。存在后备动作时，比较受到动作映射与规则覆盖差异的影响。
- 权重未公开，本仓库无法独立复现历史模型结果。

机器可读索引：[index.json](index.json)。实验目录名沿用原文件夹标识，不额外推断运行日期。

## 实验索引

### 主目录

模式：`hand`；状态：**已完成**。

[battle_report.json](battle_report.json) · [battle_ranking.csv](battle_ranking.csv) · [battle_pairwise.csv](battle_pairwise.csv)

| 模型 | 牌山组数 | 参赛次数 | 平均净点 | 平均顺位 | 胡牌率 |
|---|---:|---:|---:|---:|---:|
| ppo_rich | 4000 | 18000 | +620.65 | — | 29.29% |
| tile_transformer | 4000 | 18000 | -681.46 | — | 29.02% |

### copilot_vs_bc

模式：`hand`；状态：**未完成：中间统计**。

[battle_report.json](copilot_vs_bc/battle_report.json) · [battle_ranking.csv](copilot_vs_bc/battle_ranking.csv) · [battle_pairwise.csv](copilot_vs_bc/battle_pairwise.csv)

| 模型 | 牌山组数 | 参赛次数 | 平均净点 | 平均顺位 | 胡牌率 |
|---|---:|---:|---:|---:|---:|
| MahjongCopilot-Mortal | 1126 | 3378 | +290.23 | — | 31.71% |
| MahjongTrioAI | 1126 | 6756 | -145.12 | — | 26.04% |

Mortal 决策总数：127,021；桥接后备动作：56。

### day01_03/full_u1000_h256_e256/pairwise_g5000_seed20261001/ppo_seed_20260909

模式：`hand`；状态：**已完成**。

[battle_report.json](day01_03/full_u1000_h256_e256/pairwise_g5000_seed20261001/ppo_seed_20260909/battle_report.json) · [battle_ranking.csv](day01_03/full_u1000_h256_e256/pairwise_g5000_seed20261001/ppo_seed_20260909/battle_ranking.csv) · [battle_pairwise.csv](day01_03/full_u1000_h256_e256/pairwise_g5000_seed20261001/ppo_seed_20260909/battle_pairwise.csv)

| 模型 | 牌山组数 | 参赛次数 | 平均净点 | 平均顺位 | 胡牌率 |
|---|---:|---:|---:|---:|---:|
| ppo_seed_20260909 | 10000 | 45000 | +161.17 | — | 26.93% |
| rich_bc | 10000 | 45000 | -173.03 | — | 27.30% |

### day01_03/full_u1000_h256_e256/pairwise_g5000_seed20261001/ppo_seed_20260910

模式：`hand`；状态：**已完成**。

[battle_report.json](day01_03/full_u1000_h256_e256/pairwise_g5000_seed20261001/ppo_seed_20260910/battle_report.json) · [battle_ranking.csv](day01_03/full_u1000_h256_e256/pairwise_g5000_seed20261001/ppo_seed_20260910/battle_ranking.csv) · [battle_pairwise.csv](day01_03/full_u1000_h256_e256/pairwise_g5000_seed20261001/ppo_seed_20260910/battle_pairwise.csv)

| 模型 | 牌山组数 | 参赛次数 | 平均净点 | 平均顺位 | 胡牌率 |
|---|---:|---:|---:|---:|---:|
| ppo_seed_20260910 | 10000 | 45000 | +183.84 | — | 27.11% |
| rich_bc | 10000 | 45000 | -186.79 | — | 27.29% |

### day01_03/full_u1000_h256_e256/pairwise_g5000_seed20261001/ppo_seed_20260911

模式：`hand`；状态：**已完成**。

[battle_report.json](day01_03/full_u1000_h256_e256/pairwise_g5000_seed20261001/ppo_seed_20260911/battle_report.json) · [battle_ranking.csv](day01_03/full_u1000_h256_e256/pairwise_g5000_seed20261001/ppo_seed_20260911/battle_ranking.csv) · [battle_pairwise.csv](day01_03/full_u1000_h256_e256/pairwise_g5000_seed20261001/ppo_seed_20260911/battle_pairwise.csv)

| 模型 | 牌山组数 | 参赛次数 | 平均净点 | 平均顺位 | 胡牌率 |
|---|---:|---:|---:|---:|---:|
| ppo_seed_20260911 | 10000 | 45000 | +187.78 | — | 27.05% |
| rich_bc | 10000 | 45000 | -203.29 | — | 27.30% |

### day01_03/rich_bc_vs_ppo_rich_seed20260909

模式：`hand`；状态：**已完成**。

[battle_report.json](day01_03/rich_bc_vs_ppo_rich_seed20260909/battle_report.json) · [battle_ranking.csv](day01_03/rich_bc_vs_ppo_rich_seed20260909/battle_ranking.csv) · [battle_pairwise.csv](day01_03/rich_bc_vs_ppo_rich_seed20260909/battle_pairwise.csv)

| 模型 | 牌山组数 | 参赛次数 | 平均净点 | 平均顺位 | 胡牌率 |
|---|---:|---:|---:|---:|---:|
| ppo_rich | 10000 | 45000 | +179.77 | — | 27.64% |
| rich_bc | 10000 | 45000 | -180.76 | — | 26.76% |

### feature_ab

模式：`hand`；状态：**已完成**。

[battle_report.json](feature_ab/battle_report.json) · [battle_ranking.csv](feature_ab/battle_ranking.csv) · [battle_pairwise.csv](feature_ab/battle_pairwise.csv)

| 模型 | 牌山组数 | 参赛次数 | 平均净点 | 平均顺位 | 胡牌率 |
|---|---:|---:|---:|---:|---:|
| rich | 19664 | 88992 | +428.39 | — | 29.98% |
| legacy | 19664 | 87984 | -442.77 | — | 29.85% |

### ppo_duration_existing

模式：`south`；状态：**已完成**。

[battle_report.json](ppo_duration_existing/battle_report.json) · [battle_ranking.csv](ppo_duration_existing/battle_ranking.csv) · [battle_pairwise.csv](ppo_duration_existing/battle_pairwise.csv)

| 模型 | 牌山组数 | 参赛次数 | 平均净点 | 平均顺位 | 胡牌率 |
|---|---:|---:|---:|---:|---:|
| u300 | 1000 | 3000 | +1100.43 | 1.9683 | 27.59% |
| best_seen | 1000 | 3000 | +505.60 | 1.9970 | 26.47% |
| u0 | 1000 | 3000 | -1606.03 | 2.0347 | 25.68% |

### ppo_vs_baseline

模式：`hand`；状态：**已完成**。

[battle_report.json](ppo_vs_baseline/battle_report.json) · [battle_ranking.csv](ppo_vs_baseline/battle_ranking.csv) · [battle_pairwise.csv](ppo_vs_baseline/battle_pairwise.csv)

| 模型 | 牌山组数 | 参赛次数 | 平均净点 | 平均顺位 | 胡牌率 |
|---|---:|---:|---:|---:|---:|
| ppo | 20000 | 90000 | +154.52 | — | 31.91% |
| baseline | 20000 | 90000 | -161.04 | — | 30.89% |

### south_8h_checkpoint_league

模式：`south`；状态：**已完成**。

[battle_report.json](south_8h_checkpoint_league/battle_report.json) · [battle_ranking.csv](south_8h_checkpoint_league/battle_ranking.csv) · [battle_pairwise.csv](south_8h_checkpoint_league/battle_pairwise.csv)

| 模型 | 牌山组数 | 参赛次数 | 平均净点 | 平均顺位 | 胡牌率 |
|---|---:|---:|---:|---:|---:|
| t3_6.0h | 5999 | 17997 | +599.26 | 1.9837 | 27.85% |
| t2_4.0h | 6000 | 18000 | +289.57 | 1.9949 | 27.66% |
| t4_8.0h | 6000 | 18000 | +150.67 | 1.9966 | 27.99% |
| t1_2.0h | 5999 | 17997 | +11.62 | 1.9972 | 27.61% |
| t0_0.0h | 5999 | 17997 | -1051.20 | 2.0276 | 27.25% |

### south_8h_vs_copilot/t0_0.0h

模式：`south`；状态：**已完成**。

[battle_report.json](south_8h_vs_copilot/t0_0.0h/battle_report.json) · [battle_ranking.csv](south_8h_vs_copilot/t0_0.0h/battle_ranking.csv) · [battle_pairwise.csv](south_8h_vs_copilot/t0_0.0h/battle_pairwise.csv)

| 模型 | 牌山组数 | 参赛次数 | 平均净点 | 平均顺位 | 胡牌率 |
|---|---:|---:|---:|---:|---:|
| MahjongCopilot-Mortal | 2000 | 9000 | -38.59 | 1.9874 | 30.53% |
| t0_0.0h | 2000 | 9000 | +38.59 | 2.0126 | 26.54% |

Mortal 决策总数：2,835,999；桥接后备动作：1,182。

### south_8h_vs_copilot/t1_2.0h

模式：`south`；状态：**已完成**。

[battle_report.json](south_8h_vs_copilot/t1_2.0h/battle_report.json) · [battle_ranking.csv](south_8h_vs_copilot/t1_2.0h/battle_ranking.csv) · [battle_pairwise.csv](south_8h_vs_copilot/t1_2.0h/battle_pairwise.csv)

| 模型 | 牌山组数 | 参赛次数 | 平均净点 | 平均顺位 | 胡牌率 |
|---|---:|---:|---:|---:|---:|
| t1_2.0h | 2000 | 9000 | +706.34 | 1.9917 | 26.92% |
| MahjongCopilot-Mortal | 2000 | 9000 | -706.34 | 2.0083 | 30.17% |

Mortal 决策总数：2,840,356；桥接后备动作：1,191。

### south_8h_vs_copilot/t2_4.0h

模式：`south`；状态：**已完成**。

[battle_report.json](south_8h_vs_copilot/t2_4.0h/battle_report.json) · [battle_ranking.csv](south_8h_vs_copilot/t2_4.0h/battle_ranking.csv) · [battle_pairwise.csv](south_8h_vs_copilot/t2_4.0h/battle_pairwise.csv)

| 模型 | 牌山组数 | 参赛次数 | 平均净点 | 平均顺位 | 胡牌率 |
|---|---:|---:|---:|---:|---:|
| t2_4.0h | 2000 | 9000 | +859.28 | 1.9843 | 26.95% |
| MahjongCopilot-Mortal | 2000 | 9000 | -859.28 | 2.0157 | 30.07% |

Mortal 决策总数：2,828,753；桥接后备动作：1,222。

### south_8h_vs_copilot/t3_6.0h

模式：`south`；状态：**已完成**。

[battle_report.json](south_8h_vs_copilot/t3_6.0h/battle_report.json) · [battle_ranking.csv](south_8h_vs_copilot/t3_6.0h/battle_ranking.csv) · [battle_pairwise.csv](south_8h_vs_copilot/t3_6.0h/battle_pairwise.csv)

| 模型 | 牌山组数 | 参赛次数 | 平均净点 | 平均顺位 | 胡牌率 |
|---|---:|---:|---:|---:|---:|
| t3_6.0h | 2000 | 9000 | +1050.71 | 1.9829 | 27.55% |
| MahjongCopilot-Mortal | 2000 | 9000 | -1050.71 | 2.0171 | 30.10% |

Mortal 决策总数：2,809,535；桥接后备动作：1,172。

### south_8h_vs_copilot/t4_8.0h

模式：`south`；状态：**已完成**。

[battle_report.json](south_8h_vs_copilot/t4_8.0h/battle_report.json) · [battle_ranking.csv](south_8h_vs_copilot/t4_8.0h/battle_ranking.csv) · [battle_pairwise.csv](south_8h_vs_copilot/t4_8.0h/battle_pairwise.csv)

| 模型 | 牌山组数 | 参赛次数 | 平均净点 | 平均顺位 | 胡牌率 |
|---|---:|---:|---:|---:|---:|
| t4_8.0h | 2000 | 9000 | +874.28 | 1.9831 | 27.59% |
| MahjongCopilot-Mortal | 2000 | 9000 | -874.28 | 2.0169 | 30.28% |

Mortal 决策总数：2,809,190；桥接后备动作：1,182。

### south_screen_20260909

模式：`south`；状态：**未完成：中间统计**。

[battle_report.json](south_screen_20260909/battle_report.json) · [battle_ranking.csv](south_screen_20260909/battle_ranking.csv) · [battle_pairwise.csv](south_screen_20260909/battle_pairwise.csv)

| 模型 | 牌山组数 | 参赛次数 | 平均净点 | 平均顺位 | 胡牌率 |
|---|---:|---:|---:|---:|---:|
| south_ppo | 5000 | 15000 | +1577.59 | 1.9535 | 27.16% |
| match_bc | 5000 | 30000 | -788.80 | 2.0232 | 26.16% |
