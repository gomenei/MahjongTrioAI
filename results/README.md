# 对战结果

16 组历史实验合并为三个文件，14 组完成、2 组为中间统计：

- [battles.json](battles.json)：完整脱敏汇总，保留规则限制、置信区间、完成状态与桥接后备动作总数。
- [rankings.csv](rankings.csv)：每个实验中各模型的聚合指标。
- [pairwise.csv](pairwise.csv)：模型两两比较的点差与区间。

每个模型的 `seed_groups` 和每项配对比较的 `paired_seed_groups` 均至少为 1000。权重、逐局过程、训练日志和原始路径未公开。不同模式与阵容的结果不能直接横向排名，模型权重未提供，因此读者无法仅凭仓库复现历史成绩。

`hand` 是独立小局，`south` 是完整东南风场。`hands` 为模型参赛次数；南风场实际小局数见 `played_hands`，独立牌山组数见 `seed_groups`。

| 实验 | 模式 | 状态 | 模型与平均净点 |
|---|---|---|---|
| `initial_ppo_vs_bc` | hand | 已完成 | ppo_rich: +620.65；tile_transformer: -681.46 |
| `copilot_vs_bc` | hand | **中间统计** | MahjongCopilot-Mortal: +290.23；MahjongTrioAI: -145.12 |
| `day01_03/full_u1000_h256_e256/pairwise_g5000_seed20261001/ppo_seed_20260909` | hand | 已完成 | ppo_seed_20260909: +161.17；rich_bc: -173.03 |
| `day01_03/full_u1000_h256_e256/pairwise_g5000_seed20261001/ppo_seed_20260910` | hand | 已完成 | ppo_seed_20260910: +183.84；rich_bc: -186.79 |
| `day01_03/full_u1000_h256_e256/pairwise_g5000_seed20261001/ppo_seed_20260911` | hand | 已完成 | ppo_seed_20260911: +187.78；rich_bc: -203.29 |
| `day01_03/rich_bc_vs_ppo_rich_seed20260909` | hand | 已完成 | ppo_rich: +179.77；rich_bc: -180.76 |
| `feature_ab` | hand | 已完成 | rich: +428.39；legacy: -442.77 |
| `ppo_duration_existing` | south | 已完成 | u300: +1100.43；best_seen: +505.60；u0: -1606.03 |
| `ppo_vs_baseline` | hand | 已完成 | ppo: +154.52；baseline: -161.04 |
| `south_8h_checkpoint_league` | south | 已完成 | t3_6.0h: +599.26；t2_4.0h: +289.57；t4_8.0h: +150.67；t1_2.0h: +11.62；t0_0.0h: -1051.20 |
| `south_8h_vs_copilot/t0_0.0h` | south | 已完成 | MahjongCopilot-Mortal: -38.59；t0_0.0h: +38.59 |
| `south_8h_vs_copilot/t1_2.0h` | south | 已完成 | t1_2.0h: +706.34；MahjongCopilot-Mortal: -706.34 |
| `south_8h_vs_copilot/t2_4.0h` | south | 已完成 | t2_4.0h: +859.28；MahjongCopilot-Mortal: -859.28 |
| `south_8h_vs_copilot/t3_6.0h` | south | 已完成 | t3_6.0h: +1050.71；MahjongCopilot-Mortal: -1050.71 |
| `south_8h_vs_copilot/t4_8.0h` | south | 已完成 | t4_8.0h: +874.28；MahjongCopilot-Mortal: -874.28 |
| `south_screen_20260909` | south | **中间统计** | south_ppo: +1577.59；match_bc: -788.80 |

Copilot-Mortal3P 比较存在 MJAI 桥接后备动作和环境规则差异。JSON 内 `comparison.mortal_bridge` 保留决策总数与后备动作总数；结论仅适用于当时所用模型与模拟器条件。
