# I1_number_consistency · 跨模块数值一致性审计

> 审计范围：`pipeline/out/**/*.csv`（315 个）与全部 `result.md`（45 个）。只读，不改任何他人文件。
> 数据来源：2026-11-23 20:11 用 `srv.py getdir` 整目录同步到本地 staging（540 文件 / 26 MB）。
> 机器可读清单见 `I1_conflicts.csv`（23 行）。下表的每个数字都带 `文件:行`。

## 0. 总判决

- **真冲突：0 条。**
- 口径不同（合法，但引用时必须带口径标签）：7 条。
- 真·口径缺陷（不是冲突、但需要改表述/改契约表）：2 条（`E3 result.md:10/:41` 的 0.4080 vs 0.4095 标注；`CONVENTIONS §2.3` 表混写 676/674 两个图的 `E_full`）。
- 12 项必核指标**全部逐条给出实测值，无一漂移**。唯一一处"看起来不一致"的 1.9e-3（随机删 162 的 `E_fixed`）判为**独立抽样波动**，理由见 §1.12。

## 1. 12 项必核指标

| # | 指标 | 模块 A（值 / 出处） | 模块 B（值 / 出处） | 绝对差 | 判定 |
|---:|---|---|---|---:|---|
| 1 | 671 节点 | `D1_health_gradient` 671（`result.md:3`） | `G1_risk_matrix` 671 校正前（`result.md:23`） | 0 | 一致 |
| 2 | 676 含悬空 | `B1_louvain_stability` 676（`result.md:4`） | `G2_single_point_failure` 676（`result.md:4`） | 0 | 一致 |
| 3 | 674 GCC | `E1_attack_curves` 674（`result.md:4`） | `A2_small_world` 674（`result.md:8`） | 0 | 一致 |
| 4 | 2497 边 | `E1_attack_curves` 2497（`result.md:4`） | `E4_edge_attacks` 2497（`result.md:10`） | 0 | 一致 |
| 5 | 2498 边 | `B1_louvain_stability` 2498 全图 676（`result.md:4`） | `C1_centrality_matrix` 2498（`result.md:16`） | 0 | 一致 |
| 6 | 2788 有向边 | `A4_directed_structure` 2788（`result.md:4`） | `D2_censoring_impact` 2788（`result.md:13`） | 0 | 一致 |
| 7 | 453602 有序对 | `E1_attack_curves`（`result.md:4`） | `E2_dynamic_attack`（`result.md:20`） | 0 | 一致 |
| 8 | E_fixed 基线 0.331469 | `H4_figs_pipeline` 0.331469（`result.md:33`） | `E5_observed_scenarios` 0.331469（`scenario_metrics.csv:5`） | 0 | 一致（G2 同值见 `result.md:4`） |
| 9 | 社区数 25 / n_comms_nonsingleton | `B2_algorithm_contrast` 25 / 非单点 25（`result.md:60`） | `B6_community_semantics` 25 / 25（`result.md:9`）；`C5_functional_roles` 25 / 25（`result.md:18`） | 0 | 一致 |
| 10 | Q = 0.6904 | `B2_algorithm_contrast` 0.6904（`result.md:60`） | `B6_community_semantics` 0.690434（`result.md:9`） | 3.4e-05 | 一致（四舍五入） |
| 11 | 真 129 / 假 32 / 待定 1 | `A4_directed_structure` 129/32/1（`result.md:155`） | `H4_figs_pipeline` 129+32+1=162（`result.md:44`） | 0 | 一致 |
| 12 | 枢纽 6 个 / 505 边 | `B5_hub_removal` 6 / 505（`result.md:36`、`result.md:25`） | `C1_centrality_matrix` 6/6（`result.md:144`）；`C3_centrality_robustness` 505、删后 1993 边（`result.md:11`） | 0 | 一致（2498−505=1993 闭合） |

补充核验（支撑第 1 项"671 无零度点"）：`B4_weight_variants` 报 `Gu` 内零度孤立点 = 0（`result.md:74`），与 CONVENTIONS §1（`CONVENTIONS.md:21`）一致。

第 4/5 项不是矛盾：2497 是 674 主分量，2498 是 676 全图，差 1 条边正是 `bm—l.bm` 那个 2 点小分量。`A2_small_world` 还给出第三个值 2493（671 诱导子图，`result.md:29`），三者按点集单调，符合规范。

### 1.12 随机删 162 的 E_fixed：0.159113 vs 0.160974

| 侧 | 值 | n | sd | 出处 |
|---|---:|---:|---:|---|
| E1 | 0.159113 | 120 | 0.008698 | `E1_attack_curves/e1_curve_efficiency.csv:175`；`result.md:40` |
| E5 | 0.160974 | 100 | 0.008597 | `E5_observed_scenarios/scenario_metrics.csv:14`；`result.md:13` |

- 绝对差 **1.861e-03**。
- 各自标准误：E1 = 0.008698/√120 = **7.94e-04**；E5 = 0.008597/√100 = **8.60e-04**。
- 差的标准误 SE_diff = √(7.94e-04² + 8.60e-04²) = **1.17e-03** → 差 = **1.59 SE**，在两次独立实现的正常区间内（|z|<1.96）。
- 两侧 sd 几乎相同（0.00870 / 0.00860），说明波动来自"抽到哪 162 个点"，不是实现差异。
- `E5_observed_scenarios/result.md:71` 已自述这是"两次独立随机实现的正常波动，不是矛盾"，并规定若审计要求单一数以 E1 的 120 reps 为准。
- **判定：独立抽样波动，不是真冲突。** 但两模块存在"同一情景两个数"的可引用风险，建议 lead 在总表统一取 E1 的 0.159113（n=120）并注明 E5 的 0.160974 是另一次 100 reps。

## 2. 两个新单元的自洽性

### 2.1 E3 的 p_c 与 E1 曲线在同一 k 上是否一致 —— 一致

- E3 报 `p_c(50%) = 0.1128 / k=76`（`E3_percolation/result.md:36`，`e3_perc_thresholds.csv` 对应行）。
- E1 的确定性 GCC 曲线在同一 k 上：`k=75 → GCC=337 = 50.000%`（**未**跌破）、`k=76 → GCC=328 = 48.665%`（跌破），`e1_curve_gcc_per_step_deterministic.csv:480-481`。
- 于是 p_c(50%) = 76/674 = **0.112759**，与 E3 的 0.1128 差 **4.1e-05**（纯四舍五入）。
- E3 亦自述交叉校验：5 条静态确定性曲线与 E1 **逐点完全一致、最大差 0 个节点**（`result.md:116-117`）；随机曲线取同样 120 reps 时最大差 3.3e-05 个节点（`result.md:119-121`）。**该自述与我实测的 0.112759 相互印证。**
- 附带发现（E3 内部，不是跨模块）：§2 主表把 random 的 `p_c(50%)` 写成 **0.408012**（`e3_perc_thresholds.csv:2`，k=275）却标为"200 次均值"（`result.md:41`），而真正的 200 次 p_c 均值是 **0.409496**（`e3_random_pc_distribution.csv:2`，k_mean=276.0；`result.md:49`）。两者差 **0.001484 = 恰好 1 个网格步（1/674）**：前者是"均值 GCC 曲线首次跌破"，后者是"阈值的均值"，是两个估计量。
  - `result.md:10` 更把两种估计量写进同一句："需要删掉 **40.8%** 的节点（200 次重复，均值 **0.4095**，sd 0.0157）" —— 40.8% 与 0.4095 相差 1 个网格步。
  - **判定：口径不同，但表头"200 次均值"标注错误**。建议改法：§2 表该行改标"均值 GCC 曲线首次跌破"或直接把数值换成 0.4095；`result.md:10` 的 40.8% 改为 41.0%（或统一用 0.4095）。
- 另：E3 `k@25%` 与 §3 分布一致（0.6365，`e3_random_pc_distribution.csv:3` mean=0.636550），无此问题。

### 2.2 E6 三个分母有没有被跨分母相减 —— 没有

- 三个分母：full = 674×673 = **453602**、disparity 骨架 = 56×55 = **3080**、naive w≥10 骨架 = 367×366 = **134322**（`E6_backbone_resilience/result.md:77`；`RUN.txt:25`）。
- 三者在长表里是**列**（`attack_curves_backbone_vs_full.csv:1` 的 `backbone_denominator`），按 `graph` 取值实测为：`full→453602`、`backbone→3080`、`naive_w10→134322`，每行只带自己那一个分母，**没有被混进同一列的比较**。
- 跨图比较的量只有**图内归一化比值**：`E_fixed_ratio` / `gcc_ratio`（`attack_curves_backbone_vs_full.csv:1`）与 `Efix_ratio_at10/30/50`（`breakdown_points_backbone_vs_full.csv:1`），都是"该图第 k 步 / 该图第 0 步"，分母在比值里约掉，跨图可比合法。
- `dynamic_vs_static_betweenness.csv:1` 有 `E_fixed_static_minus_dynamic`，但它是**同一张图**的静态减动态（graph ∈ {full, backbone, naive_w10} 各自内部），分母相同，不是跨分母。
- E6 自己已声明："三个 `E_fixed` 分母不同……本单元所有跨图比较只比「删掉本图的比例」与「相对自身基线的比值」，没有任何跨分母的数值大小比较"（`result.md:145`）。
- **判定：合规。**

## 3. 口径不同（不算违规，但引用必须带标签）

| 项 | A | B | 说明 |
|---|---|---|---|
| 社区数 25 vs 32 | `B2_algorithm_contrast` 25（`result.md:60`） | `H1_figs_dist_community` 32 共识划分（`result.md:42`） | 单 seed Louvain ≠ 250-seed 共识划分。H1 已在 `result.md:46` 自行声明并要求引用时说明来源 |
| 2497 vs 2498 | 674 主分量 | 676 全图 | 规范允许，差 1 条边 |
| 2493 | 671 诱导子图（`A2_small_world/result.md:29`） | 同上 | 第三种点集 |
| Q 0.6903（中位）vs 0.6904（seed=42） | `B1_louvain_stability/result.md:31` | `B5_hub_removal/result.md:50` | 同一 50-seed 分布的不同汇总；B1 的区间 [0.6842, 0.6932] 与 B5 的区间逐位相同 |
| 社区数分布宽度 20–28 vs 23–26 | `B1_louvain_stability`（250 seed，`result.md:28`） | CONVENTIONS §3（`CONVENTIONS.md:96`，6 seed 抽样） | 后者是前者的子样本，不是矛盾 |
| E_full 基线 0.331469 vs 0.3295 | `E5`（674 口径） | `CONVENTIONS.md:69`（676 口径） | **CONVENTIONS 表自身把 676 图的 E_full 与 674 图的 E_gcc/E_fixed 写在同一行**；`E1_attack_curves/result.md:88` 已指出"不可混用" |
| E_full 删162 0.353582 vs 0.3508 | `E5/scenario_metrics.csv:14` | `CONVENTIONS.md:70` | 同源问题，契约表陈旧，建议更新 CONVENTIONS |

## 4. 无法核实的项（说明，不是豁免）

- 目录里有 `.npy`/`.npz` 缓存与 `_cache_*.csv`（如 `E5_observed_scenarios/_cache_random_controls.csv`）。缓存文件与正式 CSV 的数值**不一致**是设计如此（缓存是逐 rep 原始行），本审计不把缓存当结论来源，只在正式 CSV 缺列时交叉比对。
- 按要求**未跑任何分析脚本**，所有数字来自 CSV/`result.md` 直接读数；两处需要重算的地方（76/674、SE_diff）是手工算术，已在正文写出算式。

## 5. 产物

- `I1_conflicts.csv` — 23 行，UTF-8-sig
- `result.md` — 本文件
- `RUN.txt` — 运行记录
