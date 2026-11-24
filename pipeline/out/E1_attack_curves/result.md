# E1_attack_curves · result.md

`F:` = `out/E1_attack_curves/` 下文件与列；`c:` = `run_e1.py` 行号（RUN.txt 记录 per-stage 墙钟与口径锚点）。
基准图：主分量 **674 节点 / 2497 边**；`E_fixed` 分母恒为 **453602** 有序对（`PairSet(n0=674)`）。
攻击方式：**静态**（排序一次，删 top-k），与 CONVENTIONS §6 一致；确定性策略每档 1 次，随机策略 **120 次独立重复**。

## 1. 结构说明（口径族）

- **6 个策略**：`random`（120 reps）、`degree`（无向度）、`betweenness`（无向介数）、`pagerank_dw`（有向加权，§5 主口径）、`pagerank_uu`（无权无向，§5 对照）、`kcore`。
- **32 个 f 档**：f=0（基线，step 0）+ 30 档 0.02…0.60（step 0.02）+ 额外 1 档 f=0.05（summary 用）；**逐步** GCC/分量数在 step 1…404 每一步都记录（`F:e1_curve_efficiency.csv` 共 **192 行 = 6×32**；`F:E1_summary.json:f_grid,summary_f,max_step=404`）。
- **4 个效率口径**全给：`E_full` / `E_gcc` / `E_fixed` + `avg_pathlen_gcc`，每列各有 `_sd`（`F:e1_curve_efficiency.csv` 20 列）。
- **reps 与 sd**：`random` 32 档全部 `n_reps=120`，7 个 `_sd` 列在 random 行 **32/32 非空**；其余 5 个确定性策略 `n_reps=1`、`_sd` 列 **全空**（符合 §6"确定性跑 1 次"）。**不要写成"sd 缺失"**——是确定性策略本就不该有 sd。
- 运行：`random_pool_wall=1528.4 s`、`random_rep_cpu_sum=20306.5 s`（120 reps × 169.2 s，14 进程）、`tiebreak_pool_wall=107.6 s`、**总墙钟 1822.7 s**（`F:E1_summary.json:timing_s,total_wall_s`；RUN.txt）。

## 2. 结论表：`E_fixed` 与随机基线（主口径 = `E_fixed`）

`F:e1_summary_f5.csv`（`dE_fixed_vs_random_pct` = `100×(E_fixed/E_fixed_random − 1)`，**负值 = 比随机更疼**）：

| 策略 | f=5% `E_fixed` | f=10% | f=20% | f=30% | f=50% | 相对随机@f=20% |
|---|---:|---:|---:|---:|---:|---:|
| `random`（基准，mean±sd, n=120） | 0.289845±0.005491 | 0.251777±0.007402 | 0.183332±0.007904 | 0.127717±0.008461 | 0.049995±0.005344 | — |
| `degree` | 0.144237 | 0.060286 | **0.002362** | 0.000513 | 0.000071 | **−98.71%** |
| `betweenness` | 0.141846 | 0.063958 | 0.004507 | 0.000376 | 0.000026 | −97.54% |
| `pagerank_dw`（§5 主口径） | 0.208347 | 0.157826 | **0.082381** | 0.026623 | 0.008445 | −55.06% |
| `pagerank_uu`（§5 对照，见 §4） | 0.139222 | 0.058773 | 0.001823 | 0.000323 | 0.000013 | −99.01% |
| `kcore` | 0.206494 | 0.100063 | 0.017434 | 0.002957 | 0.000122 | −90.49% |

同一表列的 GCC 存活分数（`F:e1_summary_f5.csv:gcc_pct`）：`degree` 77.15 / 57.57 / 4.45 / 1.04 / 0.74；`betweenness` 75.07 / 54.30 / 13.20 / 0.89 / 0.30；`pagerank_dw` 85.46 / 76.85 / 57.72 / 32.94 / 16.77；`pagerank_uu` 74.93 / 55.19 / 3.56 / 0.59 / 0.30；`kcore` 85.31 / 67.80 / 28.49 / 6.23 / 0.74；`random` 93.62 / 87.39 / 74.81 / 62.61 / 39.38。

## 3. 必须报的三组数

### 3.1 每策略 reps（`F:e1_curve_efficiency.csv:n_reps`）
`random` = **120**（32 档全部）；`degree` / `betweenness` / `pagerank_dw` / `pagerank_uu` / `kcore` = **1**（32 档全部）。总行数 192。

### 3.2 随机基线 mean±sd
逐档 mean±sd 全在 `F:e1_summary_f5.csv`（`E_fixed,E_fixed_sd`）。关键档：f=0.10 → 0.251777±0.007402；**f=0.24（k=162）→ 0.159113±0.008698（n=120）**；f=0.50 → 0.049995±0.005344。
自检：随机基线 `E_fixed` 在 **31 个检查点上 0 次上升**（分母固定，删除只会扣分；`F:E1_summary.json:random_E_fixed_monotone_violations=0,random_E_fixed_checkpoints=32`），首末 0.331469 → 0.026333。

### 3.3 删 162 全挂 vs 随机删 162（同一 k=162 的跨情景比较）
- 本单元随机删 162（f=0.24，step 162，n_remaining=512）：**`E_fixed`=0.159113±0.008698**，GCC=69.79%±1.72。
- 观测情景"删 162 全挂"来自 E5：`E_fixed`=**0.203942**（`out/E5_observed_scenarios/scenario_metrics.csv`，view=as_observed）。
- 对比：0.203942 / 0.159113 = **1.2817**，即**删 162 全挂的连通效率损失比随机删 162 小 28.2%**（z=+5.15，用本单元 120 reps 的 sd；E5 用自己 100 reps 基线 0.160974 得 −26.7%，两实现一致，引用时必须写明用的是哪一个随机基线）。
- 同 k=162 下 5 个攻击策略：`pagerank_dw` 0.048255、`kcore` 0.007219、`betweenness` 0.001471、`degree` 0.001082、`pagerank_uu` 0.000771（全部远低于随机与观测情景）。**三种口径都说明：本网络的"已发生失效"远不如针对性攻击致命。**

## 4. DISSENT：PageRank 口径异议（必读，不改口径）

`out/E1_attack_curves/DISSENT.md` 记录对 CONVENTIONS §5 表第二行的异议，证据在 `F:e1_pagerank_variant_diagnostic.csv`。按 DISSENT 的要求，本文件给**三列**（GCC 存活分数）：

| PageRank 变体 | f=10% | f=20% | f=30% | 与 §5 契约最大偏差 |
|---|---:|---:|---:|---:|
| **有向加权（主口径）** | **76.85** | **57.72** | **32.94** | 5.36 pp（契约 76.3/55.8/38.3） |
| 无权无向（§5 第二行字面口径） | 55.19 | 3.56 | 0.59 | **37.94 pp**（契约 66.2/41.5/12.5，复现不出） |
| 无向加权（推测的契约真实来源） | 63.50 | 35.01 | 10.39 | 2.7–6.5 pp |

要点（照 DISSENT 表述，不得自行改口径）：
1. **结论句只能是"PageRank 攻击远弱于度/介数 只在有向加权主口径下成立"**。主口径 f=20% 时 GCC 57.7%，度 4.5%、介数 13.2%；`E_fixed` 0.0824 vs 0.0024 / 0.0045。
2. 无权无向口径**不是温和对照**：它比度攻击还略狠（55.19/3.56/0.59 vs 57.57/4.45/1.04），且与度排名几乎重合（ρ_spearman=**0.977**、τ=**0.903**，Top-k 重合 0.91–0.94）——**它实质是"度攻击的另一种写法"**。
3. α 扫描（0.50/0.70/0.85/0.90/0.95）与节点集扫描（676/671）七档全部落在 52.8–55.9 / 3.4–6.7 / 0.6–0.9，与契约值无重叠 → 不是数值噪声。
4. 在裁决前，任何引用 PageRank 攻击曲线的地方都必须标口径；三个列**不可互换**。

## 5. 关键坑的显式回应

**（坑 1）效率只能从 `E_fixed` 读，且只做同 k 跨情景比。**
- `E_gcc` 会被分母收缩污染到荒谬的程度：`degree` 在 f=20% 时 `E_gcc`=**0.341893**（比基线 0.331469 **高** 3.1%），而同期 `E_fixed`=0.002362；`pagerank_uu` 在 f=30% 时 `E_gcc`=**0.722222**（残存图只剩 4 个点时，两点间 1/d 均值必然爆表）。**禁止**把这些上升写成"效率提高"。
- `E_full` 同理（`F:e1_curve_efficiency.csv:E_full` 在部分档位上升）。本文件的排序与结论一律用 `E_fixed`，比较一律同 k。
- 本单元**没有**"删 162 全挂"的实体情景曲线（那是 E5 的职责），只有策略曲线；§3.3 的跨情景数字来自 E5，已注明。

**（坑 2）平手敏感性会动曲线高低，不能把精细差异读成结构差异。**
`F:e1_tiebreak_sensitivity.csv`（同一排名值、不同同值组内顺序，288 行）：

| 策略 | 排序变体数 | 最大极差 | 出现在 | summary 5 档极差 |
|---|---:|---:|---|---|
| `kcore` | 5 | **7.57 pp** | step 202（f=0.30） | 2.67 / 3.71 / 4.01 / **7.57** / 0.30 pp |
| `degree` | 2 | **8.75 pp** | step 108（f=0.16） | 0 / 0.74 / 0.59 / 0.15 / 0.30 pp |
| `betweenness` | 2 | 0.30 pp | step 148（f=0.22） | 全部 0.00 pp |

读法：`degree` 与 `kcore` 各有几十个同值节点，**在 f=0.16 / f=0.30 附近，"度攻击比 k-core 狠"这类 1–8 pp 的差可能整段来自打破并列的顺序**。只有 `betweenness`（连续值、几乎无并列）的曲线可以按细粒度比较。正文写"曲线高低"时必须同时给这条不确定性。

**（坑 3）静态 vs 动态，以及跨实现复核（2026-11-23 更新）。**
- 本单元全部为**静态**（§6 主口径，与葛帅可比）。静态曲线现有两个独立实现互证：`E2_dynamic_attack` 重算的静态 GCC 曲线与本单元 `e1_curve_gcc_per_step_deterministic.csv` **逐点一致**（5 个策略 `max_abs_gcc_diff_nodes=0`，`E2_summary.json:static_vs_e1_crosscheck`）；`E3_percolation` 同样报 0（`E3_summary.json:cross_check_vs_E1.per_strategy`，worst=0）。
- 随机基线也被 E3 复核：E3 用同一 seed 公式 `np.random.default_rng([SEED, rep])` 与同一份 sorted 节点表，**前 120 次重复的均值与本单元 120 次均值最大差 3.3e-05 个节点**；E3 的 **200 次**均值与本单元 120 次均值差 1.2433 个节点，这是重复数不同，**不是建图/口径漂移**（`E3_summary.json:cross_check_vs_E1`）。
- 动态攻击（每删一步重算）的结论在 E2，不在本单元：E2 报最大 GCC 差 **37.69 pp @ f=0.12**（`betweenness`，静态 46.29% vs 动态 8.61%）、`degree` 18.69 pp @ f=0.16（`E2_summary.json:headline_gap`）。**本文件只给静态曲线，不得据此外推动态结论。**

**（坑 4）因果措辞。**
只写"情景模拟显示/在同等 k 下…"，不得写"某策略导致网络崩溃"。

## 6. 可复现与交付

- `RUN.txt`：`PYTHONHASHSEED=0 python out/E1_attack_curves/run_e1.py`；python 3.12.13 / networkx 3.6.1 / pandas 3.0.2 / numpy 2.4.4；`SEED=20261113`；random 每 rep seed = `np.random.default_rng([20261113, rep])`；14 进程；总墙钟 **1822.7 s**。
- anchors：31/31 PASS；baseline 674/2497 与 `E_fixed` 0.331469 与 §2.3 表一致（RUN.txt 注明 `E_full`=0.3295 那个数是 676 图口径，不可混用）。
- 交付：`e1_curve_efficiency.csv`(192)、`e1_curve_gcc_per_step_deterministic.csv`、`e1_curve_gcc_per_step_random.csv`、`e1_summary_f5.csv`(30)、`e1_tiebreak_sensitivity.csv`(288)、`e1_pagerank_variant_diagnostic.csv`(20)、`E1_summary.json`、`DISSENT.md`、`RUN.txt`、`E1_RANDOM_DONE`（120-reps 成品哨兵，2026-11-23 13:55）。
- 无 `result.md` 之外的空缺；**PROGRESS §5 第 3 条"random 只有 n_reps=2"已过期**，以本文件 120 reps 为准。
