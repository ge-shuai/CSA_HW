# B4_weight_variants · 权重口径 × 社区

口径族（`SEEDS` = 31 个 ≥20，含文档 seed=42）：`binary` / `raw_weight` / `log1p` / `backbone_w{2,5,10,20,50}` / `disparity`。图一律取自 `csa_common.build_all()` 与 `csa_common.strong_backbone()`，开头 `C.assert_anchors()` 全 31 项通过。

## 0. 孤立点计数纠正（本单元存在的原因）

**既有结论**：

- `CSA_HW/选题分析_USTC网络拓扑/01_数据体检与预实验结论.md:111`
- `CSA_HW/选题分析_USTC网络拓扑/07_答辩提纲.md:84`（答辩提纲第 7 页，表格行 「只保留强边 w≥10 | 339 | 0.713 | 0.660」），源头代码 `CSA_HW/选题分析_USTC网络拓扑/06_代码/evidence.py:94-98`

两处都写：只保留 `weight≥10` 的强边 → **339 个社区，Q=0.713**。

**拆解**（`isolated_node_accounting.csv`，两个 seed=42 行）。边集与权重完全相同，只有节点插入顺序不同（CONVENTIONS §1 记录 Louvain 对插入顺序敏感）：`canonical` = `csa_common.strong_backbone`（键规范化成 (min,max)），`doc_order` = `evidence.py` 原样（按 CSV 首见方向累加）：

| 指标 | canonical `strong_backbone(Gu,10)` | 同学文档口径 `evidence.py` |
|---|---|---|
| Louvain 块数 `n_comms_total` | 338 | **339**（文档的 339） |
| 单点社区 `n_singleton_communities` | 307 | 307 |
| 其中零度孤立点 `singleton_isolated` | **307** | **307** |
| 单点社区中的非孤立点 | 0 | 0 |
| 骨架图内零度节点 | 307 | 307 |
| 骨架图内活跃节点 | 369 | 369 |
| **真实社区 `n_comms_nonsingleton`（主口径）** | **31** | **32** |
| Q | 0.71354 | 0.71302 |
| 闭合 `sing + nonsing == total` | `1` | `1` |
| 闭合 `所有单点都是零度孤立点` | `1` | `1` |

两条闭合校验都通过：单点社区数 = 零度孤立点数 = **307**，所以 339 个块里没有一个单点社区对应真实的连通结构。Q = 0.71302 与文档 0.713 一致。

**判决：`339` = 307 个零度孤立点 + 32 个真实社区。** 把 `n_comms_total` 当「社区数」会被孤立点抬高约 10 倍。

seed 波动（31 个 seed，`isolated_node_accounting.csv`）：

| 口径 | `n_comms_total` | 零度孤立点单点社区 | **`n_comms_nonsingleton`** |
|---|---|---|---|
| canonical | [336, 340]，中位 338 | 307（seed 间不变） | [29, 33]，中位 **31** |
| doc_order | [336, 341]，中位 337 | 307（seed 间不变） | [29, 34]，中位 **30** |

零度孤立点数在 31 个 seed 上是常数（由边集决定，与 seed 无关）。真实社区 ∈ [29, 33]（doc_order [29, 34]），中位 31：**「真实社区约 32 个」这一说法成立**（口径 = doc_order / seed=42 恰好 32；canonical 中位 31，二者在 seed 波动范围内）；但 **「339 个社区」不成立**，它是 339 个 Louvain 块，其中 307 个是孤立点。

可复现代码行（`pipeline/out/B4_weight_variants/run_b4.py`）：

```python
b  = C.build_all(); Gu = b['Gu']; links = b['links']
G  = C.strong_backbone(Gu, min_weight=10)          # canonical 边序
Gd = backbone_w10_doc_order(Gu, links, 10)          # evidence.py:94-98 边序
comms = nx.community.louvain_communities(Gd, weight='weight',
                                         resolution=1.0, seed=42, max_level=100)
iso_nodes   = {n for n, d in Gd.degree() if d == 0}                 # 307
n_total     = len(comms)                                            # 339
sing_iso    = sum(1 for c in comms if len(c) == 1 and next(iter(c)) in iso_nodes)
n_nonsing   = sum(1 for c in comms if len(c) > 1)                   # 32
```

## 1. 权重口径 × 社区（主口径 `n_comms_nonsingleton`，31 seed 分布）

| variant | 边数 | 总权重 | 活跃节点 | 孤立节点 | `n_comms_total` | **`n_comms_nonsingleton`** | Q mean(sd) | 最大社区 |
|---|---|---|---|---|---|---|---|---|
| binary | 2498 | 2498 | 676 | 0 | 12–13–15 | 12–**13**–15 | 0.4487 (0.0049) | 77–104–145 |
| raw_weight | 2498 | 63263 | 676 | 0 | 20–24–28 | 20–**24**–28 | 0.6903 (0.0017) | 66–97–134 |
| log1p | 2498 | 4716 | 676 | 0 | 13–16–17 | 13–**16**–17 | 0.5143 (0.0042) | 80–104–130 |
| backbone_w2 | 1778 | 62543 | 575 | 101 | 123–125–128 | 22–**24**–27 | 0.6947 (0.0012) | 50–70–94 |
| backbone_w5 | 879 | 60194 | 436 | 240 | 268–270–274 | 28–**30**–34 | 0.7098 (0.0014) | 40–59–82 |
| backbone_w10 | 675 | 58888 | 369 | 307 | 336–338–340 | 29–**31**–33 | 0.7163 (0.0016) | 32–44–66 |
| backbone_w20 | 506 | 56611 | 306 | 370 | 400–403–404 | 30–**33**–34 | 0.7230 (0.0014) | 31–35–45 |
| backbone_w50 | 402 | 53514 | 269 | 407 | 437–438–439 | 30–**31**–32 | 0.7381 (0.0023) | 28–39–39 |
| disparity | 89 | 20650 | 125 | 551 | 588–588–588 | 37–**37**–37 | 0.8703 (0.0000) | 14–14–14 |

参照锚点复核：`raw_weight` seed=42 → `n_comms_total`=25、Q=0.6904、最大社区=87，与 CONVENTIONS 锚点 25 / 0.6904 / 87 对齐。

社区规模分布：每口径中位 seed 的非单点社区规模（前 12 个）与规模直方图在 `weight_variant_summary.csv` 的 `size_median_top12` / `size_bins_nonsingleton` 列；全 seed 明细在 `weight_variant_per_seed.csv` 的 `comm_sizes` 列。

`binary` / `raw_weight` / `log1p` 三个全图口径的 `n_comms_total` 与 `n_comms_nonsingleton` 完全相等（例如 binary：13 vs 13）：Gu 内零度孤立点 = 0（锚点 `n_isolated_in_gu`）；骨架口径下二者显著分离（w≥10：338 vs 31），**CONVENTIONS §3 要求的一切社区数结论只看后者**。

## 2. NMI / ARI 矩阵（参照 = 原始权重）

全矩阵见 `nmi_matrix.csv` / `ari_matrix.csv`，长表 `variant_nmi_ari.csv`（含 seed 间 NMI sd，抽样网格 [0, 6, 12, 18, 24]×[0, 6, 12, 18, 24]）。NMI/ARI 一律走 `csa_common.partition_metrics()`（sklearn 口径，先取覆盖交集再重编号）。

| variant | NMI vs raw_weight | ARI vs raw_weight | NMI seed sd |
|---|---|---|---|
| raw_weight（参照，自比） | 1.0000 | 1.0000 | 0.0000 |
| binary | 0.4599 | 0.2443 | 0.0243 |
| log1p | 0.7316 | 0.5511 | 0.0270 |
| backbone_w2 | 0.7953 | 0.6056 | 0.0207 |
| backbone_w5 | 0.6901 | 0.3378 | 0.0158 |
| backbone_w10 | 0.6576 | 0.2433 | 0.0144 |
| backbone_w20 | 0.6425 | 0.1768 | 0.0110 |
| backbone_w50 | 0.6286 | 0.1406 | 0.0086 |
| disparity | 0.5942 | 0.0141 | 0.0079 |

读法：NMI 高说明两套口径把同一批节点分到同一组；ARI 高说明组间重合更接近集合级一致。所有比较都在同一 671 节点属性集上进行（CONVENTIONS §1）。

## 3. Q 在不同口径间不可比大小（显式声明）

五类口径给出的图互不相同：边集不同（骨架口径删边）、边权不同（1 / w / log(1+w)）、参与节点数不同（骨架下大量零度点）。模块度

```
Q = Σ_ij [ w_ij/m − γ·k_i·k_j/(2m) ] · δ(c_i, c_j),   m = Σ_ij w_ij
```

里的 `m`、`k_i` 与零模型 `γ_ij` 全部随图变化，因此**不同口径的 Q 不在同一尺度上**，不能比大小。表 1 的 Q 值只描述「该口径自身的划分质量」。

本模块的合法比较只有两类：**NMI / ARI**（同一节点集，`partition_metrics`）与**社区结构形状**（规模分布、最大社区、单点/孤立点构成）。

明确不做的推论：**「骨架 Q=0.71 > 原始 Q=0.69 ⇒ 骨架更好／更强」**。Q 变大只说明边集与零模型变了，不能读成结构质量提升。

## 4. disparity filter 骨架（Serrano, Boguñá & Vespignani 2009, PNAS 106:6483）

**已实现**，`build_disparity()` + `_alpha_*()`（`alpha = 0.05`）。对节点 i（度 k_i、强度 S_i），把 S_i 按 Dirichlet(1,…,1) 随机分给它的 k_i 条边 ⇒ `W_ij/S_i ~ Beta(1, k_i−1)`：

```
alpha_ij = Pr[X_ij >= w_ij] = 1 − F_Beta(1, k_i−1)(w_ij / S_i)
         = (1 − w_ij / S_i)^(k_i − 1)      # k_i >= 2；k_i == 1 记 1.0（无可检验性）
```

保留规则：**两端 alpha 同时 < 0.05**（保守交集，原文做法）。

四法一致性（`disparity_alpha_check.csv`，14 条抽样边 × 两端 = 28 行；数值积分 `np.trapz`，网格 2000；蒙特卡洛 20000 次）：

```
max |analytic − scipy|  = 6.00e-15
max |analytic − numeric| = 4.13e-07
max |analytic − MC|      = 5.89e-03
```

| 值 |
|---|
| 骨架边数 | **89**（原始图 2498 条的 3.6%） |
| 骨架总权重 | 20650（原始 32.6%） |
| 活跃节点 / 孤立节点 | 125 / 551 |
| `n_comms_total` min–median–max | 588–588–588 |
| **`n_comms_nonsingleton` min–median–max** | 37–**37**–37 |
| Q mean(sd) | 0.8703 (0.0000) |
| 最大社区 min–median–max | 14–14–14 |
| 与 w≥10 骨架共享边 | **89** 条 |
| 占 w≥10 骨架边集 | 0.1319 |
| 占 disparity 骨架边集 | 1.0000 |
| Jaccard | 0.1319 |
| **社区 NMI(disparity, w≥10)** | **0.8658** |
| 社区 ARI(disparity, w≥10) | 0.0570 |
| 社区 NMI(disparity, raw_weight) | 0.5942 |

读法：disparity 骨架（89 条）明显小于 w≥10 骨架（675 条），两者的边重叠 89 条（0.1% 的 w≥10 边）。两者社区结构在 NMI=0.8658 上仍属同族，但 disparity 把参与度低的边剔得更干净，孤立节点从 307 升到 551，真实社区数 31 → 37。两个骨架的 `n_comms_total` 都不能当社区数用（§0）。

## 5. 输出文件

| 文件 | 内容 |
|---|---|
| `weight_variant_summary.csv` | 每口径一行：边数/总权重/参与节点/双社区计数/Q 分布/规模分布 |
| `weight_variant_per_seed.csv` | 每口径 × 每 seed 的原始记录（含 `comm_sizes`） |
| `isolated_node_accounting.csv` | 孤立点计数专项：每骨架每 seed 的单点/孤立点拆解 + 两条闭合校验 |
| `nmi_matrix.csv` / `ari_matrix.csv` | 口径 × 口径 NMI / ARI 矩阵 |
| `variant_nmi_ari.csv` | 上述矩阵长表 + seed 间 NMI sd |
| `nmi_ari_vs_raw_weight.csv` | 各口径 vs 原始权重的 NMI/ARI |
| `disparity_backbone.csv` | 全图每条边的 k/S/alpha/p/keep 判决（`keep=1` 行 = 骨架边表） |
| `disparity_summary.csv` | 与 w≥10 骨架的边重叠 / NMI / ARI / Jaccard |
| `disparity_alpha_check.csv` | 解析 / scipy / 数值积分 / 蒙特卡洛 四法 alpha 一致性 |
| `RUN.txt` | 版本、seed、阈值、抽样网格、耗时 |

## 6. 局限

- 骨架口径下大量节点度为 0，`n_comms_total` 被孤立点显著抬高；社区数结论只看 `n_comms_nonsingleton`（CONVENTIONS §3）。
- disparity 用连续 Beta(1, k−1) 零模型；原文另有整数权重的离散（组合）版本，两者在小权重边上会有小幅差异。
- NMI/ARI 矩阵为控运行时间只在 5×5 seed 网格上取平均；表 1 的分布用全部 31 个 seed。
- 权重语义是「观察到的链接次数」，不是流量也不是重要性（CONVENTIONS §1）。
- 本次运行 35.7s。
