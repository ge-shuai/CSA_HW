# B2 · 算法对照矩阵（B2_algorithm_contrast）

- 口径（CONVENTIONS §1/§3/§8）：图一律 `csa_common.build_all()`；骨架 `strong_backbone(Gu, min_weight=10)`；
  相似度只用 `partition_metrics()`（覆盖交集上重编号）；社区数同报 `n_comms_total` / `n_comms_nonsingleton`；
  节点口径 **671 有属性（主分析）** 与 **676 全图（锚点/拓扑）** 两列并报。
- 锚点：`C.assert_anchors()` **31/31 PASS**。图：Gu 676n/2498e、
  Gb 676n/2498e、w>=10 骨架 676n/675e。
- 算法数：**10**（加权 Louvain、无权 Louvain、标签传播×4 变体、贪心模块度×2、KL 二分、Laplace 谱二分）。
- 全部数字可在 `algorithm_summary.csv` / `nmi_matrix.csv` / `ari_matrix.csv` / `lp_giant_community_reps.csv` /
  `lp_variant_stats.csv` / `key_numbers.json` 复算。总耗时 51.02s。

## 0. 一句话判决

**「557 节点巨型社区」是算法产物，不是数据事实。** 可核对的三组数字：

1. **权重盲的标签传播**（同学用的那一族）在 676 全图换节点遍历顺序重复 50 次：
   最大社区（671 口径）median **605**，范围 **357–622**，
   其中 **48/50**（96%）出现 ≥500 的巨型社区，
   与同学 557 集合的 Jaccard median **0.8835**（48/50 次 ≥0.80）。
   → 巨型社区**能复现**，但规模本身摆动 357–622（±8%），
   且**恰好等于 557 的次数只有 0/50**。
2. **权重一旦进入算法，巨型社区消失**：
   `asyn_lpa_communities(Gu, weight="weight")` 50 次，≥500 出现 **46/50**，
   最大社区 median **604**（范围 282–629）；
   自实现的加权投票 LPA `ext_weightedLPA_Gu` 50 次：**50/50**，median **634**；
   同一 nx 权重盲实现在 **w>=10 骨架**（权重经筛边进入）上 50 次：**0/50**，median **44**。
3. **优化模块度的两类算法从未给出 500+ 社区**：加权 Louvain 最大 87、
   无权 Louvain 最大 76、贪心模块度加权 106、
   无权 166；`algorithm_summary.csv` 里 `n_comms_ge_500` 列对全部 10 个参考划分都 = 0。
   同学的 557 巨型社区内部再跑一次加权 Louvain：557n/2155e →
   **17 个社区（非单点 17），Q=0.6742，最大 127** —— 它装得下整套模块结构。

对照基线（判决的第四根柱子）：**度保持零模型**（30 次独立重复，每轮 fresh copy 再交换）上跑
**同一个权重盲 LPA**，最大社区 mean **646.1 ± 9.96**，
观测图上同一算法给 **417**（676 口径）。
→ 结构被打散、度序列不变的图上，权重盲 LPA 给出的社区**更大**。巨型社区随「算法是否读权重」而消失、
随「结构被打散」而膨胀：它不是这组超链接的社区结构。

## 1. 算法 × 指标（参考划分，`algorithm_summary.csv`）

`n_comms_total` / `n_comms_nonsingleton`（676 全图）与 671 口径同报；`largest_671` 为主分析口径最大社区。

| algorithm | graph | n_comms_total | n_comms_nonsingleton | n_comms_nonsingleton_671 | largest_671 | largest_share_of_671_pct | n_comms_ge_500 | size_median | size_max | Q_weighted_Gu | Q_binary_Gb | coverage_weighted_Gu | coverage_binary_Gb | runtime_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| louvain_weighted | Gu(676,加权) | 25 | 25 | 25 | 87 | 12.97 | 0 | 12.0 | 87 | 0.6904 | 0.3577 | 0.7828 | 0.446 | 0.069 |
| louvain_binary | Gb(676,无权) | 14 | 14 | 14 | 76 | 11.33 | 0 | 53.5 | 76 | 0.5448 | 0.4504 | 0.6576 | 0.5396 | 0.064 |
| label_prop_binary | Gb(676,无权) | 17 | 17 | 17 | 413 | 61.55 | 0 | 5.0 | 413 | 0.2537 | 0.2798 | 0.8905 | 0.803 | 0.037 |
| label_prop_on_weighted | Gu(676,加权) | 17 | 17 | 17 | 413 | 61.55 | 0 | 5.0 | 413 | 0.2537 | 0.2798 | 0.8905 | 0.803 | 0.036 |
| label_prop_backbone_w10 | 骨架 w>=10(675e) | 379 | 72 | 71 | 57 | 8.49 | 0 | 2.0 | 57 | 0.5013 | 0.144 | 0.6039 | 0.1934 | 0.016 |
| label_prop_async_weighted | Gu(676,加权) | 78 | 78 | 78 | 91 | 13.56 | 0 | 4.0 | 91 | 0.6262 | 0.2992 | 0.7033 | 0.3487 | 0.028 |
| greedy_modularity_weighted | Gu(676,加权) | 26 | 26 | 26 | 106 | 15.8 | 0 | 11.5 | 106 | 0.6934 | 0.3568 | 0.7944 | 0.4548 | 0.481 |
| greedy_modularity_binary | Gb(676,无权) | 13 | 13 | 13 | 166 | 24.74 | 0 | 21.0 | 166 | 0.3491 | 0.4153 | 0.5737 | 0.6289 | 0.697 |
| kernighan_lin_bisection | Gu(676,加权) | 2 | 2 | 2 | 337 | 50.22 | 0 | 335.5 | 337 | 0.3836 | 0.1437 | 0.8899 | 0.6733 | 0.037 |
| spectral_bisection_fiedler | Gb(676,无权) | 2 | 2 | 2 | 336 | 50.07 | 0 | 335.5 | 336 | -0.0035 | -0.0138 | 0.842 | 0.7902 | 0.131 |

规模分布（671 口径，非单点社区）：`size_min / size_q1 / size_median / size_mean / size_q3 / size_max` 见 CSV；
表中 `size_median / size_max` 两列即每算法的分布摘要，`sizes_top8_671` 存前 8 大社区规模。

**权重贡献多少**（同一 seed=42，只把图换成无权版）：加权 Louvain
**25 社区 / 非单点 25 / 最大 87 / Q(加权) 0.6904**
vs 无权 Louvain **14 社区 / 非单点 14 / 最大 76 / Q(无权) 0.4504**；
两者互相的 NMI **0.4492** / ARI **0.2407**。
边集完全相同、权重只作为投票/合并的代价进入 —— 社区数差 11 个，
Q 差 0.2400（**不可直接比大小**，见 §4）。

## 2. NMI / ARI 交叉表（10 × 10）

### 2.1 NMI

| 算法 A \ B | louvain weighted | louvain binary | label prop binary | label prop on weighted | label prop backbone w10 | label prop async weighted | greedy modularity weighted | greedy modularity binary | kernighan lin bisection | spectral bisection fiedler |
|---|---|---|---|---|---|---|---|---|---|---|
| **louvain_weighted** | — | 0.449 | 0.322 | 0.322 | 0.588 | 0.737 | 0.789 | 0.412 | 0.138 | 0.024 |
| **louvain_binary** | 0.449 | — | 0.349 | 0.349 | 0.521 | 0.547 | 0.466 | 0.500 | 0.057 | 0.013 |
| **label_prop_binary** | 0.322 | 0.349 | — | 1.000 | 0.327 | 0.364 | 0.350 | 0.425 | 0.087 | 0.031 |
| **label_prop_on_weighted** | 0.322 | 0.349 | 1.000 | — | 0.327 | 0.364 | 0.350 | 0.425 | 0.087 | 0.031 |
| **label_prop_backbone_w10** | 0.588 | 0.521 | 0.327 | 0.327 | — | 0.736 | 0.583 | 0.442 | 0.182 | 0.149 |
| **label_prop_async_weighted** | 0.737 | 0.547 | 0.364 | 0.364 | 0.736 | — | 0.742 | 0.477 | 0.182 | 0.043 |
| **greedy_modularity_weighted** | 0.789 | 0.466 | 0.350 | 0.350 | 0.583 | 0.742 | — | 0.396 | 0.141 | 0.020 |
| **greedy_modularity_binary** | 0.412 | 0.500 | 0.425 | 0.425 | 0.442 | 0.477 | 0.396 | — | 0.066 | 0.017 |
| **kernighan_lin_bisection** | 0.138 | 0.057 | 0.087 | 0.087 | 0.182 | 0.182 | 0.141 | 0.066 | — | 0.013 |
| **spectral_bisection_fiedler** | 0.024 | 0.013 | 0.031 | 0.031 | 0.149 | 0.043 | 0.020 | 0.017 | 0.013 | — |

### 2.2 ARI

| 算法 A \ B | louvain weighted | louvain binary | label prop binary | label prop on weighted | label prop backbone w10 | label prop async weighted | greedy modularity weighted | greedy modularity binary | kernighan lin bisection | spectral bisection fiedler |
|---|---|---|---|---|---|---|---|---|---|---|
| **louvain_weighted** | — | 0.241 | 0.053 | 0.053 | 0.072 | 0.333 | 0.614 | 0.198 | 0.049 | 0.003 |
| **louvain_binary** | 0.241 | — | 0.080 | 0.080 | 0.052 | 0.219 | 0.249 | 0.283 | 0.023 | 0.003 |
| **label_prop_binary** | 0.053 | 0.080 | — | 1.000 | 0.013 | 0.055 | 0.088 | 0.209 | 0.012 | 0.003 |
| **label_prop_on_weighted** | 0.053 | 0.080 | 1.000 | — | 0.013 | 0.055 | 0.088 | 0.209 | 0.012 | 0.003 |
| **label_prop_backbone_w10** | 0.072 | 0.052 | 0.013 | 0.013 | — | 0.131 | 0.065 | 0.025 | 0.013 | 0.007 |
| **label_prop_async_weighted** | 0.333 | 0.219 | 0.055 | 0.055 | 0.131 | — | 0.391 | 0.117 | 0.028 | 0.004 |
| **greedy_modularity_weighted** | 0.614 | 0.249 | 0.088 | 0.088 | 0.065 | 0.391 | — | 0.194 | 0.047 | 0.002 |
| **greedy_modularity_binary** | 0.198 | 0.283 | 0.209 | 0.209 | 0.025 | 0.117 | 0.194 | — | 0.018 | 0.007 |
| **kernighan_lin_bisection** | 0.049 | 0.023 | 0.012 | 0.012 | 0.013 | 0.028 | 0.047 | 0.018 | — | 0.017 |
| **spectral_bisection_fiedler** | 0.003 | 0.003 | 0.003 | 0.003 | 0.007 | 0.004 | 0.002 | 0.007 | 0.017 | — |

### 2.3 怎么读这两张表（**必须先给天花板**）

- **同算法换 seed 的 NMI = 天花板**：加权 Louvain 6 个 seed 两两
  **0.8955**；无权 Louvain **0.648**。
  跨算法 NMI 只有对着这个天花板才有意义。
- 跨算法 NMI：min **0.013** / median **0.3492** / max **1.0**。
- 三个关键读数：
  1. `louvain_weighted` vs `label_prop_binary` NMI **0.3224** / ARI **0.0533**
     ——远低于同算法天花板 0.8955：**两套结论的差异远大于 seed 噪声**，
     矛盾真实存在，不能靠「换一个 seed 重跑」抹平。
  2. `louvain_weighted` vs `greedy_modularity_weighted` NMI **0.7888**
     ——两个独立最大化模块度的算法彼此**高度一致**（接近天花板）。
  3. `label_prop_binary` vs `label_prop_async_weighted` NMI **0.3637**、
     vs `label_prop_backbone_w10` NMI **0.3267**
     ——同一「标签传播」名字下，读不读权重给出**互不相干的划分**。
- KL / 谱二分那两行只有 2 个社区（块大小由「各半」约束给定，不由数据决定），它们与一切的多值 NMI 天然偏低，
  不能当成「它们不同意模块度族」的证据；放进来只为说明**目标函数族**的差别。

## 3. 标签传播重复实验（`lp_giant_community_reps.csv`，每次重复一行）

判据要求 ≥30 次；每种变体跑 50 或 50 次（canonical 顺序那次单独 1 次）。
随机源写明：nx 3.6.1 的 `label_propagation_communities` **没有随机参数**，初始标签 = `enumerate(G)` 的节点插入顺序、
更新顺序 = 按颜色分组 → 唯一可复现的随机源是把节点插入顺序置换掉（边集、边权、边次序不变，`same_structure` 已逐次校验）；
`asyn_lpa_communities` 与自实现加权 LPA 用 `seed = SEED + rep`。

| variant | n_reps | objective_graph | reads_weight | n_comms_total_median | n_comms_nonsingleton_671_median | largest_671_min | largest_671_median | largest_671_max | n_ge_500 | frac_ge_500 | n_exact_557 | jaccard_median | nmi_vs_louvain_median | Q_weighted_own_graph_median | coverage_own_graph_median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nx_LPA_binary_canonical_order | 1 | Gb(676,无权) | False | 17.0 | 17.0 | 413 | 413.0 | 413 | 0 | 0.0 | 0 | 0.6385 | 0.3224 | 0.2798 | 0.803 |
| nx_LPA_binary_nodeorder_shuffled | 50 | Gb(676,无权) | False | 13.0 | 13.0 | 357 | 605.0 | 622 | 48 | 0.96 | 0 | 0.8835 | 0.2058 | 0.0549 | 0.9756 |
| nx_LPA_weightedgraph_shuffled | 50 | Gu(676,加权) | False | 13.0 | 13.0 | 357 | 605.0 | 622 | 48 | 0.96 | 0 | 0.8835 | 0.2058 | 0.0477 | 0.9887 |
| nx_LPA_attr671_shuffled | 50 | Gu 诱导 671(加权) | False | 13.0 | 13.0 | 202 | 603.5 | 616 | 46 | 0.92 | 0 | 0.8847 | 0.208 | 0.0479 | 0.9872 |
| nx_LPA_backbone_w10_shuffled | 50 | 骨架 w>=10 | False | 375.0 | 67.0 | 24 | 43.5 | 173 | 0 | 0.0 | 0 | 0.0771 | 0.6038 | 0.5821 | 0.6955 |
| nx_asynLPA_weighted_shuffled | 50 | Gu(676,加权) | True | 13.0 | 13.0 | 282 | 604.0 | 629 | 46 | 0.92 | 0 | 0.8847 | 0.2073 | 0.0479 | 0.9886 |
| nx_asynLPA_unweighted_shuffled | 50 | Gu(676,加权) | False | 13.0 | 13.0 | 282 | 604.0 | 629 | 46 | 0.92 | 0 | 0.8847 | 0.2073 | 0.0479 | 0.9886 |
| ext_weightedLPA_Gu | 50 | Gu(676,加权) | True | 6.0 | 6.0 | 626 | 634.0 | 655 | 50 | 1.0 | 0 | 0.8764 | 0.1482 | 0.0423 | 0.9958 |

> `Q_weighted_own_graph_median` / `coverage_own_graph_median` 的目标函数都在**本变体自己的图**（`objective_graph` 列）上算：
> 671 诱导图、w≥10 骨架与 676 全图的 Q **不可跨行比较**，只能在同一 `objective_graph` 内比。

- **是否只在无权图上出现？** 更准确的事实是：**nx 的 `label_propagation_communities` 根本没有 weight 参数**，
  它按邻居**计数**投票。把加权图 `Gu` 直接喂进去，50/50 次与无权图 `Gb` 在同一置换下给出
  **逐节点完全相同的划分**（`key_numbers.json: nx_blind_identical_Gu_Gb`）。所以「巨型社区只在无权图出现」这句话不成立：
  它在两张图上出现，理由都是**算法没有读权重**。
- canonical 插入顺序（= 本项目建图口径，`edge_order`）只给 1 次：17 社区、最大 **413**（671 口径）、
  与同学 557 集合 Jaccard **0.6385** → 同学那个具体数值对应的是**另一套遍历顺序**，
  不是本流水线图上的确定输出。
- 671 主分析口径（`nx_LPA_attr671_shuffled`）：median **604**，≥500 **46/50**。
  巨型社区与节点口径几乎无关（671 口径 vs 676 口径都在同一量级），
  **同学「82.4%」的分母用的是 676**（557/676=82.4%；换成主分析口径 671 是 83.0%）——两个口径都要写明。

## 4. 分辨率 / 模块化不可比警告（判据第 4 条）

**不同算法的 Q 不能直接比大小。** 三件事同时成立：

1. **目标函数不同**：Louvain 与贪心模块度最大化 `Q(γ=1)`；标签传播**不优化任何目标函数**（多数投票收敛即停）；
   KL 二分最小化割、谱二分最小化规范化割，块大小由「各半」的约束给定。后三类划分的 Q 是**事后打分**，不是它们的优化结果。
2. **γ（分辨率）不同也不可比**：`Q(γ)` 的零模型项随 γ 缩放（见 B3 单元），换 γ 等于换打分尺。本表全部在 γ=1。
3. **加权 / 无权两套零模型**：同一划分在 `Gu`（按边权）与 `Gb`（按边数）上的 Q 数值差得很大
   （如 `louvain_weighted` 划分：Q(加权 Gu)=0.6904，同一划分 Q(无权 Gb)=0.3577），
   这只反映**分母（2m）换了**，不反映好坏。

因此本单元给每个划分**同时报两套目标函数**（同一划分、同一份数据、两种打分）：

| 划分（参考运行） | Q(加权 Gu) | Q(无权 Gb) | 覆盖率(加权) | 覆盖率(无权) | 该算法自己优化的目标 |
|---|---:|---:|---:|---:|---|
| `louvain_weighted` | 0.6904 | 0.3577 | 0.7828 | 0.446 | 模块度 Q（加权, γ=1） |
| `louvain_binary` | 0.5448 | 0.4504 | 0.6576 | 0.5396 | 模块度 Q（无权） |
| `label_prop_binary` | 0.2537 | 0.2798 | 0.8905 | 0.803 | 无（多数投票） |
| `label_prop_async_weighted` | 0.6262 | 0.2992 | 0.7033 | 0.3487 | 无（加权投票） |
| `greedy_modularity_weighted` | 0.6934 | 0.3568 | 0.7944 | 0.4548 | 模块度 Q（加权，贪心） |
| `kernighan_lin_bisection` | 0.3836 | 0.1437 | 0.8899 | 0.6733 | 割（平衡二分） |
| `spectral_bisection_fiedler` | -0.0035 | -0.0138 | 0.842 | 0.7902 | 规范化割 |

两张表**方向相反**：标签传播的**覆盖率**最高（加权 0.8905，
即把 89.0% 的权重留在社区内），而它的 **Q 最低**（加权 0.2537）。
覆盖率不含零模型项，**不惩罚「把节点并进一个大社区」**（单点社区也不受罚），所以它天然偏向巨型社区；
Q 有度序列零模型惩罚，天然偏向中等规模的多模块结构。
**所以：不能写「Louvain 的 Q 比标签传播高 ⇒ Louvain 更好」。** 本单元的判决不建立在 Q 的大小上，
建立在 §0 的三组可复算数字 + §5 的零模型 + §6 的巨型社区内部结构上。

## 5. 零模型对照（度序列保持、结构打散）

`csa_common.degree_preserving_null`，nswap=10m、max_tries=100·nswap，每轮从原图 fresh copy 再交换（CONVENTIONS §4）：

| 统计量 | 观测图 | 零模型（30 次独立重复） |
|---|---:|---:|
| 权重盲 LPA 最大社区（676 口径） | 417 | mean 646.1 ± sd 9.96 |
| 加权 LPA（自实现）最大社区 | 见 `lp_giant_community_reps.csv`（median 634，671 口径） | mean 668.87 ± sd 3.63 |

第一行是判决的核心：**把结构打散、只留度序列，权重盲 LPA 的巨型社区反而比观测图上更大**。
这说明该指标在这个算法上主要测的是「图有多连通 + 算法有多粗」，不是「有没有院系式社区」。
局限：Viger–Lat边交换并非严格均匀采样（CONVENTIONS §4 要求声明）——此处零模型只用来做**量级对照**，不做显著性检验。

## 6. 同学的 557 巨型社区：三条独立检验

1. **它属于哪一族算法**：只有权重盲 LPA 族出现 500+ 社区；其 ≥500 频率 **48/50**（676 图）/ **46/50**（671 图）；
   但**恰好 557** 只有 **0** 次；规模本身在 357–622 摆动。
2. **它与本流水线主口径划分的关系**：NMI **0.3224** / ARI **0.0533**（vs 加权 Louvain），
   同算法 seed 天花板 0.8955 → 差异是算法族层面的，不是噪声。
3. **它内部还有没有结构**：加权 Louvain 在这 557 个节点（2155 条边）上给出 **17 个社区**
   （非单点 17），Q=0.6742，最大 127 → 「557」不是模块意义上的原子单位，
   它把整套模块结构包在里面。

**结论（写进正文时允许的表述方式）**：在本数据上，「巨型社区」这一结果随算法是否读权重、
随节点遍历顺序而变，并在结构被打散的度保持零模型上变得更大；同一批图上所有优化模块度的算法
给出的最大社区都在 76–166 区间。
因此「社区=院系」的解读**只能建立在加权 Louvain（或贪心模块度）这一族划分上**，且必须配 §3 的分布与
B1 的 seed 误差棒，不能拿 557 当事实基准。

## 7. 已知坑 / 诚实记录

- `nx.community.label_propagation_communities` **无 weight 参数**：把加权图喂进去**不等于**加权标签传播。
  同学 REPORT.md §4 写「确定性顺序、最多 30 轮、6 轮收敛」，与本实现的语义（按颜色分组迭代到稳定）不完全对应，
  其 557/562 两个数字在仓库里同时出现（`node_communities.csv` 的 `community_size` 列 562、按 community 分组计数 557），
  本单元按 **557（671 口径）** 对齐，并在 §0/§3 报出摆动范围。
- 贪心模块度在全图 2498 条无向边上跑得动：加权 0.481s、
  无权 0.697s，**未降级到 w>=10 骨架**。
- `w>=10` 骨架上 50 次 LPA 的 `n_comms_total` median **375**，其中大量是零度孤立节点各成一社区；
  非单点口径 median **67**（CONVENTIONS §3：报总数会把数字抬高一个量级）。
- KL / 谱二分的社区数 = 2 由算法构造给定，不是数据结果；表中仅作目标函数族对照。
- 自实现 `ext_weightedLPA_Gu` 存在**不收敛/振荡**：50 次中 converged=True 的比例 True
  （按 max_sweeps=100 截断，`converged`/`sweeps` 列逐次写明）。这一族实现的收敛性本身是标签传播的已知弱点，不是本单元引入的错误。
- 全部结果为**静态横截面**链接快照；本单元不做任何因果陈述（CONVENTIONS §7）。

## 8. 产物

| 文件 | 内容 |
|---|---|
| `run_b2.py` | 本单元全部计算，可独立重跑 |
| `algorithm_summary.csv` | 10 算法 × 社区数（两套口径）× 分布 × 两套目标函数 × 耗时 |
| `nmi_matrix.csv` / `ari_matrix.csv` | 10×10 交叉表（`partition_metrics`，覆盖交集重编号） |
| `lp_giant_community_reps.csv` | 标签传播每次重复（351 行）：最大社区、≥500、与同学 557 的 Jaccard、Q/覆盖率 |
| `lp_variant_stats.csv` | 上述重复的按变体汇总（§3 表） |
| `key_numbers.json` | 全部判决数字（机读） |
| `RUN.txt` | 环境版本、命令、seed/重复数、分阶段耗时、运行日志 |
