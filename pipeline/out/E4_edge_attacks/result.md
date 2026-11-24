# E4_edge_attacks · result.md

> **版本注记**：本文件基于 **2026-11-23 18:47 的重跑**（lead 修复 `run_e4.py` 阶段 04 的 `KeyError` 后重跑，`ELAPSED 1073.26s, rc=0`，`RUN.txt` 由脚本自动写出）。14:50 那一版里"曲线段被强制终止、robust-yet-fragile 无法回答"的结论**已作废**，本版全部数字取自本轮重跑的新 CSV。

`F:` = `out/E4_edge_attacks/` 下文件与列；`c:` = 服务器上当前 `run_e4.py` 的行号。

## 1. 口径与规模

| 项 | 值 | 来源 |
|---|---|---|
| 基准图 | 674 节点 / 2497 边（`C.induced(Gu, gcc)`），1 个分量 | `c:268-271`；RUN.txt |
| 总观察链接次数 / 最大边权 | W=63243 / 6073（占 W 的 9.60%） | `c:272-273` |
| 基准效率 | `E_full`=`E_gcc`=`E_fixed`=0.331469（全图连通时三口径必然相等） | `c:275,282`（断言 \|E−0.3315\|<5e-3） |
| 桥 / 圈秩 | 185（7.41% of m）/ 1824 | `c:276,280-281` |
| 删边比例网格 | 19 档：0.005…0.99（`FRACS`） | `c:72-73` |
| 删边策略 | 7 个：`random_edges`、`weight_ascending`、`weight_descending`、`ebc_binarized`、`ebc_weight_as_cost`、`ebc_random_walk`、`bridge_first` | `c:74-75` |
| 删点对照策略 | 5 个：`random_nodes`、`node_degree`、`node_betweenness`、`node_kcore`、`node_pagerank_directed_weighted` | `c:76-77` |
| 主口径边介数 | `nx.edge_betweenness_centrality(G0)`，**weight=None，即先二值化**（MANIFEST 要求） | `c:117`；RUN.txt 第 7 行 |
| 随机基线 | 每档 **100 次**独立重复（`N_RANDOM=100`）；确定性策略 1 次 | `c:66`；`F:edge_attack_curves.csv:reps` |
| 曲线行数 | 228 = 7 边策略×19 档 + 5 点策略×19 档 | `F:edge_attack_curves.csv`；`F:stage_walltimes.csv` |
| 各阶段墙钟 | 静态曲线 691.0 s、动态 EBC 284.8 s、骨架 73.7 s、边普查 12.6 s、EBC 对照 8.4 s、枢纽 2.0 s、等成本 0.5 s、破坏点 0.1 s | `F:RUN.txt`；`F:stage_walltimes.csv` |

## 2. 结论表：边层攻击曲线（`E_fixed` 主口径）

`F:edge_attack_curves.csv`；`E_fixed_ratio` = `E_fixed` / 0.331469；`gcc_ratio` = GCC / 674。确定性策略 `E_fixed_sd` 为空是**正常**（1 次运行），只有 `random_edges`/`random_nodes` 有 sd（100 次）。

| 策略 | 删 1%（25 条）`E_fixed_ratio` | 5% | 10%（250） | 20%（499） | 30%（749） | 50%（1248） | 70%（1748） |
|---|---:|---:|---:|---:|---:|---:|---:|
| `random_edges` | 0.9931 | 0.9628 | 0.9245 | 0.8444 | 0.7562 | 0.5585 | 0.3131 |
| `weight_descending`（先删重边） | 0.9958 | 0.9771 | **0.9430** | 0.8397 | 0.7654 | 0.5992 | 0.3506 |
| `weight_ascending`（先删轻边） | 0.9894 | 0.9342 | 0.8963 | 0.7758 | 0.6727 | 0.4929 | 0.2483 |
| `ebc_binarized`（主口径） | 0.9762 | 0.9195 | **0.6744** | **0.5064** | 0.4565 | 0.3170 | 0.1690 |
| `ebc_weight_as_cost` | 0.9932 | 0.9697 | 0.9425 | 0.6815 | 0.5082 | 0.4032 | 0.2109 |
| `ebc_random_walk` | 0.9953 | 0.9791 | 0.9559 | 0.6789 | 0.5177 | 0.3365 | 0.1633 |
| `bridge_first`（桥优先） | **0.9392** | **0.7087** | 0.5849 | 0.5609 | 0.5305 | 0.4307 | 0.2757 |

对应 GCC 存活比例（`gcc_ratio`）：`random_edges` 0.997/0.985/0.970/0.937/0.898/0.797/0.631；`ebc_binarized` 0.997/0.987/**0.821**/**0.712**/0.702/0.623/0.479；`bridge_first` 0.963/0.815/0.724/0.723/0.715/0.675/0.577；`weight_ascending` 0.996/0.967/0.954/0.890/0.838/0.743/0.539。

三条立刻可读出的结论：
1. **按权重删边 ≈ 随机删边**（10% 时 0.9430 vs 0.9245，差 1.9%）——重边不是"结构要害"；反而是**先删轻边更伤**（10% 时 0.8963，比随机低 2.8%）。
2. **只有"按拓扑目标化"的删边才明显更狠**：主口径 `ebc_binarized` 10% 就掉到 0.6744，是随机的 1.37 倍损失；`bridge_first` 在早期最狠（1% 就 0.9392，且 25 条边把 1 个分量拆成 26 个）。
3. 即使最狠的边策略，删到 **30% 的边**（749 条）时 `E_fixed` 仍有 0.4565（基线的 45.7%）；同等份量下点攻击早已把网络打穿（见 §4）。

## 3. 必须回答（MANIFEST）：边层面是否同样 "robust yet fragile"？

**答：是，但"脆弱"来自目标化选择（介数/桥），不来自权重；而且它是"碎化"型脆弱，不是"树化"型退化。**

1. **robust（冗余大）**：随机删 30% 的边（749/2497）GCC 仍有 89.76%、`E_fixed` 仍有基线的 75.62%；即使 `ebc_binarized` 删到 30%，GCC 仍 70.18%。对照点是**点攻击**：`node_degree` 删到 9.94%（67/674）GCC 57.57%，删到 14.99%（101/674）GCC 只剩 **31.75%**、`E_fixed` 只剩 4.60%（19 档网格粗，GCC 跌破 50% 发生在这两档之间，`F:breakdown_checkpoints.csv` 的 `p50_pct_removed=14.9852` 是"首个低于 50% 的档"）。
2. **fragile（目标化）**：`bridge_first` 只用 **25 条边（1% of m）** 就把分量数从 1 拉到 **26**，GCC 掉到 96.29%；`ebc_binarized` 用 **12 条边（0.48%）** 就首次断连（`last_connected_pct=0.4806`）。也就是说**连通性在极小的删边量下就可被破坏**，前提是删对边。
3. **fragile 的机制是"碎化"不是"树化"**：GCC 跌到 50% 时，圈秩仍有 **392**（`ebc_binarized`）、**668**（`ebc_random_walk`）、**146.52**（`random_edges`）、**29**（`node_degree`），`is_tree_at_gcc50` 全部为 **False**（`F:breakdown_checkpoints.csv`）。网络不是被"削成树"，而是碎成若干仍有内部冗余的块。
4. **动态更狠**：`ebc_binarized` 的动态版本（每删一条重算介数）在 k=749 时 `E_fixed`=0.089599 vs 静态 0.151302，**低 40.8%**（`F:dynamic_vs_static_ebc.csv:E_fixed_dynamic_minus_static=-0.061703`）。所以 §2 的静态曲线对"最坏情况"是**低估**的（详见 §5）。

## 4. 同比例的点攻击对照（回答"边 vs 点，谁更疼"）

`F:edge_attack_curves.csv`（点策略的 `removed_pct_of_universe` = 删点数 / 674）：

| 点策略 | 1% `E_fixed_ratio` | 5% | 10% | 20% | 30% | 50% |
|---|---:|---:|---:|---:|---:|---:|
| `random_nodes` | 0.9737 | 0.8734 | 0.7595 | 0.5545 | 0.3851 | 0.1512 |
| `node_degree` | 0.8284 | 0.4351 | **0.1819** | 0.0071 | 0.0015 | 0.0002 |
| `node_betweenness` | 0.8106 | 0.4279 | 0.1930 | 0.0136 | 0.0011 | 0.0001 |
| `node_kcore` | 0.8284 | 0.5038 | 0.2756 | 0.0319 | 0.0067 | 0.0003 |
| `node_pagerank_directed_weighted` | 0.9228 | 0.6286 | 0.4761 | 0.2485 | 0.0803 | 0.0255 |

**同比例比较**：10% 时点攻击（`node_degree` 0.1819）比边攻击最狠的（`ebc_binarized` 0.6744）**多毁 3.7 倍**。但**注意口径**：10% 的点是 67 个节点、10% 的边是 250 条边，二者"对象数"不同（m/n=3.70，`F:equal_cost_edge_vs_node.csv:object_ratio_m_over_n`）。所以正确说法是"**按全图比例删**，边层远比点层温和"，而不是"删边无用"。

## 5. 动态 vs 静态（`F:dynamic_vs_static_ebc.csv`）

动态 = 每删一条边重算一次边介数（前 250 条为严格逐条，之后为固定重排的块删除，`c:78,408-421`）；对比对象是静态 `ebc_binarized` 曲线。

| 删除边数 | 动态 `E_fixed` | 静态 `E_fixed` | 差 | 相对差 | 模式 |
|---:|---:|---:|---:|---:|---|
| 12 | 0.320840 | 0.328471 | −0.007631 | −2.3% | strict |
| 50 | 0.294829 | 0.318733 | −0.023904 | −7.5% | strict |
| 125 | 0.266065 | 0.304785 | −0.038720 | −12.7% | strict |
| 250 | 0.213906 | 0.223534 | −0.009628 | −4.3% | strict |
| 499 | 0.127890 | 0.167865 | −0.039975 | −23.8% | block |
| 749 | 0.089599 | 0.151302 | −0.061703 | **−40.8%** | block |
| 1248 | 0.051285 | 0.105061 | −0.053776 | −51.2% | block |
| 2472 | 0.000110 | 0.000110 | 0.000000 | 0% | block |

读法：动态在所有未饱和档都比静态更狠，**峰值在 k=749（30% 边）**；到 2472 条（99%）两者都近 0，自然重合。**静态介数曲线只能当"下界"**。

## 6. 等成本比较（`F:equal_cost_edge_vs_node.csv`）

`nodes_pct_for_same_Efixed_degree` = 要让 `node_degree` 造成同样 `E_fixed` 损失所需的点删除比例：

| 边策略 @ 比例 | 删除边数 | `E_fixed` | 等损所需点比例（degree） | 等损所需点比例（random_nodes） |
|---|---:|---:|---:|---:|
| `ebc_binarized` @1% | 25 | 0.3236 | 0.445% | 1.039% |
| `ebc_binarized` @5% | 125 | 0.3048 | 1.039% | 5.045% |
| `ebc_binarized` @10% | 250 | 0.2235 | **2.967%** | 14.985% |
| `ebc_binarized` @20% | 499 | 0.1679 | 5.045% | 24.926% |
| `bridge_first` @10% | 250 | 0.1939 | 5.045% | 20.030% |
| `random_edges` @10% | 250 | 0.3064 | 1.039% | 5.045% |

`edges_cheaper_than_degree_on_Efixed` 在全部 49 行都是 **False**（`F:equal_cost_edge_vs_node.csv`）：**每删一个对象，边的破坏力都弱于点**。10% 的边约等于 3% 的点（按 degree 排序），而 m/n=3.70，说明按"对象数"折算后两者量级才接近——**这也是"边层 robust"的定量含义**。

## 7. 骨架（w≥10）复现（`F:skeleton_attack_curves.csv`）

- 骨架规模：**367 个非孤立节点 / 674 条边**（307 个节点被孤立，`c:494-499`）；骨架基线的 `E_fixed` = **0.06917**、GCC 基线 = **367**（由 `F:skeleton_attack_curves.csv` 首行的 `0.068439/0.989425` 与 `331/0.901907` 反推；脚本只把**比值** `E_fixed_ratio_vs_skeleton` 落盘，分母仍是 674 图的 453602 对，`c:500-501`）。
- 各策略的 `E_fixed_ratio_vs_skeleton`（相对**骨架自身**基线，这是唯一合法的比法）：

| 比例 | `random_edges` | `weight_descending` | `ebc_binarized` | `bridge_first` |
|---|---:|---:|---:|---:|
| 1.04% | 0.9876 | 0.9973 | 0.9227 | 0.9696 |
| 5.04% | 0.9371 | 0.9593 | 0.8450 | 0.8253 |
| 10.0% | 0.8763 | 0.8981 | 0.6991 | 0.6965 |
| 20.0% | 0.7500 | 0.7955 | 0.5716 | 0.4861 |
| 30.0% | 0.6265 | 0.6924 | 0.4461 | 0.3576 |
| 50.0% | 0.3794 | 0.3927 | 0.2029 | 0.2863 |

**结论不反转**：目标化删边（介数/桥）仍然最狠、随机与按重边删仍然最温和；但骨架把差距压缩（10% 时 random 0.8763 vs ebc 0.6991，全图是 0.9245 vs 0.6744），且到 50% 时 `ebc_binarized` 反超 `bridge_first`（0.2029 vs 0.2863）。

## 8. 导航枢纽边（模板边）敏感度（`F:hub_edge_sensitivity.csv`）

删除与 6 个导航枢纽相连的**全部 505 条边**（20.22% of m，`c:521-527`）：

| 变体 | 删边数 | `E_full` | `E_gcc` | `E_fixed` | 分量数 | GCC | 桥数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0 | 0.331469 | 0.331469 | 0.331469 | 1 | 674 | 185 |
| 剥掉全部枢纽边 | 505 | 0.284260 | 0.306599 | **0.284260** | 26 | 649 | 193 |

即：**删掉 1/5 的边（全部模板/导航边），`E_fixed` 只掉到 85.77%、GCC 仍有 96.29%**（但碎成 26 个分量、桥数升到 193）。这量化了"模板边是冗余的"——与 B5 的社区结论（删 505 条边社区结构仍大体保留）方向一致。

## 9. 关键坑的显式回应

1. **加权边介数必须先二值化**（MANIFEST 要求）：主口径用 `weight=None`（`c:117`）。三口径的排序相关性并不高——`ebc_binarized` vs `ebc_weight_as_cost` ρ=0.3008（前 5% 重叠 0.232）、vs `ebc_random_walk` ρ=0.1613、`weight_as_cost` vs `random_walk` ρ=**−0.3348**（`F:edge_metric_agreement.csv`）。所以"按边介数攻击"这句话必须写口径；本文件的主结论只在 `ebc_binarized` 下成立，§2 也给了另两列的曲线供对照。
2. **高介数边不是高权边**：Top-20 二值化介数边里，`lib–teach`（w=99）在"权重当成本"下排第 2261、`en–enbiomed`（w=101）排第 1978（`F:top_edges_by_binarized_ebc.csv`）。这与 §2 的"重边 ≠ 要害"互相印证。
3. **效率只读 `E_fixed`**：`E_gcc` 会因分母收缩上升——例如 `ebc_random_walk` 在 30% 时 `E_gcc`=0.34278（比基线 0.331469 **高** 3.4%），而同期 `E_fixed`=0.17159。任何"删边后效率上升"的读法都只能来自分母收缩。
4. **确定性策略的 `_sd` 为空是设计而非缺陷**：`reps=1` 的策略不产生 sd；只有 `random_edges`/`random_nodes` 有 100 次重复的 sd（`F:edge_attack_curves.csv:reps,E_fixed_sd`）。
5. **骨架/基线的分母不同，不能直接比绝对值**：全图 `E_fixed` 基线 0.331469、骨架 0.06917（同分母 453602，但骨架只有 367 个节点存活）；比较只能用 `E_fixed_ratio_vs_skeleton` 这类相对自身基线的比值。
6. **本轮重跑的成因要记住**：上一轮在 `_summarise()` 之后按 `q_removed` 取列触发 `KeyError`，阶段 04 起全部中断；本次修复在 `c:435-443`（改为取 `n_removed` 再改名，保持 `edge_attack_curves.csv` 列结构不变）。**该段代码是 lead 2026-11-23 的补丁**，不是原脚本。

## 10. 跨单元一致性

- **E4 的 5 条点攻击曲线与 E1 的静态曲线逐点一致**：10% 处 `node_degree` `E_fixed`=0.060286、GCC=57.57%（E1 `degree` f=0.10 为 0.060286 / 57.566766%），`node_pagerank`=0.157826（E1 `pagerank_dw` 同值）。这是一次**独立实现的互证**。
- 与 E1 的分工：E1 提供 6 策略 × 32 档的完整效率族（含 `E_full/E_gcc/E_fixed` 三口径与 sd），E4 提供**边层**策略与"边 vs 点等成本"对照；两者在点攻击部分重叠且一致，不构成重复计量。
- 与 E5 的关系：E5 是"观测到的失效情景"，E4 是"假想的边失效"；E4 的 `bridge_first` 早期碎片化（25 条边 → 26 分量）与 E5 的 `partial149`（47 个分量）量级不同，不可混引。

## 11. 可复现与交付

`PYTHONHASHSEED=0 python out/E4_edge_attacks/run_e4.py`；SEED=20261113；python 3.12.15 / networkx 3.6.1 / pandas 3.0.2 / numpy 2.4.4；**总墙钟 1073.26 s**（静态 691.0 + 动态 284.8 + 骨架 73.7 …），8 进程（`N_JOBS=8`）。
交付齐备：`RUN.txt`（脚本自动写，17 行）、`stage_walltimes.csv`、`edge_attack_curves.csv`(228)、`attack_replicates.csv`、`breakdown_checkpoints.csv`(12)、`dynamic_ebc_curve.csv`、`dynamic_vs_static_ebc.csv`(19)、`equal_cost_edge_vs_node.csv`(49)、`skeleton_attack_replicates.csv`、`skeleton_attack_curves.csv`(133)、`hub_edge_sensitivity.csv`、`baseline_edge_weight_distribution.csv`、`edge_metric_agreement.csv`、`top_edges_by_binarized_ebc.csv`。
