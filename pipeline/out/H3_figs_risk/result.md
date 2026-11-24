# H3_figs_risk · result.md

`F:` = `out/H3_figs_risk/` 下文件与列；`c:` = `run_h3.py` 行号；z–P 图由 `out/C5_functional_roles/run_h3_zp_fig.py` 生成（H3 RUN.txt §【补图】）。
**本单元不重算分数、不重新分位**：风险矩阵与排序相关的数全部来自 G1 / C1，本单元唯一派生量是 Q1 名单跨口径的 Jaccard 重合率（RUN.txt 第 4 行）。角色图只重绘 C5 的数。

## 1. 图清单（4 张，三件套齐全）

| 图 | PNG | PDF | 伴生 CSV | 行数 |
|---|---|---|---|---|
| 图 1 风险矩阵（真/假失效分色） | `h3_risk_matrix.png` 300dpi | `h3_risk_matrix.pdf` | `fig_h3_risk_matrix.csv` | 12 |
| 图 2 Q1 跨口径稳健性 | `h3_risk_quadrant_robustness.png` | `.pdf` | `fig_h3_risk_quadrant_robustness.csv` | 7 |
| 图 3 中心性排序相关热图（上三角 Spearman / 下三角 Kendall） | `h3_centrality_rank_corr.png` | `.pdf` | `fig_h3_centrality_rank_corr.csv` | 66 |
| 图 4 z–P 角色地图（MANIFEST 要求的第三张，2026-11-23 补） | `h3_zp_role_map.png` | `h3_zp_role_map.pdf` | `fig_h3_zp_role_map.csv` | 671 |

`F:h3_manifest.csv` 仍只登记前 3 张图（`fig,png_bytes,pdf_bytes,csv_rows`），**未包含补出的 z–P 图**——这是本次补图的登记缺口，见 §5。

## 2. 结论表（图 1：风险矩阵，671 站点 × 4 象限）

`F:fig_h3_risk_matrix.csv`；结构轴 = 中心性分位，可用率轴 = `1−ok_ratio_pct`；Q1 = 高结构+低可用（关键且脆弱）、Q2 = 低结构+低可用、Q3 = 高结构+高可用、Q4 = 低结构+高可用。

| 口径 | Q1 | Q2 | Q3 | Q4 | Q1 构成（healthy/partial/真失效/假失效/待判） |
|---|---:|---:|---:|---:|---|
| `in_degree` 原始权重（主口径） | **101**（15.05%） | 210（31.30%） | 165（24.59%） | 195（29.06%） | 0 / 77 / 17 / 6 / 1 |
| `in_degree` 校正后（剔 32 假失效） | **94**（14.73%） | 184（28.84%） | 165（25.86%） | 195（30.56%） | 0 / 77 / 17 / 0 / 0 |
| `deg_u` 总度（对照，含删失） | **109**（16.24%） | 202（30.10%） | 205（30.55%） | 155（23.10%） | 0 / 94 / 9 / 5 / 1 |

读法（与 G1 一致）：**关键与脆弱只在一个很小的交集里重合**——主口径 Q1 仅 101 站（15.05%），其中 17 个真失效、77 个部分失效、0 个健康（Q3/Q4 按定义全健康）。换成总度口径，Q1 变成 109 站但真失效掉到 9 个——**总度轴会把"高结构"错配给一堆死站（出度删失所致），所以主口径必须用入度**（CONVENTIONS §2.1）。

## 3. 结论表（图 2：Q1 名单跨口径稳健性）

`F:fig_h3_risk_quadrant_robustness.csv`（7 个口径 × {原始, 校正后}）：

| 口径 | Q1 人数 | Q1 真失效 | Q1 假失效 | Q1 健康 | Jaccard 最小 / 中位（对其它口径） |
|---|---:|---:|---:|---:|---|
| `in_degree` raw | 101 | 17 | 6 | 0 | 0.473 / 0.552 |
| `in_degree` corr | 94 | 17 | 0 | 0 | 0.439 / 0.547 |
| `bet_u` raw | 120 | 8 | 5 | 0 | 0.464 / 0.559 |
| `bet_u` corr | 108 | 7 | 0 | 0 | 0.429 / 0.540 |
| `pr_dw` raw | 142 | 31 | 18 | 0 | 0.429 / 0.461 |
| `pr_dw` corr | 125 | 32 | 0 | 0 | 0.463 / 0.485 |
| `deg_u` raw | 109 | 9 | 5 | 0 | 0.459 / 0.607 |

Q1 人数在 **94–142** 之间波动，Jaccard 中位仅 **0.46–0.61** —— **"最关键且最脆弱"的名单在口径之间只有约一半重合**。因此正文**禁止**把某一份 Q1 名单当成"关键站点清单"；要写"在入度+校正后口径下 94 个，换口径会换掉约一半成员"。`pr_dw` 口径把真失效捞到 31–32 个（是入度口径的近 2 倍），说明它对"高结构死站"更敏感，与 E1 的 PageRank 曲线最温和（f=20% GCC 仍有 57.7%）在机理上一致：**它把结构重要性铺得更平**。

## 4. 结论表（图 3、图 4）

**图 3 中心性排序相关**（`F:fig_h3_centrality_rank_corr.csv`，66 个指标对，spearman+kendall 双值，`a_censored/b_censored` 标记含删失风险的列）：

| 指标对 | Spearman | Kendall | 备注 |
|---|---:|---:|---|
| `deg_u` – `kcore` | **0.9857** | 0.9421 | 度与 k-core 几乎同序 |
| `out_degree` – `out_strength` | 0.9532 | 0.8412 | 两列均标记 `censored`（死站出度≡0） |
| `in_degree` – `hidx_in_in` | 0.9315 | 0.8785 | 入度与其 H-index 同族 |
| `deg_u` – `bet_u` | 0.9211 | 0.8039 | 度与介数同向 |
| `out_strength` – `hits_hub` | 0.9308 | 0.8047 | 出度侧指标互相绑定 |

读法：**"重要性"只有 2–3 个彼此独立含义**——出度侧一簇（出度/出强度/HITS hub）、入度侧一簇（入度/入度 H-index）、度-介数一簇；PageRank 有向加权与它们的相关性明显更低（见 C1）。凡引用排序，必须写明是哪一簇。

**图 4 z–P 角色地图**（`F:fig_h3_zp_role_map.csv`，671 节点，canonical = Louvain seed 42 的 25 个非单点社区）：

| 角色 | 人数 | 占比 | 12 个划分间人数 | 最大翻转率 | 其中死站 |
|---|---:|---:|---|---:|---:|
| `peripheral` 外围节点 | **447** | 66.62% | 440–454 | 4.25% | 150 |
| `non-hub_connector` 非枢纽连接者 | 151 | 22.50% | 139–161 | 16.56% | 11 |
| `non-hub_kinless` 非枢纽无亲属 | 49 | 7.30% | 41–54 | 20.41% | 1 |
| `connector_hub` 连接型枢纽 | 10 | 1.49% | 6–10 | **40.00%** | 0 |
| `kinless_hub` 无亲属枢纽 | 8 | 1.19% | 8–12 | 25.00% | 0 |
| `provincial_hub` 省级枢纽 | 5 | 0.75% | 5–7 | 0.00% | 0 |
| `network_hub` 网络枢纽 | 1 | 0.15% | 1–1 | 0.00% | 0 |

- **"连接者"确实存在但很少**：`non-hub_connector` 151 个（22.5%）+ `connector_hub` 10 个，合计 161 个（24.0%）；真正的"网络枢纽"只有 1 个（`network_hub`，z=2.88、P=0.69）。
- **角色分布极不均衡**：66.6% 是外围节点；枢纽类（4 类）合计仅 24 个（3.6%）。
- **角色对划分口径敏感**：538/671 个节点在 12 个划分里角色不变，124 个在 2 类间跳动、9 个在 3 类间跳动（`F:fig_h3_zp_role_map.csv:n_distinct_roles_across_seeds`）；`connector_hub` 的翻转率最高 40%。**因此角色标签必须与 B1 的种子误差棒一起引用，不能当固定名单。**
- **健康与角色不可作因果读**：`peripheral` 里 150/447（33.6%）是死站，但这是"死站入度低 → z 低"的机械后果（§2.1 删失），不是"外围更容易坏"。RUN.txt §【补图】也明文写了这一点。

## 5. 关键坑与登记缺口

1. **图 1/2 的"健康"是可用率轴的函数**：Q3/Q4 全健康、Q1/Q2 全非健康，所以风险矩阵本质上是"中心性分位 × 健康二分"的交叉表。任何"关键且脆弱"的表述都必须限定在 Q1 的 94–142 人名单内，并说明它随口径换掉约一半。
2. **真/假失效必须分色且分列**：`in_degree|raw` 的 Q1 里 6 个假失效、Q2 里 26 个；校正后归零（`F:fig_h3_risk_matrix.csv` 的 `n_false_fail`）。图上必须能看出这一层，否则会把认证/robots 站点算进"脆弱"。
3. **`deg_u` 轴是含删失的对照**：`fig_h3_risk_matrix.csv` 的 `table_cn` 明确写"总度（原始）—— 对照，含删失"，不得作为主口径。
4. **登记缺口**：`h3_manifest.csv` 只登记 3 张图，缺 z–P 图（补图时遵守"不动原有 3 张图与 manifest"的约束，RUN.txt 已注明）。若下游按 `h3_manifest.csv` 收图会漏一张。
5. **上游依赖**：图 1/2 依赖 `G1_risk_matrix/g1_nodes_scored.csv`、`g1_quadrant_summary.csv`；图 3 依赖 `C1_centrality_matrix/spearman_matrix.csv`、`kendall_matrix.csv`；图 4 依赖 C5 的 4 张表。上游若改数，H3 必须重画（本单元不缓存数值）。

## 6. 可复现

- 图 1–3：`python out/H3_figs_risk/run_h3.py`（只读上游 CSV，出图 + `h3_manifest.csv`）；`RUN.txt` 记录"不重算分数、不重新分位"。
- 图 4：`python out/C5_functional_roles/run_h3_zp_fig.py`；C5 侧环境 python 3.12.15 / networkx 3.6.1、`SEED=20261113`、canonical seed=42 → 25 社区 / Q=0.69043 / 最大社区 87，12 个划分社区数 23–26、最大社区 83–116，anchors 31/31 PASS（H3 `RUN.txt` §【补图】）。
- 出图规范：PNG 300dpi + 矢量 PDF + `fig_*.csv` 三件套，配色统一，中文 SimHei，无 emoji。
