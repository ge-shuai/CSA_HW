# F2_template_families · result.md

`F:` = `out/F2_template_families/` 下文件与列；`c:` = `run_f2.py` 行号（`F:mark_lines.csv` 给出 16 个关键段的代码行：eligible=84、distance=96、cut_grid=105、rule_r1=121、primary_partition=144、louvain_cross=241、random_baseline=268、edge_enrichment=329）。

## 1. 结论表

| 指标 | 值 | 来源 |
|---|---|---|
| 可用站点集 | `listed_urls>=10` → **394**（MANIFEST 声称 394，实测复核一致） | `c:84-90`；`F:key_numbers.csv:eligible_matches_manifest=True` |
| 被排除的站点 | 277 个（`listed_urls<10`）；其中 352 个站点正好卡在 20-URL 上限 | `F:family_sizes.csv` 口径；`F:key_numbers.csv:n_url_rows_cap20=352` |
| 距离 | `1 − Jaccard(路径指纹模板集)`，average linkage | `c:96`；RUN.txt |
| 阈值 | `t* = 0.29`（silhouette 在 {0.05…1.00 step 0.01} 上取最大；平手取更小 t） | `c:105-140`；`F:cut_sensitivity.csv:t,silhouette=0.404930` |
| 家族总数 | **230** | `F:cut_sensitivity.csv:n_groups` |
| 非单例家族 | **37** | `F:cut_sensitivity.csv:n_nonsingleton` |
| 单例家族 | **193**（占 394 站的 48.98%） | `F:family_sizes.csv:n_families,share_of_sites` |
| **最大家族规模** | **49**（49/394 = 12.44%；family_id=32） | `F:family_sizes.csv:family_size=49`；`F:family_examples.csv` |
| 第二大 | 47（11.93%；family_id=2） | 同上 |
| 其余非单例 | 13×1、6×2、5×2、4×2、3×6、2×22 | `F:family_sizes.csv` |
| 前两大家族合计占比 | 24.37%（96/394） | 由 `F:family_sizes.csv:share_of_sites` 相加 |
| **与超链接社区的 NMI** | **0.5068**（Louvain seed=42），ARI=0.0059 | `F:louvain_cross.csv:NMI_vs_family,ARI_vs_family`；`c:241` |
| NMI 随机基线 | null mean **0.491912** ± 0.005369，z=2.77，双侧 p=0.007，单侧 p=0.003（1000 次） | `F:random_baseline.csv`；`c:268` |
| ARI 随机基线 | null mean 0.000389 ± 0.004924，z=1.12，双侧 p=0.249 | 同上 |
| 家族内边富集 | 同家族点对占 3.148%（2437/77421）；家族内实测边 90，随机期望 60.751，**lift=1.4815**，超几何单侧 p=**0.000174** | `F:family_edge_enrichment.csv`；`c:329-339` |
| 边密度比（内/外） | 3.693% vs 2.454% → **1.505** | 同上 |

## 2. 必须回答：平台/建站模板的集中度

**答：集中度是"两极"的——一半站点没有任何同族伙伴，另一半里有 2 个超大"家族"，但这两个家族恰好是信息最稀薄的那一类。**

1. **量级**：230 个家族 / 37 个非单例 / 最大 49（12.44%）/ 第二大 47（11.93%）。单例占 48.98%（193/394），也就是说**近一半站点在路径指纹上没有可归并的伙伴**。真正有"平台"含义的中等家族是：13 个站点的 `family_id=0`（`/ | /list.htm | /main.htm | page.htm` 系列，家族内平均 Jaccard=0.8737，`F:family_examples.csv:mean_internal_jaccard`）、6 个站点的 `family_id=20`（`/index.html | /class_#/achievement.html` 系列，Jaccard=1.0000）、6 个站点的 `family_id=50`（`/_upload/tpl/#/#/#/template#/contact.htm`，Jaccard=1.0000）。
2. **两个大"家族"要读成"站点被少抓了"，不是"平台统一"**：`family_id=32`（49 站）的定义模板是 `/`、`/ccp.html`、`/hr#.html`、`/prof.html`（家族内平均 Jaccard=1.0000，其成员大多只有根页面 + 1–2 个浅层页）；`family_id=2`（47 站）是 `/ | /#/list.htm | /main.htm`（Jaccard=0.9275）。这两个家族合计 24.37%，与 `listed_urls` 深度强相关（见 §3 稳定性），因此**不能**写成"这一个 CMS 平台占全网四分之一"。
3. **与超链接社区的关系：NMI=0.5068 看起来"中等相关"，但它是机械效应。** 决定性的证据是随机基线：把 37 个非单例家族的标签随机重排后，NMI 的 null 均值就有 **0.4919**（1000 次，sd=0.0054），观测值只比 null 高 2.77 个 sd（p=0.007）。而 ARI 只有 0.0059（null 0.0004±0.0049，p=0.249，完全不可区分）。**高 NMI + 近零 ARI = 家族划分太细（230 组）在大社区里到处"命中"，不是对齐。** 结论必须写成"家族划分与链接社区 **不** 是同一件事（ARI≈0），NMI 的 0.51 是粒度产物"。
4. **但点对层面有一点真实（虽小）的信号**：同家族点对的连边率是跨家族的 1.48–1.51 倍（lift 1.4815，超几何 p=1.74e-04；家族内边数 90 vs 期望 60.75）。这与 F1 的 Mantel r=+0.0419（S1 token Jaccard × 二值邻接）量级一致，两者不矛盾：**极小的点对级关联可以共存于"社区级不对齐"**。注意 487 条边两端都是单例家族（`F:family_edge_enrichment.csv`），即大部分连边发生在"无同族伙伴"的站点之间。

## 3. 关键坑的显式回应

**（坑 1）只用 `listed_urls>=10` 的站点（394），需复核。** 已复核：`F:key_numbers.csv` 中 `n_eligible_listed_ge10=394`、`manifest_claimed_eligible=394`、`eligible_matches_manifest=True`（`c:86-88`）。被排除的 277 站里，绝大多数受抓取深度限制（352 站卡 20-URL 上限；162 个死站的 `listed_urls` 中位数极低），排除是对的——但如果把它们放进来，家族会被"只有根页面"的站点淹没。

**（坑 2）家族划分对 URL 抓取深度极敏感（必须报，否则会被当成结构结论）。**
`F:stability_url_subsample.csv`（按 URL 条数上限子采样后重做整套聚类）：

| url_cap | 家族数 | 非单例 | 最大 | ARI vs 主口径 | NMI vs 主口径 |
|---:|---:|---:|---:|---:|---:|
| 3 | 194 | 31 | 59 | 0.1693 | 0.7914 |
| 5 | 191 | 33 | 56 | 0.2086 | 0.8132 |
| 10 | 227 | 37 | 54 | 0.3826 | 0.8816 |
| 20 / full | 230 | 37 | 49 | 1.0000 | 1.0000 |

读法：把每个站点的 URL 限制到 3 条后，家族划分与主口径只有 ARI=0.17（几乎重排），NMI 0.79 但 ARI 0.17 同样说明"组数都很多，标签不可比"。**所以"哪 49 个站是一家"这种名单只能在 cap=20 口径下引用，且必须写明它随抓取深度变化。**

**（坑 3）阈值选择必须给敏感性（不许只报一套）。**
`F:cut_sensitivity.csv` 与 `F:partition_alternatives.csv`：
- R1（average, silhouette t*=0.29）：230 组 / 37 非单例 / 最大 49 / NMI 0.5068 / ARI 0.0059。
- R2（complete, t*=0.29）：240 组 / 37 / 49 / NMI 0.5162 / ARI 0.0067。
- R4（换 S1 token 指纹）：295 组 / 30 / 50 / NMI 0.5638 / ARI 0.0027。
- R3（按距离分位数切）：q50→30 组（最大 281！）/ q75→17 组（最大 375）/ q90→12 组（最大 383），NMI 只有 0.19/0.08/0.06。
即：silhouette 口径给出"细碎"解（230 组），分位数口径给出"一锅粥"解（12–30 组，最大组吞掉 71%–97% 的站点）。**"最大家族 49"这句话只在 silhouette 口径下成立**；换分位数口径最大组会是 375–383。两种口径都报，主口径取 R1（RUN.txt 明确 primary=R1）。

**（坑 4）社区口径必须是 `csa_common.partition_metrics`，且报 NMI+ARI 两个。**
`F:louvain_cross.csv` 的 20 个 Louvain seed 给出 NMI 0.481–0.527、ARI 0.0059–0.0187（n_compared=394），社区数 20–26、最大社区 80–116 —— 与本项目 CONVENTIONS §3 记录的分布一致，说明家族-NMI 的结论不依赖某一个 seed。**只报 NMI=0.51 而不报 ARI≈0.006 会严重误导（NMI 对细粒度划分天然偏高）。**

## 4. 结论的可用性边界

- 可以说：**模板家族在点对层面与连边弱正相关（lift 1.48，p=1.7e-04）**；**家族划分与链接社区不对齐（ARI≈0）**；**集中度呈两极（近半单例 + 两个低信息量大家族）**。
- 不可以说：家族就是平台/院系；某家族"占全网 X%"；NMI 0.51 说明两者一致。
- 与 F1/F3 的接口：F1 的 |r|≤0.043 与本单元的 lift 1.48 是同一现象的两个尺度（点对相关 vs 边富集），F3 的功效分析应基于这个量级（r≈0.04）判断"值不值得做主线"。

## 5. 可复现与交付

- `RUN.txt`：`PYTHONHASHSEED=0 python out/F2_template_families/run_f2.py`，SEED=20261113，wall 79.7s，anchors 31/31 PASS，且**签名矩阵与 F1 的 `sim_matrices.npz` 数值核对一致**（RUN.txt `signature matrices checked against F1 npz = YES`）。
- 交付齐备：`result.md` + `cut_sensitivity.csv` / `cut_grid.csv` / `partition_alternatives.csv` / `family_sizes.csv` / `family_assignment.csv`(394 行) / `family_examples.csv` / `louvain_cross.csv` / `random_baseline.csv` / `stability_url_subsample.csv` / `family_edge_enrichment.csv` / `key_numbers.csv` / `mark_lines.csv` / `RUN.txt`。
- `url_signature.py` 是同目录的指纹构造模块（非运行入口），`cut_grid*.csv` 是 silhouette 网格原始值，供复核 t*=0.29。
