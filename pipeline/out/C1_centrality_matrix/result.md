# C1 · 中心性矩阵：这个网络支持多少种彼此一致的"重要性"

## 0. 一句话结论

12 个重要性指标在两两秩相关（Spearman）意义上折叠为 **8 种彼此一致的含义**（阈值 ρ≥0.90，主口径）；
但若要求"同一含义"还须体现在 Top-20 名单层面（ρ≥0.90 **且** Jaccard≥0.6），66 对指标里只有 **2 对**过关。
没有任何节点同时进入全部 12 张 Top-20 榜（run_c1.py:L381）。

---

## 1. 口径（全部来自本单元实跑）

| 项 | 值 | 出处 |
|---|---|---|
| 图构建 | 一律 `C.build_all()`；锚点 `C.assert_anchors()` 31/31 PASS | run_c1.py:L70-73 |
| Gu / Gd | 676 节点 2498 边 / 676 节点 2788 弧 | run_c1.py:L76-78 |
| 分析节点集 | 671 个有属性站点（图内 676，5 个无属性站点被排除在所有统计之外） | run_c1.py:L76, L167 |
| 度与强度 | 由图重算（`C.add_degrees`）；`nodes_full.csv` 的 `degree`/`color` 列未进入本模块 | run_c1.py:L146-147, L161-163 |
| 中心性主族 | `C.centralities(Gu, Gd)` 给 deg_u / bet_u / pr_dw / pr_uu / hits_hub / hits_auth / kcore | run_c1.py:L138 |
| 本单元新增 | in_degree、out_degree、in_strength、out_strength、strength_u、hidx_in_in、hidx_in_deg、bet_uw | run_c1.py:L146-166 |
| 进入相关矩阵的指标 | 12 个（`strength_u`、`hidx_in_deg`、`bet_uw` 只作描述/敏感性，不进 MEAS） | run_c1.py:L176-191 |
| Top-K | K=20；并列按 FQDN 字典升序打破（候选池升序断言） | run_c1.py:L46, L228-236, L240 |
| 并列可辨识性 | 200 次并列随机打散，同一随机序用于所有指标 | run_c1.py:L47, L256-265 |
| 阈值 θ（"同一含义"） | 0.95 / 0.90 / 0.85 / 0.80 四档 | run_c1.py:L48 |
| 随机性 | `SEED=20261113`，`rng=np.random.default_rng(SEED+r)`，`PYTHONHASHSEED=0` | run_c1.py:L44, L259 |
| HITS 可复现处理 | `nx.hits(..., nstart=全1)` 固定 ARPACK 起点，与 `csa_common` 同算法同参数，\|Δ\|<1e-9 断言通过 | run_c1.py:L115-128, L142-145 |
| 复现验证 | 两次独立进程运行 `c1_summary.json` 逐字节一致（差异仅计时字段） | RUN.txt |

⚠ 与 CONVENTIONS 的分歧（HITS 跨进程不可复现）不在此文展开，见 `DISSENT.md`。

---

## 2. 全表与 12 个指标的一致性或孤立性

`centrality_matrix.csv`：671 行 × 21 列（run_c1.py:L388）。每个指标与其余 11 个的相关性、其 Top-20 的
平均 Jaccard、θ=0.90 下的簇编号（average / complete）如下（run_c1.py:L329-350，`measure_profile.csv`）：

| 指标 | ρ_s 与其他指标 min / **median** / max | τ_b 中位 | Top-20 平均 J | 与之 ρ≥0.90 的指标数 | 簇号(avg/comp) |
|---|---|---|---|---|---|
| deg_u | 0.5236 / **0.7650** / 0.9857 | 0.6380 | 0.3725 | 2 | 2/4 |
| kcore | 0.5062 / **0.7578** / 0.9857 | 0.6390 | 0.2044 | 1 | 2/4 |
| out_strength | 0.3218 / **0.7388** / 0.9532 | 0.5935 | 0.1659 | 2 | 1/3 |
| out_degree | 0.3288 / **0.7059** / 0.9532 | 0.5607 | 0.2964 | 2 | 1/3 |
| pr_uu | 0.4948 / **0.7059** / 0.7714 | 0.5607 | 0.3505 | 0 | 6/6 |
| bet_u | 0.4804 / **0.7020** / 0.9211 | 0.5685 | 0.2853 | 1 | 3/5 |
| in_degree | 0.4280 / **0.6631** / 0.9315 | 0.5226 | 0.3300 | 1 | 7/7 |
| hits_hub | 0.2781 / **0.6490** / 0.9308 | 0.4934 | 0.1372 | 2 | 1/3 |
| hidx_in_in | 0.3945 / **0.6368** / 0.9315 | 0.5094 | 0.3074 | 1 | 7/7 |
| in_strength | 0.3687 / **0.5946** / 0.8086 | 0.4670 | 0.3323 | 0 | 4/1 |
| pr_dw | 0.2781 / **0.5062** / 0.8086 | 0.3762 | 0.2492 | 0 | 5/2 |
| hits_auth | 0.3761 / **0.4948** / 0.6518 | 0.3620 | 0.1774 | 0 | 8/8 |

- 最"通用"的度量：`deg_u`（与其他指标相关性中位数最高 0.7650）；最"孤立"的度量：`hits_auth`（0.4948）（run_c1.py:L335）。
- `pr_dw`（规范主口径）与 `in_strength` 各自独立成簇，说明**有向加权随机游走的重要性**在本网络上
  与度类、HITS 类只呈弱到中等秩序对应（ρ 上限 0.8086）。

---

## 3. 秩相关矩阵：分布与两端

66 个指标对的分布（run_c1.py:L429-441）：

| 统计量 | Spearman ρ_s | Kendall τ_b |
|---|---|---|
| min | **0.2781**（pr_dw ↔ hits_hub） | **0.1992**（pr_dw ↔ hits_hub） |
| p25 | 0.4840 | 0.3558 |
| **median** | **0.6516** | **0.5099** |
| p75 | 0.7648 | 0.6273 |
| max | **0.9857**（deg_u ↔ kcore） | **0.9421**（deg_u ↔ kcore） |

完整矩阵：`spearman_matrix.csv`、`kendall_matrix.csv`（run_c1.py:L389-390）；
ρ_s 的全部统计量（含 p25）与 τ_b 的 min/median/p75/max 存于 `c1_summary.json`（run_c1.py:L429-441）；
τ_b 的 p25 = 0.3558 是对 `kendall_matrix.csv` 上三角（run_c1.py:L390）现场取分位数得到，未写入 JSON。
长表（含 τ_b、Top-20 交集数、Jaccard、是否同义）：`pairwise_stats.csv`（run_c1.py:L391）。
pandas 与 scipy 口径互校通过：|Δρ|、|Δτ_b| 均 <1e-9（run_c1.py:L197-203）。

ρ_s ≥ 0.90 的全部 6 对（run_c1.py:L367）：

| m1 ↔ m2 | ρ_s | τ_b | Top-20 共有 | Jaccard | 名单层面也算同一含义？ |
|---|---|---|---|---|---|
| deg_u ↔ kcore | 0.9857 | 0.9421 | 9 | 0.2903 | 否 |
| out_degree ↔ out_strength | 0.9532 | 0.8412 | 7 | 0.2121 | 否 |
| in_degree ↔ hidx_in_in | 0.9315 | 0.8785 | 15 | **0.6000** | **是** |
| out_strength ↔ hits_hub | 0.9308 | 0.8047 | 6 | 0.1765 | 否 |
| out_degree ↔ hits_hub | 0.9236 | 0.7987 | 6 | 0.1765 | 否 |
| deg_u ↔ bet_u | 0.9211 | 0.8039 | 16 | **0.6667** | **是** |

最不相关的 6 对（run_c1.py:L353-366）：pr_dw↔hits_hub 0.2781、out_strength↔pr_dw 0.3218、
out_degree↔pr_dw 0.3288、in_strength↔hits_hub 0.3687、in_strength↔out_strength **0.3751**、pr_dw↔hits_auth 0.3761。
注意 `in_strength ↔ out_strength` 只有 0.3751：**入向与出向的链接量在本网络里几乎独立**，
这与"高度节点是同一批"并不矛盾（deg_u 与两者都相关，但两个方向彼此不相关）。

---

## 4. "几种彼此一致的重要性"：阈值扫描

1−ρ 距离上的层次聚类（average 与 complete 两种连接；run_c1.py:L298-315）：

| θ（ρ≥θ 记为同义） | 含义数（average） | 含义数（complete） | θ=0.90 的分块 |
|---|---|---|---|
| 0.95 | 10 | 10 | {out_degree,out_strength} \| {hits_hub} \| {deg_u,kcore} \| {bet_u} \| {in_strength} \| {pr_dw} \| {pr_uu} \| {in_degree} \| {hidx_in_in} \| {hits_auth} |
| **0.90（主口径）** | **8** | **8** | {hits_hub,out_degree,out_strength} \| {deg_u,kcore} \| {bet_u} \| {in_strength} \| {pr_dw} \| {pr_uu} \| {hidx_in_in,in_degree} \| {hits_auth} |
| 0.85 | 7 | 7 | {hits_hub,out_degree,out_strength} \| {bet_u,deg_u,kcore} \| {in_strength} \| {pr_dw} \| {pr_uu} \| {hidx_in_in,in_degree} \| {hits_auth} |
| 0.80 | 6 | 6 | {hits_hub,out_degree,out_strength} \| {bet_u,deg_u,kcore} \| {in_strength,pr_dw} \| {pr_uu} \| {hidx_in_in,in_degree} \| {hits_auth} |

严格判据（两两 ρ≥0.90 的最大团，`nx.find_cliques`，run_c1.py:L320-323）给出 4 个可互换小组：
`{hits_hub, out_degree, out_strength}`、`{deg_u, kcore}`、`{bet_u, deg_u}`、`{hidx_in_in, in_degree}`。

**答案**：本网络在秩层面支持 **8 种**（θ=0.90）互不一致的重要性含义，阈值宽松到 0.80 时降到 6 种，
收紧到 0.95 时升到 10 种；8 种含义分别为
①出向扩散（out_degree/out_strength/hits_hub）②无向连通量（deg_u/kcore）③中介位置（bet_u）
④入向链接总量（in_strength）⑤有向加权随机游走（pr_dw）⑥无向随机游走（pr_uu）
⑦入向被引（in_degree/hidx_in_in）⑧HITS 权威（hits_auth）。
其中 ①②⑦ 三组内部的名单重合度差异很大：只有 ②（deg_u↔bet_u 0.6667）与 ⑦（in_degree↔hidx 0.6000）
在 Top-20 名单层面也一致（run_c1.py:L368）。

---

## 5. Top-20 名单的两两重合度

`top20_jaccard.csv`（run_c1.py:L282-288, L392），66 对：
min **0.0526** / median **0.2311** / max **0.6667**（run_c1.py:L442-444）。
即典型的一对"重要性指标"在 20 人榜单上只共享 **7.5 个节点**（`top20_shared` 列中位数 7.5，
区间 2–16；run_c1.py:L358-366）。最高的一对是 `deg_u ↔ bet_u`（共 16 个，J=0.6667），最低 0.0526（共 2 个）。

---

## 6. 共识结构：谁在所有含义下都重要

按"进入 12 张 Top-20 榜的张数"统计（run_c1.py:L372-384，`top20_membership.csv`）：

| 进榜张数 | 节点数 | 节点（张数） |
|---|---|---|
| 12（全票） | **0** | — |
| ≥10 | **1** | en.ustc.edu.cn (10) |
| 9 | 5 | cs / faculty / staff / teach / zsb |
| 8 | 3 | employment / gradschool / yz |
| 7 | 3 | hsss / news / som |
| 6 | 5 | bs / fusep / lib / physics / sist |
| ≥6 合计 | **17** | — |
| 0 | **594** | 671 个有属性站点中 88.5% 一张榜都没进 |

`en.ustc.edu.cn` 未进的两张榜是 `out_strength` 与 `hits_hub`；`cs.ustc.edu.cn` 未进 `hits_hub`、`hits_auth`、`hidx_in_in`；
6 个命名导航枢纽（`C.hub_nodes()` 精确匹配，run_c1.py:L70-78）的进榜张数为
en 10、staff 9、teach 9、news 7、fusep 6、**wpsearch 4**——
即"名字叫导航枢纽"与"在多套重要性口径下都靠前"并不同义，wpsearch 只在出向族与 deg_u 类口径下进榜。

---

## 7. H-index 定义与实测

**主口径 `hidx_in_in`**（run_c1.py:L82-95）：把 Hirsch 指数搬到有向入边——
h(u) = max{ h : u 的**入邻域**中存在 ≥h 个站点 x，且 x 的**入度** ≥ h }。
语义："被 u 指向的站点里，有 h 个各自至少被 h 个站点指向"，对应引文网络的原始定义
（论文↔站点、被引次数↔入度），只是把"文献"换成"入邻域"（有向版本要求区分引用方向，
此处只统计入边，与 in_degree 构成"量 vs 质"的对照）。

**备选口径 `hidx_in_deg`**（run_c1.py:L98-111）：内层评价量换成无向度 degree_u(x)，仅作敏感性。

| 量 | 值 | 出处 |
|---|---|---|
| h 最大值 | 14（teach.ustc.edu.cn） | run_c1.py:L456 |
| h 均值（671 站点） | 2.313 | run_c1.py:L456 |
| h = 0 的站点数 | 10 | run_c1.py:L457 |
| ρ_s(hidx_in_in, in_degree) | 0.9315 | run_c1.py:L458 |
| ρ_s(两种 h 变体之间) | 0.9343 | run_c1.py:L459 |
| τ_b(hidx_in_in, in_degree) | 0.8785 | run_c1.py:L353-366 |
| Top-20 与 in_degree 交集 | 15 / 20（J=0.6000） | run_c1.py:L282-288 |

H-index 与入度高度相关但**不重合**：它把 in_degree 榜单里的 5 个成员换成了被"更硬的"入邻域支撑的站点。

---

## 8. HITS 噪声地板（把尾部视为真并列后的敏感性）

`nx.hits` 的向量尾部在 |v| ≤ 1e-12·max 量级只是浮点噪声（run_c1.py:L131-134, L208-224）：

| 指标 | 低于地板的节点数 | 独立取值数：原始 → 合并后 | 地板绝对值 |
|---|---|---|---|
| hits_hub | **337 / 671** | 346 → 330 | 8.56e-13 |
| hits_auth | 32 / 671 | 546 → 528 | 8.30e-13 |

把这部分并入 0（即承认为真并列）后重算全部 66 对相关（run_c1.py:L218-224, L420-424）：
|Δρ_s| 最大 **0.0112**、中位 0.0；|Δτ_b| 最大 0.0115；**跨过 θ=0.90 阈值的指标对 0 个**；
**12 张 Top-20 榜单的成员改变数全为 0**。
→ 本单元所有结论对 HITS 尾部分辨力不敏感；但 `hits_hub` 的秩在 337 个节点上没有可解释的顺序，
引用该指标时只能用头部（详见 `DISSENT.md`）。逐对明细：`hits_noise_sensitivity.csv`（run_c1.py:L399-407）。

---

## 9. 并列与榜单不可辨识（本单元最主要的口径风险）

第 20 名的值上并列了多少节点（run_c1.py:L246-254）+ 200 次并列随机打散后榜单自身的抖动（run_c1.py:L256-278）：

| 指标 | 第 20 名的值 | 并列在该值的节点数 | 严格在上的节点数 | 必进榜的核心 | 打散后榜内自相似 J |
|---|---|---|---|---|---|
| deg_u | 37 | 3 | 19 | 19 | 0.9390 |
| out_degree | 25 | 3 | 18 | 18 | 0.9438 |
| in_degree | 24 | 2 | 18 | 18（并列 2 个正好补满） | 1.0000 |
| in_strength | 693 | 2 | 18 | 18（同上） | 1.0000 |
| out_strength | 704 | 1 | 19 | 19 | 1.0000 |
| bet_u / pr_dw / pr_uu / hits_hub / hits_auth | — | 1 | 19 | 19 | 1.0000 |
| hidx_in_in | 8 | 5 | 16 | 16（20 席争 4 席） | 0.9267 |
| **kcore** | **11** | **45** | **0** | **0** | **0.3044** |

`kcore` 的 Top-20 在本网络里是"45 个 k=11 的节点里任取 20 个"，**这份名单不可引用**（其成员在打散下
平均只重合 0.3044，最多 45 个节点曾出现在榜上，0 个节点稳定在榜）；这也解释了 §4 中
`deg_u ↔ kcore` 的 ρ=0.9857 与 J=0.2903 之间的落差（run_c1.py:L282-288, L353-366）。
C3 单元在三种图口径下复核了这个结论并给出判据。

---

## 10. 局限

1. 相关系数只在 671 个有属性节点上计算，图内 5 个无属性站点未参与任何统计（run_c1.py:L167）。
2. 死站（162 个）的出度与出强度为删失值（恒 0），因此 out_* 族与 `hits_hub`、`pr_dw` 的相关结构里
   混有观测缺失；分类层面的处理见 C2，图口径层面的稳健性见 C3。
3. 本单元所有陈述为**秩层面的对应关系**，不含任何因果解释；"含义数"依赖 θ 的选取，已给 4 档扫描。
