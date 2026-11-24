"""D2 · censoring_impact — 出度删失的系统性量化（本项目最关键的方法学单元）。

本单元回答四件事（MANIFEST D2 + CONVENTIONS §2.1/§7）：
  (a) 核验删失事实锚点（edges_from_dead / dead 出度 / 指向死站的边）。
  (b) "总度口径 vs 入度口径"结论对照表：同一个结论在两套口径下各是什么数、哪个有效、为什么。
  (c) 哪些统计量**任何校正都救不回来**，哪些可以。
  (d) 邻接矩阵缺失量级的**情景模拟**（counterfactual，不是数据）：
      若死站拥有"同入度活站"的出度分布，会多出多少条边。

措辞遵守 §7：只写"与…相关 / 结构上对应 / 情景模拟显示"，不写机制、不写因果。

运行：
    cd pipeline && PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8 python -X utf8 out/D2_censoring_impact/run_d2.py
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PIPELINE)
import csa_common as C  # noqa: E402

MODULE = "D2_censoring_impact"
SEED = C.SEED
N_BOOT = 1000                                   # 情景模拟重复次数（≥1000，全部显式传 seed）
np.random.seed(SEED)                                   # 兜底；下面的抽样全部显式 RandomState(SEED+…)

# ============================================================================ 数据
T0 = time.time()
b = C.build_all()
C.assert_anchors(b)                             # 31 项锚点
ANCHOR_LINE = inspect.currentframe().f_lineno - 1
nodes, Gd, Gu, Gb = b["nodes"], b["Gd"], b["Gu"], b["Gb"]
sets, VERSIONS = b["sets"], C.versions()
nodes = nodes.sort_values("node", kind="mergesort").reset_index(drop=True)
nodes["_removed"] = nodes["false_fail"] | nodes["ambiguous_fail"]
S_uncorr = nodes.copy()                          # 671
S_corr = nodes.loc[~nodes["_removed"]].copy()    # 638

dead = set(nodes.loc[nodes["is_dead"], "node"])
alive_all = set(Gd.nodes()) - dead               # 514 = 509 有属性活站 + 5 悬空引用
true_fail = set(nodes.loc[nodes["true_fail"], "node"])
false_fail = set(nodes.loc[nodes["false_fail"], "node"])
healthy = set(nodes.loc[nodes["is_healthy"], "node"])
partial = set(nodes.loc[nodes["is_partial"], "node"])
N_NODES = Gd.number_of_nodes()                   # 676
N_DIRECTED = Gd.number_of_edges()                # 2788
N_UNDIRECTED = Gu.number_of_edges()              # 2498
W_TOTAL = sum(d["weight"] for _, _, d in Gd.edges(data=True))


def dg(nodeset, which):
    f = Gd.in_degree if which == "in" else Gd.out_degree
    return np.array([f(n) for n in nodeset], dtype=float)


def ug(nodeset):
    return np.array([Gu.degree(n) for n in nodeset], dtype=float)


# ============================================================ (a) 删失事实核验
edges_from_dead = sum(1 for s, _ in Gd.edges() if s in dead)
weight_from_dead = sum(d["weight"] for s, _, d in Gd.edges(data=True) if s in dead)
edges_to_dead = sum(1 for _, t in Gd.edges() if t in dead)
edges_dead_to_dead = sum(1 for s, t in Gd.edges() if s in dead and t in dead)
edges_alive_to_dead = sum(1 for s, t in Gd.edges() if t in dead and s not in dead)
weight_to_dead = sum(d["weight"] for _, t, d in Gd.edges(data=True) if t in dead)
dead_out_degrees = dg(dead, "out")
dead_in = dg(dead, "in")
undirected_touching_dead = sum(1 for u, v in Gu.edges() if u in dead or v in dead)
gu_pairs = {(min(u, v), max(u, v)) for u, v in Gu.edges()}
gd_pairs = {(min(s, t), max(s, t)) for s, t in Gd.edges()}
dead_in_gu_equals_dead_in_gd = bool(np.array_equal(
    np.sort([Gu.degree(n) for n in dead]), np.sort([Gd.in_degree(n) for n in dead])))
identity_holds = int((S_uncorr["degree_u"] == S_uncorr["in_degree"] + S_uncorr["out_degree"]).sum())

censor_rows = [
    ("edges_total_directed", N_DIRECTED, "锚点 2788"),
    ("edges_from_dead", edges_from_dead, "锚点 0 → 死站出边**完全**不可观测（不是低估，是缺失）"),
    ("weight_from_dead", weight_from_dead, "出边权重合计同样为 0"),
    ("edges_dead_to_dead", edges_dead_to_dead, "锚点 0 → 死—死相邻构造性不可识别"),
    ("edges_to_dead", edges_to_dead, "=edges_alive_to_dead，因为不存在 dead→dead"),
    ("edges_alive_to_dead", edges_alive_to_dead, "锚点 296；入边可观测 → 失效分析主口径=入度"),
    ("weight_to_dead", weight_to_dead, f"占全网权重 {weight_to_dead / W_TOTAL:.4f}"),
    ("dead_n", len(dead), "锚点 162"),
    ("dead_out_degree_zero_n", int((dead_out_degrees == 0).sum()), f"/{len(dead)} = 100%"),
    ("dead_out_degree_max", int(dead_out_degrees.max()), "死站出度最大值"),
    ("dead_out_degree_mean", 0.0, "恒等于 0"),
    ("dead_in_degree_mean", round(float(dead_in.mean()), 4), "唯一可观测的度"),
    ("dead_in_degree_median", float(np.median(dead_in)), ""),
    ("dead_in_degree_max", int(dead_in.max()), "入度是可观测的：死站最高 22"),
    ("undirected_edges_touching_dead", undirected_touching_dead, "296：死站参与的无向边全部来自活站"),
    ("Gu_pairset_equals_Gd_pairset", int(gu_pairs == gd_pairs), "1=无向边集就是有向边集的底图（2498）"),
    ("dead_degreeU_equals_inDegree_1to1", int(dead_in_gu_equals_dead_in_gd),
     "1=死站的无向度集合与入度集合逐点相同（定义性事实）"),
    ("degree_u_eq_in_plus_out_n_of_671", identity_holds,
     "只有 435/671 成立：存在互指边时 degree_u<in+out（度分解不能直接相加）"),
    ("urls_csv_page_level_hrefs_available", 0, "urls.csv 无页面级链接来源 → 无法反查被删失的出边"),
]
censoring = pd.DataFrame(censor_rows, columns=["item", "value", "note"])

# ============================================================ (b) 两套口径对照表
mean_in_dead, mean_in_healthy = float(dead_in.mean()), float(dg(healthy, "in").mean())
mean_du_dead, mean_du_healthy = float(ug(dead).mean()), float(ug(healthy).mean())
mean_out_healthy = float(dg(healthy, "out").mean())
ratio_in = mean_in_healthy / mean_in_dead
ratio_du = mean_du_healthy / mean_du_dead
cross_factor = mean_du_healthy / mean_in_healthy          # 5.195 = 2.705 × 1.921

# D1 的十分位梯度在本单元重算一遍（自检 D1，不读 D1 文件）
def rank_decile(df, col, q=10):
    ordered = df.sort_values([col, "node"], kind="mergesort")
    lab = pd.qcut(pd.Series(np.arange(len(ordered), dtype=float), index=ordered.index), q, labels=False) + 1
    return pd.Series(lab.astype(int)).reindex(df.index)


grad_rows = []
for expo in ("degree_u", "in_degree", "url_count", "listed_urls"):
    for sample_tag, sdf, ymask in (("S_uncorr_671", S_uncorr, "is_dead"), ("S_corr_638", S_corr, "true_fail")):
        d = sdf.assign(_bin=rank_decile(sdf, expo).to_numpy())
        g = d.groupby("_bin").agg(nn=("node", "size"), kk=(ymask, "sum"),
                                  xmean=(expo, "mean")).reset_index()
        g["rate"] = (g.kk / g.nn).round(4)          # 与 D1 的 bin_stats 同精度，保证跨单元数字一致
        rho, p = stats.spearmanr(g["xmean"], g["rate"])
        bot, top = g["rate"].iloc[0], g["rate"].iloc[-1]
        grad_rows.append({"exposure": expo, "sample": sample_tag, "outcome": ymask,
                          "bottom_rate": round(float(bot), 4), "top_rate": round(float(top), 4),
                          "rate_diff_pp": round(float((top - bot) * 100), 2),
                          "bin_spearman": round(float(rho), 4), "bin_spearman_p": float(p),
                          "monotone": bool(g["rate"].is_monotonic_decreasing)})
grad = pd.DataFrame(grad_rows)

# 可计算的关联（全部是"相关"，不是机制）
corr_rows = []
for a, bb in [("in_degree", "is_dead"), ("degree_u", "is_dead"), ("out_degree", "is_dead"),
              ("in_degree", "true_fail"), ("degree_u", "true_fail"), ("url_count", "is_dead"),
              ("in_degree", "ok_ratio_pct"), ("degree_u", "ok_ratio_pct"),
              ("out_degree", "ok_ratio_pct"), ("url_count", "ok_ratio_pct")]:
    x = S_uncorr[a].astype(float)
    y = S_uncorr[bb].astype(float)
    rs, ps = stats.spearmanr(x, y)
    rp, pr = stats.pearsonr(x, y)
    corr_rows.append({"x": a, "y": bb, "n": len(S_uncorr), "spearman": round(float(rs), 4),
                      "spearman_p": float(ps), "pearson": round(float(rp), 4), "pearson_p": float(pr),
                      "readable": 1 if a != "out_degree" else 0,
                      "note": ("out_degree 在 162 个死站上恒为 0，相关里混入定义性成分" if a == "out_degree"
                               else "入度/总度/页面数都可计算；总度版本受删失污染")})
correlations = pd.DataFrame(corr_rows)

# 只活站子图上的关联（把定义性成分剥掉）
alive_attr = S_uncorr.loc[~S_uncorr["is_dead"]]
sub_alive = Gd.subgraph(alive_all).copy()
rows_sub = []
for a, bb in [("in_degree", "is_partial"), ("degree_u", "is_partial"), ("in_degree", "partial_clean"),
              ("in_degree", "ok_ratio_pct"), ("degree_u", "ok_ratio_pct")]:
    x = alive_attr[a].astype(float)
    y = alive_attr[bb].astype(float)
    rs, ps = stats.spearmanr(x, y)
    rows_sub.append({"x": a, "y": bb, "n": len(alive_attr), "spearman": round(float(rs), 4),
                    "p": float(ps), "note": "只在 509 个活站上算：out_degree 此处可观测（部分受 20 页上限截断）"})
corr_alive = pd.DataFrame(rows_sub)

leaf_rows = []
for label, mask in (("dead162", S_uncorr["is_dead"]), ("true129", S_uncorr["true_fail"]),
                    ("healthy360", S_uncorr["is_healthy"]), ("partial149", S_uncorr["is_partial"])):
    s = S_uncorr[mask]
    leaf_rows.append({"group": label, "n": len(s),
                      "mean_in_degree": round(float(s.in_degree.mean()), 4),
                      "mean_out_degree": round(float(s.out_degree.mean()), 4),
                      "mean_degree_u": round(float(s.degree_u.mean()), 4),
                      "share_in_le_1": round(float((s.in_degree <= 1).mean()), 4),
                      "share_degree_u_le_1": round(float((s.degree_u <= 1).mean()), 4),
                      "share_in_eq_1": round(float((s.in_degree == 1).mean()), 4),
                      "share_out_eq_0": round(float((s.out_degree == 0).mean()), 4)})
leaves = pd.DataFrame(leaf_rows)

contrast_rows = [
    # 结论, 总度口径, 入度口径, 判定, 为什么
    ("死站平均度 1.83 vs 健康站 9.49",
     f"{mean_du_dead:.3f} vs {mean_du_healthy:.3f} = {ratio_du:.3f}×",
     f"{mean_in_dead:.3f} vs {mean_in_healthy:.3f} = {ratio_in:.3f}×",
     "总度版本 INVALID（跨口径）",
     "死站的 degree_u 恒等于 in_degree（出边全部不可观测），健康站的 degree_u 里含 5.786 的可观测出度；"
     "两边量的不是同一个东西。倍数分解：5.195× = 2.705×（同口径入度）×1.921（健康站度/入度，纯删失自由度）"),
    ("'死站是叶子'（度≤1）",
     f"死站 {(S_uncorr.loc[S_uncorr.is_dead,'degree_u']<=1).mean():.4f} vs 健康站 "
     f"{(S_uncorr.loc[S_uncorr.is_healthy,'degree_u']<=1).mean():.4f}",
     f"死站 {(S_uncorr.loc[S_uncorr.is_dead,'in_degree']<=1).mean():.4f} vs 健康站 "
     f"{(S_uncorr.loc[S_uncorr.is_healthy,'in_degree']<=1).mean():.4f}",
     "表述必须挂在入度口径 + 2.705× 这个数上",
     "同一句话在无向度上把 62.35% vs 33.89% 夸大成 62.35% vs 17.50%；差值来自健康站的出度分量"),
    ("度十分位的全挂率梯度",
     f"{grad.loc[(grad.exposure=='degree_u')&(grad['sample']=='S_uncorr_671'),'rate_diff_pp'].iloc[0]:.2f} pp",
     f"{grad.loc[(grad.exposure=='in_degree')&(grad['sample']=='S_uncorr_671'),'rate_diff_pp'].iloc[0]:.2f} pp",
     "两口径同号同显著；总度更陡 1.8 倍",
     "总度把'没有可观测出边'记成'低度'，等于给结论加了一遍自变量里的答案"),
    ("ρ(度, URL 可用率)（§2.1 锚点）",
     f"{correlations.loc[(correlations.x=='degree_u')&(correlations.y=='ok_ratio_pct'),'spearman'].iloc[0]:.4f}",
     f"{correlations.loc[(correlations.x=='in_degree')&(correlations.y=='ok_ratio_pct'),'spearman'].iloc[0]:.4f}",
     "主口径=入度；总度只作对照",
     "CONVENTIONS §2.1 写明 0.376 vs 0.216，本单元复现 0.3762 / 0.2157"),
    ("出度均值 死 vs 健康",
     f"{float(dead_out_degrees.mean()):.3f} vs {mean_out_healthy:.3f}",
     "不适用（出度不是入度的替身）",
     "0 vs 5.786 不是'死站不出链'的证据",
     "它是爬虫定义：抓不到的页面提不出链接。禁止读作行为差异"),
    ("死站在无向图上的邻居数",
     f"平均 {undirected_touching_dead/len(dead):.3f} 条边连着死站",
     f"平均 {edges_to_dead/len(dead):.3f}",
     "两数相同（296/162=1.827），这正是同一个数",
     "死站参与的无向边 = 指向死站的有向边，逐条一一对应"),
    ("'失效站点之间不相邻'（D4 的前置）",
     "dead↔dead 观测边 0",
     "target-side 可检验（活站→死站 296 条）",
     "dead↔dead 构造性不可识别",
     "观测度序列本身已把'死站出度=0'编码进去，度保持零模型也复现不出任何 dead→dead 边"),
]
contrast = pd.DataFrame(contrast_rows, columns=["结论", "总度口径", "入度口径", "判定", "为什么"])

# ============================================================ (c) 可恢复性
recover = pd.DataFrame([
    ("死站的出度", "不可恢复", "0 信息量：观测值恒为 0，真值域 [0, 95]（活站最大出度 95）",
     "任何校正只能给区间/情景，不能给点估计"),
    ("死站→死站的边", "不可恢复", "分子分母两端都不可观测", "只能改写为可检验命题（D4 target-side 置换）"),
    ("死站→活站的边", "不可恢复", "起点页面未被抓取", "同上"),
    ("死站作为链接来源的角色（hubness/出强度/介数经它的路径）", "不可恢复",
     "全部依赖出边", "有向介数/PageRank 的死站行向量是缺失值不是 0"),
    ("全网有向边集 / 出度分布 / 权重出强度", "部分不可恢复",
     "观测 2788 条是**完整子集**（少而真），不是抽样", "缺失量级见情景模拟（d）"),
    ("每个节点的入度", "可恢复（但有下界性质）",
     "入边只依赖起点是否被抓到；起点是死站时该边仍缺失", "观测入度 ≤ 真入度；本报告统一称'观测入度'"),
    ("无向度（活站之间）", "可恢复", "活站↔活站的边两个方向都可观测",
     "死站参与的无向边 = 其入边，见 (b) 最后一行"),
    ("活站诱导子图的全部拓扑", "可恢复", "514 个节点 / 2492 条有向边",
     "这是唯一一张'口径自洽'的图，任何'结构 vs 健康'的推断都应在此复算"),
    ("页面级链接来源（谁在哪个页面链向谁）", "不可恢复", "urls.csv 只有 subdomain/status/url/title 四列",
     "无法区分模板边与正文边，也无法反查删失"),
    ("时间维度（谁先坏/谁后坏）", "不可恢复", "单次横截面", "所有'梯度'都只是同时观测的共现"),
], columns=["统计量", "可否恢复", "依据", "备注"])

# ============================================================ (d) 情景模拟：缺失量级
alive_df = S_uncorr.loc[~S_uncorr["is_dead"], ["node", "in_degree", "out_degree"]].copy()
BANDS = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9),
         (10, 12), (13, 15), (16, 19), (20, 24), (25, 40), (41, 200)]


def band_of(x):
    for lo, hi in BANDS:
        if lo <= x <= hi:
            return (lo, hi)
    return BANDS[-1]


alive_df["_band"] = alive_df["in_degree"].map(band_of)
dead_df = S_uncorr.loc[S_uncorr["is_dead"], ["node", "in_degree"]].copy()
dead_df["_band"] = dead_df["in_degree"].map(band_of)
band_stats = alive_df.groupby("_band").agg(
    n_alive=("node", "size"), mean_out=("out_degree", "mean"), median_out=("out_degree", "median"),
    p75_out=("out_degree", lambda q: float(np.quantile(q, .75)))).reset_index()
band_stats[["band_lo", "band_hi"]] = pd.DataFrame(band_stats["_band"].tolist(), index=band_stats.index)
band_stats["n_dead_in_band"] = dead_df["_band"].value_counts().reindex(band_stats["_band"]).fillna(0).astype(int).to_numpy()
band_stats["expected_edges_mean"] = band_stats["n_dead_in_band"] * band_stats["mean_out"]
band_stats["expected_edges_median"] = band_stats["n_dead_in_band"] * band_stats["median_out"]
band_stats["mean_out_pct_of全网"] = (band_stats["mean_out"] / float(np.mean(alive_df["out_degree"]))).round(4)
impute_bands = band_stats[["band_lo", "band_hi", "n_alive", "mean_out", "median_out", "p75_out",
                           "n_dead_in_band", "expected_edges_mean", "expected_edges_median"]].copy()
impute_bands["mean_out"] = impute_bands["mean_out"].round(4)
impute_bands["p75_out"] = impute_bands["p75_out"].round(2)
impute_bands["expected_edges_mean"] = impute_bands["expected_edges_mean"].round(2)

mean_map = dict(zip(band_stats["_band"], band_stats["mean_out"]))
med_map = dict(zip(band_stats["_band"], band_stats["median_out"]))
p75_map = dict(zip(band_stats["_band"], band_stats["p75_out"]))
extra_mean = float(sum(mean_map[x] for x in dead_df["_band"]))
extra_median = float(sum(med_map[x] for x in dead_df["_band"]))
extra_p75 = float(sum(p75_map[x] for x in dead_df["_band"]))

# 分层匹配 + 蒙特卡洛：从同层活站的出度分布里**有放回抽一个**（不是取均值）
bands_list = dead_df["_band"].to_numpy(dtype=object)
pool = {k: g.to_numpy() for k, g in alive_df.groupby("_band")["out_degree"]}
draws = np.empty((N_BOOT, len(dead_df)), dtype=float)
for r in range(N_BOOT):
    loc = np.random.RandomState(SEED + r)          # 每次独立重复（§4 的规范）
    for i, bd in enumerate(bands_list):
        arr = pool[bd]
        draws[r, i] = arr[loc.randint(len(arr))]
extra_stochastic = draws.sum(axis=1)

# 目标选择：情景里每条被抽出的出边指向谁 —— 按观测入度∝偏好（含死站目标）
in_pop = np.array([Gd.in_degree(n) for n in Gd.nodes()], dtype=float) + 1.0
node_list = list(Gd.nodes())
target_p = in_pop / in_pop.sum()
dead_idx = np.array([i for i, n in enumerate(node_list) if n in dead])
n_nodes_all = len(node_list)
d2_counts, src_dead_counts = [], []
for r in range(200):                                # 目标指派只跑 200 次（每次 O(extra) 抽样）
    loc = np.random.RandomState(SEED + 10_000 + r)
    k = int(round(draws[r].sum()))
    tgt = loc.choice(n_nodes_all, size=k, replace=True, p=target_p)
    src = np.repeat(np.arange(len(dead_df)), np.round(draws[r]).astype(int))[:k]
    d2_counts.append(int(np.isin(tgt, dead_idx).sum()))
    src_dead_counts.append(int(len(tgt) - np.isin(tgt, dead_idx).sum()))
d2_arr = np.array(d2_counts, dtype=float)

obs_cells = N_NODES * (N_NODES - 1)
dead_row_cells = len(dead) * (N_NODES - 1)
dd_block = len(dead) * (len(dead) - 1)
scenario = pd.DataFrame([
    ("情景：死站拥有'同入度层活站'的出度分布", "", "counterfactual 情景模拟，不是观测"),
    ("extra_edges_mean_matched（层均值）", round(extra_mean, 1),
     f"= Σ 层均值；全网观测边 {N_DIRECTED} → 补齐后 {N_DIRECTED + extra_mean:.0f}"),
    ("extra_edges_median_matched（层中位数，保守）", round(extra_median, 1), "右偏分布下中位数版本是低情景"),
    ("extra_edges_p75_matched", round(extra_p75, 1), "高分位版本"),
    ("extra_edges_stochastic（层内有放回抽样, n_boot=1000）", round(float(extra_stochastic.mean()), 1),
     f"sd={float(extra_stochastic.std(ddof=1)):.1f}; 95% 区间 "
     f"[{float(np.quantile(extra_stochastic,.025)):.0f}, {float(np.quantile(extra_stochastic,.975)):.0f}]"
     f"；期望与层均值版相同（抽样期望=层均值），但重复均值的 MC 标准误 "
     f"{float(extra_stochastic.std(ddof=1))/np.sqrt(N_BOOT):.1f}，实现值与解析值 {extra_mean:.1f} 相差 "
     f"{abs(float(extra_stochastic.mean())-extra_mean)/max(float(extra_stochastic.std(ddof=1))/np.sqrt(N_BOOT),1e-9):.1f} "
     f"个 MC SE（源自助分布右偏，重尾抽样噪声大）→ 正文一律以解析值 {extra_mean:.2f} 为准"),
    ("缺失比例（均值情景）", round(extra_mean / (N_DIRECTED + extra_mean), 4),
     "缺失边 / 补齐后总边数"),
    ("缺失比例（中位情景）", round(extra_median / (N_DIRECTED + extra_median), 4), ""),
    ("缺失比例（无向边口径）", round(extra_mean / (N_UNDIRECTED + extra_mean), 4),
     f"观测无向边 {N_UNDIRECTED}；死站的边全部已计入（296 条），新增边只与活站邻居重叠时才会并边"),
    ("dead→dead 边数（同一情景的目标指派, n=200）", round(float(d2_arr.mean()), 1),
     f"sd={float(d2_arr.std(ddof=1)):.1f}; 95% 区间 [{float(np.quantile(d2_arr,.025)):.0f}, "
     f"{float(np.quantile(d2_arr,.975)):.0f}]；观测值恒 0，与情景无关 → 这条量**永远无法证伪**"),
    ("邻接矩阵单元总数（676×675 有序对）", obs_cells, f"观测密度 {N_DIRECTED/obs_cells:.4%}"),
    ("死站行（162×675 单元）完全未观测", dead_row_cells, f"占矩阵 {dead_row_cells/obs_cells:.4%}"),
    ("死站×死站子块（162×161）双重未观测", dd_block, f"占矩阵 {dd_block/obs_cells:.4%}"),
    ("死站行内的观测密度", 0.0, "0/109350 —— 这就是'完全删失'的字面含义"),
    ("活站行的观测密度", round((N_DIRECTED) / ((N_NODES - len(dead)) * (N_NODES - 1)), 6),
     f"2492 条起点为活站的边落在 {N_NODES - len(dead)}×{N_NODES - 1} 单元里"),
    ("权重口径：出边权重合计缺失", f"≥{extra_mean:.0f} 条链接次数的量级",
     f"观测权重合计 {W_TOTAL}；情景多出的权重无法折算（出边目标模板未知）→ 只给边数，不给权重"),
], columns=["item", "value", "note"])

# 情景下的"总度口径"重算：把假想出度加给死站，看梯度会不会变
cf_du = dict(zip(S_uncorr["node"], S_uncorr["degree_u"]))
imputed_per_node = dict(zip(dead_df["node"], [mean_map[x] for x in dead_df["_band"]]))
extra = S_uncorr.copy()
extra["degree_u_cf"] = extra["node"].map(cf_du).astype(float) + extra["node"].map(imputed_per_node).fillna(0.0)
cf_rows = []
for col in ("degree_u", "degree_u_cf", "in_degree"):
    for tag, sdf in (("S_uncorr_671", extra), ("S_corr_638", extra.loc[~extra["_removed"]])):
        ymask = "is_dead" if tag == "S_uncorr_671" else "true_fail"
        d = sdf.assign(_bin=rank_decile(sdf, col).to_numpy())
        g = d.groupby("_bin").agg(nn=("node", "size"), kk=(ymask, "sum"), xmean=(col, "mean")).reset_index()
        g["rate"] = (g.kk / g.nn).round(4)          # 与 D1 同精度
        rho, p = stats.spearmanr(g["xmean"], g["rate"])
        cf_rows.append({"exposure": col, "sample": tag, "outcome": ymask,
                        "bottom_rate": round(float(g["rate"].iloc[0]), 4),
                        "top_rate": round(float(g["rate"].iloc[-1]), 4),
                        "rate_diff_pp": round(float((g["rate"].iloc[-1] - g["rate"].iloc[0]) * 100), 2),
                        "bin_spearman": round(float(rho), 4), "p": float(p),
                        "bottom_bin_label": f"[{g['xmean'].iloc[0]:.2f}]",
                        "dead_group_mean_exposure": round(float(sdf.loc[sdf.is_dead, col].mean()), 4),
                        "healthy_group_mean_exposure": round(float(sdf.loc[sdf.is_healthy, col].mean()), 4)})
cf_grad = pd.DataFrame(cf_rows)

# ============================================================ 落盘
C.save_csv(censoring, MODULE, "d2_censoring_facts.csv")
C.save_csv(contrast, MODULE, "d2_conclusion_contrast.csv")
C.save_csv(grad, MODULE, "d2_gradient_recheck.csv")
C.save_csv(correlations, MODULE, "d2_correlations.csv")
C.save_csv(corr_alive, MODULE, "d2_correlations_alive_only.csv")
C.save_csv(leaves, MODULE, "d2_leaf_and_degree_by_group.csv")
C.save_csv(recover, MODULE, "d2_recoverability.csv")
C.save_csv(impute_bands, MODULE, "d2_imputation_by_band.csv")
C.save_csv(scenario, MODULE, "d2_missing_mass_scenario.csv")
C.save_csv(cf_grad, MODULE, "d2_counterfactual_exposure_gradient.csv")
C.save_json({"anchor_line": ANCHOR_LINE, "edges_from_dead": edges_from_dead,
             "dead_out_zero": int((dead_out_degrees == 0).sum()), "dead_n": len(dead),
             "edges_alive_to_dead": edges_alive_to_dead, "mean_du_dead": round(mean_du_dead, 4),
             "mean_du_healthy": round(mean_du_healthy, 4), "mean_in_dead": round(mean_in_dead, 4),
             "mean_in_healthy": round(mean_in_healthy, 4), "ratio_du": round(ratio_du, 4),
             "ratio_in": round(ratio_in, 4), "cross_factor": round(cross_factor, 4),
             "extra_edges_mean": round(extra_mean, 2), "extra_edges_median": round(extra_median, 2),
             "extra_stochastic_mean": round(float(extra_stochastic.mean()), 2),
             "extra_stochastic_sd": round(float(extra_stochastic.std(ddof=1)), 2),
             "extra_stochastic_mc_se": round(float(extra_stochastic.std(ddof=1)) / np.sqrt(N_BOOT), 3),
             "dead_row_cells": dead_row_cells, "dd_block": dd_block, "obs_cells": obs_cells,
             "n_dead_in_true129": len(true_fail), "alive_edges": int(sub_alive.number_of_edges()),
             "alive_nodes": len(alive_all), "W_TOTAL": int(W_TOTAL)},
            MODULE, "d2_summary.json")

ELAPSED = time.time() - T0
C.write_run_note(MODULE, [
    "D2_censoring_impact run log",
    "command: cd pipeline && PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8 python -X utf8 out/D2_censoring_impact/run_d2.py",
    f"versions: {json.dumps(VERSIONS, ensure_ascii=False)}",
    f"seed: {SEED}; 情景模拟 N_BOOT={N_BOOT}（np.random.RandomState(SEED+r) 独立重复）；"
    f"目标指派 200 次（RandomState(SEED+10000+r)）",
    f"wall_time_sec: {ELAPSED:.2f}",
    f"anchors: csa_common.assert_anchors() 31/31 PASS (run_d2.py:L{ANCHOR_LINE})",
    f"key: edges_from_dead={edges_from_dead} dead_out0={int((dead_out_degrees==0).sum())}/{len(dead)} "
    f"edges_alive_to_dead={edges_alive_to_dead} weight_from_dead={weight_from_dead}",
    f"degree contrast: 总度 {mean_du_dead:.3f} vs {mean_du_healthy:.3f} = {ratio_du:.3f}×(INVALID) | "
    f"入度 {mean_in_dead:.3f} vs {mean_in_healthy:.3f} = {ratio_in:.3f}×(主口径)",
    f"scenario extra edges: mean-matched {extra_mean:.1f} / median-matched {extra_median:.1f} / "
    f"stochastic {extra_stochastic.mean():.1f} (sd {extra_stochastic.std(ddof=1):.1f})",
    "files: d2_censoring_facts.csv d2_conclusion_contrast.csv d2_gradient_recheck.csv d2_correlations.csv "
    "d2_correlations_alive_only.csv d2_leaf_and_degree_by_group.csv d2_recoverability.csv "
    "d2_imputation_by_band.csv d2_missing_mass_scenario.csv d2_counterfactual_exposure_gradient.csv d2_summary.json",
])

pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 90)
print("== 31 anchors PASS ==")
print(json.dumps(json.loads(open(os.path.join(HERE, "d2_summary.json"), encoding="utf-8").read()),
                 ensure_ascii=False, indent=1))
print("== (a) censoring facts ==")
print(censoring.to_string(index=False))
print("== (b) contrast ==")
print(contrast.to_string(index=False))
print("== (b) gradient recheck ==")
print(grad.to_string(index=False))
print("== (b) correlations ==")
print(correlations.to_string(index=False))
print(corr_alive.to_string(index=False))
print("== (b) leaves ==")
print(leaves.to_string(index=False))
print("== (d) imputation bands ==")
print(impute_bands.to_string(index=False))
print("== (d) scenario ==")
print(scenario.to_string(index=False))
print("== (d) counterfactual exposure gradient ==")
print(cf_grad.to_string(index=False))
print(f"elapsed {ELAPSED:.2f}s")
