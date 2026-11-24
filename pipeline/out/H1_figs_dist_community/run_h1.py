# -*- coding: utf-8 -*-
"""H1 · 分布与社区图（读 A/B 组已产出的 CSV，不重算任何统计量）。

三张图，每张 PNG(300dpi) + 矢量 PDF + fig_*.csv 数据表：
  1) h1_ccdf_degree_strength —— 度与强度的 CCDF + 四种尾部分布拟合（数据源 A1）
  2) h1_community_layout     —— 共识社区布局（标签来自 B1 的推荐划分）
  3) h1_community_sensitivity—— 社区结论的敏感性矩阵（数据源 B1/B2/B3/B4）

唯一在这里"算"的东西是节点坐标：布局是为了画图的几何，不是分析量；
社区归属一律取 B1/consensus_partitions.csv 的 recommended 行，没有重新跑 Louvain。
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import networkx as nx  # noqa: E402
import matplotlib  # noqa: E402

import csa_common as cs  # noqa: E402
import figstyle as fs  # noqa: E402

MOD = "H1_figs_dist_community"
T0 = time.time()
LOG = []


def log(s):
    LOG.append(s)
    print(s, flush=True)


fs.setup(verbose=True)
D = fs.OUT_DIR
fig_dir = fs.make_dir(MOD)
log("figure dir: %s" % fig_dir)

FIT_COLOR = {
    "powerlaw": ("#c0392b", "-"),
    "trunc_powerlaw": ("#d98a4a", "--"),
    "lognormal": ("#4878a8", "-."),
    "exponential": ("#7f7f7f", ":"),
    "nbinomial": ("#8e44ad", (0, (3, 1, 1, 1))),
}

# ---------------------------------------------------------------- 图 1
ccdf = fs.read_csv("A1_degree_heavy_tail", "ccdf_tables.csv")
fits = fs.read_csv("A1_degree_heavy_tail", "fits.csv")
summ = fs.read_csv("A1_degree_heavy_tail", "summary.csv")
log("A1 ccdf rows=%d  fits rows=%d" % (len(ccdf), len(fits)))

PANELS = [("degree_u_671", "无权度 k"), ("strength_u_671", "加权强度 s")]
fig, axes = fs.plt.subplots(1, 2, figsize=(11.0, 4.4))
tbl_rows = []
for ax, (ser, name) in zip(axes, PANELS):
    d = ccdf[ccdf.series == ser].sort_values("x")
    xs = d.x.to_numpy(float)
    emp = d.ccdf_emp.to_numpy(float)
    # 经验 CCDF 的分辨率下限就是 1/n：拟合曲线画到比它还低的位置没有任何数据支撑，
    # 只会把 y 轴拉到 1e-21 让主区间挤成一条线。所以下限取 1/n，曲线截到这里为止。
    n_tot = int(summ[summ.series == ser].iloc[0]["n"])
    y_floor = 1.0 / n_tot
    ax.step(xs, emp, where="post", color=fs.C_OBS, lw=1.8, zorder=5,
            label="经验 CCDF（n=%d）" % n_tot)
    # 每族分布可能有多个 xmin 的拟合行；主口径取 xmin_source=common_pl_ks_search 的那批
    main = fits[(fits.series == ser) & (fits.xmin_source == "common_pl_ks_search")]
    if len(main) == 0:                      # 强度只在 xmin=1 拟合，可能没有该来源标记
        main = fits[(fits.series == ser) & (fits.x_min == fits[fits.series == ser].x_min.min())]
    for _, r in main.iterrows():
        col = "ccdf_fit_" + r["dist"]
        if col not in d.columns:
            continue
        y = d[col].to_numpy(float)
        ok = np.isfinite(y) & (y >= y_floor)
        if ok.sum() < 3:
            continue
        c, ls = FIT_COLOR[r["dist"]]
        lab = "%s x_min=%g" % (fs.DIST_LABELS.get(r["dist"], r["dist"]), r["x_min"])
        al = r["par_alpha"]
        if np.isfinite(al):
            lab += " α=%.2f" % al
        elif np.isfinite(r["par_x_c"]):
            lab += " x_c=%.0f" % r["par_x_c"]
        elif np.isfinite(r["par_mu"]):
            lab += " μ=%.2f σ=%.2f" % (r["par_mu"], r["par_sigma"])
        lab += " KS=%.3f" % r["KS"]
        ax.plot(xs[ok], y[ok], color=c, ls=ls, lw=1.5, label=lab, zorder=3)
    xmin = float(main.x_min.iloc[0]) if len(main) else np.nan
    if np.isfinite(xmin):
        ax.axvline(xmin, color="k", ls="--", lw=0.9, alpha=0.55, zorder=2)
    row = summ[summ.series == ser].iloc[0]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(y_floor * 0.8, 1.45)
    ax.set_xlim(xs.min() * 0.85, xs.max() * 1.15)
    ax.set_xlabel("%s（对数轴）" % name)
    ax.set_ylabel("P(X ≥ x)")
    fs.log_decimal_axis(ax, "y")
    ax.set_title("%s：尾部形状与四族拟合对比（x_min=%g 起拟合）\n"
                 "Gini=%.3f  CV=%.2f  max=%g  top10 份额=%.1f%%  n=%d"
                 % (name, xmin, row.gini_pos, row.cv, row["max"],
                    100 * row.top10_share, n_tot))
    ax.legend(fontsize=7.0, loc="lower left", ncol=1)
    keep = ["x", "ccdf_emp", "ccdf_fit_powerlaw", "ccdf_fit_trunc_powerlaw",
            "ccdf_fit_lognormal", "ccdf_fit_exponential", "ccdf_fit_nbinomial"]
    t = d[keep].copy()
    t.insert(0, "series", ser)
    tbl_rows.append(t)
fig1 = fs.save_fig(fig, "h1_ccdf_degree_strength", fig_dir, pd.concat(tbl_rows, ignore_index=True))
log("fig1 %s" % fig1)

fig_note = (
    "读图须知：双对数下近似直线不等于幂律。判据是 KS + 自举 p 值 + 配对似然比（见 A1 "
    "lr_tests.csv）。指数族的 ks_boot_p=0.000 被拒；截断幂律/对数正态与幂律彼此无法区分，"
    "因此这里能说的是'重尾、尾部指数约 2 的幂律不被拒绝'，不是'确认无标度'。"
)
log(fig_note)

# ---------------------------------------------------------------- 图 2
part = fs.read_csv("B1_louvain_stability", "consensus_partitions.csv")
rec = part[part.recommended == True]  # noqa: E712
if len(rec) == 0:
    raise RuntimeError("B1 没有 recommended 划分")
log("B1 recommended: method=%s tau=%s n_seeds=%s nodes=%d comms=%d"
    % (rec.method.iloc[0], rec.tau.iloc[0], rec.n_seeds_used.iloc[0],
       rec.node.nunique(), rec.community_id.nunique()))
comm = rec.set_index("node").community_id.to_dict()
size_tbl = fs.read_csv("B1_louvain_stability", "recommended_partition_communities.csv")
size_tbl = size_tbl[(size_tbl.method == rec.method.iloc[0]) & (size_tbl.tau == rec.tau.iloc[0])]

b = cs.build_all()                    # 只取图与节点属性，不产生新的分析量
Gu = b["Gu"]
nodes = sorted(Gu.nodes())
pos = nx.spring_layout(Gu, seed=cs.SEED, k=1.9 / np.sqrt(len(nodes)), iterations=120)

labs = np.array([comm.get(n, -1) for n in nodes])
d_u = np.array([Gu.degree(n) for n in nodes], float)
F = b["fail"]
true_fail = set(F["true_fail"])
false_fail = set(F["false_fail"])
amb = set(F["ambiguous"])

fig, (ax, ax2) = fs.plt.subplots(1, 2, figsize=(13.6, 6.4),
                                 gridspec_kw={"width_ratios": [1.65, 1]})
uniq = sorted({c for c in labs if c >= 0})
# 32 个社区：8 色循环会大量撞色，用 figstyle 的色相环取色（matplotlib 3.11 没有 tab30）
cmap = {c: col for c, col in zip(uniq, fs.cat_colors(len(uniq)))}
xy = np.array([pos[n] for n in nodes])
idx = {n: i for i, n in enumerate(nodes)}
# 边一定要画：只有点的话这张图退化成散点云，看不出"社区"是连出来的还是摆出来的
segs = [(xy[idx[u]], xy[idx[v]]) for u, v in Gu.edges()]
ax.add_collection(matplotlib.collections.LineCollection(
    segs, colors="#444444", linewidths=0.18, alpha=0.16, zorder=1))
for c in uniq:                        # 同社区节点共用一种底色
    sel = labs == c
    ax.scatter(xy[sel, 0], xy[sel, 1], s=8 + 6.0 * d_u[sel], c=cmap[c],
               edgecolors="none", alpha=0.9, linewidths=0, zorder=2)
tf = np.isin(nodes, list(true_fail))
ff = np.isin(nodes, list(false_fail))
am = np.isin(nodes, list(amb))
for mask, colr, tag in ((tf, fs.C_DEAD, "真失效 %d 个（连不上）"),
                        (ff, fs.C_FALSE_FAIL, "假失效 %d 个（认证跳转等）"),
                        (am, fs.C_AMBIG, "口径不明 %d 个")):
    ax.scatter(xy[mask, 0], xy[mask, 1], s=20 + 6.0 * d_u[mask], facecolors="none",
               edgecolors=colr, linewidths=1.3, zorder=4, label=tag % int(mask.sum()))
for n in cs.hub_nodes(Gu):            # 导航枢纽：形状区分，不靠颜色
    if n in comm:
        p = pos[n]
        ax.scatter([p[0]], [p[1]], marker="s", s=70, c="#ffffff",
                   edgecolors=fs.C_HUB, linewidths=1.8, zorder=6)
size_of = pd.Series(labs).value_counts()
for c in list(size_of.index)[:6]:     # 只标最大的 6 个社区，其余靠颜色
    sel = labs == c
    p = xy[sel].mean(axis=0)
    ax.annotate("#%d (%d)" % (c, size_of[c]), p, fontsize=8, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#888888", alpha=0.9),
                zorder=7)
ax.set_title("加权无向图（%d 节点，%d 条边）的共识社区布局\n%s τ=%s，%d 个 seed 共识 → "
             "%d 个社区；点面积∝度，灰线=连边，方块=%d 个导航枢纽\n"
             "（共识划分的 %d 个社区 ≠ 单 seed Louvain 的 25 个：两种口径，各有适用场景）"
             % (Gu.number_of_nodes(), Gu.number_of_edges(),
                rec.method.iloc[0], rec.tau.iloc[0], int(rec.n_seeds_used.iloc[0]),
                len(uniq), len(cs.hub_nodes(Gu)), len(uniq)))
ax.set_xticks([]); ax.set_yticks([])
ax.grid(False)
ax.legend(loc="lower left", fontsize=8)

ss = size_tbl.sort_values("size", ascending=False)
ax2.barh(range(len(ss)), ss["size"], color=[cmap.get(c, "#999") for c in ss.community_id])
ax2.set_yticks(range(len(ss)))
ax2.set_yticklabels(["%s (%s)" % (c, s.split(".")[0])
                     for c, s in zip(ss.community_id, ss.strongest_member)], fontsize=7.5)
ax2.invert_yaxis()
ax2.set_xlabel("社区规模（节点数）")
ax2.set_title("推荐划分的社区规模降序\n共 %d 社区，覆盖 %d 节点"
              % (len(ss), int(ss["size"].sum())))
fig2 = fs.save_fig(fig, "h1_community_layout", fig_dir,
                   fs.table_from_columns({
                       "node": nodes, "community_id": labs,
                       "degree_u": d_u, "x": xy[:, 0], "y": xy[:, 1],
                       "is_true_fail": tf, "is_false_fail": ff,
                       "is_ambiguous": am,
                       "is_nav_hub": np.isin(nodes, list(cs.hub_nodes(Gu)))}))
log("fig2 %s" % fig2)
log("布局口径：坐标=spring_layout(seed=%d)，只用于摆放节点，不是分析结果；"
    "社区标签=B1 共识划分 avglink τ=%s（%d 个 seed），与单 seed Louvain 的 25 个社区"
    "不是同一套数，图上的 %d 是共识社区数。"
    % (cs.SEED, rec.tau.iloc[0], int(rec.n_seeds_used.iloc[0]), len(uniq)))

# ---------------------------------------------------------------- 图 3
nmi = fs.read_csv("B2_algorithm_contrast", "nmi_matrix.csv")
ari = fs.read_csv("B2_algorithm_contrast", "ari_matrix.csv")
sweep = fs.read_csv("B1_louvain_stability", "seed_sweep.csv")
stab = fs.read_csv("B3_resolution_scan", "resolution_stability.csv")
wvar = fs.read_csv("B4_weight_variants", "weight_variant_summary.csv")

fig, axes = fs.plt.subplots(2, 2, figsize=(13.2, 10.0))
axA, axB, axC, axD = axes.ravel()

# (A) 算法 × 算法 NMI 热图
# nmi_matrix.csv 是无行名的对称方阵：行序 = 列序 = algorithm_summary 的算法顺序，实测核对。
alg = fs.read_csv("B2_algorithm_contrast", "algorithm_summary.csv")
cols = list(nmi.columns)
assert list(alg.algorithm) == cols, "nmi_matrix 的列序与 algorithm_summary 不一致，不能假定行序=列序"
assert nmi.shape[0] == nmi.shape[1]
nmi.index = cols
M = nmi.to_numpy(float)
assert np.allclose(M, M.T), "NMI 矩阵不对称"
im = axA.imshow(M, cmap=fs.HEATMAP_CMAP, vmin=0, vmax=1)
short = [c.replace("_", " ") for c in cols]
SHORT_A = {"louvain weighted": "Louvain 加权", "louvain binary": "Louvain 无权",
           "label prop binary": "标签传播 无权", "label prop on weighted": "标签传播 加权",
           "label prop backbone w10": "标签传播 骨架w≥10", "label prop async weighted": "标签传播 异步加权",
           "greedy modularity weighted": "贪心模块度 加权", "greedy modularity binary": "贪心模块度 无权",
           "kernighan lin bisection": "KL 二分", "spectral bisection fiedler": "谱二分 Fiedler"}
short = [SHORT_A.get(c, c) for c in short]
axA.set_xticks(range(len(cols)))
axA.set_xticklabels(short, fontsize=6.0, rotation=55, ha="right")
axA.set_yticks(range(len(cols)))
axA.set_yticklabels(short, fontsize=6.2)
for i in range(len(cols)):
    for j in range(len(cols)):
        if i != j:
            axA.text(j, i, "%.2f" % M[i, j], ha="center", va="center", fontsize=6.0,
                     color="w" if M[i, j] < 0.55 else "k")
axA.grid(False)
axA.set_title("(A) 10 种社区划分算法两两 NMI\n中位数=%.3f 最小=%.3f：划分高度依赖算法"
              % (np.median(M[np.triu_indices(len(cols), 1)]),
                 M[np.triu_indices(len(cols), 1)].min()))
fig.colorbar(im, ax=axA, fraction=0.046, pad=0.02)

# (B) seed 敏感性：社区数与最大社区
al = sweep[sweep.sweep == "all250"] if (sweep.sweep == "all250").any() else sweep
axB.hist(al.n_comms_total, bins=np.arange(al.n_comms_total.min() - 0.5,
           al.n_comms_total.max() + 1.5, 1.0), color=fs.C_EDGE1, label="社区数")
axB.set_xlabel("Louvain 社区数（%d 个 seed）" % len(al))
axB.set_ylabel("频次")
axB2 = axB.twinx()
axB2.hist(al.largest, bins=18, color=fs.C_DEAD, alpha=0.45, label="最大社区规模")
axB2.set_ylabel("最大社区规模分布", color=fs.C_DEAD, labelpad=8)
axB2.grid(False)
axB.set_title("(B) 只换随机种子：%d 个 seed\n社区数 %d–%d（中位 %.0f），最大社区 %d–%d"
              "（中位 %.0f）→ 单个数值不能当结论"
              % (len(al), al.n_comms_total.min(), al.n_comms_total.max(),
                 al.n_comms_total.median(), al.largest.min(), al.largest.max(),
                 al.largest.median()))

# (C) 分辨率扫描
st = stab.sort_values("gamma")
axC.semilogx(st.gamma, st.n_comms_nonsingleton_mean, "o-", color=fs.C_OBS,
             ms=3.5, label="非单点社区数（均值）")
axC.fill_between(st.gamma, st.n_comms_nonsingleton_mean - st.n_comms_nonsingleton_sd,
                 st.n_comms_nonsingleton_mean + st.n_comms_nonsingleton_sd,
                 color=fs.C_OBS, alpha=0.18, lw=0)
axC.set_xlabel("分辨率 γ（对数轴）")
axC.set_ylabel("社区数")
fs.log_decimal_axis(axC, "x")
axC2 = axC.twinx()
axC2.plot(st.gamma, st.nmi_vs_gamma1_mean, "s--", color=fs.C_MODEL, ms=3.2,
          label="与 γ=1 划分的 NMI")
axC2.set_ylabel("NMI vs γ=1", color=fs.C_MODEL)
axC2.set_ylim(0, 1.02)
axC2.grid(False)
g1 = st[(st.gamma - 1.0).abs() == (st.gamma - 1.0).abs().min()]
axC.axvline(1.0, color="k", ls=":", lw=0.9)
# seed 噪声天花板从 B2 的 key_numbers 取，不在这里抄一遍数字（抄了就会和 B2 漂移）
CEIL = fs.read_json("B2_algorithm_contrast", "key_numbers.json")["nmi_seed_ceiling_louvain_weighted"]
axC2.axhline(CEIL, color=fs.C_NULL, ls="--", lw=0.9)
axC2.annotate("seed 噪声天花板 %.4f（B2 实测）" % CEIL, (1.0, CEIL), xytext=(-150, 6),
              textcoords="offset points", fontsize=7, color=fs.C_NULL)
axC.set_title("(C) 分辨率扫描：γ 0.05→4 社区数 %d→%d；\nγ=1 处 NMI=%.2f（≈ seed 天花板 %.2f，"
              "即参照本身就有这么大噪声）"
              % (int(st.n_comms_nonsingleton_mean.iloc[0]),
                 int(st.n_comms_nonsingleton_mean.iloc[-1]),
                 float(g1.nmi_vs_gamma1_mean.iloc[0]), CEIL))

# (D) 权重口径
wv = wvar.copy()
y = np.arange(len(wv))
axD.barh(y - 0.19, wv.n_comms_nonsingleton_median, height=0.36, color=fs.C_EDGE1,
         label="非单点社区数（seed 中位）")
axD.barh(y + 0.19, wv.max_comm_median, height=0.36, color=fs.C_MODEL,
         label="最大社区规模（seed 中位）")
axD.set_yticks(y)
axD.set_yticklabels(wv.variant, fontsize=8)
axD.invert_yaxis()
for i, r in wv.iterrows():
    axD.text(r.n_comms_nonsingleton_median, i - 0.19, " %.0f" % r.n_comms_nonsingleton_median,
             va="center", fontsize=7)
    axD.text(r.max_comm_median, i + 0.19, " %.0f" % r.max_comm_median, va="center", fontsize=7)
axD.set_xlabel("数量（个 / 节点）")
axD.set_title("(D) 权重口径换一下，同一张图给出 %d–%d 个社区\n（每行 31 个 seed 的中位数；"
              "孤立点不计入非单点社区数）"
              % (int(wv.n_comms_nonsingleton_median.min()), int(wv.n_comms_nonsingleton_median.max())))

fig.suptitle("社区结论的四个敏感性来源：算法、种子、分辨率、权重口径（数据源 B1/B2/B3/B4）",
             fontsize=12, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.98))
sens_rows = ([{"panel": "A_nmi", "row": c, **{k: v for k, v in zip(cols, nmi.loc[c].astype(float))}}
              for c in nmi.index]
             + [{"panel": "B_seed", "n_comms_total": r.n_comms_total, "largest": r.largest,
                 "Q": r.Q, "seed": r.seed} for _, r in al.iterrows()]
             + [{"panel": "C_gamma", "gamma": r.gamma,
                 "n_comms_nonsingleton_mean": r.n_comms_nonsingleton_mean,
                 "nmi_vs_gamma1_mean": r.nmi_vs_gamma1_mean,
                 "largest_mean": r.largest_mean} for _, r in st.iterrows()]
             + [{"panel": "D_weight", "variant": r.variant,
                 "n_comms_nonsingleton_median": r.n_comms_nonsingleton_median,
                 "max_comm_median": r.max_comm_median, "Q_mean": r.Q_mean}
                for _, r in wv.iterrows()])
fig3 = fs.save_fig(fig, "h1_community_sensitivity", fig_dir, pd.DataFrame(sens_rows))
log("fig3 %s" % fig3)

# ---------------------------------------------------------------- 收尾
rows = []
for tag, r in (("ccdf", fig1), ("layout", fig2), ("sensitivity", fig3)):
    rows.append({"figure": tag, "png_bytes": r["png_bytes"], "pdf_bytes": r["pdf_bytes"],
                 "csv_rows": r["csv_rows"]})
pd.DataFrame(rows).to_csv(fig_dir / "h1_manifest.csv", index=False, encoding="utf-8-sig")

fs.note(fig_dir, [
    "H1_figs_dist_community —— 度/强度 CCDF + 拟合、社区布局、社区敏感性矩阵",
    "生成时间: %s  用时 %.1fs" % (time.strftime("%Y-%m-%d %H:%M:%S"), time.time() - T0),
    "环境: %s" % cs.versions(),
    "",
    "读 CSV 清单:",
    "  A1_degree_heavy_tail/ccdf_tables.csv, fits.csv, summary.csv",
    "  B1_louvain_stability/consensus_partitions.csv, recommended_partition_communities.csv, seed_sweep.csv",
    "  B2_algorithm_contrast/nmi_matrix.csv",
    "  B3_resolution_scan/resolution_stability.csv",
    "  B4_weight_variants/weight_variant_summary.csv",
    "csa_common 只提供图对象与节点属性（用于画布局），未重算任何统计量。",
    "布局坐标 = nx.spring_layout(Gu, seed=%d)；社区标签 = B1 recommended 划分（%s τ=%s）。"
    % (cs.SEED, rec.method.iloc[0], rec.tau.iloc[0]),
    "",
    "图注:",
    "  h1_ccdf_degree_strength: 双对数 CCDF + 四族拟合；虚线是 x_min。" + fig_note,
    "  h1_community_layout: 点面积∝度；空心圈按失效类型分色（红=真失效 129、"
    "橙=假失效 32、紫=口径不明 1）；白方块=6 个导航枢纽；数字=共识社区号。",
    "  h1_community_sensitivity: (A) 算法两两 NMI；(B) 换 seed 的社区数与最大社区分布；"
    "(C) γ 扫描 vs 与 γ=1 的 NMI（灰虚线=seed 噪声天花板）；(D) 权重口径。",
    "",
    "每张图三件套：PNG(300dpi) + 矢量 PDF + fig_<name>.csv（图背后的数据）。",
] + LOG)
log("done in %.1fs" % (time.time() - T0))
