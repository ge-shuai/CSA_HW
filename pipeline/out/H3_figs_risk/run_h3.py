# -*- coding: utf-8 -*-
"""H3 · 风险矩阵与中心性一致性的成图单元。

只读上游 CSV（G1 风险矩阵、C1 排序相关矩阵），本单元不产生任何新的分析量。
唯一从原始列派生的是「同一张风险矩阵换个口径后 Q1 名单还剩多少重合」——
它是对 G1 已经写好的 __quadrant 列做集合比较，不重新打分、不重新分位。

MANIFEST 还要求一张 z–P 角色地图，那是 C5_functional_roles 的产物。
C5 本轮没有跑（无数据），所以这张图**主动缺席**，不做任何示意性替代。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import figstyle as fs  # noqa: E402

fs.setup()
import matplotlib.pyplot as plt  # noqa: E402

OUT = fs.make_dir("H3_figs_risk")
rows = []

ns = fs.read_csv("G1_risk_matrix", "g1_nodes_scored.csv")
qs = fs.read_csv("G1_risk_matrix", "g1_quadrant_summary.csv")
rho = fs.read_csv("C1_centrality_matrix", "spearman_matrix.csv")
tau = fs.read_csv("C1_centrality_matrix", "kendall_matrix.csv")


def as_bool(s):
    """G1 的 0/1 列在不同版本里可能是 bool 也可能是 float，两种都要认。"""
    num = pd.to_numeric(s, errors="coerce")
    txt = s.astype(str).str.strip().str.lower()
    return (num == 1) | txt.isin(["true", "yes"])


# 分类别着色：健康 / 部分失效 / 真失效 / 假失效 / 口径不明。
# 假失效（认证跳转那一类）必须和真失效分开画，否则风险矩阵右上角
# 会被读成「高结构 + 已宕机」，而其中一部分只是探针被跳转到登录页。
CATS = [
    ("healthy", "is_healthy", "存活", fs.C_ALIVE),
    ("partial", "is_partial", "部分失效", fs.C_PARTIAL),
    ("true_fail", "true_fail", "真失效", fs.C_DEAD),
    ("false_fail", "false_fail", "假失效（认证跳转等）", fs.C_FALSE_FAIL),
    ("ambiguous_fail", "ambiguous_fail", "口径不明", fs.C_AMBIG),
]
flags = {k: as_bool(ns[c]) for k, c, _, _ in CATS}
n_cat = {k: int(v.sum()) for k, v in flags.items()}

# ---------------------------------------------------------------- 图 1
PANELS = [("in_degree|raw", "入度（原始权重）"),
          ("in_degree|corr", "入度（校正后）"),
          ("deg_u|raw", "总度（原始）—— 对照，含删失")]
fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.4), sharey=True)
tbl = []
for ax, (tb, lab) in zip(axes, PANELS):
    xp, up = ns[tb + "__x_pct"], ns[tb + "__u_pct"]
    for key, _, legend, colr in CATS:
        m = flags[key]
        ax.scatter(xp[m], up[m], s=16, c=colr, alpha=0.8,
                   edgecolors="none", label="%s %d 个" % (legend, n_cat[key]))
    ax.axvline(50, color="#888888", lw=0.9, ls=":")
    ax.axhline(50, color="#888888", lw=0.9, ls=":")
    q1 = (xp > 50) & (up > 50)
    ax.text(0.97, 0.03, "右上 Q1 = %d 个\n其中真失效 %d / 假失效 %d"
            % (int(q1.sum()),
               int((q1 & flags["true_fail"]).sum()),
               int((q1 & flags["false_fail"]).sum())),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbbbbb", alpha=0.9))
    ax.set_xlabel("%s：结构分位 x_pct" % lab)
    ax.set_xlim(-2, 102)
    ax.legend(loc="upper left", fontsize=7.5)
    sub = qs[qs.table == tb]
    for _, r in sub.iterrows():
        tbl.append(dict(table=tb, table_cn=lab, quadrant=str(r.quadrant),
                        n=int(r.n), share_pct=100.0 * float(r.share_of_scored),
                        n_healthy=int(r.n_healthy), n_partial=int(r.n_partial),
                        n_true_fail=int(r.n_true_fail),
                        n_false_fail=int(r.n_false_fail),
                        n_ambiguous=int(r.n_ambiguous)))
axes[0].set_ylabel("不可用分位 u_pct（50 = 中位数切）")
fig.suptitle("同一批 671 个站点、三种口径的风险矩阵：横轴=结构位置分位，纵轴=不可用程度分位\n"
             "读法只到「同时出现」为止：右上格是「结构位置靠前且探不到」的站点清单，"
             "不是说结构位置造成了宕机", fontsize=11)
fig.text(0.5, 0.085, "注意第三张（总度）：死站的出度被采集截断成恒 0，"
         "总度把这一列 0 当信息用了，所以它的分位本身就和「已宕机」绑死，只能当反面教材。",
         ha="center", fontsize=8.5, color="#7a3b00")
fig.tight_layout(rect=(0, 0.11, 1, 0.94))
rows.append(fs.save_fig(fig, "h3_risk_matrix", OUT, tbl))

# ---------------------------------------------------------------- 图 2
tables = list(qs.table.unique())
q1sets = {}
for tb in tables:
    col = ns[tb + "__quadrant"].astype(str)
    q1sets[tb] = set(np.where(col.str.startswith("Q1"))[0])
M = np.zeros((len(tables), len(tables)))
for i, a in enumerate(tables):
    for j, b in enumerate(tables):
        u = len(q1sets[a] | q1sets[b])
        M[i, j] = (len(q1sets[a] & q1sets[b]) / u) if u else np.nan

fig, (axa, axb) = plt.subplots(1, 2, figsize=(13.4, 5.6),
                               gridspec_kw={"width_ratios": [1.0, 1.25]})
comp = ["n_healthy", "n_partial", "n_true_fail", "n_false_fail", "n_ambiguous"]
lab_comp = ["存活", "部分失效", "真失效", "假失效", "口径不明"]
col_comp = [fs.C_ALIVE, fs.C_PARTIAL, fs.C_DEAD, fs.C_FALSE_FAIL, fs.C_AMBIG]
sub = qs[qs.quadrant.astype(str).str.startswith("Q1")].set_index("table")
sub = sub.reindex(tables)
y = np.arange(len(tables))
left = np.zeros(len(tables))
for c, lc, colr in zip(comp, lab_comp, col_comp):
    v = sub[c].to_numpy(dtype=float)
    axa.barh(y, v, left=left, color=colr, edgecolor="white", linewidth=0.6, label=lc)
    for yy, (vv, ll) in enumerate(zip(v, left)):
        if vv >= 4:
            axa.text(ll + vv / 2, yy, "%d" % int(vv), ha="center", va="center",
                     fontsize=8, color="white")
    left += v
axa.set_yticks(y)
axa.set_yticklabels(tables, fontsize=8.5)
axa.invert_yaxis()
axa.set_xlabel("Q1（高结构 × 高不可用）里的站点数")
axa.set_title("(A) 风险矩阵右上格换了口径以后：总数在 %d–%d 之间跳，\n"
              "而且「其中真失效」的个数也跟着跳——口径选择本身就是结论的一部分"
              % (int(left.min()), int(left.max())), fontsize=9.5)
axa.legend(loc="lower right", fontsize=8, ncol=2)

im = axb.imshow(M, cmap=fs.HEATMAP_CMAP, vmin=0, vmax=1)
for i in range(len(tables)):
    for j in range(len(tables)):
        axb.text(j, i, "%.2f" % M[i, j], ha="center", va="center", fontsize=7,
                 color="white" if M[i, j] < 0.55 else "#111111")
axb.set_xticks(range(len(tables)))
axb.set_xticklabels(tables, rotation=42, ha="right", fontsize=8)
axb.set_yticks(range(len(tables)))
axb.set_yticklabels(tables, fontsize=8)
axb.set_title("(B) 两套口径的 Q1 名单 Jaccard 重合率\n"
              "1.00 = 同一份名单；0.5 上下意味着「高风险名单」不是唯一答案", fontsize=9.5)
axb.grid(False)
fig.colorbar(im, ax=axb, fraction=0.046, pad=0.03, label="Q1 名单 Jaccard")
fig.tight_layout()
rows.append(fs.save_fig(
    fig, "h3_risk_quadrant_robustness", OUT,
    fs.table_from_columns({
        "table": tables,
        "n_Q1": [int(sub.loc[t, "n"]) for t in tables],
        "n_Q1_true_fail": [int(sub.loc[t, "n_true_fail"]) for t in tables],
        "n_Q1_false_fail": [int(sub.loc[t, "n_false_fail"]) for t in tables],
        "n_Q1_healthy": [int(sub.loc[t, "n_healthy"]) for t in tables],
        "jaccard_min_vs_others": [float(np.nanmin(np.delete(M[i], i)))
                                  for i in range(len(tables))],
        "jaccard_median_vs_others": [float(np.nanmedian(np.delete(M[i], i)))
                                     for i in range(len(tables))]})))

# ---------------------------------------------------------------- 图 3
MEAS_CN = {
    "deg_u": "总度 k", "in_degree": "入度", "out_degree": "出度",
    "in_strength": "入强度", "out_strength": "出强度", "bet_u": "介数",
    "pr_dw": "PageRank 有向加权", "pr_uu": "PageRank 无向无权",
    "hits_hub": "HITS hub", "hits_auth": "HITS auth", "kcore": "k-core",
    "hidx_in_in": "H-index(入-入)",
}
CENSORED = {"out_degree", "out_strength"}
for df in (rho, tau):
    # C1 把指标名写在第一列（读进来成了 'Unnamed: 0'），先转成索引再按名字对齐。
    if "Unnamed: 0" in df.columns:
        df.set_index("Unnamed: 0", inplace=True)
    df.index = [str(x).strip() for x in df.index]
    df.columns = [str(c).strip() for c in df.columns]
assert list(rho.index) == list(rho.columns), "Spearman 矩阵不是方阵"
S = rho.to_numpy(dtype=float)
K = tau.reindex(index=rho.index, columns=rho.columns).to_numpy(dtype=float)
order = list(rho.index)
n = len(order)
Z = np.full((n, n), np.nan)
for i in range(n):
    for j in range(n):
        if j > i:
            Z[i, j] = S[i, j]        # 上三角 Spearman
        elif j < i:
            Z[i, j] = K[i, j]        # 下三角 Kendall
fig, ax = plt.subplots(figsize=(10.4, 8.8))
mask = np.isnan(Z)
cax = ax.imshow(np.ma.masked_array(Z, mask), cmap=fs.HEATMAP_CMAP,
                vmin=0, vmax=1, interpolation="nearest")
for i in range(n):
    for j in range(n):
        if not mask[i, j]:
            ax.text(j, i, "%.2f" % Z[i, j], ha="center", va="center", fontsize=7,
                    color="white" if Z[i, j] < 0.55 else "#111111")
labels = [MEAS_CN.get(x, x) for x in order]
ax.set_xticks(range(n))
ax.set_xticklabels(labels, rotation=48, ha="right", fontsize=8.5)
ax.set_yticks(range(n))
ax.set_yticklabels(labels, fontsize=8.5)
for i, x in enumerate(order):
    if x in CENSORED:
        ax.get_xticklabels()[i].set_color(fs.C_DEAD)
        ax.get_yticklabels()[i].set_color(fs.C_DEAD)
ax.plot([0, n - 1], [0, n - 1], color="white", lw=1.2)
ax.grid(False)
ax.set_title("12 个「重要性」指标的排序相关：上三角 Spearman ρ，下三角 Kendall τ\n"
             "红字的两个指标（出度、出强度）被采集截断污染，它们彼此的高相关\n"
             "有一部分是「同一批死站出度都是 0」这件事，不是两条独立的信息", fontsize=10.5)
fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.03, label="秩相关系数")
fig.tight_layout()
pairs = []
for i in range(n):
    for j in range(i + 1, n):
        pairs.append(dict(measure_a=order[i], measure_b=order[j],
                          label_a=MEAS_CN.get(order[i], order[i]),
                          label_b=MEAS_CN.get(order[j], order[j]),
                          spearman=S[i, j], kendall=K[i, j],
                          a_censored=order[i] in CENSORED,
                          b_censored=order[j] in CENSORED))
pairs.sort(key=lambda d: -d["spearman"])
rows.append(fs.save_fig(fig, "h3_centrality_rank_corr", OUT, pairs))

pd.DataFrame(rows).to_csv(OUT / "h3_manifest.csv", index=False, encoding="utf-8-sig")
fs.note(OUT, [
    "H3 · 风险矩阵 + 中心性排序相关 成图单元",
    "上游只读：G1_risk_matrix/g1_nodes_scored.csv, g1_quadrant_summary.csv;",
    "        C1_centrality_matrix/spearman_matrix.csv, kendall_matrix.csv",
    "本单元不重算分数、不重新分位；唯一派生量是 Q1 名单跨口径的 Jaccard 重合率。",
    "",
    "MANIFEST 要求的第三张图（z–P 角色地图）本单元**没有产出**：",
    "  它依赖 C5_functional_roles，本轮 C5 未运行、目录里没有数据。",
    "  宁缺毋滥——不拿示意图冒充实测。补 C5 之后再回来加这张图。",
    "",
    "图 1 h3_risk_matrix：三种口径的风险矩阵散点，真/假失效分开着色。",
    "图 2 h3_risk_quadrant_robustness：Q1 格的人数构成 + 跨口径名单 Jaccard。",
    "图 3 h3_centrality_rank_corr：Spearman 上三角 / Kendall 下三角。",
    "",
    "出图三件套：PNG 300dpi + 矢量 PDF + fig_*.csv 伴生数据表。",
    "产物：%s" % ", ".join(r["fig"] for r in rows),
])
for r in rows:
    print(r)
print("[H3] done")
