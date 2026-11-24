# -*- coding: utf-8 -*-
"""H2 · 失效相关图（读 D1/D2/E5 的 CSV，不重算）。

三张图，每张 PNG(300dpi) + 矢量 PDF + fig_*.csv：
  1) h2_health_gradient      —— 健康梯度：总度 vs 入度 vs 抓取量 并排
  2) h2_outdegree_censoring  —— 出度删失示意图（本项目最关键的方法学结论）
  3) h2_observed_failures    —— 观测失效情景 × 三种效率口径对照

图里每个数字都来自上游 CSV；这里只做布局与配色。
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import csa_common as cs  # noqa: E402
import figstyle as fs  # noqa: E402

MOD = "H2_figs_failure"
T0 = time.time()
LOG = []


def log(s):
    LOG.append(s)
    print(s, flush=True)


fs.setup(verbose=False)
fig_dir = fs.make_dir(MOD)

# 中文引号包术语，避免在 ASCII 字符串里嵌套引号
Q = lambda s: "「" + s + "」"  # noqa: E731

OUTCOME = {  # outcome -> (率列, 下界列, 上界列, 计数列, 图例, 颜色)
    "true_fail": ("true_fail_rate", "true_fail_ci_lo", "true_fail_ci_hi",
                  "true_fail_n", "真失效率", fs.C_DEAD),
    "partial": ("partial_rate", "partial_ci_lo", "partial_ci_hi",
                "partial_n", "部分失效率", fs.C_PARTIAL),
}

# ---------------------------------------------------------------- 图 1
dec = fs.read_csv("D1_health_gradient", "d1_decile_table.csv")
gs = fs.read_csv("D1_health_gradient", "d1_gradient_summary.csv")
# 用 sample_tag 选口径，不拿中文列值做匹配（correction 在部分终端会显示成乱码，匹配易出错）
d = dec[(dec.sample_tag == "S_uncorr_671") & (dec.scheme == "rank10")]
PANELS = [("degree_u", "无向总度 k（含出度分量）"),
          ("in_degree", "入度 k_in（主口径）"),
          ("url_count", "URL 数（抓取规模代理）")]
fig, axes = fs.plt.subplots(1, 3, figsize=(14.2, 4.8), sharey=True)
rows1 = []
for ax, (exp, name) in zip(axes, PANELS):
    e = d[d.exposure == exp].sort_values("bin").reset_index(drop=True)
    x = e.bin.to_numpy(float)
    for oc, (yc, lo_c, hi_c, n_c, lab, col) in OUTCOME.items():
        y = e[yc].to_numpy(float)
        lo = e[lo_c].to_numpy(float)
        hi = e[hi_c].to_numpy(float)
        ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o-", color=col,
                    ms=4.5, capsize=2.5, lw=1.4, label=lab)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("%s 的十分位（1 = 最小）" % name)
    g = gs[(gs.exposure == exp) & (gs.scheme == "rank10")
           & (gs.sample_tag == "S_uncorr_671") & (gs.outcome == "true_fail")]
    ax.set_title("%s\n底档→顶档 真失效 %.1f%% → %.1f%%（差 %+.1f pp）"
                 % (exp, 100 * e.true_fail_rate.iloc[0], 100 * e.true_fail_rate.iloc[-1],
                    100 * (e.true_fail_rate.iloc[-1] - e.true_fail_rate.iloc[0])))
    ax.text(0.03, 0.95, "ρ=%.3f  p=%.1e\n档间单调：%s"
            % (g.spearman_binlevel_rho.iloc[0], g.spearman_binlevel_p.iloc[0],
               "是" if bool(g.monotone_in_bin_order.iloc[0]) else "否"),
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="#f2f2f2", ec="none"))
    keep = e[["bin", "bin_label", "n_sample", "true_fail_n", "true_fail_rate",
              "true_fail_ci_lo", "true_fail_ci_hi", "partial_n", "partial_rate",
              "partial_ci_lo", "partial_ci_hi"]].copy()
    keep.insert(0, "exposure", exp)
    rows1.append(keep)
axes[0].set_ylabel("比率（误差棒 = Wilson 95% 区间）")
axes[0].legend(fontsize=8, loc="center right")
fig.suptitle("健康梯度：同一句" + Q("边缘更脆弱") + "，换成入度口径后斜率明显变缓——"
             "总度口径里混进了不可观测的出度分量", fontsize=11.5)
fig.tight_layout(rect=(0, 0, 1, 0.92))
r1 = fs.save_fig(fig, "h2_health_gradient", fig_dir, pd.concat(rows1, ignore_index=True))
log("fig1 %s" % r1)

# ---------------------------------------------------------------- 图 2
facts = fs.read_csv("D2_censoring_impact", "d2_censoring_facts.csv").set_index("item").value
leaf = fs.read_csv("D2_censoring_impact", "d2_leaf_and_degree_by_group.csv")
cont = fs.read_csv("D2_censoring_impact", "d2_conclusion_contrast.csv")

fig, (axa, axb) = fs.plt.subplots(1, 2, figsize=(13.8, 5.4),
                                  gridspec_kw={"width_ratios": [1.15, 1]})
groups = ["dead162", "true129", "partial149", "healthy360"]
gl = leaf.set_index("group").loc[groups]
xx = np.arange(len(gl))
axa.bar(xx - 0.18, gl.mean_in_degree, width=0.36, color=fs.C_EDGE1, label="平均入度（可观测）")
axa.bar(xx + 0.18, gl.mean_out_degree, width=0.36, color=fs.C_DEAD, label="平均出度")
for i, v in enumerate(gl.mean_out_degree.to_numpy(float)):
    # 出度恒 0 时柱子没有高度，标签放到 0 线上方，否则会压住 x 轴刻度
    axa.annotate("%.4f" % v, (xx[i] + 0.18, 0.06 if v == 0 else v), ha="center",
                 fontsize=8, va="bottom", color=fs.C_DEAD)
axa.set_xticks(xx)
axa.set_xticklabels(["%s\nn=%d" % (g, gl.loc[g, "n"]) for g in groups], fontsize=8.5)
axa.set_ylabel("度（条）")
axa.set_ylim(0, 7.4)                    # 顶部留白放结论框，避免压住柱子
axa.legend(fontsize=8.5, loc="upper left")
axa.set_title("(a) 出度删失：死站/真失效站的出度恒等于 0\n"
              "%d/%d 个整站不可用节点出度为 0；以死站为起点的有向边 = %d 条（锚点）"
              % (int(facts["dead_out_degree_zero_n"]), int(facts["dead_n"]),
                 int(facts["edges_from_dead"])))
axa.text(0.985, 0.97, "不是" + Q("低估") + "，是" + Q("缺失") + "\n→ 死—死相邻构造性不可识别",
         transform=axa.transAxes, ha="right", va="top", fontsize=8.5, color=fs.C_DEAD,
         bbox=dict(boxstyle="round,pad=0.3", fc="#fdf0f0", ec=fs.C_DEAD, lw=0.8))

# (b) 同一句话在两种度口径下的数值差——倍数直接从 leaf 表算，不抄 D2 正文。
# 方向取 健康站÷死站（与 D2 的 5.195× / 2.705× 同一方向），反过来会变成 0.19 这种倒数。
d_dead = float(gl.loc["healthy360", "mean_degree_u"]) / float(gl.loc["dead162", "mean_degree_u"])
d_in = float(gl.loc["healthy360", "mean_in_degree"]) / float(gl.loc["dead162", "mean_in_degree"])
lbl = ["健康站÷死站\n平均度倍数", "死站叶子占比\n（度≤1）", "健康站叶子占比\n（度≤1）"]
total_v = [d_dead, float(gl.loc["dead162", "share_degree_u_le_1"]),
           float(gl.loc["healthy360", "share_degree_u_le_1"])]
indeg_v = [d_in, float(gl.loc["dead162", "share_in_le_1"]),
           float(gl.loc["healthy360", "share_in_le_1"])]
yy = np.arange(len(lbl))
axb.barh(yy - 0.19, total_v, height=0.36, color=fs.C_MODEL, label="无向总度口径")
axb.barh(yy + 0.19, indeg_v, height=0.36, color=fs.C_EDGE1, label="入度口径")
for i, (a, b) in enumerate(zip(total_v, indeg_v)):
    axb.text(a, i - 0.19, " %.3f" % a, va="center", fontsize=8)
    axb.text(b, i + 0.19, " %.3f" % b, va="center", fontsize=8)
axb.set_yticks(yy)
axb.set_yticklabels(lbl, fontsize=8.5)
axb.invert_yaxis()
axb.legend(fontsize=8.5, loc="lower right")
axb.set_title("(b) 同一句话换个度，数字差这么多\n%.3f× = %.3f×（同口径）× %.3f（纯删失自由度）"
              % (d_dead, d_in, d_dead / d_in))
axb.set_xlabel("倍数 / 占比")
rows2 = [{"panel": "a_group_means", "group": g,
          "n": int(gl.loc[g, "n"]),
          "mean_in_degree": float(gl.loc[g, "mean_in_degree"]),
          "mean_out_degree": float(gl.loc[g, "mean_out_degree"]),
          "mean_degree_u": float(gl.loc[g, "mean_degree_u"]),
          "share_degree_u_le_1": float(gl.loc[g, "share_degree_u_le_1"]),
          "share_in_le_1": float(gl.loc[g, "share_in_le_1"]),
          "share_out_eq_0": float(gl.loc[g, "share_out_eq_0"])} for g in groups]
rows2 += [{"panel": "b_degree_scale_contrast", "quantity": q,
           "undirected_total_degree": a, "in_degree": b}
          for q, a, b in zip(lbl, total_v, indeg_v)]
rows2 += [{"panel": "c_conclusion_text_contrast", "结论": r["结论"],
           "总度口径": r["总度口径"], "入度口径": r["入度口径"],
           "判定": r["判定"], "为什么": r["为什么"]} for _, r in cont.iterrows()]
rows2 += [{"panel": "d_censoring_facts", "item": k, "value": v}
          for k, v in facts.items()]
r2 = fs.save_fig(fig, "h2_outdegree_censoring", fig_dir, pd.DataFrame(rows2))
log("fig2 %s" % r2)

# ---------------------------------------------------------------- 图 3
sm = fs.read_csv("E5_observed_scenarios", "scenario_metrics.csv")
sm = sm[sm.view == "as_observed"].sort_values("k_removed").reset_index(drop=True)
fig, (axc, axd) = fs.plt.subplots(1, 2, figsize=(13.8, 5.4),
                                  gridspec_kw={"width_ratios": [1.3, 1]})
x = np.arange(len(sm))
w = 0.26
axc.bar(x - w, sm.dE_full_pct, width=w, color=fs.C_NULL, label="Δ E_full（分母随删除收缩）")
axc.bar(x, sm.dE_gcc_pct, width=w, color=fs.C_EDGE2, label="Δ E_gcc（只在最大连通片内）")
axc.bar(x + w, sm.dE_fixed_pct, width=w, color=fs.C_DEAD, label="Δ E_fixed（分母固定 453602 对）")
axc.axhline(0, color="k", lw=0.8)
axc.set_xticks(x)
axc.set_xticklabels(["%s\nk=%d" % (s, k) for s, k in zip(sm.scenario, sm.k_removed)],
                    fontsize=7.2, rotation=20, ha="right")
axc.set_ylabel("相对基线的效率变化（%）")
axc.legend(fontsize=7.6, loc="lower left")
for i, r in enumerate(sm.itertuples()):
    if r.E_full_rises:
        axc.annotate("E_full 反而上升\n= 分母假象", (i - w, r.dE_full_pct),
                     xytext=(0, 5), textcoords="offset points",
                     ha="center", fontsize=6.8, color=fs.C_NULL)
axc.set_title("(a) 同一批真实失效，三种效率口径能给出符号相反的答案\n"
              "判据只读 E_fixed（CONVENTIONS §2.3）；另两口径仅作对照")

axd.plot(sm.k_removed, sm.E_fixed_ratio_vs_random_same_k, "o-", color=fs.C_OBS, lw=1.6,
         ms=5, label="E_fixed ÷ 同 k 随机基线")
axd.axhline(1.0, color=fs.C_NULL, ls="--", lw=1.0)
axd.set_xscale("log")
axd.set_xlabel("删除节点数 k（对数轴）")
axd.set_ylabel("相对同 k 随机删除的损伤倍数")
for _, r in sm.iterrows():
    axd.annotate(r.scenario, (r.k_removed, r.E_fixed_ratio_vs_random_same_k),
                 xytext=(4, 3), textcoords="offset points", fontsize=6.6)
axd.set_title("(b) 观测失效 vs 同规模随机删除\n>1 = 比随机更疼；含真失效的情景全部在 1 之上")
axd.legend(fontsize=8, loc="lower right")
fig.tight_layout()
r3 = fs.save_fig(fig, "h2_observed_failures", fig_dir, sm[[
    "scenario", "k_removed", "E_full", "E_gcc", "E_fixed", "dE_full_pct", "dE_gcc_pct",
    "dE_fixed_pct", "E_fixed_ratio_vs_random_same_k", "rel_damage_vs_random_pct",
    "rand_mean_E_fixed", "rand_sd_E_fixed", "z_vs_random_same_k", "gcc_ratio_vs_N0",
    "E_full_rises", "artifact_gap_Efull_minus_Efixed_pct_points",
    "significant_at_0.05", "conclusion_read_from"]])
log("fig3 %s" % r3)

# ---------------------------------------------------------------- 收尾
pd.DataFrame([{"figure": "h2_health_gradient", **r1},
              {"figure": "h2_outdegree_censoring", **r2},
              {"figure": "h2_observed_failures", **r3}]).to_csv(
    fig_dir / "h2_manifest.csv", index=False, encoding="utf-8-sig")

fs.note(fig_dir, [
    "H2_figs_failure —— 健康梯度 / 出度删失示意 / 观测失效三效率对照",
    "生成时间 %s  用时 %.1fs" % (time.strftime("%Y-%m-%d %H:%M:%S"), time.time() - T0),
    "环境 %s" % cs.versions(),
    "",
    "数据源（全部读 CSV，未重算任何统计量）:",
    "  D1_health_gradient/d1_decile_table.csv, d1_gradient_summary.csv",
    "  D2_censoring_impact/d2_censoring_facts.csv, d2_leaf_and_degree_by_group.csv, d2_conclusion_contrast.csv",
    "  E5_observed_scenarios/scenario_metrics.csv (view=as_observed)",
    "",
    "口径选择:",
    "  D1: sample_tag=S_uncorr_671（校正前）、scheme=rank10（秩十分位）；outcome 取 true_fail 与 partial。",
    "  D2: 倍数不抄正文数字，直接由 d2_leaf_and_degree_by_group.csv 的组均值相除得到。",
    "  E5: 只画 as_observed 视图；corrected_SS22 视图留在 E5 自己的 CSV 里。",
    "  效率判据一律读 E_fixed（CONVENTIONS §2.3）；E_full/E_gcc 只作对照，不上升为结论。",
    "",
    "图注:",
    "  h2_health_gradient: 三个暴露量的十分位 × 真失效率/部分失效率，误差棒为 Wilson 95% 区间；"
    "左上角是档间 Spearman ρ 与其 p 值。",
    "  h2_outdegree_censoring: (a) 分组平均入度 vs 平均出度，死站出度恒 0；"
    "(b) 同一结论在无向总度口径与入度口径下的数值差；(c) 文字版判定见数据表。",
    "  h2_observed_failures: (a) 9 个观测失效情景 × 三种效率口径的百分比变化；"
    "(b) E_fixed 相对同 k 随机删除的损伤倍数（log-k 轴）。",
    "",
    "注意：本图不使用 nodes_full.csv 的 degree / color 禁用列。",
    "",
] + LOG)
log("done %.1fs" % (time.time() - T0))
