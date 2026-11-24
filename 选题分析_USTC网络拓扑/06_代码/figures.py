# -*- coding: utf-8 -*-
"""生成选题分析用图"""
import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter

BASE = r"d:\科大\作业\复杂系统分析\大作业\USTC网络拓扑分析"
OUT = r"d:\科大\作业\复杂系统分析\大作业\选题分析_USTC网络拓扑\05_图表"
os.makedirs(OUT, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

nodes = pd.read_csv(os.path.join(BASE, "nodes_full.csv"), encoding="utf-8-sig")
links = pd.read_csv(os.path.join(BASE, "links.csv"), encoding="utf-8-sig")
node = pd.read_csv(os.path.join(r"d:\科大\作业\复杂系统分析\大作业\选题分析_USTC网络拓扑\02_预实验", "node_metrics_enriched.csv"), encoding="utf-8-sig").set_index("subdomain")

G = nx.Graph()
for _, r in links.iterrows():
    s, t, w = str(r["source"]), str(r["target"]), float(r["weight"])
    if G.has_edge(s, t): G[s][t]["weight"] += w
    else: G.add_edge(s, t, weight=w)
Gd = nx.DiGraph()
for _, r in links.iterrows():
    s, t, w = str(r["source"]), str(r["target"]), float(r["weight"])
    if Gd.has_edge(s, t): Gd[s][t]["weight"] += w
    else: Gd.add_edge(s, t, weight=w)

# ---------- 图1：度与强度 CCDF ----------
deg = np.array([d for _, d in G.degree()])
strength = np.array([d for _, d in G.degree(weight="weight")])
def ccdf(x):
    x = np.sort(x); n = len(x)
    u, idx = np.unique(x, return_index=True)
    return u, 1.0 - idx / n
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, x, label, color in [(axes[0], deg, "度 k", "#2b6cb0"), (axes[1], strength, "点强度 s（边权之和）", "#c53030")]:
    u, y = ccdf(x)
    ax.loglog(u, y, marker=".", linestyle="none", markersize=4, color=color, alpha=.75)
    ax.set_xlabel(label); ax.set_ylabel("P(X ≥ x)")
    ax.grid(alpha=.25, which="both")
    fmt = FuncFormatter(lambda v, p: f"{v:g}")
    ax.xaxis.set_major_formatter(fmt); ax.yaxis.set_major_formatter(fmt)
    ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
axes[0].set_title("(a) 度分布：重尾，但非干净幂律")
axes[1].set_title("(b) 强度分布：比度更集中")
fig.suptitle("USTC 子域超链接网络的度与强度分布（676 节点，2498 条无向边）", y=1.02, fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_degree_strength_ccdf.png"), bbox_inches="tight")
plt.close(fig)

# ---------- 图2：Louvain 社区布局（社区网格，避免力导向毛线球） ----------
GC = G  # 全图 676 节点（含 2 节点小岛），与正文表格口径一致
comms = sorted(nx.community.louvain_communities(GC, weight="weight", seed=42), key=len, reverse=True)
cmap = {n: i for i, c in enumerate(comms) for n in c}
pos = {}
cols = 5
for i, c in enumerate(comms):
    cx = (i % cols) * 1.75
    cy = -(i // cols) * 1.75
    members = sorted(c)
    rr = 0.06 + 0.075 * np.sqrt(len(members))
    for j, n in enumerate(members):
        ang = 2 * np.pi * j / len(members)
        pos[n] = (cx + rr * np.cos(ang), cy + rr * np.sin(ang))
fig, ax = plt.subplots(figsize=(10.5, 9.0))
palette = plt.cm.tab20(np.linspace(0, 1, 20))
node_colors = [palette[cmap[n] % 20] for n in GC.nodes()]
node_sizes = [5 + 1.8 * np.sqrt(GC.degree(n)) for n in GC.nodes()]
strong_edges = [(u, v) for u, v, d in GC.edges(data=True) if d["weight"] >= 5]
nx.draw_networkx_edges(GC, pos, edgelist=strong_edges, ax=ax, alpha=.10, width=.35, edge_color="#777")
nx.draw_networkx_nodes(GC, pos, ax=ax, node_color=node_colors, node_size=node_sizes, linewidths=0)
top = sorted(GC.degree, key=lambda x: -x[1])[:8]
for n, _ in top:
    ax.annotate(n.replace(".ustc.edu.cn", ""), pos[n], fontsize=7.5, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=.8))
# 标注每个社区的大小
for i, c in enumerate(comms):
    cx = (i % cols) * 1.75
    cy = -(i // cols) * 1.75
    ax.text(cx, cy + 0.5, f"{len(c)}", fontsize=7, color="#666", ha="center", va="bottom")
ax.set_title("加权 Louvain：25 个社区（Q=0.690），而不是一个 82% 的巨型社区\n（每个团簇=一个社区，数字为社区规模；点大小=度）", fontsize=10.5)
ax.axis("off"); ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_louvain_communities.png"), bbox_inches="tight")
plt.close(fig)

# ---------- 图3：鲁棒性曲线 ----------
rob = pd.read_csv(os.path.join(r"d:\科大\作业\复杂系统分析\大作业\选题分析_USTC网络拓扑\02_预实验", "robustness_curves.csv"), encoding="utf-8-sig")
agg = rob.groupby(["strategy", "frac"]).agg(gcc=("gcc", "mean"), eff=("eff", "mean")).reset_index()
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
colors = {"degree": "#c53030", "betweenness": "#dd6b20", "pagerank": "#805ad5", "random": "#2b6cb0"}
labels = {"degree": "按度攻击", "betweenness": "按介数攻击", "pagerank": "按 PageRank 攻击", "random": "随机失效"}
for strat, g in agg.groupby("strategy"):
    axes[0].plot(g.frac * 100, g.gcc * 100, marker="o", ms=3, color=colors.get(strat, "gray"), label=labels.get(strat, strat))
    axes[1].plot(g.frac * 100, g.eff, marker="o", ms=3, color=colors.get(strat, "gray"), label=labels.get(strat, strat))
axes[0].set_xlabel("移除节点比例 (%)"); axes[0].set_ylabel("最大连通分量比例 (%)")
axes[0].set_title("(a) 连通性：10% 枢纽移除即腰斩")
axes[1].set_xlabel("移除节点比例 (%)"); axes[1].set_ylabel("全局效率 E")
axes[1].set_title("(b) 传输效率：下降更快")
for ax in axes:
    ax.grid(alpha=.25); ax.legend(fontsize=8, frameon=False)
fig.suptitle("鲁棒且脆弱：随机失效 vs 蓄意攻击（674 节点主分量）", y=1.02, fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_robustness.png"), bbox_inches="tight")
plt.close(fig)

# ---------- 图4：健康度 vs 度 ----------
node["deg_decile"] = pd.qcut(node["degree"], 10, labels=False, duplicates="drop")
tab = node.groupby("deg_decile").agg(mean_deg=("degree", "mean"), dead_rate=("dead", "mean"), n=("sub", "size"))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
axes[0].bar(range(len(tab)), tab.dead_rate * 100, color="#c53030", alpha=.8)
axes[0].set_xticks(range(len(tab)))
axes[0].set_xticklabels([f"{int(x)}" for x in tab.mean_deg], fontsize=8)
axes[0].set_xlabel("度十分位（组内平均度）"); axes[0].set_ylabel("全挂站点比例 (%)")
axes[0].set_title("(a) 度越低，越容易整站失效")
axes[0].grid(axis="y", alpha=.25)
axes[1].scatter(node["degree"] + 1, node["ok_ratio"], s=14, alpha=.45,
                c=np.where(node["dead"] == 1, "#c53030", "#2b6cb0"))
axes[1].set_xscale("log"); axes[1].set_xlabel("度 k（log）"); axes[1].set_ylabel("URL 可用率 (%)")
axes[1].set_title("(b) 度与可用率正相关（Spearman ρ=0.38）")
axes[1].grid(alpha=.25)
fig.suptitle("结构性边缘与站点健康：相关不等于因果（失效站点的度可能被低估）", y=1.02, fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_health_vs_degree.png"), bbox_inches="tight")
plt.close(fig)

# ---------- 图5：风险矩阵 ----------
fig, ax = plt.subplots(figsize=(8.6, 6.4))
sub = node.dropna(subset=["betweenness", "ok_ratio"]).copy()
deadcat = pd.read_csv(os.path.join(BASE, "dead_sites_category.csv"), encoding="utf-8-sig")
catmap = dict(zip(deadcat["sub"], deadcat["cat"]))
true_fail = {"A_conn_fail", "D_not_found", "F_server_error"}
def status(row):
    if row["dead"] == 1:
        c = catmap.get(row["sub"], "")
        return "true_dead" if c in true_fail else "false_dead"
    if row["ok_ratio"] < 100: return "partial"
    return "healthy"
sub["status"] = sub.apply(status, axis=1)
colors_map = {"true_dead": "#c53030", "false_dead": "#d69e2e", "partial": "#dd6b20", "healthy": "#2b6cb0"}
labels_map = {"true_dead": "真失效（连接失败/404/5xx）", "false_dead": "假失效（认证/robots）",
              "partial": "部分失效", "healthy": "全部正常"}
for st, g in sub.groupby("status"):
    ax.scatter(g["betweenness"], g["ok_ratio"], s=26, alpha=.6, c=colors_map[st],
               label=labels_map[st], edgecolors="none")
# 标注高风险：介数前 15 且可用率<100
cand = sub[(sub["betweenness"] > sub["betweenness"].quantile(.90)) & (sub["ok_ratio"] < 100)].nlargest(15, "risk")
for _, r in cand.iterrows():
    ax.annotate(r["sub"], (r["betweenness"], r["ok_ratio"]), fontsize=7.5,
                xytext=(4, 3), textcoords="offset points")
ax.set_xlabel("介数中心性（导航桥梁作用）"); ax.set_ylabel("URL 可用率 (%)")
ax.set_title("关键与脆弱并不重合：高介数节点多健康，失效集中在边缘", fontsize=11)
ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.text(0.5, 0.005, "注：passport / wp / id 等低可用率源于认证或 robots，属假失效；正式分析需先剔除",
         ha="center", fontsize=7.5, color="#555")
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(os.path.join(OUT, "fig5_risk_matrix.png"), bbox_inches="tight")
plt.close(fig)

# ---------- 图6：社区划分对“模板边”的敏感性 ----------
rows = [
    ("全图（加权）", 25, 0.690, 1.0),
    ("去掉导航枢纽边", 76, 0.759, 0.648),
    ("只保留强边 w≥10", 339, 0.713, 0.660),
]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
names = [r[0] for r in rows]
axes[0].bar(names, [r[1] for r in rows], color=["#2b6cb0", "#c53030", "#dd6b20"], alpha=.85)
axes[0].set_ylabel("社区数"); axes[0].set_title("(a) 去掉模板/弱边后，社区数暴涨")
axes[0].tick_params(axis="x", labelsize=8)
axes[1].bar(names, [r[2] for r in rows], color=["#2b6cb0", "#c53030", "#dd6b20"], alpha=.85)
axes[1].set_ylabel("模块度 Q"); axes[1].set_ylim(0.6, 0.8)
axes[1].set_title("(b) 模块度上升，但划分已明显改变（NMI≈0.65）")
axes[1].tick_params(axis="x", labelsize=8)
for ax in axes: ax.grid(axis="y", alpha=.25)
fig.suptitle("社区结构对“模板边”的敏感性：原报告的巨型社区不是唯一答案", y=1.02, fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig6_community_sensitivity.png"), bbox_inches="tight")
plt.close(fig)

# ---------- 图7：观测失效情景 ----------
from matplotlib.patches import Patch
deadcat = pd.read_csv(os.path.join(BASE, "dead_sites_category.csv"), encoding="utf-8-sig")
catmap = dict(zip(deadcat["sub"], deadcat["cat"]))
true_fail = {"A_conn_fail", "D_not_found", "F_server_error"}
dead_set = set(nodes.loc[nodes["ok"] == 0, "subdomain"])
true_dead = set(nodes.loc[(nodes["ok"] == 0) & (nodes["sub"].map(catmap).isin(true_fail)), "subdomain"])
partial_set = set(nodes.loc[(nodes["ok"] > 0) & (nodes["bad"] > 0), "subdomain"])
rng = np.random.default_rng(42)

def scen_stats(H):
    comps = sorted(nx.connected_components(H), key=len, reverse=True)
    gcc = len(comps[0]) / G.number_of_nodes()
    return gcc, nx.global_efficiency(H)

base = scen_stats(G)
scen = {"基线": base}
H = G.copy(); H.remove_nodes_from(dead_set); scen[f"移除全部全挂\n({len(dead_set)})"] = scen_stats(H)
H = G.copy(); H.remove_nodes_from(true_dead); scen[f"移除真失效\n({len(true_dead)})"] = scen_stats(H)
H = G.copy(); H.remove_nodes_from(dead_set | partial_set); scen[f"移除全挂+部分\n({len(dead_set | partial_set)})"] = scen_stats(H)
# 同规模随机对照
for name, k in [(f"随机移除\n{len(dead_set)}", len(dead_set)), (f"随机移除\n{len(true_dead)}", len(true_dead))]:
    vals = []
    for _ in range(100):
        H = G.copy(); H.remove_nodes_from(rng.choice(list(G.nodes()), size=k, replace=False))
        vals.append(scen_stats(H))
    vals = np.array(vals); scen[name] = (vals[:,0].mean(), vals[:,1].mean())

fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
names = list(scen.keys()); xs = np.arange(len(names))
gccs = [scen[n][0]*100 for n in names]; effs = [scen[n][1] for n in names]
colors = ["#2b6cb0", "#c53030", "#dd6b20", "#805ad5", "#718096", "#718096"]
axes[0].bar(xs, gccs, color=colors, alpha=.85)
axes[0].axhline(base[0]*100, color="#333", ls="--", lw=1, label="基线")
axes[0].set_ylabel("最大连通分量 (%)"); axes[0].set_title("(a) 连通性：观测失效比同规模随机更‘轻’")
axes[1].bar(xs, effs, color=colors, alpha=.85)
axes[1].axhline(base[1], color="#333", ls="--", lw=1, label="基线")
axes[1].set_ylabel("全局效率 E"); axes[1].set_title("(b) 效率：移除全挂站点后反而略升")
for ax in axes:
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=7.5)
    ax.grid(axis="y", alpha=.25); ax.legend(frameon=False, fontsize=8)
fig.suptitle("观测失效情景：全挂的多是叶子；‘半坏不坏’的 149 个站点才在主干上", y=1.03, fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig7_observed_failure.png"), bbox_inches="tight")
plt.close(fig)

print("figures saved to", OUT)
