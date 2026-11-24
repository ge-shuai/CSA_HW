# -*- coding: utf-8 -*-
"""Independent re-check of the 7 headline claims in 选题分析_USTC网络拓扑/01_*.md
Reads only the raw CSVs in USTC网络拓扑分析/ (仓库根). Does not import their code.
"""
import os, re, sys, json
import numpy as np, pandas as pd, networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
_SUB = "USTC网络拓扑分析"
# 与 csa_common 同样的候选，但**故意不 import 它**——这个脚本的价值就在于
# 独立实现；如果连路径解析都共用，口径错了会一起错。找不到就抛错，不静默换目录。
# 仓库内的原料优先于外部的 CSA_HW 兄弟目录，与 csa_common 的取舍一致。
_cands = [os.environ.get("CSA_DATA_DIR"),
          os.path.join(HERE, "..", _SUB),
          os.path.join(HERE, "..", "CSA_HW", _SUB)]
RAW = next((c for c in _cands if c and os.path.isdir(c)), None)
if RAW is None:
    raise FileNotFoundError("找不到原始数据目录（试了 %s）；可设 CSA_DATA_DIR 显式指定。" % _cands)
nodes = pd.read_csv(os.path.join(RAW, "nodes_full.csv"), encoding="utf-8-sig")
links = pd.read_csv(os.path.join(RAW, "links.csv"), encoding="utf-8-sig")
urls = pd.read_csv(os.path.join(RAW, "urls.csv"), encoding="utf-8-sig")
deadcat = pd.read_csv(os.path.join(RAW, "dead_sites_category.csv"), encoding="utf-8-sig")

def h(t): print("\n" + "=" * 70 + "\n" + t + "\n" + "=" * 70)

# 两类判据要分开，否则这个脚本既不能当"复现同学数字"的对照，也不能当"契约没破"的门禁。
#   claim    = 同学 01_*.md 里的原文数字。本项目实测与它有零点几的漂移是已知的，
#              只报 MISMATCH、不让脚本失败——它的价值是把漂移摆出来。
#   contract = 本项目 CONVENTIONS/csa_common 定的口径。违反就是真事故，退出码非 0。
_results = []

def chk(label, got, want, tol, kind="claim"):
    ok = abs(float(got) - float(want)) <= tol
    _results.append((kind, label, float(got), float(want), ok))
    print("  [%s] %-46s got %-12.6g want %-12.6g tol %g"
          % ("PASS" if ok else "MISMATCH" if kind == "claim" else "FAIL",
             label, got, want, tol))
    return ok

# ---------- graph ----------
Gd = nx.DiGraph()
for s, t, w in links.itertuples(index=False):
    Gd.add_edge(s, t, weight=float(w))
G = nx.Graph()
for s, t, d in Gd.edges(data=True):
    G.add_edge(s, t, weight=d["weight"])

h("CLAIM: 671 nodes / 2788 directed links / 2498 undirected / 676 graph nodes / comps 674+2")
print("nodes_full rows        :", len(nodes))
print("links rows             :", len(links), "| unique ordered pairs:", Gd.number_of_edges())
print("undirected edges       :", G.number_of_edges())
print("graph nodes            :", G.number_of_nodes())
print("components sizes       :", sorted((len(c) for c in nx.connected_components(G)), reverse=True)[:5])
print("density                : %.5f" % nx.density(G))
print("avg degree             : %.2f | median %d | max %d" %
      (np.mean([d for _, d in G.degree()]), np.median([d for _, d in G.degree()]),
       max(d for _, d in G.degree())))
print("clustering / trans     : %.3f / %.3f" % (nx.average_clustering(G), nx.transitivity(G)))
print("degree assortativity   : %.3f" % nx.degree_assortativity_coefficient(G))
print("reciprocity (directed) : %.3f" % nx.reciprocity(Gd))

h("CONTRACT: 建图口径（csa_common 锚点，独立实现必须一致）")
chk("n_attributed", len(nodes), 671, 0, "contract")
chk("n_edges_directed", Gd.number_of_edges(), 2788, 0, "contract")
chk("n_edges_undirected", G.number_of_edges(), 2498, 0, "contract")
chk("n_in_graph", G.number_of_nodes(), 676, 0, "contract")
chk("gcc_size", len(max(nx.connected_components(G), key=len)), 674, 0, "contract")
chk("max_degree_u", max(dict(G.degree()).values()), 119, 0, "contract")

h("CLAIM: nodes_full.degree max=146 but recomputed undirected max=119")
print("csv degree max         :", nodes["degree"].max())
deg = dict(G.degree())
print("recomputed max         :", max(deg.values()), "->", max(deg, key=deg.get))
print("teach csv/recomputed   :", int(nodes.loc[nodes.subdomain == "teach.ustc.edu.cn", "degree"].iloc[0]),
      deg.get("teach.ustc.edu.cn"))

h("CLAIM: weight median 2, p75 11, max 4870 (bs->som)")
print(links["weight"].describe().loc[["mean", "50%", "75%", "max"]].to_dict())
top = links.nlargest(5, "weight")[["source", "target", "weight"]]
print(top.to_string(index=False))

h("CLAIM: 49.5% of URLs match /list*.htm ; 3.3% /page.htm ; 1.9% /_upload/")
path = urls["url"].astype(str).str.replace(r"^https?://[^/]+", "", regex=True)
print("list %.4f | page %.4f | upload %.4f | n=%d" % (
    path.str.contains(r"/list\d*\.htm", regex=True).mean(),
    path.str.contains(r"/page\.htm", regex=True).mean(),
    path.str.contains(r"/_upload/", regex=True).mean(), len(urls)))

h("CLAIM: 162 dead / 149 partial / 360 healthy")
nodes["dead"] = (nodes["ok"] == 0).astype(int)
nodes["partial"] = ((nodes["ok"] > 0) & (nodes["bad"] > 0)).astype(int)
nodes["healthy"] = ((nodes["bad"] == 0) & (nodes["ok"] > 0)).astype(int)
print("dead %d | partial %d | healthy %d | sum %d" %
      (nodes.dead.sum(), nodes.partial.sum(), nodes.healthy.sum(), len(nodes)))
print("dead_sites_category counts:\n", deadcat["cat"].value_counts().to_string())
true_subs = set(deadcat.loc[deadcat.cat.isin(["A_conn_fail", "D_not_found", "F_server_error"]), "sub"])
true_dead = set(nodes.loc[nodes["sub"].isin(true_subs), "subdomain"])
print("true-fail (A+D+F) matched:", len(true_dead), "of", len(true_subs))

h("CONTRACT: 失效集合与出度删失（D2/D4 的立论基础）")
chk("n_dead", int(nodes.dead.sum()), 162, 0, "contract")
chk("n_partial", int(nodes.partial.sum()), 149, 0, "contract")
chk("n_healthy", int(nodes.healthy.sum()), 360, 0, "contract")
chk("n_true_fail", len(true_dead), 129, 0, "contract")
# 162 个死站的出度必须恒为 0：这不是"死站不链人"，是采集时根本取不到它们的出链。
# 一旦这条破了，D2 的删失结论和 D4 的"死—死相邻不可识别"都要重写。
dead_subs = set(nodes.loc[nodes.dead == 1, "subdomain"])
edges_from_dead = int(sum(1 for s, _ in Gd.edges() if s in dead_subs))
chk("edges_from_dead", edges_from_dead, 0, 0, "contract")
chk("dead_with_outdegree_zero",
    sum(1 for v, d in Gd.out_degree() if v in dead_subs and d == 0),
    len(dead_subs), 0, "contract")

# ---------- communities ----------
h("CLAIM: weighted Louvain ~25 communities, Q~0.690 (seed=42)")
qs, ns = [], []
for seed in [42, 0, 1, 7, 2026]:
    comms = nx.community.louvain_communities(G, weight="weight", seed=seed)
    ns.append(len(comms))
    qs.append(nx.community.modularity(G, comms, weight="weight"))
print("seeds 42/0/1/7/2026 -> n_comms", ns)
print("modularity                       ", ["%.4f" % q for q in qs])
print("largest community size (seed42) :", max(len(c) for c in nx.community.louvain_communities(G, weight="weight", seed=42)))

HUBS = {"teach", "staff", "wpsearch", "news", "en", "fusep"}
hub_nodes = set(nodes.loc[nodes["sub"].isin(HUBS), "subdomain"])
Gs = G.copy(); Gs.remove_edges_from([(u, v) for u, v in G.edges() if u in hub_nodes or v in hub_nodes])
cs = nx.community.louvain_communities(Gs, weight="weight", seed=42)
print("drop hub-incident edges: edges %d -> %d, n_comms %d, Q %.4f" %
      (G.number_of_edges(), Gs.number_of_edges(), len(cs), nx.community.modularity(Gs, cs, weight="weight")))

# ---------- health vs degree ----------
h("CLAIM: lowest degree decile dead-rate 49.5%; Spearman rho(degree, ok_ratio)=0.376")
m = nodes.copy()
m["degree_u"] = m["subdomain"].map(deg).fillna(0)
m["ok_ratio"] = pd.to_numeric(m["ok_ratio_pct"], errors="coerce")
m["_bins"] = pd.qcut(m["degree_u"], 8, duplicates="drop")
print(m.groupby("_bins", observed=True).agg(n=("sub", "size"), avg_deg=("degree_u", "mean"),
                                            dead_rate=("dead", "mean")).round(4).to_string())
from scipy.stats import spearmanr
print("Spearman rho(degree, ok_ratio) : %.4f" % spearmanr(m["degree_u"], m["ok_ratio"]).statistic)
print("Spearman rho(in-degree, ok)    : %.4f" % spearmanr(m["subdomain"].map(dict(Gd.in_degree())).fillna(0), m["ok_ratio"]).statistic)

h("CLAIM: dead assortativity -0.063; degree-preserving null mean -0.009 sd 0.018 p<0.001")
Gx = G.copy()
dm = dict(zip(m["subdomain"], m["dead"]))
nx.set_node_attributes(Gx, {n: int(dm.get(n, 0)) for n in Gx.nodes()}, "dead")
obs = nx.attribute_assortativity_coefficient(Gx, "dead")
print("observed dead assortativity    : %.4f" % obs)
null = []
for s in range(200):
    H = nx.double_edge_swap(Gx, max_tries=100, seed=s)
    d2 = dict(H.nodes(data=True))
    Hc = H.copy()
    for n, a in d2.items():
        Hc.nodes[n]["dead"] = a["dead"]
    null.append(nx.attribute_assortativity_coefficient(Hc, "dead"))
null = np.array(null)
print("null mean %.4f sd %.4f  | two-sided p = %.4f" %
      (null.mean(), null.std(), (np.abs(null - null.mean()) >= abs(obs - null.mean())).mean()))

# ---------- robustness ----------
h("CLAIM: GCC after removal — random10% 87.6 / deg10% 57.3 / bet10% 54.3 / deg20% 4.5")
GC = max(nx.connected_components(G), key=len)
Gc = G.subgraph(GC).copy()
n0 = Gc.number_of_nodes()
bt = nx.betweenness_centrality(Gc)
pr = nx.pagerank(Gc, weight="weight")
rng = np.random.default_rng(0)
def curve(order_fn, fracs, reps=20):
    out = {}
    for f in fracs:
        k = int(round(f * n0)); vals = []
        for r in range(reps if order_fn is None else 1):
            H = Gc.copy()
            if order_fn is None:
                rm = rng.choice(list(H.nodes()), size=k, replace=False).tolist()
            else:
                rm = order_fn(H)[:k]
            H.remove_nodes_from(rm)
            vals.append(max(len(c) for c in nx.connected_components(H)) / n0)
        out[f] = np.mean(vals)
    return out
def by_deg(H): return sorted(H.nodes(), key=lambda x: -H.degree(x))
def by_bt(H): return sorted(H.nodes(), key=lambda x: -bt[x])
def by_pr(H): return sorted(H.nodes(), key=lambda x: -pr[x])
fr = [0.05, 0.10, 0.15, 0.20, 0.30]
res = {"degree": curve(by_deg, fr), "betweenness": curve(by_bt, fr),
       "pagerank": curve(by_pr, fr), "random": curve(None, fr)}
print(pd.DataFrame(res).round(3).to_string())
# random 这里只跑 20 reps（本脚本是快速对照，不是 E1 的 120 reps），所以容差放宽。
chk("GCC@deg10%", res["degree"][0.10], 0.573, 0.005, "claim")
chk("GCC@bet10%", res["betweenness"][0.10], 0.543, 0.005, "claim")
chk("GCC@deg20%", res["degree"][0.20], 0.045, 0.005, "claim")
chk("GCC@random10% (20 reps)", res["random"][0.10], 0.876, 0.02, "claim")

h("CLAIM: dead avg deg 1.83 / partial 8.58 / healthy 9.49")
print(m.groupby(["dead", "partial", "healthy"], as_index=False).agg(
    n=("sub", "size"), avg_deg=("degree_u", "mean")) if False else
    m.assign(state=np.where(m.dead == 1, "dead", np.where(m.partial == 1, "partial", "healthy")))
     .groupby("state")["degree_u"].agg(["size", "mean", "median"]).round(3).to_string())

h("CLAIM: baseline E=0.3295; drop 162 dead -> GCC 75.7%, E 0.3508 (rises); drop dead+partial 311 -> GCC 50.6%")
# 同学原文那个 "E" 是在剩余最大连通块上算 global_efficiency，等于本项目的 E_gcc：
# 分母随删除收缩，所以它会"上升"。CONVENTIONS §2.3 规定结论只从 E_fixed 读——
# 分母恒为基线 GCC 的有序对数 n0*(n0-1)，端点被删或删后不连通的对一律记 0。
DENOM = float(n0 * (n0 - 1))
print("固定分母 n0*(n0-1) = %d*%d = %.0f" % (n0, n0 - 1, DENOM))

def eff_fixed(H):
    if H.number_of_nodes() < 2:
        return 0.0
    tot = 0.0
    for comp in nx.connected_components(H):
        if len(comp) < 2:
            continue
        for s in comp:
            for t, d in nx.single_source_shortest_path_length(H, s).items():
                if d:
                    tot += 1.0 / d
    return float(tot / DENOM)

def scen(subset):
    H = Gc.subgraph([v for v in Gc.nodes() if v not in subset]).copy()
    removed = n0 - H.number_of_nodes()
    gcc = max(len(c) for c in nx.connected_components(H)) / n0
    big = max(nx.connected_components(H), key=len)
    return removed, gcc, nx.global_efficiency(H.subgraph(big).copy()), eff_fixed(H)

base_E_gcc = nx.global_efficiency(Gc)
base_E_fix = eff_fixed(Gc)
print("baseline                 : removed 0, GCC 1.000, E_gcc %.4f, E_fixed %.6f"
      % (base_E_gcc, base_E_fix))
# 基线图上三口径必须重合，否则说明分母算错了。
chk("E_gcc == E_fixed at baseline", base_E_gcc - base_E_fix, 0.0, 1e-12, "contract")
chk("baseline E (同学原文 0.3295)", base_E_fix, 0.3295, 0.005, "claim")
rows = []
for name, ss in [("all dead", set(m.loc[m.dead == 1, "subdomain"])),
                 ("true-fail only", true_dead & set(Gc.nodes())),
                 ("dead+partial", set(m.loc[(m.dead == 1) | (m.partial == 1), "subdomain"]))]:
    k, gcc, eg, ef = scen(ss)
    rows.append((name, k, gcc, eg, ef))
    print("%-22s: removed %3d, GCC %.3f, E_gcc %.4f, E_fixed %.6f  (%+.2f%% vs 基线)"
          % (name, k, gcc, eg, ef, 100 * (ef / base_E_fix - 1)))
rnd = set(rng.choice(list(Gc.nodes()), size=162, replace=False))
k, gcc, eg, ef = scen(rnd)
print("%-22s: removed %3d, GCC %.3f, E_gcc %.4f, E_fixed %.6f  (%+.2f%% vs 基线)"
      % ("random 162", k, gcc, eg, ef, 100 * (ef / base_E_fix - 1)))

h("CONTRACT: 删失与效率口径（本项目自己的红线）")
# 同学那句"删掉全挂站点效率反而上升"只在 E_gcc/E_full 上成立。
# 换成 E_fixed 必须消失——这条破了就是 CONVENTIONS §2.3 被违反。
dead_row = [r for r in rows if r[0] == "all dead"][0]
chk("E_gcc(删162死站) 上升（复现同学假象）", dead_row[3] - base_E_gcc, 0.0213, 0.01, "claim")
print("  注：上一行 PASS 意味着「上升」这个假象被成功复现，不是认可它。")
if dead_row[4] >= base_E_fix:
    print("  [FAIL] E_fixed 在删死站后没有下降——口径实现有问题，§2.3 的立论失效")
    _results.append(("contract", "E_fixed(dead162) < baseline", dead_row[4], base_E_fix, False))
else:
    print("  [PASS] E_fixed(删162死站) %.6f < 基线 %.6f，降幅 %.2f%%："
          "「删死站让效率上升」在主口径下不成立" %
          (dead_row[4], base_E_fix, 100 * (1 - dead_row[4] / base_E_fix)))
    _results.append(("contract", "E_fixed(dead162) < baseline", 0.0, 0.0, True))

h("判决")
bad = [r for r in _results if r[0] == "contract" and not r[4]]
soft = [r for r in _results if r[0] == "claim" and not r[4]]
print("契约项 contract：%d 项，违反 %d 项" % (sum(1 for r in _results if r[0] == "contract"), len(bad)))
print("同学原文 claim：%d 项，复现偏差 %d 项（只报不拦）"
      % (sum(1 for r in _results if r[0] == "claim"), len(soft)))
for kind, label, got, want, ok in soft:
    if not ok:
        print("   MISMATCH %-46s got %.6g want %.6g" % (label, got, want))
if bad:
    print("\n有契约项违反，退出码 1。")
    sys.exit(1)
print("\n契约项全部通过。")

