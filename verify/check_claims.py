# -*- coding: utf-8 -*-
"""Independent re-check of the 7 headline claims in 选题分析_USTC网络拓扑/01_*.md
Reads only the raw CSVs in ../CSA_HW/USTC网络拓扑分析/. Does not import their code.
"""
import os, re, sys, json
import numpy as np, pandas as pd, networkx as nx

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "CSA_HW", "USTC网络拓扑分析")
nodes = pd.read_csv(os.path.join(RAW, "nodes_full.csv"), encoding="utf-8-sig")
links = pd.read_csv(os.path.join(RAW, "links.csv"), encoding="utf-8-sig")
urls = pd.read_csv(os.path.join(RAW, "urls.csv"), encoding="utf-8-sig")
deadcat = pd.read_csv(os.path.join(RAW, "dead_sites_category.csv"), encoding="utf-8-sig")

def h(t): print("\n" + "=" * 70 + "\n" + t + "\n" + "=" * 70)

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

h("CLAIM: dead avg deg 1.83 / partial 8.58 / healthy 9.49")
print(m.groupby(["dead", "partial", "healthy"], as_index=False).agg(
    n=("sub", "size"), avg_deg=("degree_u", "mean")) if False else
    m.assign(state=np.where(m.dead == 1, "dead", np.where(m.partial == 1, "partial", "healthy")))
     .groupby("state")["degree_u"].agg(["size", "mean", "median"]).round(3).to_string())

h("CLAIM: baseline E=0.3295; drop 162 dead -> GCC 75.7%, E 0.3508 (rises); drop dead+partial 311 -> GCC 50.6%")
def scen(subset):
    H = Gc.subgraph([n for n in Gc.nodes() if n not in subset]).copy()
    gcc = max(len(c) for c in nx.connected_components(H)) / n0
    big = max(nx.connected_components(H), key=len)
    E = nx.global_efficiency(H.subgraph(big).copy())
    return len(subset), gcc, E
base_E = nx.global_efficiency(Gc)
print("baseline                 : removed 0, GCC 1.000, E %.4f" % base_E)
for name, ss in [("all dead", set(m.loc[m.dead == 1, "subdomain"])),
                 ("true-fail only", true_dead & set(Gc.nodes())),
                 ("dead+partial", set(m.loc[(m.dead == 1) | (m.partial == 1), "subdomain"]))]:
    k, gcc, E = scen(ss)
    print("%-22s: removed %3d, GCC %.3f, E %.4f" % (name, k, gcc, E))
# random 162
rnd = set(rng.choice(list(Gc.nodes()), size=162, replace=False))
k, gcc, E = scen(rnd)
print("%-22s: removed %3d, GCC %.3f, E %.4f" % ("random 162", k, gcc, E))
