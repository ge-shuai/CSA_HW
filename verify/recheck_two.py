# -*- coding: utf-8 -*-
"""Focused re-check of the two claims that did NOT reproduce:
   (A) dead-assortativity degree-preserving null  -> they report mean -0.009, sd 0.018, p<0.001
   (B) PageRank attack curve                      -> they report 10% 76.9%, 20% 57.7%, 30% 32.9%
"""
import os
import numpy as np, pandas as pd, networkx as nx

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SUB = "USTC网络拓扑分析"
# 与 csa_common 同样的候选，仓库内的原料优先；不 import 它，保持独立实现。
_cands = [os.environ.get("CSA_DATA_DIR"),
          os.path.join(_ROOT, _SUB),
          os.path.join(_ROOT, "CSA_HW", _SUB)]
RAW = next((c for c in _cands if c and os.path.isdir(c)), None)
if RAW is None:
    raise FileNotFoundError("找不到原始数据目录（试了 %s）；可设 CSA_DATA_DIR 显式指定。" % _cands)
nodes = pd.read_csv(os.path.join(RAW, "nodes_full.csv"), encoding="utf-8-sig")
links = pd.read_csv(os.path.join(RAW, "links.csv"), encoding="utf-8-sig")

Gd = nx.DiGraph()
for s, t, w in links.itertuples(index=False):
    Gd.add_edge(s, t, weight=float(w))
G = nx.Graph()
for s, t, d in Gd.edges(data=True):
    G.add_edge(s, t, weight=d["weight"])
dead = dict(zip(nodes["subdomain"], (nodes["ok"] == 0).astype(int)))
nx.set_node_attributes(G, {n: int(dead.get(n, 0)) for n in G.nodes()}, "dead")
obs = nx.attribute_assortativity_coefficient(G, "dead")
print("observed dead assortativity: %.4f" % obs)

print("\n--- (A) degree-preserving null, more reps ---")
for reps in [50, 200, 500]:
    vals = []
    for s in range(reps):
        H = nx.double_edge_swap(G, max_tries=200, seed=s)
        vals.append(nx.attribute_assortativity_coefficient(H, "dead"))
    v = np.array(vals)
    p = (np.abs(v - v.mean()) >= abs(obs - v.mean())).mean()
    print("reps=%3d  null mean %.4f sd %.4f  |  two-sided p %.4f  |  rank of obs: %d/%d"
          % (reps, v.mean(), v.std(), p, int((v <= obs).sum()), reps))

print("\n--- (A2) is the null well mixed? autocorrelation of successive swap states ---")
H = G.copy(); trace = []
for s in range(30):
    H = nx.double_edge_swap(H, max_tries=200, seed=1000 + s)
    trace.append(nx.attribute_assortativity_coefficient(H, "dead"))
print("cumulative-swap trace (first 10): ", ["%.3f" % x for x in trace[:10]])
print("cumulative-swap trace (last 10):  ", ["%.3f" % x for x in trace[-10:]])

print("\n--- (A3) skipped: configuration-model null not used by them ---")

print("\n--- (A4) simple sanity: raw (non-degree-preserving) random relabelling ---")
rng = np.random.default_rng(0)
labels = np.array([dead.get(n, 0) for n in G.nodes()])
vals = []
for _ in range(300):
    perm = rng.permutation(len(labels))
    H = G.copy()
    nx.set_node_attributes(H, dict(zip(G.nodes(), perm_labels := labels[perm])), "dead")
    vals.append(nx.attribute_assortativity_coefficient(H, "dead"))
v = np.array(vals)
print("label-permutation null mean %.4f sd %.4f  p %.4f" %
      (v.mean(), v.std(), (np.abs(v - v.mean()) >= abs(obs - v.mean())).mean()))

print("\n--- (B) PageRank attack, three ranking sources ---")
GC = max(nx.connected_components(G), key=len)
Gc = G.subgraph(GC).copy()
n0 = Gc.number_of_nodes()
orders = {
    "pr_weighted_directed (their eda.py)": nx.pagerank(Gd, weight="weight"),
    "pr_unweighted_undirected":            nx.pagerank(Gc),
    "pr_weighted_undirected":              nx.pagerank(Gc, weight="weight"),
    "pr_directed_unweighted":              nx.pagerank(Gd),
}
fr = [0.05, 0.10, 0.15, 0.20, 0.30]
rows = {}
for name, sc in orders.items():
    ranked = sorted(Gc.nodes(), key=lambda x: -sc.get(x, 0.0))
    rows[name] = {f: max(len(c) for c in nx.connected_components(
        Gc.subgraph(ranked[int(round(f * n0)):]).copy())) / n0 for f in fr}
print(pd.DataFrame(rows).T.round(3).to_string())
print("\ntheir reported: 5%% 85.8 / 10%% 76.9 / 15%% 65.0 / 20%% 57.7 / 30%% 32.9")
