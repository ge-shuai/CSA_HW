# -*- coding: utf-8 -*-
"""Definitive check of 发现4 (dead clustering vs degree-preserving null).
Their extra.py: numeric_assortativity_coefficient + 300 CUMULATIVE swaps on one graph.
Here: both assortativity definitions, independent replicates, proper mixing.
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

G = nx.Graph()
for s, t, w in links.itertuples(index=False):
    if G.has_edge(s, t): G[s][t]["weight"] += float(w)
    else: G.add_edge(s, t, weight=float(w))

GC = G.subgraph(max(nx.connected_components(G), key=len)).copy()
dead = dict(zip(nodes["subdomain"], (nodes["ok"] == 0).astype(int)))
nx.set_node_attributes(GC, {n: int(dead.get(n, 0)) for n in GC.nodes()}, "dead")
print("GC nodes", GC.number_of_nodes(), "edges", GC.number_of_edges(),
      "dead in GC", sum(a["dead"] for _, a in GC.nodes(data=True)))

obs_num = nx.numeric_assortativity_coefficient(GC, "dead")
obs_cat = nx.attribute_assortativity_coefficient(GC, "dead")
print("\nOBSERVED  numeric=%.4f  categorical=%.4f" % (obs_num, obs_cat))

def null_independent(fn, reps, nswap, seed0=0):
    vals = []
    for i in range(reps):
        H = fn(GC.copy())
        vals.append(fn_attr(H, ))
    return np.array(vals)

def swap_once(H, seed, nswap):
    nx.double_edge_swap(H, nswap=nswap, max_tries=100 * nswap, seed=seed)
    return H

def fn_attr(H):
    return nx.numeric_assortativity_coefficient(H, "dead")

NSWAP = 10 * GC.number_of_edges()
print("\n--- degree-preserving null, INDEPENDENT replicates (nswap=%d each) ---" % NSWAP)
for reps in [100, 300]:
    v = np.array([fn_attr(swap_once(GC.copy(), 10000 + i, NSWAP)) for i in range(reps)])
    z = (obs_num - v.mean()) / v.std()
    p = (np.abs(v - v.mean()) >= abs(obs_num - v.mean())).mean()
    print("reps=%3d null mean %.4f sd %.4f | obs %.4f z %.2f two-sided p %.4f | min %.4f max %.4f"
          % (reps, v.mean(), v.std(), obs_num, z, p, v.min(), v.max()))

print("\n--- their procedure reproduced: CUMULATIVE swaps, nswap=500, max_tries=20000 ---")
H = GC.copy(); traj = []
for i in range(300):
    try: nx.double_edge_swap(H, nswap=500, max_tries=20000, seed=i)
    except Exception: pass
    traj.append(nx.numeric_assortativity_coefficient(H, "dead"))
t = np.array(traj)
print("trajectory mean %.4f sd %.4f | first10 %s | last10 %s"
      % (t.mean(), t.std(), np.round(t[:5], 4), np.round(t[-5:], 4)))
print("their p formula (|vals|>=|obs|): %.4f" % (np.abs(t) >= abs(obs_num)).mean())

print("\n--- NON-degree-preserving control: random label permutation ---")
rng = np.random.default_rng(0)
lab = np.array([a["dead"] for _, a in GC.nodes(data=True)])
order = list(GC.nodes())
v = []
for _ in range(300):
    H = GC.copy()
    nx.set_node_attributes(H, dict(zip(order, rng.permutation(lab))), "dead")
    v.append(nx.numeric_assortativity_coefficient(H, "dead"))
v = np.array(v)
print("permutation null mean %.4f sd %.4f | z %.2f p %.4f"
      % (v.mean(), v.std(), (obs_num - v.mean()) / v.std(),
         (np.abs(v - v.mean()) >= abs(obs_num - v.mean())).mean()))

print("\n--- how much do swaps actually move the graph? (edge overlap after nswap) ---")
H2 = swap_once(GC.copy(), 1, NSWAP)
e0 = set(map(frozenset, GC.edges())); e1 = set(map(frozenset, H2.edges()))
print("edges changed: %d / %d (%.1f%%)" % (len(e0 - e1), len(e0), 100 * len(e0 - e1) / len(e0)))
H3 = GC.copy(); nx.double_edge_swap(H3, nswap=500, max_tries=20000, seed=1)
e2 = set(map(frozenset, H3.edges()))
print("their setting (nswap=500): %d edges changed (%.1f%%)" % (len(e0 - e2), 100 * len(e0 - e2) / len(e0)))
