# -*- coding: utf-8 -*-
"""观测失效情景：把实际坏掉的节点移除，网络还连通吗？"""
import os
import numpy as np, pandas as pd, networkx as nx

BASE = r"d:\科大\作业\复杂系统分析\大作业\USTC网络拓扑分析"
OUT = r"d:\科大\作业\复杂系统分析\大作业\选题分析_USTC网络拓扑\02_预实验"
rng = np.random.default_rng(42)

nodes = pd.read_csv(os.path.join(BASE, "nodes_full.csv"), encoding="utf-8-sig")
links = pd.read_csv(os.path.join(BASE, "links.csv"), encoding="utf-8-sig")
deadcat = pd.read_csv(os.path.join(BASE, "dead_sites_category.csv"), encoding="utf-8-sig")
catmap = dict(zip(deadcat["sub"], deadcat["cat"]))
true_fail = {"A_conn_fail", "D_not_found", "F_server_error"}

G = nx.Graph()
for _, r in links.iterrows():
    s, t, w = str(r["source"]), str(r["target"]), float(r["weight"])
    if G.has_edge(s, t): G[s][t]["weight"] += w
    else: G.add_edge(s, t, weight=w)

dead = set(nodes.loc[(nodes["ok"] == 0), "subdomain"])
true_dead = set(nodes.loc[(nodes["ok"] == 0) & (nodes["sub"].map(catmap).isin(true_fail)), "subdomain"])
print("all nodes", G.number_of_nodes(), "dead", len(dead), "true dead", len(true_dead))

def stats(H):
    if H.number_of_nodes() == 0: return 0, 0, 0
    comps = sorted(nx.connected_components(H), key=len, reverse=True)
    gcc = len(comps[0]) / G.number_of_nodes()
    eff = nx.global_efficiency(H)
    return gcc, eff, len(comps)

base = stats(G)
print("baseline GCC/E:", base)
for name, rem in [("all_dead", dead), ("true_dead", true_dead)]:
    H = G.copy(); H.remove_nodes_from(rem)
    print(name, "removed", len(rem), "GCC/E/ncomp:", stats(H))

# 随机移除同样数量，做对照
for name, k in [("random_162", len(dead)), ("random_129", len(true_dead))]:
    vals = []
    for _ in range(200):
        H = G.copy()
        H.remove_nodes_from(rng.choice(list(G.nodes()), size=k, replace=False))
        vals.append(stats(H))
    vals = np.array(vals)
    print(name, "GCC mean %.4f sd %.4f | E mean %.4f sd %.4f" % (
        vals[:,0].mean(), vals[:,0].std(), vals[:,1].mean(), vals[:,1].std()))

# 移除所有 dead + 所有 partial
partial = set(nodes.loc[(nodes["ok"] > 0) & (nodes["bad"] > 0), "subdomain"])
H = G.copy(); H.remove_nodes_from(dead | partial)
print("dead+partial removed", len(dead|partial), "GCC/E/ncomp:", stats(H))
