# -*- coding: utf-8 -*-
"""USTC 子域超链接网络：预实验数据体检"""
import os, json, re, math
import numpy as np
import pandas as pd
import networkx as nx

BASE = r"d:\科大\作业\复杂系统分析\大作业\USTC网络拓扑分析"
OUT = r"d:\科大\作业\复杂系统分析\大作业\选题分析_USTC网络拓扑\02_预实验"
os.makedirs(OUT, exist_ok=True)

nodes = pd.read_csv(os.path.join(BASE, "nodes_full.csv"), encoding="utf-8-sig")
links = pd.read_csv(os.path.join(BASE, "links.csv"), encoding="utf-8-sig")
urls = pd.read_csv(os.path.join(BASE, "urls.csv"), encoding="utf-8-sig")

print("nodes", nodes.shape, "links", links.shape, "urls", urls.shape)
print(nodes.columns.tolist())
print(links.head())
print("weight describe:\n", links["weight"].describe())
print("weight quantiles:", links["weight"].quantile([.5,.9,.95,.99,1]).to_dict())

# ---------- 构图 ----------
Gd = nx.DiGraph()
for _, r in links.iterrows():
    s, t, w = str(r["source"]), str(r["target"]), float(r["weight"])
    if Gd.has_edge(s, t):
        Gd[s][t]["weight"] += w
    else:
        Gd.add_edge(s, t, weight=w)
G = nx.Graph()
for s, t, d in Gd.edges(data=True):
    w = d["weight"]
    if G.has_edge(s, t):
        G[s][t]["weight"] += w
    else:
        G.add_edge(s, t, weight=w)

print("\n=== 拓扑 ===")
print("directed nodes/edges", Gd.number_of_nodes(), Gd.number_of_edges())
print("undirected nodes/edges", G.number_of_nodes(), G.number_of_edges())
print("self loops", list(nx.selfloop_edges(G))[:5])
comps = sorted(nx.connected_components(G), key=len, reverse=True)
print("components", [len(c) for c in comps][:10])
print("density", nx.density(G))
print("avg clustering", nx.average_clustering(G))
print("transitivity", nx.transitivity(G))
print("degree assortativity", nx.degree_assortativity_coefficient(G))
print("reciprocity (directed)", nx.reciprocity(Gd))
print("is DAG?", nx.is_directed_acyclic_graph(Gd))

# 度/强度
deg = dict(G.degree())
strength = dict(G.degree(weight="weight"))
degv = np.array(list(deg.values()))
strv = np.array(list(strength.values()))
print("degree: mean %.2f max %d min %d" % (degv.mean(), degv.max(), degv.min()))
print("strength: mean %.1f max %.0f" % (strv.mean(), strv.max()))
print("degree top10", sorted(deg.items(), key=lambda x: -x[1])[:10])
print("strength top10", sorted(strength.items(), key=lambda x: -x[1])[:10])

# 幂律粗拟合（log-log 线性回归）
d = degv[degv > 0]
coef = np.polyfit(np.log(d), np.log(np.arange(len(d), 0, -1)), 1)
print("degree CCDF slope approx", coef[0])

# 全局效率 / 平均路径（用最大连通分量）
GC = G.subgraph(comps[0]).copy()
print("GC nodes", GC.number_of_nodes())
print("avg shortest path (unweighted)", nx.average_shortest_path_length(GC))
print("diameter", nx.diameter(GC))
print("global efficiency", nx.global_efficiency(GC))

# 中心性
pr = nx.pagerank(Gd, weight="weight")
bt = nx.betweenness_centrality(GC, weight=None)
hits_h, hits_a = nx.hits(Gd, max_iter=1000, normalized=True)
print("\nPageRank top10", sorted(pr.items(), key=lambda x: -x[1])[:10])
print("Betweenness top10", sorted(bt.items(), key=lambda x: -x[1])[:10])
print("HITS hubs top10", sorted(hits_h.items(), key=lambda x: -x[1])[:10])
print("HITS auth top10", sorted(hits_a.items(), key=lambda x: -x[1])[:10])

# ---------- 节点属性 ----------
nodes["ok_ratio"] = pd.to_numeric(nodes["ok_ratio_pct"], errors="coerce")
nodes["dead"] = (nodes["ok"] == 0).astype(int)
nodes["partial"] = ((nodes["ok"] > 0) & (nodes["bad"] > 0)).astype(int)
nodes["healthy"] = ((nodes["bad"] == 0) & (nodes["ok"] > 0)).astype(int)
print("\nhealth counts dead/partial/healthy:",
      nodes["dead"].sum(), nodes["partial"].sum(), nodes["healthy"].sum())
print(nodes.groupby("color").agg(n=("sub", "size"), mean_ok=("ok_ratio", "mean")).to_string())

# 度 vs 健康
m = nodes.set_index("subdomain").drop(columns=["degree"], errors="ignore")
m["degree_u"] = pd.Series(deg)
m["strength_u"] = pd.Series(strength)
m["degree"] = m["degree_u"]
for col in ["ok_ratio", "dead", "partial"]:
    print("corr degree vs", col, m[["degree", col]].corr().iloc[0, 1])

# 同配性：健康度
G2 = G.copy()
for n in G2.nodes():
    row = m.loc[n] if n in m.index else None
    G2.nodes[n]["ok_ratio"] = float(row["ok_ratio"]) if row is not None and pd.notna(row["ok_ratio"]) else 100.0
try:
    print("assortativity ok_ratio", nx.numeric_assortativity_coefficient(G2, "ok_ratio"))
except Exception as e:
    print("assort err", e)

# ---------- 社区 ----------
print("\n=== 社区（Louvain, 加权） ===")
comms = nx.community.louvain_communities(G, weight="weight", seed=42, resolution=1.0)
comms = sorted(comms, key=len, reverse=True)
print("n comms", len(comms), "sizes", [len(c) for c in comms][:15])
print("modularity", nx.community.modularity(G, comms, weight="weight"))

# 去掉"全局导航枢纽"的边后再看社区
HUB_SUBS = {"teach", "staff", "wpsearch", "news", "en", "fusep"}
hub_nodes = set(nodes.loc[nodes["sub"].isin(HUB_SUBS), "subdomain"])
Gsub = G.copy()
removed = 0
for u, v in list(Gsub.edges()):
    if u in hub_nodes or v in hub_nodes:
        Gsub.remove_edge(u, v); removed += 1
print("\nremove edges touching global nav hubs:", removed, "remain edges", Gsub.number_of_edges())
comps2 = sorted(nx.connected_components(Gsub), key=len, reverse=True)
print("components after hub-edge removal", [len(c) for c in comps2][:15])
if Gsub.number_of_edges() > 0:
    comms2 = sorted(nx.community.louvain_communities(Gsub, weight="weight", seed=42), key=len, reverse=True)
    print("comms after", len(comms2), [len(c) for c in comms2][:15])
    print("modularity after", nx.community.modularity(Gsub, comms2, weight="weight"))
    print("largest comm members sample", [n.replace(".ustc.edu.cn","") for n in list(comms2[0])[:25]])

# ---------- URL 模板/路径 ----------
print("\n=== URL 路径特征 ===")
urls["path"] = urls["url"].astype(str).str.replace(r"^https?://[^/]+", "", regex=True)
urls["ext"] = urls["path"].str.extract(r"\.([a-zA-Z0-9]{1,5})(?:$|\?)")[0].str.lower()
print("ext top", urls["ext"].value_counts().head(12).to_dict())
urls["has_list"] = urls["path"].str.contains(r"/list\d*\.htm", regex=True)
urls["has_page"] = urls["path"].str.contains(r"/page\.htm", regex=True)
urls["has_upload"] = urls["path"].str.contains(r"/_upload/", regex=True)
print("template-ish share: list %.3f page %.3f upload %.3f" % (
    urls["has_list"].mean(), urls["has_page"].mean(), urls["has_upload"].mean()))
# 每个子域的模板特征
g = urls.groupby("subdomain").agg(n=("url", "size"), list_share=("has_list", "mean"),
                                  page_share=("has_page", "mean"), upload_share=("has_upload", "mean"))
print("subdomain template feature describe:\n", g.describe().to_string())

# 每个子域 URL 的路径首段集合（用于相似度）
def first_seg(p):
    p = re.sub(r"^/+", "", str(p))
    return p.split("/")[0].lower() if p else ""
urls["seg1"] = urls["path"].map(first_seg)
seg_counts = urls.groupby("subdomain")["seg1"].nunique()
print("seg1 unique per subdomain describe:\n", seg_counts.describe().to_string())

# 标题关键词分类（粗分类，仅用于预实验）
def cat(title):
    t = str(title)
    rules = [
        ("招生", ["招生", "本科招生", "研究生招生", "录取", "admission"]),
        ("教务/培养", ["教务", "教学", "培养", "课程", "网络课堂", "研究生院"]),
        ("科研/平台", ["科研", "实验室", "重点实验室", "研究中心", "研究院", "平台"]),
        ("院系", ["学院", "系", "学部", "school", "department"]),
        ("新闻/宣传", ["新闻", "新闻网", "宣传", "媒体", "报"]),
        ("服务/门户", ["门户", "服务", "搜索", "图书馆", "邮箱", "一卡通", "缴费", "招聘", "人才"]),
        ("附属单位", ["附中", "附小", "幼儿园", "医院"]),
        ("英文", ["english", "international"]),
    ]
    for name, kws in rules:
        if any(k.lower() in t.lower() for k in kws):
            return name
    return "其他"

nodes["cat"] = nodes["title"].map(cat)
print("\ncategory counts:\n", nodes["cat"].value_counts().to_string())

# 类别同配性
G3 = G.copy()
catmap = dict(zip(nodes["subdomain"], nodes["cat"]))
for n in G3.nodes():
    G3.nodes[n]["cat"] = catmap.get(n, "其他")
from networkx.algorithms.assortativity import attribute_assortativity_coefficient
print("categorical assortativity (title cat)", attribute_assortativity_coefficient(G3, "cat"))

# 类别 vs 社区
node_comm = {}
for i, c in enumerate(comms):
    for n in c:
        node_comm[n] = i
nodes["louvain"] = nodes["subdomain"].map(node_comm)
print("\ncat x louvain crosstab (top 8 comms):")
top_comms = nodes["louvain"].value_counts().head(8).index
print(pd.crosstab(nodes["cat"], nodes["louvain"])[top_comms].to_string())

# 保存中间结果
m.to_csv(os.path.join(OUT, "node_metrics.csv"), encoding="utf-8-sig")
pd.DataFrame([{"metric": k, "value": v} for k, v in {
    "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
    "density": nx.density(G), "avg_clustering": nx.average_clustering(G),
    "degree_assortativity": nx.degree_assortativity_coefficient(G),
    "reciprocity": nx.reciprocity(Gd), "avg_path_gc": nx.average_shortest_path_length(GC),
    "diameter_gc": nx.diameter(GC), "global_efficiency": nx.global_efficiency(GC),
    "n_communities": len(comms), "modularity": nx.community.modularity(G, comms, weight="weight"),
    "edges_removed_hub": removed, "modularity_after_hub_removal": nx.community.modularity(Gsub, comms2, weight="weight") if Gsub.number_of_edges()>0 else None,
}.items()]).to_csv(os.path.join(OUT, "topology_metrics.csv"), index=False, encoding="utf-8-sig")

print("\nDONE ->", OUT)
