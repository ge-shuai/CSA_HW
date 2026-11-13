# -*- coding: utf-8 -*-
"""补充证据：度保持零模型、集中度、导航效率、链路推荐、风险矩阵"""
import os, re, json, random
import numpy as np
import pandas as pd
import networkx as nx

BASE = r"d:\科大\作业\复杂系统分析\大作业\USTC网络拓扑分析"
OUT = r"d:\科大\作业\复杂系统分析\大作业\选题分析_USTC网络拓扑\02_预实验"
rng = np.random.default_rng(7)

nodes = pd.read_csv(os.path.join(BASE, "nodes_full.csv"), encoding="utf-8-sig")
links = pd.read_csv(os.path.join(BASE, "links.csv"), encoding="utf-8-sig")
urls = pd.read_csv(os.path.join(BASE, "urls.csv"), encoding="utf-8-sig")
nodes["ok_ratio"] = pd.to_numeric(nodes["ok_ratio_pct"], errors="coerce")
nodes["dead"] = (nodes["ok"] == 0).astype(int)

G = nx.Graph()
for _, r in links.iterrows():
    s,t,w = str(r["source"]), str(r["target"]), float(r["weight"])
    if G.has_edge(s,t): G[s][t]["weight"] += w
    else: G.add_edge(s,t,weight=w)
Gd = nx.DiGraph()
for _, r in links.iterrows():
    s,t,w = str(r["source"]), str(r["target"]), float(r["weight"])
    if Gd.has_edge(s,t): Gd[s][t]["weight"] += w
    else: Gd.add_edge(s,t,weight=w)

print("=== 0. 数据一致性 ===")
gnodes = set(G.nodes()); nsubs = set(nodes["subdomain"])
print("graph-only nodes:", sorted(gnodes-nsubs))
print("table-only nodes:", sorted(nsubs-gnodes))
print("degree column vs recomputed: corr=%.3f, max reported=%d max recomputed=%d" % (
    pd.Series(dict(G.degree())).reindex(nodes.subdomain).corr(nodes["degree"]),
    nodes["degree"].max(), max(dict(G.degree()).values())))
print("listed_urls value counts top:", nodes["listed_urls"].value_counts().head(5).to_dict())
print("url_count vs listed corr:", nodes[["url_count","listed_urls"]].corr().iloc[0,1])

node = nodes.set_index("subdomain").copy()
node["degree"] = pd.Series(dict(G.degree()))
node["strength"] = pd.Series(dict(G.degree(weight="weight")))

print("\n=== 1. 失效同配性：度保持零模型 ===")
GCn = max(nx.connected_components(G), key=len)
GC = G.subgraph(GCn).copy()
dead_map = {n: int(node.loc[n,"dead"]) if n in node.index else 0 for n in GC.nodes()}
nx.set_node_attributes(GC, dead_map, "dead")
obs = nx.numeric_assortativity_coefficient(GC, "dead")
print("observed dead assortativity (GC):", round(obs,4))
vals=[]
H = GC.copy()
for i in range(300):
    try:
        nx.double_edge_swap(H, nswap=500, max_tries=20000, seed=i)
    except Exception:
        pass
    vals.append(nx.numeric_assortativity_coefficient(H, "dead"))
vals=np.array(vals)
print("degree-preserving null: mean=%.4f sd=%.4f p(two-sided)=%.4f" % (vals.mean(), vals.std(), (np.abs(vals)>=abs(obs)).mean()))

print("\n=== 2. 集中度 / 不平等 ===")
def gini(x):
    x=np.sort(np.asarray(x,float)); n=len(x)
    if n==0 or x.sum()==0: return 0
    return float((2*np.arange(1,n+1)-n-1).dot(x)/(n*x.sum()))
pr = nx.pagerank(Gd, weight="weight")
for name, vals_ in [("degree", node.degree), ("strength", node.strength), ("pagerank", pd.Series(pr).reindex(node.index).fillna(0))]:
    arr=np.asarray(vals_,float)
    top1=arr[np.argsort(-arr)[:max(1,len(arr)//100)]].sum()/arr.sum()
    print("%s: gini=%.3f top1%%share=%.3f max/mean=%.1f" % (name,gini(arr),top1,arr.max()/arr.mean()))

print("\n=== 3. 有向性：返回性 / 导航陷阱 ===")
outd = dict(Gd.out_degree()); ind = dict(Gd.in_degree())
rec = nx.reciprocity(Gd)
print("reciprocity=%.3f" % rec)
both = sum(1 for n in Gd if outd.get(n,0)>0 and ind.get(n,0)>0)
sink = sum(1 for n in Gd if outd.get(n,0)==0 and ind.get(n,0)>0)
source = sum(1 for n in Gd if outd.get(n,0)>0 and ind.get(n,0)==0)
iso = sum(1 for n in Gd if outd.get(n,0)==0 and ind.get(n,0)==0)
print("nodes both in/out=%d, sink(out=0)=%d, source(in=0)=%d, isolated=%d" % (both,sink,source,iso))
print("out-degree top:", sorted(outd.items(), key=lambda x:-x[1])[:8])
print("in-degree top:", sorted(ind.items(), key=lambda x:-x[1])[:8])

print("\n=== 4. 导航效率与链路推荐 ===")
N = GC.number_of_nodes()
print("GC n=%d, m=%d, avg path=%.3f, diameter=%d" % (N,GC.number_of_edges(),nx.average_shortest_path_length(GC),nx.diameter(GC)))
# 候选边：共同邻居多但无连边（仅取共同邻居>=3，限制数量）
nonedges = list(nx.non_edges(GC))
print("non-edges:", len(nonedges))
rng.shuffle(nonedges)
sample = nonedges[:20000]
cn = {(u,v): len(list(nx.common_neighbors(GC,u,v))) for u,v in sample}
top_cand = sorted(cn.items(), key=lambda x:-x[1])[:30]
print("top candidate links by common neighbors:")
for (u,v),c in top_cand[:10]:
    print("  %s -- %s : CN=%d" % (u.replace(".ustc.edu.cn",""), v.replace(".ustc.edu.cn",""), c))

# 贪心加边：每次加一条使平均路径下降最多的边（候选集限制，抽样 BFS 近似）
src_sample = list(GC.nodes())[:150]
def approx_ap(H):
    total=0.0; cnt=0
    for s in src_sample:
        d = nx.single_source_shortest_path_length(H, s)
        total += sum(d.values()); cnt += len(d)
    return total/cnt
base = approx_ap(GC)
print("base approx avg path", base)
H = GC.copy()
chosen=[]
cands = [e for e,_ in sorted(cn.items(), key=lambda x:-x[1])[:100] if not H.has_edge(*e)]
for step in range(5):
    best=None; best_ap=base
    for e in cands:
        if H.has_edge(*e): continue
        H.add_edge(*e)
        ap = approx_ap(H)
        H.remove_edge(*e)
        if ap < best_ap - 1e-6:
            best_ap=ap; best=e
    if best is None: break
    H.add_edge(*best, weight=1.0); chosen.append((best, base-best_ap)); base=best_ap
    print("add %s -- %s -> approx avg path %.4f (delta %.4f)" % (best[0].replace(".ustc.edu.cn",""), best[1].replace(".ustc.edu.cn",""), best_ap, chosen[-1][1]))
print("after 5 links: approx avg path %.4f" % approx_ap(H))

print("\n=== 5. 重要且脆弱（风险矩阵） ===")
bt = nx.betweenness_centrality(GC, normalized=True)
node["betweenness"]=pd.Series(bt)
node["pagerank"]=pd.Series(pr)
node["deg_rank"]=node["degree"].rank(pct=True)
node["bt_rank"]=node["betweenness"].rank(pct=True)
node["unhealthy"]=1-node["ok_ratio"]/100
node["risk"]=node["bt_rank"]*node["unhealthy"]
node["criticality"]=node["bt_rank"]
def cat(title, sub):
    t=str(title); s=str(sub)
    rules=[("招生",["招生","录取","admission","zsb","yz"]),("教务培养",["教务","教学","培养","课程","课堂","研究生院","teach","gradschool"]),
           ("科研平台",["科研","实验室","研究中心","研究院","平台","lab","sklfs"]),("院系",["学院","系","学部","school","department","dept"]),
           ("新闻宣传",["新闻","宣传","媒体","news","press"]),("服务门户",["门户","服务","搜索","图书馆","邮箱","一卡通","缴费","招聘","人才","lib","mail","job"]),
           ("附属单位",["附中","附小","幼儿园","医院","kdfz"]),("英文国际",["english","international","en."]),("个人主页",["个人主页","homepage","~"])]
    for name,kws in rules:
        if any(k.lower() in t.lower() or k.lower()==s.lower() for k in kws): return name
    return "其他"
node["cat"]=[cat(t,s) for t,s in zip(node["title"],node["sub"])]
cols=["sub","cat","degree","betweenness","ok_ratio","dead","risk"]
print(node.sort_values("risk",ascending=False)[cols].head(25).to_string())
node.to_csv(os.path.join(OUT,"node_metrics_enriched.csv"), encoding="utf-8-sig")

print("\n=== 6. 失效 URL 的状态构成（按类别） ===")
dead = pd.read_csv(os.path.join(BASE,"dead_sites_category.csv"), encoding="utf-8-sig")
print(dead["cat"].value_counts().to_string())
print("dead category by url_count sum:")
print(dead.groupby("cat")["url_count"].agg(["count","sum","mean"]).to_string())

print("\nDONE")
