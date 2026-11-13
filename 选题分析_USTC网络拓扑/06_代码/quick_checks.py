# -*- coding: utf-8 -*-
import os
import pandas as pd, numpy as np, networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"d:\科大\作业\复杂系统分析\大作业\USTC网络拓扑分析"
nodes = pd.read_csv(os.path.join(BASE,"nodes_full.csv"), encoding="utf-8-sig")
links = pd.read_csv(os.path.join(BASE,"links.csv"), encoding="utf-8-sig")
nodes["dead"]=(nodes["ok"]==0).astype(int)
nodes["partial"]=((nodes["ok"]>0)&(nodes["bad"]>0)).astype(int)
nodes["healthy"]=((nodes["bad"]==0)&(nodes["ok"]>0)).astype(int)

Gd=nx.DiGraph()
for _,r in links.iterrows():
    s,t,w=str(r["source"]),str(r["target"]),float(r["weight"])
    if Gd.has_edge(s,t): Gd[s][t]["weight"]+=w
    else: Gd.add_edge(s,t,weight=w)
G=nx.Graph(Gd)

outd=dict(Gd.out_degree()); ind=dict(Gd.in_degree())
df=pd.DataFrame({"subdomain":list(Gd.nodes())})
df["out"]=df.subdomain.map(outd).fillna(0).astype(int)
df["in"]=df.subdomain.map(ind).fillna(0).astype(int)
df=df.merge(nodes[["subdomain","sub","dead","partial","healthy","ok_ratio_pct","color","url_count"]],on="subdomain",how="left")
df["sink"]=(df["out"]==0)&(df["in"]>0)
df["source"]=(df["in"]==0)&(df["out"]>0)
print("sink x health:")
print(pd.crosstab(df["sink"], df["healthy"].fillna(0)).to_string())
print("\nsink nodes health counts:", df.loc[df.sink,"healthy"].value_counts(dropna=False).to_dict())
print("sink examples healthy:")
print(df[(df.sink)&(df.healthy==1)][["sub","in","out","ok_ratio_pct","url_count"]].head(20).to_string(index=False))
print("\nsource nodes:")
print(df[df.source][["sub","in","out","healthy","ok_ratio_pct"]].to_string(index=False))

# 只保留健康节点的诱导子图，看向上导航
Gh = G.subgraph(df.loc[df.healthy==1,"subdomain"]).copy()
print("\nhealthy subgraph: n=%d m=%d comps=%s" % (Gh.number_of_nodes(),Gh.number_of_edges(),
      [len(c) for c in sorted(nx.connected_components(Gh),key=len,reverse=True)[:5]]))

# 字体测试
plt.rcParams["font.sans-serif"]=["SimHei","Microsoft YaHei","DejaVu Sans"]
plt.rcParams["axes.unicode_minus"]=False
fig,ax=plt.subplots(figsize=(4,2))
ax.text(0.1,0.5,"中文测试 USTC 网络",fontsize=14)
fig.savefig(r"d:\科大\作业\复杂系统分析\大作业\选题分析_USTC网络拓扑\02_预实验\font_test.png",dpi=100)
print("font test saved")
