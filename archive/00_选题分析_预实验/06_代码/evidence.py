# -*- coding: utf-8 -*-
"""针对候选选题的证据计算：模板层、语义层、失效层、鲁棒性、传播"""
import os, re, json, random, itertools
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

BASE = r"d:\科大\作业\复杂系统分析\大作业\USTC网络拓扑分析"
OUT = r"d:\科大\作业\复杂系统分析\大作业\选题分析_USTC网络拓扑\02_预实验"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)

nodes = pd.read_csv(os.path.join(BASE, "nodes_full.csv"), encoding="utf-8-sig")
links = pd.read_csv(os.path.join(BASE, "links.csv"), encoding="utf-8-sig")
urls = pd.read_csv(os.path.join(BASE, "urls.csv"), encoding="utf-8-sig")
nodes["ok_ratio"] = pd.to_numeric(nodes["ok_ratio_pct"], errors="coerce")
nodes["dead"] = (nodes["ok"] == 0).astype(int)

# 图
G = nx.Graph()
for _, r in links.iterrows():
    s, t, w = str(r["source"]), str(r["target"]), float(r["weight"])
    if G.has_edge(s, t):
        G[s][t]["weight"] += w
    else:
        G.add_edge(s, t, weight=w)
Gd = nx.DiGraph()
for _, r in links.iterrows():
    s, t, w = str(r["source"]), str(r["target"]), float(r["weight"])
    if Gd.has_edge(s, t):
        Gd[s][t]["weight"] += w
    else:
        Gd.add_edge(s, t, weight=w)

print("=== 1. 边的权重与模板性质 ===")
print("top 20 edges:")
print(links.sort_values("weight", ascending=False).head(20).to_string(index=False))

# 边权是否与源/目标"模板相似度"相关（先做粗略：目标是否为全局导航站）
HUBS = {"teach.ustc.edu.cn","staff.ustc.edu.cn","wpsearch.ustc.edu.cn","news.ustc.edu.cn",
        "en.ustc.edu.cn","fusep.ustc.edu.cn","faculty.ustc.edu.cn","lib.ustc.edu.cn",
        "gradschool.ustc.edu.cn","employment.ustc.edu.cn"}
links["target_is_hub"] = links["target"].isin(HUBS)
links["source_is_hub"] = links["source"].isin(HUBS)
print("\nweight by target hub vs not:")
print(links.groupby("target_is_hub")["weight"].describe().to_string())
print("share of total weight to top10 hubs: %.3f" % (links.loc[links.target_is_hub,"weight"].sum()/links.weight.sum()))

print("\n=== 2. 度/强度/健康 ===")
deg = dict(G.degree()); strength = dict(G.degree(weight="weight"))
node = nodes.set_index("subdomain").copy()
node["degree"] = pd.Series(deg); node["strength"] = pd.Series(strength)
node["deg_decile"] = pd.qcut(node["degree"], 10, labels=False, duplicates="drop")
tab = node.groupby("deg_decile").agg(n=("sub","size"), mean_deg=("degree","mean"),
                                     dead_rate=("dead","mean"), mean_ok=("ok_ratio","mean"))
print(tab.to_string())
print("spearman degree vs ok_ratio", node[["degree","ok_ratio"]].corr(method="spearman").iloc[0,1])

# 失效同配性 + 置换检验
G2 = G.copy()
for n in G2.nodes():
    G2.nodes[n]["dead"] = int(node.loc[n,"dead"]) if n in node.index else 0
obs = nx.numeric_assortativity_coefficient(G2, "dead")
vals = list(nx.get_node_attributes(G2, "dead").values())
perm = []
for _ in range(1000):
    rng.shuffle(vals)
    nx.set_node_attributes(G2, dict(zip(G2.nodes(), vals)), "dead")
    perm.append(nx.numeric_assortativity_coefficient(G2, "dead"))
perm = np.array(perm)
print("dead assortativity obs=%.4f, perm mean=%.4f, p=%.4f" % (obs, perm.mean(), (np.abs(perm)>=abs(obs)).mean()))

print("\n=== 3. 社区：全图 vs 去掉导航枢纽边 vs 只保留强边 ===")
def part_stats(H, name, seed=42, res=1.0):
    if H.number_of_edges() == 0:
        return None
    comms = sorted(nx.community.louvain_communities(H, weight="weight", seed=seed, resolution=res), key=len, reverse=True)
    q = nx.community.modularity(H, comms, weight="weight")
    lab = {}
    for i,c in enumerate(comms):
        for n in c: lab[n] = i
    sizes = [len(c) for c in comms]
    print("%s: nodes=%d edges=%d ncomm=%d sizes=%s Q=%.3f" % (name, H.number_of_nodes(), H.number_of_edges(), len(comms), sizes[:12], q))
    return lab, comms, q

lab_full, comms_full, q_full = part_stats(G, "full weighted")

# 去导航枢纽边
Gn = G.copy()
Gn.remove_edges_from([(u,v) for u,v in Gn.edges() if u in HUBS or v in HUBS])
lab_nav, comms_nav, q_nav = part_stats(Gn, "no hub edges")

# 只保留 weight>=10 的强边（导航模板通常重复出现）
Gs = nx.Graph()
Gs.add_nodes_from(G.nodes())
Gs.add_edges_from([(u,v,d) for u,v,d in G.edges(data=True) if d["weight"]>=10])
lab_str, comms_str, q_str = part_stats(Gs, "strong edges w>=10")
Gs2 = nx.Graph()
Gs2.add_nodes_from(G.nodes())
Gs2.add_edges_from([(u,v,d) for u,v,d in G.edges(data=True) if d["weight"]>=50])
lab_s50, comms_s50, q_s50 = part_stats(Gs2, "strong edges w>=50")

# 社区间 NMI
common = set(lab_full) & set(lab_nav)
print("NMI full vs no-hub", normalized_mutual_info_score([lab_full[n] for n in common],[lab_nav[n] for n in common]))
common = set(lab_full) & set(lab_str)
print("NMI full vs strong>=10", normalized_mutual_info_score([lab_full[n] for n in common],[lab_str[n] for n in common]))

print("\n=== 4. 语义层：标题分类 vs 社区 ===")
def cat(title, sub):
    t = str(title); s = str(sub)
    rules = [
        ("招生", ["招生","录取","admission","zsb","yz"]),
        ("教务培养", ["教务","教学","培养","课程","课堂","研究生院","teach","gradschool"]),
        ("科研平台", ["科研","实验室","研究中心","研究院","平台","lab","sklfs","cas"]),
        ("院系", ["学院","系","学部","school","department","dept"]),
        ("新闻宣传", ["新闻","宣传","媒体","news","press"]),
        ("服务门户", ["门户","服务","搜索","图书馆","邮箱","一卡通","缴费","招聘","人才","lib","mail","job"]),
        ("附属单位", ["附中","附小","幼儿园","医院","kdfz"]),
        ("英文国际", ["english","international","en."]),
        ("个人主页", ["个人主页","homepage","~"]),
    ]
    for name,kws in rules:
        if any(k.lower() in t.lower() or k.lower()==s.lower() for k in kws): return name
    return "其他"
nodes["cat"] = [cat(t,s) for t,s in zip(nodes["title"], nodes["sub"])]
print(nodes["cat"].value_counts().to_string())
G3 = G.copy()
cm = dict(zip(nodes["subdomain"], nodes["cat"]))
for n in G3.nodes(): G3.nodes[n]["cat"] = cm.get(n,"其他")
print("categorical assortativity:", nx.attribute_assortativity_coefficient(G3,"cat"))
common = [n for n in G if n in cm]
print("NMI louvain(full) vs cat:", normalized_mutual_info_score([lab_full[n] for n in common],[cm[n] for n in common]))
# 类别划分的模块度
from collections import defaultdict
cat_parts = defaultdict(list)
for n in G:
    cat_parts[cm.get(n,"其他")].append(n)
cat_parts = list(cat_parts.values())
print("modularity of category partition:", nx.community.modularity(G, cat_parts, weight="weight"))

print("\n=== 5. URL 模板相似度 vs 超链接邻接（Mantel） ===")
# 路径模板：把数字替换为 #，去掉扩展名
def tmpl(p):
    p = str(p).lower()
    p = re.sub(r"^https?://[^/]+", "", p)
    p = re.sub(r"\d+", "#", p)
    p = re.sub(r"\.(htm|html|php|aspx?|jsp|psp|shtml)$", "", p)
    return p
urls["tmpl"] = urls["url"].map(tmpl)
sub_tmpl = urls.groupby("subdomain")["tmpl"].agg(lambda x: set(x))
n_urls = urls.groupby("subdomain").size()
eligible = [s for s in sub_tmpl.index if n_urls.get(s,0) >= 10 and s in G]
print("eligible subdomains with >=10 listed urls:", len(eligible))
# Jaccard
def jac(a,b):
    if not a or not b: return 0.0
    return len(a&b)/len(a|b)
subs = eligible
idx = {s:i for i,s in enumerate(subs)}
n = len(subs)
T = np.zeros((n,n))
for i,a in enumerate(subs):
    for j in range(i+1,n):
        T[i,j] = T[j,i] = jac(sub_tmpl[a], sub_tmpl[subs[j]])
# 超链接邻接（限定在 eligible 上，binary/weighted）
A = np.zeros((n,n))
for u,v,d in G.edges(data=True):
    if u in idx and v in idx:
        i,j = idx[u],idx[v]
        A[i,j] = A[j,i] = d["weight"]
# 只对上三角
iu = np.triu_indices(n,1)
x = T[iu]; y = A[iu]
print("Pearson template-sim vs hyperlink weight: %.3f" % np.corrcoef(x,y)[0,1])
print("Spearman: %.3f" % pd.Series(x).corr(pd.Series(y), method="spearman"))
# 置换
obs_r = np.corrcoef(x,y)[0,1]
perm_r=[]
for _ in range(500):
    yp = rng.permutation(y)
    perm_r.append(np.corrcoef(x,yp)[0,1])
perm_r=np.array(perm_r)
print("Mantel p=%.4f" % ((np.abs(perm_r)>=abs(obs_r)).mean()))

# 模板家族：对 eligible 做阈值图 + Louvain
GT = nx.Graph()
GT.add_nodes_from(subs)
for i,a in enumerate(subs):
    for j in range(i+1,n):
        if T[i,j] >= 0.5:
            GT.add_edge(a, subs[j], weight=float(T[i,j]))
print("template-sim graph: nodes=%d edges=%d" % (GT.number_of_nodes(), GT.number_of_edges()))
if GT.number_of_edges()>0:
    tc = sorted(nx.community.louvain_communities(GT, weight="weight", seed=42), key=len, reverse=True)
    print("template families:", len(tc), [len(c) for c in tc][:12])
    print("largest family sample:", [x.replace(".ustc.edu.cn","") for x in list(tc[0])[:20]])

print("\n=== 6. 鲁棒性曲线（GCC / 效率） ===")
GCnodes = max(nx.connected_components(G), key=len)
GC = G.subgraph(GCnodes).copy()
N0 = GC.number_of_nodes()
bt = nx.betweenness_centrality(GC, normalized=True)
pr = nx.pagerank(Gd, weight="weight")
strat = {
    "degree": sorted(GC.nodes(), key=lambda x: -GC.degree(x)),
    "betweenness": sorted(GC.nodes(), key=lambda x: -bt.get(x,0)),
    "pagerank": sorted(GC.nodes(), key=lambda x: -pr.get(x,0)),
}
# 随机基线
rows=[]
FRACS = np.arange(0,0.31,0.05)
for name, order in strat.items():
    for frac in FRACS:
        k = int(frac*N0)
        H2 = GC.copy()
        if k>0:
            H2.remove_nodes_from(order[:k])
        if H2.number_of_nodes()==0:
            gcc=0; eff=0
        else:
            gcc = max((len(c) for c in nx.connected_components(H2)), default=0)/N0
            eff = nx.global_efficiency(H2)
        rows.append({"strategy":name,"frac":round(float(frac),2),"gcc":gcc,"eff":eff})
# random
for seed in range(5):
    order = list(GC.nodes()); rng.shuffle(order)
    for frac in FRACS:
        k=int(frac*N0)
        H2=GC.copy(); H2.remove_nodes_from(order[:k])
        gcc = max((len(c) for c in nx.connected_components(H2)), default=0)/N0
        eff = nx.global_efficiency(H2)
        rows.append({"strategy":"random","frac":round(float(frac),2),"gcc":gcc,"eff":eff,"seed":seed})
rob = pd.DataFrame(rows)
rob.to_csv(os.path.join(OUT,"robustness_curves.csv"), index=False, encoding="utf-8-sig")
print(rob.groupby(["strategy","frac"]).agg(gcc=("gcc","mean"),eff=("eff","mean")).head(40).to_string())

# 关键节点：同时高中心性 + 低健康
node["betweenness"]=pd.Series(bt); node["pagerank"]=pd.Series(pr)
node["cat"] = pd.Series(cm).reindex(node.index)
node["risk"] = node["betweenness"].rank(pct=True) * (1-node["ok_ratio"]/100)
print("\n=== 7. '重要且脆弱'候选 ===")
cols=["sub","degree","betweenness","pagerank","ok_ratio","dead","cat"]
print(node.sort_values("risk",ascending=False)[cols].head(20).to_string(index=False))

# 保存
node.to_csv(os.path.join(OUT,"node_metrics_enriched.csv"), encoding="utf-8-sig")
print("\nDONE")
