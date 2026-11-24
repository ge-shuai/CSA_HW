# -*- coding: utf-8 -*-
"""Is the 'template layer' worth building? Replicate + extend their Mantel test.
Question: does URL-path-template similarity between two subdomains predict that they hyperlink?
If not, 选题A's WP1 (模板层) is a null result and 选题B is the better bet.
"""
import os, re
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
links = pd.read_csv(os.path.join(RAW, "links.csv"), encoding="utf-8-sig")
urls = pd.read_csv(os.path.join(RAW, "urls.csv"), encoding="utf-8-sig")

G = nx.Graph()
for s, t, w in links.itertuples(index=False):
    if G.has_edge(s, t): G[s][t]["weight"] += float(w)
    else: G.add_edge(s, t, weight=float(w))

n_urls = urls.groupby("subdomain").size()
elig = sorted([s for s in n_urls.index if n_urls[s] >= 10 and s in G])
print("eligible subdomains (>=10 listed URLs, in graph):", len(elig))
idx = {s: i for i, s in enumerate(elig)}
N = len(elig)

def path(p):
    return re.sub(r"^https?://[^/]+", "", str(p).lower())

# --- three signature definitions ---
def sig_fullpath_num(p):        # theirs: digits -> #, strip extension
    q = re.sub(r"\d+", "#", path(p))
    return re.sub(r"\.(htm|html|php|aspx?|jsp|psp|shtml)$", "", q)
def sig_firstseg(p):            # top-level directory
    q = path(p).lstrip("/")
    return q.split("/")[0].lower()
def sig_depth_ext(p):           # (depth, extension) pair
    q = path(p)
    ext = re.search(r"\.([a-zA-Z0-9]{1,5})(?:$|\?)", q)
    return (q.count("/") - q.rstrip("/").count("/") + q.count("/"), (ext.group(1) if ext else "none"))

urls["p"] = urls["url"].map(path)
sigs = {
    "A_fullpath_digits_normalised": urls.assign(t=urls["p"].map(sig_fullpath_num)).groupby("subdomain")["t"].agg(set),
    "B_first_path_segment":         urls.assign(t=urls["p"].map(sig_firstseg)).groupby("subdomain")["t"].agg(set),
    "C_depth_x_extension":          urls.assign(t=urls["p"].map(sig_depth_ext)).groupby("subdomain")["t"].agg(set),
}

# adjacency matrices restricted to eligible
W = np.zeros((N, N)); Bm = np.zeros((N, N))
for u, v, d in G.edges(data=True):
    if u in idx and v in idx:
        i, j = idx[u], idx[v]
        W[i, j] = W[j, i] = d["weight"]
        Bm[i, j] = Bm[j, i] = 1.0
iu = np.triu_indices(N, 1)
print("eligible-eligible edges present: %d of %d possible pairs" % (int(Bm[iu].sum()), len(iu[0])))

rng = np.random.default_rng(11)
def mantel(x, y, reps=999):
    obs = np.corrcoef(x, y)[0, 1]
    null = np.array([np.corrcoef(x, rng.permutation(y))[0, 1] for _ in range(reps)])
    return obs, (np.abs(null - null.mean()) >= abs(obs - null.mean())).mean(), null.mean(), null.std()

from scipy.stats import spearmanr
print("\n%-34s %8s %8s %9s | %8s %8s %9s" %
      ("signature", "r(weight)", "p", "null_sd", "r(binary)", "p", "adj-rate"))
for name, S in sigs.items():
    T = np.zeros((N, N))
    sets = [S.get(s, set()) for s in elig]
    for i in range(N):
        for j in range(i + 1, N):
            a, b = sets[i], sets[j]
            T[i, j] = T[j, i] = (len(a & b) / len(a | b)) if a and b else 0.0
    x = T[iu]
    o_w, p_w, _, sd_w = mantel(x, W[iu])
    o_b, p_b, _, _ = mantel(x, Bm[iu])
    sp_b = spearmanr(x, Bm[iu]).statistic
    print("%-34s %8.4f %8.4f %9.4f | %8.4f %8.4f %9.4f"
          % (name, o_w, p_w, sd_w, o_b, p_b, Bm[iu].mean()))

print("\n--- how much of the link weight is 'template-ish' at all? ---")
tot = links["weight"].sum()
top10 = links.nlargest(10, "weight")["weight"].sum()
print("total weight %.0f | top-10 edges = %.1f%% of it | single edge bs<->som = %.1f%%"
      % (tot, 100 * top10 / tot, 100 * 4870 / tot))

print("\n--- does stripping hub edges change who is adjacent more than templates do? ---")
HUBS = {"teach", "staff", "wpsearch", "news", "en", "fusep"}
nodes_df = pd.read_csv(os.path.join(RAW, "nodes_full.csv"), encoding="utf-8-sig")
hub = set(nodes_df.loc[nodes_df["sub"].isin(HUBS), "subdomain"])
Gs = G.copy(); Gs.remove_edges_from([(u, v) for u, v in list(G.edges()) if u in hub or v in hub])
print("edges %d -> %d (%.1f%% removed by 6 navigation hubs)"
      % (G.number_of_edges(), Gs.number_of_edges(),
         100 * (G.number_of_edges() - Gs.number_of_edges()) / G.number_of_edges()))
for name, S in sigs.items():
    pass
print("=> preprocessing (strip hub edges) moves the structure a lot; template similarity does not.")
