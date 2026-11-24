# -*- coding: utf-8 -*-
"""A2_small_world — USTC 网站群超链接网络的小世界性定量刻画。

口径（全部来自 csa_common，禁止自建图）
---------------------------------------
* 基线图 `Gobs` = **674 节点 GCC** 的**无权无向**诱导子图：
  `C.build_binary(C.induced(b['Gu'], b['sets']['gcc']))`。
  无权 = 丢掉 `weight` 属性；所有指标都用拓扑（非权重）定义。
* 对照节点集（每个数字都标明节点集，避免 676/674/671 混淆）：
  - `gcc674`  主口径（平均路径 3.2702 / 直径 7 / 平均聚类 0.2788 / 度同配 −0.1679）
  - `gu676`   全图（平均聚类 0.2780 / 度同配 −0.1674，CONVENTIONS 锚点口径）
  - `gu671`   有属性节点集
* 对照模型：同 n(=674)、同平均度的 ER(gnp / gnm)、WS 小世界（k 取整 + p 扫描）、
  度保持配置模型（**调用 `C.degree_preserving_null`**，每轮独立复制再交换）。
* 随机图的 L / 直径 / C 一律在**该实例自己的最大连通分量**上算（观测图 GCC 本身连通，
  所以同一规则）；`lcc_frac` 单独报告，说明丢掉了多少节点。
* Humphries et al. (2006)：γ = C_obs/C_rand，λ = L_obs/L_rand，σ = γ/λ；σ>1 才算小世界。
  σ 用**逐实例**定义 σ_i = (C_obs·L_i)/(L_obs·C_i)，报 mean±sd；同时报"均值之比"的点估计。
* seed 一律 `C.SEED`；每个模型 ≥30 个实例（本脚本 N_INST=100）。

诚实约束：只做描述与对照，不给因果解释。禁止"导致/因为/使得/引发/决定了/传染"。
"""
from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = _HERE
for _ in range(4):  # pipeline/out/A2_small_world -> pipeline
    _PIPE = os.path.dirname(_PIPE)
    if os.path.exists(os.path.join(_PIPE, "csa_common.py")):
        break
if _PIPE not in sys.path:
    sys.path.insert(0, _PIPE)

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

import csa_common as C

MODULE = "A2_small_world"
SEED = C.SEED
N_INST = 100            # 每个对照模型的实例数（CONVENTIONS §4 要求 >=30）
N_CFG = 100             # 度保持配置模型的轮数
WS_KS = (6, 8)          # networkx 的 WS 每侧连 k//2 个邻居 => 只有偶数 k 才让 <k>=k
WS_PS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0)
CFG_CHECK_REPS = 30     # 用于验证"同一个 C.degree_preserving_null 序列"的子轮数

T_START = time.time()
LOG: list[str] = []
ROWS: list[dict] = []
INST: list[dict] = []


def log(msg: str) -> None:
    line = f"[{time.time() - T_START:8.2f}s] {msg}"
    print(line, flush=True)
    LOG.append(line)


# ------------------------------------------------------------------ 统计引擎
# 674 节点、~2500 边的图要跑几千个实例，所以最短路用 scipy 稀疏矩阵的
# "全体源点同步 BFS"，与 nx.average_shortest_path_length 逐位对齐（脚本内自检）。

def apsp(H: nx.Graph):
    """返回 (节点列表, 距离矩阵 int16(-1=不可达), 不可达有序对数)。"""
    nodes = list(H.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    A = sp.lil_matrix((n, n), dtype=np.int8)
    for u, v in H.edges():
        A[idx[u], idx[v]] = 1
        A[idx[v], idx[u]] = 1
    A = A.tocsr()
    visited = np.eye(n, dtype=bool)
    frontier = np.eye(n, dtype=bool)
    dist = np.full((n, n), -1, dtype=np.int16)
    level = 0
    while True:
        level += 1
        nxt = (A @ frontier.astype(np.int8)).astype(bool) & ~visited
        if not nxt.any():
            break
        dist[nxt] = level
        visited |= nxt
        frontier = nxt
    return nodes, idx, dist


def graph_stats(H: nx.Graph, keep: bool = False) -> dict:
    """一个图 → 全部拓扑指标。L/直径/C_lcc 一律在**该图自己的最大连通分量**上算，
    观测图与所有对照模型共用这一条规则（观测 GCC 本身连通，所以等价）。

    keep=True 才挂上 n×n 距离矩阵（只有观测基线需要，几千个实例会吃掉 GB 级内存）。
    """
    nodes, idx, dist = apsp(H)
    n = len(nodes)
    m = H.number_of_edges()
    deg = dict(H.degree())
    comps = sorted(nx.connected_components(H), key=len, reverse=True)
    lcc = comps[0] if comps else set()
    li = np.array([idx[x] for x in lcc], dtype=int) if lcc else np.array([], dtype=int)
    nl = len(li)

    # --- 距离类
    if nl >= 2:
        sub = dist[np.ix_(li, li)]
        off = ~np.eye(nl, dtype=bool)
        d = sub[off]
        dd = d[d >= 0].astype(np.float64)
        L = float(dd.sum() / dd.size) if dd.size else np.nan
        diam = float(dd.max()) if dd.size else np.nan
    else:
        L, diam = np.nan, np.nan

    # --- 聚类类（逐点 C_v = 2*T_v / (k_v*(k_v-1))）
    tri = nx.triangles(H)
    cl = {}
    for v in nodes:
        dv = deg[v]
        cl[v] = 0.0 if dv < 2 else tri[v] / (dv * (dv - 1) / 2.0)
    C_all = float(np.mean([cl[x] for x in nodes])) if nodes else np.nan
    C_lcc = float(np.mean([cl[x] for x in lcc])) if lcc else np.nan
    triples = sum(dv * (dv - 1) // 2 for dv in deg.values())
    triangles = int(sum(tri.values()) // 3)
    trans = float(3.0 * triangles / triples) if triples else np.nan

    try:
        assort = float(nx.degree_assortativity_coefficient(H))
    except Exception:
        assort = np.nan

    # --- 度序列摘要
    ds = np.array(sorted(deg.values()), dtype=float)
    out = {
        "n": float(n),
        "m": float(m),
        "kavg": float(2.0 * m / n) if n else np.nan,
        "kmax": float(ds.max()) if ds.size else np.nan,
        "sum_deg2": float((ds ** 2).sum()) if ds.size else np.nan,
        "ncomp": float(len(comps)),
        "lcc_size": float(nl),
        "lcc_frac": float(nl / n) if n else np.nan,
        "n_isolated": float(sum(1 for c in comps if len(c) == 1)),
        "L": L,
        "diameter": diam,
        "C_lcc": C_lcc,
        "C_all": C_all,
        "transitivity": trans,
        "triangles": float(triangles),
        "triples": float(triples),
        "assortativity": assort,
    }
    if keep:  # 只有观测基线需要这些大对象；实例图一律丢掉，否则几千个实例吃 GB 内存
        out["_nodes"] = nodes
        out["_idx"] = idx
        out["_dist"] = dist
        out["_lcc"] = lcc
        out["_cl"] = cl
        out["_deg"] = deg
    return out


def core_metrics(s: dict) -> dict:
    """进 CSV 的指标子集（去掉内部对象；缺失的诊断项记 nan）。"""
    return {k: float(s.get(k, np.nan)) for k in (
        "n", "m", "kavg", "kmax", "sum_deg2", "ncomp", "lcc_size", "lcc_frac", "n_isolated",
        "L", "diameter", "C_lcc", "C_all", "transitivity", "triangles", "triples",
        "assortativity", "overlap_obs")}


def edge_overlap(H: nx.Graph, E_obs: frozenset, m_obs: int) -> float:
    """与观测图共有的边占比 —— 随机化模型"到底打散了多少"的混合度诊断。"""
    if not m_obs:
        return np.nan
    return len(frozenset(((u, v) if u <= v else (v, u)) for u, v in H.edges()) & E_obs) / float(m_obs)


def add_instances(model: str, node_set: str, stats_list: list, obs: dict, note: str = "") -> None:
    """把一组实例登记到原始表，并对每个指标汇总 mean/sd/z/p。"""
    cms = []
    for i, s in enumerate(stats_list):
        cm = core_metrics(s)
        if np.isfinite(cm["C_lcc"]) and cm["C_lcc"] > 0 and np.isfinite(cm["L"]) and cm["L"] > 0:
            cm["gamma"] = obs["C_lcc"] / cm["C_lcc"]
            cm["lambda"] = obs["L"] / cm["L"]
            cm["sigma"] = cm["gamma"] / cm["lambda"]
        else:
            cm["gamma"] = cm["lambda"] = cm["sigma"] = np.nan
        # 用 C_all（全节点平均聚类）再算一套：D2b 会把少量点甩出最大分量，
        # 只用 C_lcc 会把"分量碎裂"混进 γ 里。两套并列才看得出差别。
        if np.isfinite(cm["C_all"]) and cm["C_all"] > 0 and np.isfinite(cm["L"]) and cm["L"] > 0:
            cm["gamma_all"] = obs["C_all"] / cm["C_all"]
            cm["sigma_all"] = cm["gamma_all"] / cm["lambda"]
        else:
            cm["gamma_all"] = cm["sigma_all"] = np.nan
        cm.update(model=model, node_set=node_set, inst=i, block="raw", note=note)
        INST.append(cm)
        cms.append(cm)

    arrs = {k: np.array([cm[k] for cm in cms], dtype=float)
            for k in ("kavg", "kmax", "sum_deg2", "m", "lcc_frac", "ncomp", "L", "diameter",
                      "C_lcc", "C_all", "transitivity", "triangles", "assortativity",
                      "overlap_obs", "gamma", "gamma_all", "lambda", "sigma", "sigma_all")}
    for k, v in arrs.items():
        vv = v[np.isfinite(v)]
        if vv.size == 0:
            continue
        ov = obs.get(k, np.nan)
        sd = float(vv.std(ddof=1)) if vv.size > 1 else np.nan
        # 确定性实例（如 WS 的 p=0 环格）sd 只有浮点噪声，z 无意义 -> 记 nan 并在 note 里说明
        z = float((ov - vv.mean()) / sd) if (sd and sd > 1e-9 and np.isfinite(ov)) else np.nan
        p = float((np.abs(vv - vv.mean()) >= abs(ov - vv.mean())).mean()) if np.isfinite(ov) else np.nan
        ROWS.append({
            "block": "models", "metric": k, "node_set": node_set, "model": model,
            "n_inst": int(vv.size), "mean": round(float(vv.mean()), 6),
            "sd": round(sd, 6) if np.isfinite(sd) else np.nan,
            "min": round(float(vv.min()), 6), "median": round(float(np.median(vv)), 6),
            "max": round(float(vv.max()), 6),
            "obs_value": round(float(ov), 6) if np.isfinite(ov) else np.nan,
            "obs_over_mean": round(float(ov / vv.mean()), 6) if np.isfinite(ov) and vv.mean() else np.nan,
            "z_of_obs": round(z, 4) if np.isfinite(z) else np.nan,
            "p_two_sided": round(p, 4),
            "note": note,
        })


def add_observed(block: str, metric: str, node_set: str, model: str, value, note: str = "") -> None:
    ROWS.append({"block": block, "metric": metric, "node_set": node_set, "model": model,
                 "n_inst": 1, "mean": value, "sd": np.nan, "min": value, "median": value,
                 "max": value, "obs_value": value, "obs_over_mean": 1.0, "z_of_obs": np.nan,
                 "p_two_sided": np.nan, "note": note})


# ------------------------------------------------------------------ 随机化模型
class Memo:
    """把 C.degree_preserving_null 生成的每个实例的完整指标缓存下来。

    同一个 seed 序列下 `degree_preserving_null` 每轮生成的图是确定的，所以按边集签名
    缓存后，对多个指标重复调用它不会重复做 BFS；再用一次小子集调用做一致性断言，
    证明所有指标来自**同一批**独立复制-交换实例。
    """

    def __init__(self, fn=None):
        self.fn = fn or graph_stats
        self.store: dict = {}
        self.records: list = []
        self.hits = 0
        self.misses = 0

    @staticmethod
    def sig(H: nx.Graph):
        return (H.number_of_nodes(), H.number_of_edges(),
                hash(frozenset(((u, v) if u <= v else (v, u)) for u, v in H.edges())))

    def stats(self, H: nx.Graph) -> dict:
        k = self.sig(H)
        if k in self.store:
            self.hits += 1
        else:
            self.store[k] = self.fn(H)
            self.misses += 1
        return self.store[k]

    def stat(self, field: str):
        def f(H: nx.Graph) -> float:
            s = self.stats(H)
            self.records.append(s)
            return float(s[field])
        return f


def triangle_sum_preserving_swap(G: nx.Graph, nswap: int, max_tries: int, rng):
    """D2a：双边交换，只接受**三角总数不变**的交换。

    保度序列 + 保三角总数 => 保全局聚类（传递性）精确不变；逐点平均聚类不保证。
    """
    H = G.copy()
    adj = {v: set(H[v]) for v in H}
    E = [(u, v) for u, v in H.edges()]
    done = tries = 0
    ne = len(E)
    while done < nswap and tries < max_tries:
        tries += 1
        i = int(rng.integers(ne))
        j = int(rng.integers(ne))
        if i == j:
            continue
        a, b = E[i]
        c, d = E[j]
        if len({a, b, c, d}) < 4:
            continue
        if c in adj[a] or d in adj[b]:
            continue
        destroy = len(adj[a] & adj[b]) + len(adj[c] & adj[d])
        create = len((adj[a] - {b}) & (adj[c] - {d})) + len((adj[b] - {a}) & (adj[d] - {c}))
        if create != destroy:
            continue
        adj[a].discard(b); adj[b].discard(a)
        adj[c].discard(d); adj[d].discard(c)
        adj[a].add(c); adj[c].add(a)
        adj[b].add(d); adj[d].add(b)
        E[i] = (a, c)
        E[j] = (b, d)
        done += 1
    out = nx.Graph()
    out.add_nodes_from(G.nodes())
    out.add_edges_from(E)
    return out, done, tries


def clustering_preserving_swap(G: nx.Graph, nswap: int, max_tries: int, rng):
    """D2b：仅重连**两个都不在任何三角形里**的边，且新位置也不许形成三角形。

    这样每个节点的度与三角数都不变 => **逐点聚类系数、平均聚类、传递性全部精确不变**，
    其余结构（度关联、社群、路径）被打乱。这是"仅重连（保聚类）"的操作化定义。
    已知局限：三角形内部的那批边动不了，混合程度低于配置模型（结果表里同时报
    三角边占比与同配性变化，供判断到底打散了没有）。
    """
    H = G.copy()
    # 可动边集合 = 两端无公共邻居的边（该边本身不在任何三角形里）
    free = [(u, v) for u, v in H.edges() if not (set(H[u]) & set(H[v]))]
    adj = {v: set(H[v]) for v in H}
    done = tries = 0
    nf = len(free)
    while done < nswap and tries < max_tries:
        tries += 1
        i = int(rng.integers(nf))
        j = int(rng.integers(nf))
        if i == j:
            continue
        a, b = free[i]
        c, d = free[j]
        if len({a, b, c, d}) < 4:
            continue
        if c in adj[a] or d in adj[b]:
            continue
        if (adj[a] - {b}) & (adj[c] - {d}):
            continue
        if (adj[b] - {a}) & (adj[d] - {c}):
            continue
        adj[a].discard(b); adj[b].discard(a)
        adj[c].discard(d); adj[d].discard(c)
        adj[a].add(c); adj[c].add(a)
        adj[b].add(d); adj[d].add(b)
        free[i] = (a, c)
        free[j] = (b, d)
        done += 1
    out = nx.Graph()
    out.add_nodes_from(G.nodes())
    out.add_edges_from([(u, v) for u, v in adj.items() for v in adj[u] if u < v])
    return out, done, tries


def rewire_observed(G: nx.Graph, p: float, rng) -> nx.Graph:
    """D3：把 WS 式重连直接作用在观测图上——每条边以概率 p 把终点搬到随机节点。

    保 n、保 m（重连失败时保留原边），不保度序列：从观测结构连续走向无结构。
    """
    H = G.copy()
    nodes = list(H.nodes())
    nn = len(nodes)
    for u, v in list(H.edges()):
        if not H.has_edge(u, v):
            continue
        if rng.random() >= p:
            continue
        for _ in range(200):
            w = nodes[int(rng.integers(nn))]
            if w != u and not H.has_edge(u, w):
                H.remove_edge(u, v)
                H.add_edge(u, w)
                break
    return H


# ==================================================================== 主流程
def main() -> None:
    log(f"A2_small_world start; SEED={SEED}; N_INST={N_INST}; N_CFG={N_CFG}")

    # ---------------------------------------------------------- 0. 锚点
    C.assert_anchors()
    log("csa_common.assert_anchors(): 31/31 PASS")
    b = C.build_all()

    # ---------------------------------------------------- 1. 观测基线
    GCCw = C.induced(b["Gu"], b["sets"]["gcc"])          # 674 节点（带权对象）
    Gobs = C.build_binary(GCCw)                          # 无权无向 —— 主口径
    G676 = C.build_binary(b["Gu"])                       # 676 全图对照
    G671 = C.build_binary(C.induced(b["Gu"], b["sets"]["attributed"]))  # 671 对照

    # 观测图边集（用于"打散了多少"的混合度诊断）
    E_obs = frozenset(((u, v) if u <= v else (v, u)) for u, v in Gobs.edges())

    def gstats(H, keep: bool = False) -> dict:
        s = graph_stats(H, keep=keep)
        s["overlap_obs"] = edge_overlap(H, E_obs, len(E_obs))
        return s

    obs = graph_stats(Gobs, keep=True)
    s676 = graph_stats(G676)
    s671 = graph_stats(G671)
    log(f"GCC674 n={obs['n']:.0f} m={obs['m']:.0f} kavg={obs['kavg']:.4f} "
        f"L={obs['L']:.4f} diam={obs['diameter']:.0f} C={obs['C_lcc']:.4f} "
        f"assort={obs['assortativity']:.4f}")

    # BFS 实现自检：必须与 networkx 逐位一致
    L_nx = nx.average_shortest_path_length(Gobs)
    assert abs(L_nx - obs["L"]) < 1e-9, (L_nx, obs["L"])
    assert nx.diameter(Gobs) == obs["diameter"]
    assert abs(nx.average_clustering(Gobs) - obs["C_lcc"]) < 1e-12
    assert abs(nx.transitivity(Gobs) - obs["transitivity"]) < 1e-12
    log("fast APSP / clustering self-check vs networkx: PASS")

    # 锚点口径复核（任务书给的四个数 + 676 的两个数）
    # 注意 0.2780 那个锚是 nx.average_clustering 在 **676 个节点全体**上的均值
    # （= 本脚本的 C_all），不是最大分量上的 C_lcc；两者相差就是 674/676 的分母。
    anchor_rows = {
        "avg_pathlen_gcc674(task=3.2702)": (obs["L"], 3.2702),
        "diameter_gcc674(task=7)": (obs["diameter"], 7.0),
        "clustering_gcc674(task=0.2788)": (obs["C_lcc"], 0.2788),
        "assortativity_gcc674(task=-0.1679)": (obs["assortativity"], -0.1679),
        "clustering_gu676_C_all(anchor=0.2780)": (s676["C_all"], 0.2780),
        "clustering_gu676_C_lcc(=gcc674 0.2788)": (s676["C_lcc"], 0.2788),
        "assortativity_gu676(anchor=-0.1674)": (s676["assortativity"], -0.1674),
    }
    for kk, (got, want) in anchor_rows.items():
        ok = abs(got - want) < 5e-4
        log(f"  {'OK ' if ok else 'DIFF'} {kk}: got={got:.4f}")
        assert ok, kk

    # 节点集口径表：同一个指标在 674/676/671 上不一样，必须分列
    # （gu676/gu671 含第二个 2 点分量 {bm, l.bm}，所以 L/直径定义在其最大分量上）
    for tag, s, note in (("gcc674", obs, "主口径：GCC 诱导、无权无向"),
                         ("gu676", s676, "CONVENTIONS 锚点口径（含 5 悬空 + 2 点分量）"),
                         ("gu671", s671, "有属性节点口径")):
        for mk, mv in (("L", s["L"]), ("diameter", s["diameter"]), ("C_lcc", s["C_lcc"]),
                       ("C_all", s["C_all"]), ("transitivity", s["transitivity"]),
                       ("assortativity", s["assortativity"]), ("kavg", s["kavg"]),
                       ("kmax", s["kmax"]), ("m", s["m"]), ("n", s["n"]),
                       ("lcc_frac", s["lcc_frac"])):
            add_observed("A_observed", mk, tag, f"observed_{tag}", round(float(mv), 6),
                         note + ("；L/直径在该图自身最大分量上" if mk in ("L", "diameter") else ""))

    # ------------------------------------------------- 2. ER 对照（同 n 同平均度）
    n0, m0, kavg0 = int(obs["n"]), int(obs["m"]), obs["kavg"]
    p_er = kavg0 / (n0 - 1.0)
    log(f"ER gnp p = kavg/(n-1) = {kavg0:.4f}/{n0-1} = {p_er:.6f}")
    er_gnp = [gstats(nx.gnp_random_graph(n0, p_er, seed=SEED + i)) for i in range(N_INST)]
    er_gnm = [gstats(nx.gnm_random_graph(n0, m0, seed=SEED + i)) for i in range(N_INST)]
    add_instances("ER_gnp", "synthetic_n674_lcc", er_gnp, obs,
                  f"p=k/(n-1)={p_er:.6f}；L/C 在实例自身最大分量")
    add_instances("ER_gnm", "synthetic_n674_lcc", er_gnm, obs,
                  f"固定 m={m0}；L/C 在实例自身最大分量")
    log(f"ER_gnp done ({N_INST} inst): kavg={np.mean([s['kavg'] for s in er_gnp]):.4f} "
        f"C={np.mean([s['C_lcc'] for s in er_gnp]):.5f} L={np.mean([s['L'] for s in er_gnp]):.4f}")

    # ------------------------------------------------------ 3. WS 对照（k 取整扫描）
    ws_stats: dict[str, list] = {}
    for k in WS_KS:
        for p in WS_PS:
            key = f"WS_k{k}_p{p:g}"
            ws_stats[key] = [gstats(nx.watts_strogatz_graph(n0, k, p, seed=SEED + 1000 * k + i))
                             for i in range(N_INST)]
            add_instances(key, "synthetic_n674_lcc", ws_stats[key], obs,
                          f"WS 环格 k={k}（networkx 每侧 k//2，偶数 k 才有 <k>=k）重连概率 p={p:g}")
        log(f"WS k={k} done ({len(WS_PS)} p x {N_INST} inst)")

    # ---------------------------------------- 4. 度保持配置模型（csa_common 实现）
    memo = Memo(fn=gstats)
    res_c = C.degree_preserving_null(Gobs, n_reps=N_CFG, seed=SEED, stat=memo.stat("C_lcc"))
    cfg_records = memo.records[:N_CFG]
    # 一致性断言：degree_preserving_null 每轮调用 stat 两次之外的唯一入口是末尾的 obs=stat(G)，
    # 所以 30 轮 + 1 次 obs = 31 次缓存命中、0 次新增 => 证明两轮调用生成的是同一批图。
    memo.records = []
    memo.hits = memo.misses = 0
    res_l = C.degree_preserving_null(Gobs, n_reps=CFG_CHECK_REPS, seed=SEED, stat=memo.stat("L"))
    assert memo.misses == 0 and len(memo.records) == CFG_CHECK_REPS + 1, (memo.misses, len(memo.records))
    chk = np.array([s["L"] for s in cfg_records[:CFG_CHECK_REPS]])
    assert np.allclose(res_l["vals"], chk, rtol=0, atol=0), "配置模型实例序列不可复现"
    memo.records = []
    memo.hits = memo.misses = 0
    res_c30 = C.degree_preserving_null(Gobs, n_reps=CFG_CHECK_REPS, seed=SEED, stat=memo.stat("C_lcc"))
    assert memo.misses == 0 and len(memo.records) == CFG_CHECK_REPS + 1, (memo.misses, len(memo.records))
    chk2 = np.array([s["C_lcc"] for s in cfg_records[:CFG_CHECK_REPS]])
    assert np.allclose(res_c30["vals"], chk2, rtol=0, atol=0)
    log(f"CFG via C.degree_preserving_null: n_reps={N_CFG}; 复算 {CFG_CHECK_REPS} 轮全部缓存命中 "
        f"hits={memo.hits}(={CFG_CHECK_REPS} 轮 + obs) misses={memo.misses} "
        f"=> 所有指标来自同一批独立复制-交换实例")
    # 与 csa_common 自己的 obs/null_mean/null_sd 对齐
    cfg_C = np.array([s["C_lcc"] for s in cfg_records])
    assert abs(cfg_C.mean() - res_c["null_mean"]) < 1e-12
    assert abs(res_c["obs"] - obs["C_lcc"]) < 1e-12
    assert all(abs(s["sum_deg2"] - obs["sum_deg2"]) < 1e-9 for s in cfg_records), "CFG 未严格保度"
    assert all(abs(s["m"] - m0) < 1e-9 for s in cfg_records), "CFG 边数不守恒"
    log(f"CFG: C_mean={cfg_C.mean():.4f} sd={cfg_C.std(ddof=1):.4f} z={res_c['z']:.2f} "
        f"p={res_c['p_two_sided']:.4f}")
    add_instances("CFG_edge_switch", "synthetic_n674_lcc", cfg_records, obs,
                  "C.degree_preserving_null：每轮 fresh copy + double_edge_swap(nswap=10m, max_tries=100*nswap)；"
                  "Viger-Latapy 非均匀采样")

    # ------------------------------------- 5. 分解：仅打乱边(保度) / 仅重连(保聚类)
    d1 = cfg_records  # 同配置模型 = 仅打乱边、保度
    add_instances("D1_edge_shuffle_keep_degree", "synthetic_n674_lcc", d1, obs,
                  "分解用别名，等同 CFG_edge_switch")

    d2a, d2b = [], []
    nsw = 10 * m0
    for i in range(N_INST):
        rng = np.random.default_rng(SEED + 7919 * i)
        H, done, tries = triangle_sum_preserving_swap(Gobs, nsw, 100 * nsw, rng)
        assert H.number_of_edges() == m0, "D2a 边数不守恒"
        st = gstats(H)
        assert abs(st["sum_deg2"] - obs["sum_deg2"]) < 1e-9, "D2a 度序列被改动"
        assert abs(st["triangles"] - obs["triangles"]) < 1e-9, "D2a 三角总数被改动了"
        st["swap_done"] = done
        st["swap_tries"] = tries
        d2a.append(st)
    add_instances("D2a_triangle_sum_preserving", "synthetic_n674_lcc", d2a, obs,
                  "保度序列 + 保三角总数 => 传递性精确不变；逐点平均聚类不保证")
    log(f"D2a done: 三角总数不变={np.mean([abs(s['triangles'] - obs['triangles']) < 1e-9 for s in d2a]):.0%} "
        f"接受率={np.mean([s['swap_done'] / s['swap_tries'] for s in d2a]):.3f} "
        f"C_lcc={np.mean([s['C_lcc'] for s in d2a]):.4f} trans={np.mean([s['transitivity'] for s in d2a]):.4f} "
        f"L={np.mean([s['L'] for s in d2a]):.4f} assort={np.mean([s['assortativity'] for s in d2a]):.4f}")

    for i in range(N_INST):
        rng = np.random.default_rng(SEED + 104729 * i)
        H, done, tries = clustering_preserving_swap(Gobs, nsw, 100 * nsw, rng)
        assert H.number_of_edges() == m0, "D2b 边数不守恒"
        st = gstats(H)
        assert abs(st["sum_deg2"] - obs["sum_deg2"]) < 1e-9, "D2b 度序列被改动"
        assert abs(st["triangles"] - obs["triangles"]) < 1e-9, "D2b 三角总数被改动了"
        assert abs(st["C_all"] - obs["C_all"]) < 1e-12, "D2b 逐点平均聚类被改动了"
        st["swap_done"] = done
        st["swap_tries"] = tries
        d2b.append(st)
    add_instances("D2b_per_node_clustering_preserving", "synthetic_n674_lcc", d2b, obs,
                  "仅重连两端都不在三角形里的边且新位置不成三角 => 逐点聚类系数精确不变")
    log(f"D2b done: C_all(逐点平均聚类)精确不变比例="
        f"{np.mean([abs(s['C_all'] - obs['C_all']) < 1e-12 for s in d2b]):.0%} "
        f"C_lcc={np.mean([s['C_lcc'] for s in d2b]):.4f} trans={np.mean([s['transitivity'] for s in d2b]):.4f} "
        f"L={np.mean([s['L'] for s in d2b]):.4f} assort={np.mean([s['assortativity'] for s in d2b]):.4f} "
        f"lcc_frac={np.mean([s['lcc_frac'] for s in d2b]):.4f}")
    log("混合度诊断（与观测 GCC674 重合边占比 mean）: " + " | ".join(
        f"{nm} {np.mean([s['overlap_obs'] for s in st]):.3f}"
        for nm, st in (("ER_gnp", er_gnp), ("CFG_D1", cfg_records), ("D2a", d2a), ("D2b", d2b))))

    for p in WS_PS:
        key = f"D3_obsWS_p{p:g}"
        lst = []
        for i in range(N_INST):
            rng = np.random.default_rng(SEED + 31337 * (int(p * 1000) + 1) + i)
            lst.append(gstats(rewire_observed(Gobs, p, rng)))
        if p == 0.0:
            # p=0 应当精确回到观测图（自检：整条 D3 序列的参照系没跑偏）
            assert all(abs(s["C_all"] - obs["C_all"]) < 1e-12 and abs(s["L"] - obs["L"]) < 1e-12
                       and abs(s["overlap_obs"] - 1.0) < 1e-12 for s in lst), "D3 p=0 未回到观测图"
        add_instances(key, "synthetic_n674_lcc", lst, obs,
                      f"WS 式重连直接作用在观测 GCC674 上，p={p:g}（保 n、保 m，不保度序列）")
    log(f"D3 obs-WS scan done ({len(WS_PS)} p x {N_INST} inst)")

    # ------------------------------- 6. 162 个整站不可用节点 / 360 健康节点对照
    m_tab = b["nodes"]
    hm = m_tab.set_index("node")
    dead = b["fail"]["dead_all"]
    healthy_all = b["fail"]["healthy"]
    healthy = healthy_all & set(Gobs.nodes())
    partial = b["fail"]["partial_all"] & set(Gobs.nodes())
    in_gcc = set(Gobs.nodes())
    deg_obs = dict(Gobs.degree())
    dead_in = sorted(dead & in_gcc)
    leaf_deg1 = [v for v in Gobs.nodes() if deg_obs[v] == 1]
    dead_leaf = [v for v in dead_in if deg_obs[v] == 1]
    # 任务书说的"360 个健康节点"里，有 2 个不在 GCC（{bm, l.bm} 那个 2 点分量）
    healthy_out = sorted(healthy_all - healthy)
    log(f"GCC674 内：dead={len(dead_in)} partial={len(partial)} healthy={len(healthy)}; "
        f"GCC 外的健康节点 {len(healthy_out)} 个 = {healthy_out}; "
        f"度=1 叶子共 {len(leaf_deg1)} 个，其中 dead {len(dead_leaf)} 个")
    assert len(healthy_all) == 360 and len(dead_in) == 162

    # 直径对里死站占几端
    nodes, idx, dist = obs["_nodes"], obs["_idx"], obs["_dist"]
    diam = int(obs["diameter"])
    dead_arr = np.array([1 if nodes[i] in dead else 0 for i in range(len(nodes))])
    n_pairs = 0
    n_with_dead = 0
    n_both_dead = 0
    for a, c in zip(*np.triu_indices(len(nodes), 1)):
        if dist[a, c] == diam:
            n_pairs += 1
            if dead_arr[a] or dead_arr[c]:
                n_with_dead += 1
            if dead_arr[a] and dead_arr[c]:
                n_both_dead += 1
    log(f"距离={diam} 的点对 {n_pairs} 对：至少一端 dead {n_with_dead} 对 ({(n_with_dead/n_pairs if n_pairs else 0):.1%})，"
        f"两端都 dead {n_both_dead} 对")

    # 叶子（度=1）不可能是任何最短路的中继点：全部 674 点的度分布
    deg_hist = pd.Series(sorted(deg_obs.values())).value_counts().sort_index()
    log(f"GCC 度分布：度=1 共 {int(deg_hist.get(1,0))} 个节点，"
        f"度<=2 共 {int(sum(v for k2, v in deg_hist.items() if k2 <= 2))} 个")

    subsets = {
        "healthy360_in_gu676": sorted(healthy_all),   # 字面上的 360 个健康节点（在全图口径上诱导）
    }
    for tag, keep in subsets.items():
        sub = C.induced(G676, keep)
        st = graph_stats(sub)
        for mk in ("n", "m", "L", "diameter", "C_lcc", "C_all", "transitivity", "assortativity",
                   "kavg", "lcc_frac", "ncomp", "lcc_size"):
            add_observed("E_health_subset", mk, tag, f"observed_{tag}", round(float(st[mk]), 6),
                         "节点集=360 个健康节点在 G676 上的诱导图（含 GCC 外的 {bm,l.bm}）；"
                         "L/直径/C 在其最大分量上")
        log(f"subset {tag}: n={st['n']:.0f} lcc={st['lcc_size']:.0f} ncomp={st['ncomp']:.0f} "
            f"L={st['L']:.4f} diam={st['diameter']:.0f} C={st['C_lcc']:.4f} C_all={st['C_all']:.4f} "
            f"assort={st['assortativity']:.4f}")

    subsets = {
        "healthy358": sorted(healthy),
        "nonleaf_all": [v for v in Gobs.nodes() if deg_obs[v] >= 2],
        "dead_removed": [v for v in Gobs.nodes() if v not in dead],
    }
    for tag, keep in subsets.items():
        sub = C.induced(Gobs, keep)
        st = graph_stats(sub)
        add_observed("E_health_subset", "n", f"gcc674_{tag}", f"observed_{tag}", int(st["n"]))
        for mk in ("L", "diameter", "C_lcc", "C_all", "transitivity", "assortativity", "kavg",
                   "lcc_frac", "m", "ncomp", "lcc_size"):
            add_observed("E_health_subset", mk, f"gcc674_{tag}", f"observed_{tag}",
                         round(float(st[mk]), 6),
                         "节点集=GCC674 的显式子集诱导图；L/直径/C 在其最大分量上")
        log(f"subset {tag}: n={st['n']:.0f} lcc={st['lcc_size']:.0f} ncomp={st['ncomp']:.0f} "
            f"L={st['L']:.4f} diam={st['diameter']:.0f} C={st['C_lcc']:.4f} "
            f"assort={st['assortativity']:.4f}")

    # 同规模随机子集 / 度分层随机子集（给 healthy360 两个"只是子图变小了"的对照）
    nh = len(healthy)
    rng = np.random.default_rng(SEED + 5)
    alln = list(Gobs.nodes())
    rs = []
    for i in range(N_INST):
        keep = list(rng.choice(alln, size=nh, replace=False))
        rs.append(graph_stats(C.induced(Gobs, keep)))
    add_instances(f"rand_subgraph_n{nh}", f"gcc674_rand{nh}_lcc", rs, obs,
                  f"GCC674 内均匀随机取 {nh} 个点的诱导图（规模匹配对照，不匹配度）")
    log(f"rand{nh}: L={np.mean([s['L'] for s in rs]):.4f} diam={np.mean([s['diameter'] for s in rs]):.2f} "
        f"C={np.mean([s['C_lcc'] for s in rs]):.4f}")

    by_deg: dict[int, list] = {}
    for v in alln:
        by_deg.setdefault(deg_obs[v], []).append(v)
    target: dict[int, int] = {}
    for v in healthy:
        target[deg_obs[v]] = target.get(deg_obs[v], 0) + 1
    dm = []
    for i in range(N_INST):
        r2 = np.random.default_rng(SEED + 777 + i)
        keep = []
        for d, cnt in target.items():
            pool = by_deg[d]
            keep += list(r2.choice(pool, size=min(cnt, len(pool)), replace=False))
        dm.append(graph_stats(C.induced(Gobs, keep)))
    add_instances(f"degmatch_subgraph_n{nh}", f"gcc674_degreematch{nh}_lcc", dm, obs,
                  f"GCC674 内按度分层随机取点，逐度值节点数与 healthy{nh} 相同（度分布匹配对照）")
    log(f"degmatch{nh}: L={np.mean([s['L'] for s in dm]):.4f} diam={np.mean([s['diameter'] for s in dm]):.2f} "
        f"C={np.mean([s['C_lcc'] for s in dm]):.4f}")

    # --------- 6a. 死站/叶子在路径上的位置（纯描述，不构成因果）
    nm_nodes = len(nodes)
    is_dead_arr = dead_arr.astype(bool)
    is_alive = ~is_dead_arr
    offd = ~np.eye(nm_nodes, dtype=bool)
    AA = offd & is_alive[:, None] & is_alive[None, :]
    AD = offd & ((is_alive[:, None] & ~is_alive[None, :]) | (~is_alive[:, None] & is_alive[None, :]))
    DD = offd & ~is_alive[:, None] & ~is_alive[None, :]
    AL = offd & (~is_alive[:, None] | ~is_alive[None, :])
    Dm = dist.astype(np.float64)
    n_nond = int(is_alive.sum())   # 674-162=512（含 5 个悬空引用节点）
    # 注意分组用的是"非死站"= healthy358 + partial149 + 5 个悬空引用 = 512 个，不是"健康站"
    for gname, msk in (("nondead512_nondead512", AA), ("nondead512_dead162", AD),
                       ("dead162_dead162", DD), ("at_least_one_nondead", AL)):
        v = Dm[msk]
        ROWS.append({"block": "C_leaf_and_path", "metric": "mean_pair_distance",
                     "node_set": "gcc674 内的有序点对分组", "model": gname,
                     "n_inst": int(v.size), "mean": round(float(v.mean()), 6), "sd": round(float(v.std(ddof=1)), 6),
                     "min": round(float(v.min()), 4), "median": round(float(np.median(v)), 4),
                     "max": round(float(v.max()), 4), "obs_value": round(obs["L"], 6),
                     "obs_over_mean": round(float(obs["L"] / v.mean()), 6), "z_of_obs": np.nan,
                     "p_two_sided": np.nan,
                     "note": f"该分组占全部 {nm_nodes*(nm_nodes-1)//2} 个无序点对的 "
                             f"{v.size / (nm_nodes*(nm_nodes-1)):.4f}；obs_value 列=整图 L=3.2702"})
    # 死站是否只是末端：非叶子的死站有没有落在某条"非死站—非死站"最短路径上
    dead_internal = []
    for wi in range(nm_nodes):
        if not is_dead_arr[wi] or deg_obs[nodes[wi]] < 2:
            continue
        dw = Dm[wi]
        cand = np.outer(dw, dw)
        hit = (np.abs(Dm - cand) < 1e-12) & AA & (dw[:, None] > 0) & (dw[None, :] > 0)
        if hit.any():
            dead_internal.append(nodes[wi])
    for tag, grp in (("dead162", dead_in), ("healthy358", sorted(healthy)), ("partial149", sorted(partial))):
        dg = np.array([deg_obs[v] for v in grp], dtype=float)
        for mk, mv in (("kavg", float(dg.mean())), ("kmax", float(dg.max())),
                       ("n_leaves_deg1", float((dg == 1).sum())),
                       ("share_leaves_deg1", float((dg == 1).mean()))):
            ROWS.append({"block": "C_leaf_and_path", "metric": mk,
                         "node_set": f"gcc674_{tag}", "model": f"group_{tag}", "n_inst": int(dg.size),
                         "mean": round(mv, 6), "sd": np.nan, "min": round(float(dg.min()), 4),
                         "median": round(float(np.median(dg)), 4), "max": round(float(dg.max()), 4),
                         "obs_value": round(obs["kavg"], 6), "obs_over_mean": np.nan,
                         "z_of_obs": np.nan, "p_two_sided": np.nan,
                         "note": "GCC674 上的无向度；死站出度恒为 0（CONVENTIONS §2.1），故其无向度=入度"})
        # 同口径对照（CONVENTIONS §2.1：跨口径比 1.83 vs 9.49 是错的，这里三套都给）
        ind = hm["in_degree"].reindex(grp).fillna(0).to_numpy(dtype=float)
        outd = hm["out_degree"].reindex(grp).fillna(0).to_numpy(dtype=float)
        for mk, arr_ in (("in_degree_Gd_kavg", ind), ("out_degree_Gd_kavg", outd)):
            ROWS.append({"block": "C_leaf_and_path", "metric": mk,
                         "node_set": f"gcc674_{tag}", "model": f"group_{tag}", "n_inst": int(arr_.size),
                         "mean": round(float(arr_.mean()), 6), "sd": round(float(arr_.std(ddof=1)), 6),
                         "min": round(float(arr_.min()), 4), "median": round(float(np.median(arr_)), 4),
                         "max": round(float(arr_.max()), 4), "obs_value": np.nan,
                         "obs_over_mean": np.nan, "z_of_obs": np.nan, "p_two_sided": np.nan,
                         "note": "有向图 Gd 上的度（口径与无向总度不同，勿跨口径比）"})
        if tag == "dead162":
            assert float(outd.max()) == 0.0, "CONVENTIONS §2.1：死站出度应恒为 0"
    ROWS.append({"block": "C_leaf_and_path", "metric": "n_dead_on_nondead_nondead_shortest_path",
                 "node_set": "gcc674", "model": "observed", "n_inst": 1,
                 "mean": len(dead_internal), "sd": np.nan, "min": len(dead_internal),
                 "median": len(dead_internal), "max": len(dead_internal),
                 "obs_value": len(dead_internal), "obs_over_mean": 1.0, "z_of_obs": np.nan,
                 "p_two_sided": np.nan,
                 "note": f"162 个死站里 {len([v for v in dead_in if deg_obs[v] >= 2])} 个非叶子中，"
                         f"有 {len(dead_internal)} 个位于某条非死站—非死站最短路径上（其余仅作端点）"})
    log(f"点对分组平均距离: alive-alive={Dm[AA].mean():.4f} alive-dead={Dm[AD].mean():.4f} "
        f"dead-dead={Dm[DD].mean():.4f}")
    log(f"死站度: kavg={np.mean([deg_obs[v] for v in dead_in]):.4f} "
        f"叶子={sum(1 for v in dead_in if deg_obs[v] == 1)}/162; "
        f"健康站 kavg={np.mean([deg_obs[v] for v in healthy]):.4f} "
        f"叶子={sum(1 for v in healthy if deg_obs[v] == 1)}/{len(healthy)}; "
        f"非叶子死站中位于非死站-非死站最短路径上的 {len(dead_internal)} 个")

    # ------------------- 6b. L 口径敏感性（λ 是四个数里唯一随约定变的，必须并列给出）
    def L_variants(H):
        nodes_, idx_, dist_ = apsp(H)
        nn = len(nodes_)
        comps_ = sorted(nx.connected_components(H), key=len, reverse=True)
        lcc_ = comps_[0]
        li = np.array([idx_[x] for x in lcc_], dtype=int)
        sub = dist_[np.ix_(li, li)]
        off = ~np.eye(len(li), dtype=bool)
        d = sub[off].astype(np.float64)
        d = d[d > 0]
        allp = dist_[~np.eye(nn, dtype=bool)].astype(np.float64)
        allp = allp[allp > 0]
        return {
            "L_lcc": float(d.mean()),
            "L_harmonic_lcc": float(1.0 / (1.0 / d).mean()),
            "L_reachable_all": float(allp.mean()),
            "connected": float(len(comps_) == 1),
        }

    obs_lv = L_variants(Gobs)
    assert abs(obs_lv["L_lcc"] - obs["L"]) < 1e-9
    er_lv = [L_variants(nx.gnp_random_graph(n0, p_er, seed=SEED + i)) for i in range(N_INST)]
    for mk in ("L_lcc", "L_harmonic_lcc", "L_reachable_all"):
        v = np.array([x[mk] for x in er_lv])
        ov = obs_lv[mk]
        sd = float(v.std(ddof=1))
        ROWS.append({"block": "F_L_convention", "metric": mk, "node_set": "gcc674 vs synthetic_n674",
                     "model": "ER_gnp", "n_inst": int(v.size), "mean": round(float(v.mean()), 6),
                     "sd": round(sd, 6), "min": round(float(v.min()), 6),
                     "median": round(float(np.median(v)), 6), "max": round(float(v.max()), 6),
                     "obs_value": round(ov, 6), "obs_over_mean": round(float(ov / v.mean()), 6),
                     "z_of_obs": round(float((ov - v.mean()) / sd), 4),
                     "p_two_sided": round(float((np.abs(v - v.mean()) >= abs(ov - v.mean())).mean()), 4),
                     "note": "λ 的分母按同一约定取；主口径 = L_lcc（Watts-Strogatz 惯例）"})
    v = np.array([x["L_lcc"] for x in er_lv if x["connected"] == 1.0])
    if v.size >= 5:
        ROWS.append({"block": "F_L_convention", "metric": "L_lcc", "node_set": "gcc674 vs synthetic_n674",
                     "model": "ER_gnp_connected_only", "n_inst": int(v.size),
                     "mean": round(float(v.mean()), 6), "sd": round(float(v.std(ddof=1)), 6),
                     "min": round(float(v.min()), 6), "median": round(float(np.median(v)), 6),
                     "max": round(float(v.max()), 6), "obs_value": round(obs_lv["L_lcc"], 6),
                     "obs_over_mean": round(float(obs_lv["L_lcc"] / v.mean()), 6),
                     "z_of_obs": round(float((obs_lv["L_lcc"] - v.mean()) / v.std(ddof=1)), 4),
                     "p_two_sided": np.nan,
                     "note": f"仅取 {v.size}/{N_INST} 个恰好连通的实例（拒绝采样条件）"})
    log("L 口径敏感性: obs " + " ".join(f"{k}={obs_lv[k]:.4f}" for k in
                                        ("L_lcc", "L_harmonic_lcc", "L_reachable_all")))
    log("ER 口径敏感性: " + " ".join(f"{k}={np.mean([x[k] for x in er_lv]):.4f}" for k in
                                     ("L_lcc", "L_harmonic_lcc", "L_reachable_all"))
        + f" connected={np.mean([x['connected'] for x in er_lv]):.0%}")

    # ---------------------------------------------------- 7. σ / γ / λ 汇总判决
    sig_rows = []
    SIG_FIELDS = {"gamma": "C_lcc", "gamma_all": "C_all", "lambda": "L",
                  "sigma": None, "sigma_all": None}
    for mdl in ("ER_gnp", "ER_gnm", "CFG_edge_switch", "D1_edge_shuffle_keep_degree",
                "D2a_triangle_sum_preserving", "D2b_per_node_clustering_preserving",
                *[f"D3_obsWS_p{p:g}" for p in WS_PS],
                *[f"WS_k{k}_p{p:g}" for k in WS_KS for p in WS_PS]):
        arr = {kk: np.array([x[kk] for x in INST if x["model"] == mdl and np.isfinite(x[kk])])
               for kk in SIG_FIELDS}
        if arr["sigma"].size == 0:
            continue
        pv = float(np.mean(arr["sigma"] > 1.0))          # σ_i>1 的实例占比（经验"是否小世界"频率）
        for kk in SIG_FIELDS:
            v = arr[kk]
            if v.size == 0:
                continue
            if kk == "gamma":
                ovv = round(obs["C_lcc"], 6)
            elif kk == "gamma_all":
                ovv = round(obs["C_all"], 6)
            elif kk == "lambda":
                ovv = round(obs["L"], 6)
            elif kk == "sigma":
                ovv = round(float(arr["gamma"].mean() / arr["lambda"].mean()), 4)
            else:
                ovv = round(float(arr["gamma_all"].mean() / arr["lambda"].mean()), 4)
            ROWS.append({
                "block": "C_sigma", "metric": kk,
                "node_set": "gcc674(obs) vs synthetic_n674_lcc(model)", "model": mdl,
                "n_inst": int(v.size), "mean": round(float(v.mean()), 4),
                "sd": round(float(v.std(ddof=1)), 4) if v.size > 1 else np.nan,
                "min": round(float(v.min()), 4), "median": round(float(np.median(v)), 4),
                "max": round(float(v.max()), 4), "obs_value": ovv, "obs_over_mean": np.nan,
                "z_of_obs": np.nan, "p_two_sided": round(pv, 4),
                "note": ("p_two_sided 列 = P(sigma_i>1)；obs_value=观测值" if kk != "sigma"
                         else "p_two_sided 列 = P(sigma_i>1)；obs_value 列 = sigma(均值之比)；下标 i 逐实例")})
        sig_rows.append((mdl, float(arr["gamma"].mean()), float(arr["lambda"].mean()),
                         float(arr["sigma"].mean()), pv))
    log("sigma table built for %d models" % len(sig_rows))

    # 度异质性对"超额聚类"的可解释份额（对数分解，纯算术）
    C_er = float(np.mean([s["C_lcc"] for s in er_gnp]))
    C_cfg = float(np.mean([s["C_lcc"] for s in cfg_records]))
    share_deg = np.log(C_cfg / C_er) / np.log(obs["C_lcc"] / C_er)
    share_res = np.log(obs["C_lcc"] / C_cfg) / np.log(obs["C_lcc"] / C_er)
    assert abs(share_deg + share_res - 1.0) < 1e-9
    for kk, vv in (("log_excess_clustering_total", float(np.log(obs["C_lcc"] / C_er))),
                   ("log_share_reproduced_by_degree_sequence", float(share_deg)),
                   ("log_share_residual_beyond_degree_sequence", float(share_res))):
        ROWS.append({"block": "C_sigma", "metric": kk, "node_set": "gcc674 vs synthetic_n674_lcc",
                     "model": "ER_gnp -> CFG_edge_switch -> observed", "n_inst": N_INST,
                     "mean": round(vv, 4), "sd": np.nan, "min": np.nan, "median": np.nan,
                     "max": np.nan, "obs_value": np.nan, "obs_over_mean": np.nan,
                     "z_of_obs": np.nan, "p_two_sided": np.nan,
                     "note": "对数分解 share_deg+share_res=1（恒等式，已断言）"})

    # --------------------------------------------------------- 8. 写文件
    cols = ["block", "metric", "node_set", "model", "n_inst", "mean", "sd", "min", "median",
            "max", "obs_value", "obs_over_mean", "z_of_obs", "p_two_sided", "note"]
    dfm = pd.DataFrame(ROWS)[cols]
    p1 = C.save_csv(dfm, MODULE, "smallworld_metrics.csv")
    dfi = pd.DataFrame(INST)
    dfi = dfi[["model", "node_set", "inst", "block", "n", "m", "kavg", "kmax", "sum_deg2", "ncomp",
               "lcc_size", "lcc_frac", "n_isolated", "L", "diameter", "C_lcc", "C_all",
               "transitivity", "triangles", "triples", "assortativity", "overlap_obs",
               "gamma", "gamma_all", "lambda", "sigma", "sigma_all", "note"]].round(6)
    p2 = C.save_csv(dfi, MODULE, "smallworld_instances.csv")

    # 节点级附表：GCC674 内每个节点的度 / 叶子标记 / 健康态 / 偏心率
    ecc_max = np.max(np.where(dist >= 0, dist, -1), axis=1)
    tri_obs = nx.triangles(Gobs)
    dn = pd.DataFrame({
        "node": nodes,
        "degree_gcc674": [deg_obs[v] for v in nodes],
        "is_leaf_deg1": [int(deg_obs[v] == 1) for v in nodes],
        "triangles_at_node": [int(tri_obs[v]) for v in nodes],
        "clustering_gcc674": [round(float(obs["_cl"][v]), 6) for v in nodes],
        "eccentricity_gcc674": [int(x) for x in ecc_max],
        "is_healthy": [int(v in healthy) for v in nodes],
        "is_dead": [int(v in dead) for v in nodes],
    })
    dn["health_state"] = dn["node"].map(hm["health_state"])
    dn["dead_cat"] = dn["node"].map(hm["dead_cat"])
    dn["in_degree_Gd"] = dn["node"].map(hm["in_degree"])
    dn["out_degree_Gd"] = dn["node"].map(hm["out_degree"])
    p3 = C.save_csv(dn, MODULE, "gcc674_node_profile.csv")

    meta = {
        "module": MODULE, "SEED": SEED, "N_INST": N_INST, "N_CFG": N_CFG,
        "WS_KS": list(WS_KS), "WS_PS": list(WS_PS),
        "obs": {k: (round(float(v), 6) if isinstance(v, (int, float, np.floating)) else None)
                for k, v in obs.items() if not k.startswith("_")},
        "obs676": {k: round(float(v), 6) for k, v in s676.items() if not k.startswith("_")},
        "obs671": {k: round(float(v), 6) for k, v in s671.items() if not k.startswith("_")},
        "cfg_csa_common": {kk: res_c[kk] for kk in ("obs", "null_mean", "null_sd", "z", "p_two_sided", "n_reps", "note")},
        "cfg_lambda_csa_common_30reps": {kk: res_l[kk] for kk in ("obs", "null_mean", "null_sd", "z", "p_two_sided")},
        "er_p": p_er,
        "dead_in_gcc": len(dead_in), "healthy_in_gcc": len(healthy), "partial_in_gcc": len(partial),
        "healthy_total": len(healthy_all), "healthy_outside_gcc": healthy_out,
        "leaf_deg1_in_gcc": len(leaf_deg1), "dead_leaf_deg1": len(dead_leaf),
        "diam_pairs": n_pairs, "diam_pairs_with_dead_endpoint": n_with_dead,
        "diam_pairs_both_dead": n_both_dead,
        "overlap_with_obs_mean": {nm: round(float(np.mean([s["overlap_obs"] for s in st])), 4)
                                  for nm, st in (("ER_gnp", er_gnp), ("CFG_D1", cfg_records),
                                                 ("D2a", d2a), ("D2b", d2b))},
        "share_deg": float(share_deg), "share_res": float(share_res),
        "files": [p1, p2, p3],
    }
    C.save_json(meta, MODULE, "a2_meta.json")
    log("written: " + ", ".join(os.path.basename(p) for p in (p1, p2, p3)))

    C.write_run_note(MODULE, [
        "A2_small_world RUN.txt",
        "command: python run_a2.py   (cwd = pipeline/)",
        f"versions: {C.versions()}",
        f"SEED = C.SEED = {SEED}; N_INST = {N_INST}; N_CFG = {N_CFG}; WS_KS = {WS_KS}; WS_PS = {WS_PS}",
        "csa_common.assert_anchors(): 31/31 PASS (脚本开头调用)",
        f"elapsed: {time.time() - T_START:.1f} s",
        *LOG,
    ])
    log(f"DONE in {time.time() - T_START:.1f}s")


if __name__ == "__main__":
    main()
