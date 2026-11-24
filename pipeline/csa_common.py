"""csa_common — 全项目唯一的建图/口径实现。

规则见 pipeline/CONVENTIONS.md。任何分析脚本都必须 `from csa_common import *`，
不得自己重写建图、清洗、效率或零模型逻辑。
"""
from __future__ import annotations

import os
import sys
import json
import random
import numpy as np
import pandas as pd
import networkx as nx

SEED = 20261113

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_RAW_SUB = "USTC网络拓扑分析"


def _find_data_dir():
    """定位原始数据目录。只改路径解析，不改任何分析口径。

    1) 仓库自带的 `USTC网络拓扑分析/`（交付布局，原料与 pipeline 同仓）
    2) 兄弟目录 `CSA_HW/USTC网络拓扑分析/`（本机开发期的两仓布局，向后兼容）

    顺序上仓库内优先：仓内那份与本仓锚点是同一份原料，若让外部的兄弟目录抢先生效，
    读到过期副本时锚点仍会"自洽地"通过，等于失去校验意义。
    找不到就抛错，绝不静默退回到别的目录：全项目 31 个锚点的前提是"读的是同一份原料"，
    静默换目录会让锚点校验失去意义。也可以用环境变量 CSA_DATA_DIR 显式指定。
    """
    cand = [os.environ.get("CSA_DATA_DIR"),
            os.path.join(ROOT, _RAW_SUB),
            os.path.join(ROOT, "CSA_HW", _RAW_SUB)]
    for c in cand:
        if c and os.path.isdir(c):
            return c
    raise FileNotFoundError(
        "找不到原始数据目录（试了 %s）。原料应在仓库根的 %s/ 里；"
        "或设 CSA_DATA_DIR 指向含 %s 的目录。" % (cand[1:], _RAW_SUB, _RAW_SUB))


DATA_DIR = _find_data_dir()
CLEAN_DIR = os.path.join(HERE, "data")
OUT_ROOT = os.path.join(HERE, "out")

os.environ.setdefault("PYTHONHASHSEED", "0")

# 6 个全局导航枢纽（用于"去枢纽边"敏感性分析，不是真值，只是一个口径）
NAV_HUBS = ["teach", "staff", "wpsearch", "news", "en", "fusep"]

# 失效语义分类：A/D/F 是站点真的坏了；B/C 是抓取器被挡在外面
TRUE_FAIL_CATS = ["A_conn_fail", "D_not_found", "F_server_error"]
FALSE_FAIL_CATS = ["B_auth_or_skip", "C_robots_blocked"]
AMBIGUOUS_CATS = ["G_other_400"]


# ---------------------------------------------------------------- 载入原始数据

def _fqdn(sub: str) -> str:
    return sub if sub.endswith(".ustc.edu.cn") else sub + ".ustc.edu.cn"


def load_nodes() -> pd.DataFrame:
    """671 个有属性节点。节点主键用 FQDN（`node` 列），与 links.csv 对齐。

    注意：nodes_full.csv 的 `degree` 列与 links.csv 重算不一致，禁止使用。
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "nodes_full.csv"))
    df["node"] = df["subdomain"]
    dup = df["node"].duplicated().sum()
    if dup:
        raise ValueError(f"nodes_full.csv 有 {dup} 个重复 FQDN")
    return df


def load_links() -> pd.DataFrame:
    """2788 条有向边（source/target 为 FQDN，weight = 观察到的链接次数）。

    `edge_order` = 在 links.csv 里的行号。它不是装饰：加权 Louvain 对建图时的
    节点插入顺序敏感，而节点顺序由边的顺序决定。同一张图、同一个 seed=42，
    按抓取顺序建图 = 25 社区 / Q=0.6904 / 最大 87；按 (source,target) 字典序
    建图 = 24 社区 / Q=0.6898 / 最大 84。所以清洗产物必须保留 edge_order，
    并且所有建图入口都按它排序。
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "links.csv"))
    df["edge_order"] = np.arange(len(df))
    return df


def load_urls() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "urls.csv"))


def load_dead_categories() -> pd.DataFrame:
    """162 个"整站不可用"子域的失败原因分类。主键是 `sub`（短名）。"""
    d = pd.read_csv(os.path.join(DATA_DIR, "dead_sites_category.csv"))
    d["node"] = d["sub"].map(_fqdn)
    return d


# ------------------------------------------------------------------- 建图

def _ordered_links(links: pd.DataFrame) -> pd.DataFrame:
    """建图前统一按 edge_order 排序，使节点插入顺序与 CSV 行序无关。"""
    if "edge_order" in links.columns:
        return links.sort_values("edge_order")
    return links


def build_directed(links: pd.DataFrame, nodes: pd.DataFrame):
    """有向图：676 个节点（含 5 个只在边里出现的悬空引用）+ 671 个有属性节点。

    links.csv 无自环、无重复 (source,target) 对（已核验），因此 DiGraph 足够。
    """
    G = nx.DiGraph()
    for s, t, w in _ordered_links(links)[["source", "target", "weight"]].itertuples(index=False):
        G.add_edge(s, t, weight=int(w))
    G.add_nodes_from(nodes["node"])
    return G


def build_undirected(links: pd.DataFrame, nodes: pd.DataFrame):
    """无向加权图：**互为反向的两条边权重必须相加**。

    这是本项目最容易搞错的一步。用 add_edge(..., weight=w) 直接覆盖会得到
    19 个社区 / Q≈0.677；相加得到 2498 条边、25 个社区 / Q=0.6904（与作者一致）。

    建图顺序也是口径的一部分：先加边、后补节点。反过来（先 add_nodes_from(671)
    再 add_edges_from）**边集完全相同**，但加权 Louvain 在同一 seed 下会给
    24 个社区 / Q=0.6898 / 最大社区 87。也就是说本图的 Louvain 对节点插入顺序
    敏感，"25 个社区"不是一个稳定事实，必须报分布（见 louvain_robust）。
    """
    agg: dict = {}
    for s, t, w in _ordered_links(links)[["source", "target", "weight"]].itertuples(index=False):
        k = (s, t) if s <= t else (t, s)
        agg[k] = agg.get(k, 0) + int(w)
    G = nx.Graph()
    for (s, t), w in agg.items():
        G.add_edge(s, t, weight=w)
    G.add_nodes_from(nodes["node"])  # 兜底：保证有属性节点一定在图里（当前是 no-op）
    return G


def build_binary(Gu: nx.Graph) -> nx.Graph:
    """无权无向图（拓扑指标默认口径）。"""
    H = nx.Graph()
    H.add_nodes_from(Gu.nodes())
    H.add_edges_from(Gu.edges())
    return H


def induced(G: nx.Graph, keep) -> nx.Graph:
    """子图诱导；保持节点集口径显式。"""
    return G.subgraph(list(keep)).copy()


# ------------------------------------------------------------------ 节点集合

def node_sets(Gu: nx.Graph, nodes: pd.DataFrame) -> dict:
    """三套节点口径，报告里必须写清用的是哪一套。

    - attributed (671)：主分析口径，有健康/标题属性
    - in_graph   (676)：含 5 个悬空节点，附录/拓扑对照
    - gcc        (674)：无向图最大连通分量，鲁棒性实验基线
    """
    attributed = set(nodes["node"])
    in_graph = set(Gu.nodes())
    comps = sorted(nx.connected_components(Gu), key=len, reverse=True)
    return {
        "attributed": attributed,
        "in_graph": in_graph,
        "dangling": in_graph - attributed,
        "gcc": comps[0],
        "n_components": len(comps),
        "comp_sizes": [len(c) for c in comps],
    }


# --------------------------------------------------------------- 度 / 强度

def add_degrees(nodes: pd.DataFrame, Gd: nx.DiGraph, Gu: nx.Graph) -> pd.DataFrame:
    """三种度同时给出。健康梯度分析**必须同时报入度**，见 CONVENTIONS §2。

    关键事实（已核验）：162 个整站不可用节点的**出度全为 0**，
    所以它们的无向度 == 入度。"平均度 1.83 vs 9.49 (5.2×)" 是跨口径比较，
    同口径应比入度：1.83 vs 4.94 (2.7×)。
    """
    m = nodes.copy()
    m = m.drop(columns=[c for c in ("in_degree", "out_degree", "degree_u", "strength_u",
                                    "in_strength", "out_strength") if c in m.columns])
    m["in_degree"] = m["node"].map(Gd.in_degree).fillna(0).astype(int)
    m["out_degree"] = m["node"].map(Gd.out_degree).fillna(0).astype(int)
    m["degree_u"] = m["node"].map(Gu.degree).fillna(0).astype(int)
    m["strength_u"] = m["node"].map(dict(Gu.degree(weight="weight"))).fillna(0).astype(int)
    m["in_strength"] = m["node"].map(dict(Gd.in_degree(weight="weight"))).fillna(0).astype(int)
    m["out_strength"] = m["node"].map(dict(Gd.out_degree(weight="weight"))).fillna(0).astype(int)
    return m


# --------------------------------------------------------------- 健康 / 失效

def add_health(nodes: pd.DataFrame, dead_cat: pd.DataFrame) -> pd.DataFrame:
    """健康分层 + 真/假失效校正。

    用 `ok_ratio_pct`（URL 可用率）而不是 `color`：color 与可用率不一致
    （32 个可用率为 0 的站点在 color 里被标成 blue）。
    """
    m = nodes.copy()
    m = m.drop(columns=[c for c in ("is_dead", "is_partial", "is_healthy", "health_state",
                                    "dead_cat", "true_fail", "false_fail", "ambiguous_fail",
                                    "partial_has_auth_like", "partial_clean") if c in m.columns])
    m["is_dead"] = m["ok_ratio_pct"] == 0
    m["is_healthy"] = m["ok_ratio_pct"] == 100
    m["is_partial"] = (~m["is_dead"]) & (~m["is_healthy"])
    m["health_state"] = np.select(
        [m["is_dead"], m["is_partial"], m["is_healthy"]],
        ["dead", "partial", "healthy"],
        default="unknown",
    )

    cat = m["node"].map(dead_cat.set_index("node")["cat"])
    m["dead_cat"] = cat
    m["true_fail"] = m["is_dead"] & cat.isin(TRUE_FAIL_CATS)
    m["false_fail"] = m["is_dead"] & cat.isin(FALSE_FAIL_CATS)
    m["ambiguous_fail"] = m["is_dead"] & cat.isin(AMBIGUOUS_CATS)
    # "干净的部分失效"：部分失效且 URL 状态码里没有 -1/-2/403（认证/robots/ACL）
    m["partial_has_auth_like"] = m["is_partial"] & m["status_codes"].fillna("").astype(str).str.contains(
        r"(?:^|[\s,])-1:|(?:^|[\s,])-2:|403:", regex=True
    )
    m["partial_clean"] = m["is_partial"] & (~m["partial_has_auth_like"])
    return m


def failure_sets(m: pd.DataFrame) -> dict:
    """常用失效节点集合（FQDN），供鲁棒性/风险矩阵直接取用。"""
    g = lambda mask: set(m.loc[mask, "node"])  # noqa: E731
    return {
        "dead_all": g(m["is_dead"]),
        "true_fail": g(m["true_fail"]),
        "false_fail": g(m["false_fail"]),
        "ambiguous": g(m["ambiguous_fail"]),
        "partial_all": g(m["is_partial"]),
        "partial_clean": g(m["partial_clean"]),
        "partial_auth_like": g(m["partial_has_auth_like"]),
        "healthy": g(m["is_healthy"]),
    }


# -------------------------------------------------------------------- 效率
# 三种口径必须同时报。只报 E_full 会得出"删掉全挂站点效率反而上升"的假结论。

def eff_full(H: nx.Graph) -> float:
    """整张剩余图的全局效率（不可达对记 0，分母随删除收缩）。作者用的就是这个。"""
    return nx.global_efficiency(H) if H.number_of_nodes() > 1 else 0.0


def largest_component(H: nx.Graph) -> nx.Graph:
    comps = sorted(nx.connected_components(H), key=len, reverse=True)
    return H.subgraph(comps[0]).copy() if comps else H.copy()


def eff_gcc(H: nx.Graph, which: int = 0) -> float:
    """只在本图最大的第 `which` 个分量上算效率——没有不可达对，也不受分母收缩影响。"""
    comps = sorted(nx.connected_components(H), key=len, reverse=True)
    if len(comps) <= which or len(comps[which]) < 2:
        return 0.0
    return nx.global_efficiency(H.subgraph(comps[which]).copy())


class PairSet:
    """基线连通对样本集：固定分母 = n0*(n0-1) 个有序对（n0 = 基线 GCC 规模）。"""

    def __init__(self, gcc_nodes):
        self.nodes = list(gcc_nodes)
        self.idx = {n: i for i, n in enumerate(self.nodes)}
        self.n0 = len(self.nodes)
        self.denom = float(self.n0 * (self.n0 - 1))

    def __repr__(self):
        return f"PairSet(n0={self.n0}, ordered_pairs={int(self.denom)})"


def eff_fixed(H: nx.Graph, ps: PairSet) -> float:
    """固定样本口径（**主口径**）。

    分母恒为基线 GCC 的有序对数 |P0|；端点被删除或删后不连通的对都记 0。
    这样"分母随删除收缩"不可能抬高数值，作者那条"删掉全挂站点效率反而上升"就消失了。

    代价：它把"节点被移除"本身也算成损失，所以只能**在同一移除数量下跨策略比较**，
    不能拿它跟基线比大小来论证"网络变好了"。
    """
    if ps.denom == 0:
        return 0.0
    present = [n for n in H.nodes() if n in ps.idx]
    if len(present) < 2:
        return 0.0
    sub = H.subgraph(present)
    tot = 0.0
    for comp in nx.connected_components(sub):
        if len(comp) < 2:
            continue
        for s in comp:
            for t, d in nx.single_source_shortest_path_length(sub, s).items():
                if d:
                    tot += 1.0 / d
    return float(tot / ps.denom)


def eff_three(H: nx.Graph, ps: PairSet | None = None) -> dict:
    out = {"E_full": eff_full(H), "E_gcc": eff_gcc(H)}
    if ps is not None:
        out["E_fixed"] = eff_fixed(H, ps)
    return out


# ----------------------------------------------------------------- 社区工具


def louvain(G: nx.Graph, seed: int = SEED, resolution: float = 1.0):
    """加权 Louvain。返回 (partition dict, Q)。

    报告时必须同时给 `n_comms_total` 与 `n_comms_nonsingleton`（CONVENTIONS §3）：
    强边骨架下大量孤立零度节点会被各算成一个社区，直接报总数会虚高十倍。
    """
    part = nx.community.louvain_communities(
        G, weight="weight", resolution=resolution, seed=seed, max_level=100
    )
    sizes = sorted((len(c) for c in part), reverse=True)
    Q = nx.community.modularity(G, part, weight="weight", resolution=resolution)
    return {
        "partition": part,
        "Q": float(Q),
        "n_comms_total": len(part),
        "n_comms_nonsingleton": int(sum(1 for s in sizes if s > 1)),
        "n_singleton": int(sum(1 for s in sizes if s == 1)),
        "n_covered": int(sum(sizes)),
        "largest": sizes[0] if sizes else 0,
        "sizes": sizes,
    }


def louvain_robust(G: nx.Graph, seeds=range(20), resolution: float = 1.0) -> dict:
    """多 seed Louvain → 报分布，不报单个数。

    本图的社区划分在 seed 之间会波动（实测 23~26 个社区、最大社区 80~113），
    所以任何"社区数 = X"的结论都必须配这张分布表。
    """
    runs = [louvain(G, seed=s, resolution=resolution) for s in seeds]
    ks = np.array([r["n_comms_total"] for r in runs])
    nsc = np.array([r["n_comms_nonsingleton"] for r in runs])
    Qs = np.array([r["Q"] for r in runs])
    lg = np.array([r["largest"] for r in runs])
    node_list = list(G.nodes())
    pos = {n: i for i, n in enumerate(node_list)}
    members = np.full((len(runs), len(node_list)), -1, dtype=np.int32)
    for k, r in enumerate(runs):
        for ci, c in enumerate(r["partition"]):
            for n in c:
                members[k, pos[n]] = ci
    co = np.zeros((len(node_list), len(node_list)), dtype=np.float32)
    for k in range(len(runs)):
        lab = members[k]
        co += (lab[:, None] == lab[None, :])
    co /= len(runs)
    return {
        "n_seeds": len(runs),
        "comms_total": {"min": int(ks.min()), "median": float(np.median(ks)), "max": int(ks.max())},
        "comms_nonsingleton": {"min": int(nsc.min()), "median": float(np.median(nsc)), "max": int(nsc.max())},
        "Q": {"min": round(float(Qs.min()), 4), "mean": round(float(Qs.mean()), 4),
              "sd": round(float(Qs.std(ddof=1)), 4), "max": round(float(Qs.max()), 4)},
        "largest": {"min": int(lg.min()), "median": float(np.median(lg)), "max": int(lg.max())},
        "per_seed": [{"seed": int(s), "comms": int(runs[i]["n_comms_total"]),
                      "Q": round(runs[i]["Q"], 4), "largest": runs[i]["largest"]}
                     for i, s in enumerate(seeds)],
        "consensus": co,
        "node_list": node_list,
    }


def hub_nodes(G: nx.Graph, hubs=NAV_HUBS) -> set:
    """把短名解析成**精确** FQDN。

    坑：用 `n.split('.')[0] in hubs` 会把 `en.physics`、`en.cs` 等 30 个单位英文站
    当成 `en` 枢纽一起删掉（实测多删到 36 个节点 / 678 条边）。必须精确匹配。
    """
    return {n for n in G.nodes() if n in {_fqdn(h) for h in hubs}}


def to_labels(partition, nodes) -> list:
    """社区划分 → 与 `nodes` 对齐的整数标签向量（未覆盖的节点标 -1）。"""
    lab = {n: i for i, c in enumerate(partition) for n in c}
    return [lab.get(n, -1) for n in nodes]


def partition_metrics(G: nx.Graph, commsA, commsB) -> dict:
    """两套划分的 NMI / ARI（sklearn 口径，在**两套都覆盖了**的节点交集上算）。

    sklearn 的 NMI/ARI 要求等长标签向量，所以必须先取交集再删标签，
    否则 -1（未覆盖）会被当成一个真实社区，把 NMI 拉低。
    """
    from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_rand_score as ARI
    nodes = list(G.nodes())
    la, lb = to_labels(commsA, nodes), to_labels(commsB, nodes)
    keep = [i for i, (x, y) in enumerate(zip(la, lb)) if x >= 0 and y >= 0]
    a = [la[i] for i in keep]
    b = [lb[i] for i in keep]
    re_a = {v: i for i, v in enumerate(sorted(set(a)))}
    re_b = {v: i for i, v in enumerate(sorted(set(b)))}
    a = [re_a[v] for v in a]
    b = [re_b[v] for v in b]
    return {
        "n_compared": len(keep),
        "n_total": len(nodes),
        "NMI": round(float(NMI(a, b)), 4),
        "ARI": round(float(ARI(a, b)), 4),
    }


def strip_hub_edges(G: nx.Graph, hubs=NAV_HUBS) -> nx.Graph:
    """删掉与导航枢纽相连的全部边（保留节点）——模板边敏感性的廉价代理。"""
    hn = hub_nodes(G, hubs)
    H = G.copy()
    H.remove_edges_from([(u, v) for u, v in H.edges() if u in hn or v in hn])
    return H


def strong_backbone(G: nx.Graph, min_weight: int = 10) -> nx.Graph:
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        if d.get("weight", 1) >= min_weight:
            H.add_edge(u, v, weight=d["weight"])
    return H


# ------------------------------------------------------------------ 零模型

def degree_preserving_null(G: nx.Graph, n_reps: int = 300, seed: int = SEED, stat=None):
    """度保持零模型：**每轮从原图独立复制再 swap**，不能在同一条 swap 轨迹上取点。

    extra.py:52-59 的写法是 `H=GC.copy()` 之后对同一个 H 连续 swap 300 次，
    把累积轨迹当成 300 个独立样本，标准差被低估。数值碰巧接近，但方法不成立。

    nswap=10|E|, max_tries=100·nswap 是实测能真正打散的配置（小值会几乎不交换）。
    Viger–Latapy 的边交换采样并非严格均匀，报告里要声明。
    """
    m = G.number_of_edges()
    vals = []
    for r in range(n_reps):
        H = G.copy()
        nx.double_edge_swap(H, nswap=10 * m, max_tries=100 * 10 * m, seed=seed + r)
        H.remove_edges_from(nx.selfloop_edges(H))
        vals.append(float(stat(H)))
    vals = np.asarray(vals)
    obs = float(stat(G))
    z = (obs - vals.mean()) / vals.std(ddof=1) if vals.std(ddof=1) > 0 else np.nan
    p_two = float((np.abs(vals - vals.mean()) >= abs(obs - vals.mean())).mean())
    return {
        "obs": obs,
        "null_mean": float(vals.mean()),
        "null_sd": float(vals.std(ddof=1)),
        "z": float(z),
        "p_two_sided": p_two,
        "n_reps": n_reps,
        "vals": vals,
        "note": "edge-switching is not an exactly uniform sampler (Viger & Latapy 2005)",
    }


# -------------------------------------------------------------- 中心性快捷

def centralities(Gu: nx.Graph, Gd: nx.DiGraph) -> pd.DataFrame:
    """一次算全，避免各模块口径漂移。

    PageRank 主口径 = **有向 + 加权**（`nx.pagerank(Gd, weight="weight")`），
    与作者 eda.py:60 一致。换成无权无向会让"PageRank 攻击弱于度攻击"反转，
    所以每张表都要写明变体。
    """
    rows = {}
    rows["deg_u"] = dict(Gu.degree())
    rows["bet_u"] = nx.betweenness_centrality(Gu)
    rows["pr_dw"] = nx.pagerank(Gd, weight="weight")
    rows["pr_uu"] = nx.pagerank(nx.Graph(Gu))  # 无权无向对照
    try:
        hubs, auth = nx.hits(Gd, max_iter=200, tol=1e-8)
    except nx.PowerIterationFailedConvergence:
        hubs, auth = nx.hits(Gd, max_iter=2000, tol=1e-6)
    rows["hits_hub"] = hubs
    rows["hits_auth"] = auth
    rows["kcore"] = dict(nx.core_number(Gu))
    df = pd.DataFrame(rows)
    df.index.name = "node"
    return df


# ---------------------------------------------------------------- 输出工具

def module_dir(module: str) -> str:
    d = os.path.join(OUT_ROOT, module)
    os.makedirs(d, exist_ok=True)
    return d


def save_csv(df: pd.DataFrame, module: str, name: str) -> str:
    p = os.path.join(module_dir(module), name)
    df.to_csv(p, index=False, encoding="utf-8-sig")
    return p


def save_json(obj, module: str, name: str) -> str:
    p = os.path.join(module_dir(module), name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    return p


def write_run_note(module: str, lines: list) -> str:
    p = os.path.join(module_dir(module), "RUN.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return p


def versions() -> dict:
    return {
        "python": sys.version.split()[0],
        "networkx": nx.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "SEED": SEED,
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
    }


# --------------------------------------------------------------- 一键装配

def build_all(use_clean: bool = True) -> dict:
    """返回全套标准对象。所有分析模块都从这里取，不要自己建图。"""
    nodes, links, dc = load_clean() if use_clean else (load_nodes(), load_links(), load_dead_categories())
    Gd = build_directed(links, nodes)
    Gu = build_undirected(links, nodes)
    Gb = build_binary(Gu)
    sets = node_sets(Gu, nodes)
    m = add_degrees(nodes, Gd, Gu)
    m = add_health(m, dc)
    ps = PairSet(sets["gcc"])
    return {
        "nodes": m,
        "links": links,
        "dead_cat": dc,
        "Gd": Gd,
        "Gu": Gu,
        "Gb": Gb,
        "sets": sets,
        "fail": failure_sets(m),
        "ps": ps,
        "versions": versions(),
    }


def load_clean():
    """优先读 pipeline/data 的清洗产物；没有则回退到原始 CSV（结果应完全一致）。

    只读回**原始列**；度/健康派生列由 build_all 从图重算，避免二次加工漂移。
    """
    cn = os.path.join(CLEAN_DIR, "clean_nodes.csv")
    if os.path.exists(cn):
        nodes = pd.read_csv(cn)
        links = pd.read_csv(os.path.join(CLEAN_DIR, "clean_edges.csv"))
        if "edge_order" in links.columns:
            links = links.sort_values("edge_order")
        dc = pd.read_csv(os.path.join(CLEAN_DIR, "dead_categories.csv"))
        derived = [c for c in nodes.columns if c in _DERIVED]
        nodes = nodes.drop(columns=derived)
        nodes["node"] = nodes["subdomain"]
        dc = dc.drop(columns=[c for c in dc.columns if c in _DERIVED])
        dc["node"] = dc["sub"].map(_fqdn)
        return nodes, links, dc
    return load_nodes(), load_links(), load_dead_categories()


_DERIVED = {
    "node", "in_degree", "out_degree", "degree_u", "strength_u", "in_strength",
    "out_strength", "is_dead", "is_partial", "is_healthy", "health_state",
    "dead_cat", "true_fail", "false_fail", "ambiguous_fail",
    "partial_has_auth_like", "partial_clean", "in_gcc", "is_dangling_ref",
}


# ---------------------------------------------------------------- 锚点自检
# 下面是本项目所有已知"事实锚"。任何模块跑完先调 assert_anchors()，
# 对不上就是建图/清洗漂移了，禁止继续往下写结论。

ANCHORS = {
    "n_attributed": 671,
    "n_in_graph": 676,
    "n_edges_undirected": 2498,
    "n_edges_directed": 2788,
    "n_dangling": 5,
    "n_isolated_in_gu": 0,
    "gcc_size": 674,
    "n_components": 2,
    "comp_sizes": [674, 2],
    "max_degree_u": 119,
    "argmax_degree_u": "teach.ustc.edu.cn",
    "mean_degree_u_gu676": 7.3905,
    "n_urls": 8115,
    "n_dead": 162,
    "n_partial": 149,
    "n_healthy": 360,
    "n_true_fail": 129,
    "n_false_fail": 32,
    "n_ambiguous": 1,
    "dead_with_outdegree_zero": 162,
    "edges_from_dead": 0,
    "edges_dead_to_dead": 0,
    "edges_alive_to_dead": 296,
    # Louvain 只有在"建图顺序 + seed"都固定时才复现作者的 25/0.6904/87；
    # 换成项目默认 seed 就是 23/0.6922/110。所以这两个锚都保留，且强制 seed=42。
    "louvain_seed42_comms": 25,
    "louvain_seed42_Q": 0.6904,
    "louvain_seed42_largest": 87,
    "avg_pathlen_gcc": 3.2702,
    "diameter_gcc": 7,
    "clustering_gu676": 0.2780,
    "degree_assortativity_gu676": -0.1674,
    "reciprocity_gd": 0.2081,
}


def check_anchors(b: dict | None = None) -> pd.DataFrame:
    """返回逐项 PASS/FAIL 表；不抛异常，方便写进 RUN.txt。

    注意 `clustering_gu676` / `degree_assortativity_gu676` 在 676 节点图上算
    （作者文档里的 0.278 / −0.167 就是这个口径）；换成 674 的 GCC 会变成
    0.2788 / −0.1679。两者都对，但报告必须写清是哪个节点集。
    """
    b = b or build_all()
    m, Gu, Gd, sets = b["nodes"], b["Gu"], b["Gd"], b["sets"]
    GCC = induced(Gu, sets["gcc"])
    lou42 = louvain(Gu, seed=42)
    dead = set(m.loc[m["is_dead"], "node"])
    alive = set(Gu.nodes()) - dead
    cur = {
        "n_attributed": len(m),
        "n_in_graph": Gu.number_of_nodes(),
        "n_edges_undirected": Gu.number_of_edges(),
        "n_edges_directed": Gd.number_of_edges(),
        "n_dangling": len(sets["dangling"]),
        "n_isolated_in_gu": sum(1 for _, d in Gu.degree() if d == 0),
        "gcc_size": len(sets["gcc"]),
        "n_components": sets["n_components"],
        "comp_sizes": sets["comp_sizes"],
        "max_degree_u": int(m["degree_u"].max()),
        "argmax_degree_u": m.loc[m["degree_u"].idxmax(), "node"],
        "mean_degree_u_gu676": round(2 * Gu.number_of_edges() / Gu.number_of_nodes(), 4),
        "n_urls": len(load_urls()),
        "n_dead": int(m["is_dead"].sum()),
        "n_partial": int(m["is_partial"].sum()),
        "n_healthy": int(m["is_healthy"].sum()),
        "n_true_fail": int(m["true_fail"].sum()),
        "n_false_fail": int(m["false_fail"].sum()),
        "n_ambiguous": int(m["ambiguous_fail"].sum()),
        "dead_with_outdegree_zero": sum(1 for n, d in Gd.out_degree(dead) if d == 0),
        "edges_from_dead": sum(1 for s, _ in Gd.edges() if s in dead),
        "edges_dead_to_dead": sum(1 for s, t in Gd.edges() if s in dead and t in dead),
        "edges_alive_to_dead": sum(1 for s, t in Gd.edges() if t in dead and s not in dead),
        "louvain_seed42_comms": lou42["n_comms_total"],
        "louvain_seed42_Q": round(lou42["Q"], 4),
        "louvain_seed42_largest": lou42["largest"],
        "avg_pathlen_gcc": round(nx.average_shortest_path_length(GCC), 4),
        "diameter_gcc": nx.diameter(GCC),
        "clustering_gu676": round(nx.average_clustering(Gu), 4),
        "degree_assortativity_gu676": round(float(nx.degree_assortativity_coefficient(Gu)), 4),
        "reciprocity_gd": round(float(nx.reciprocity(Gd)), 4),
    }
    rows = []
    for k, want in ANCHORS.items():
        got = cur.get(k)
        ok = abs(want - got) < 1e-3 if isinstance(want, float) and isinstance(got, float) else want == got
        rows.append({"anchor": k, "expected": want, "got": got, "ok": ok})
    return pd.DataFrame(rows)


def assert_anchors(b: dict | None = None) -> None:
    df = check_anchors(b)
    bad = df[~df["ok"]]
    if len(bad):
        raise AssertionError("锚点校验失败：\n" + bad.to_string(index=False))
