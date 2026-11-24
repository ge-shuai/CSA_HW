# -*- coding: utf-8 -*-
"""
E4 — EDGE-LEVEL ATTACKS: is the network "robust yet fragile" at the edge level too?
==================================================================================
Contract: pipeline/CONVENTIONS.md — clauses that bind this module:
  SS1   baseline graph = the 674-node / 2497-edge main component, taken as
        C.induced(B["Gu"], B["sets"]["gcc"]). Graphs are never rebuilt by hand.
  SS2.3 three efficiency口径 together; conclusions from E_fixed only, and only at
        EQUAL cost. Only C.eff_three / C.PairSet are used.
  SS4   random strategies: >=100 seeded replicates, mean +- sd.
  SS6   main口径 = STATIC (rank once on the baseline, delete the top-k). Betweenness
        additionally gets a DYNAMIC re-ranking variant as the robustness control.
        Deterministic strategies 1 run, random strategies >=100 runs with sd.
  SS7   no causal wording.
  SS9   run_e4.py / *.csv (UTF-8-sig) / result.md / RUN.txt / DISSENT.md.

Strategies, all on the SAME baseline graph
  edge side (q edges removed, n held at 674)
    random_edges          100 seeded replicates
    weight_ascending      fragile-first (observed link count = 1 goes first)
    weight_descending     strong-first
    ebc_binarized         nx.edge_betweenness_centrality(G0)  weight=None  <- MAIN口径
    ebc_weight_as_cost    nx.edge_betweenness_centrality(G0, weight="weight")
                          (networkx reads `weight` as a path COST, so a large observed
                          link count becomes a SHORT hop: this ranks strong edges high)
    ebc_random_walk       nx.edge_betweenness_centrality(G0, weight={e: 1/w})
                          (Codebook-style random-walk bias: strong edges carry more paths)
    bridge_first          every cut-edge first (max damage per edge), then random filler
  node side (k nodes removed; recomputed here, not imported from another module)
    random_nodes / node_degree / node_betweenness / node_kcore /
    node_pagerank_directed_weighted

Cycle bookkeeping at every step, so "does it become a tree before it fragments" is
answerable: cyclomatic number of the surviving largest component (m_G - n_G + 1),
feedback edge count of the whole remaining graph (m - n + c), number and share of
bridges, and the component count.

CAVEAT stated up front (see result.md): E_fixed scores a pair as 0 when an ENDPOINT was
deleted. An edge attack never deletes a node, so E_fixed carries a structural head
start for edge attacks at equal "number of objects removed". Equal-cost comparisons in
stage 06 therefore also use GCC size and the number of connected components, which are
口径-neutral.

Run:  PYTHONHASHSEED=0 python out/E4_edge_attacks/run_e4.py
"""
from __future__ import annotations

import os
import random
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

import csa_common as C

SEED = 20261113            # CONVENTIONS SS8
N_RANDOM = 100             # CONVENTIONS SS4/SS6
N_JOBS = 8
W_STRONG = 10              # CONVENTIONS SS3 reference skeleton
MODULE = "E4_edge_attacks"
OUT = SCRIPT_DIR

FRACS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30,
         0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
EDGE_STRATS = ["random_edges", "weight_ascending", "weight_descending", "ebc_binarized",
               "ebc_weight_as_cost", "ebc_random_walk", "bridge_first"]
NODE_STRATS = ["random_nodes", "node_degree", "node_betweenness", "node_kcore",
               "node_pagerank_directed_weighted"]
STRICT_DYNAMIC_UPTO = 250  # exact per-edge Girvan-Newman for the first 250 edges (~10%)

_W = {}


# ---------------------------------------------------------------- worker side
def _rank_desc(items):
    """Descending-value order, ties broken on the (sorted) edge tuple.
    Keys are normalised to (min,max) so they can be joined against W / bridges."""
    return [tuple(sorted(e)) for e, _ in sorted(items, key=lambda kv: (-kv[1], tuple(sorted(kv[0]))))]


def _rwl_graph(G):
    """Copy with edge attribute 'rwl' = 1 / observed link count (Codebook random-walk
    convention: a strong link is a SHORT hop).  networkx cannot take a dict for
    `weight=`, hence the explicit attribute."""
    H = G.copy()
    for u, v, d in H.edges(data=True):
        H[u][v]["rwl"] = 1.0 / max(1, d.get("weight", 1))
    return H


def _init_worker(strong_only=False):
    B = C.build_all()
    G0 = C.induced(B["Gu"], B["sets"]["gcc"])
    if strong_only:
        S = C.strong_backbone(G0, W_STRONG)
        G0 = C.induced(S, [x for x in S if S.degree(x) > 0])
    _W["G0"] = G0
    _W["ps"] = B["ps"]
    E = sorted(tuple(sorted((u, v))) for u, v in G0.edges())
    W = {tuple(sorted((u, v))): d.get("weight", 1) for u, v, d in G0.edges(data=True)}
    _W["E"], _W["w"] = E, W
    _W["npool"] = sorted(G0.nodes())
    # random-walk variant: edge length = 1 / observed link count (Codebook convention)
    Grw = _rwl_graph(G0)
    _W["rank"] = {
        "weight_ascending": sorted(E, key=lambda e: (W[e], e)),
        "weight_descending": sorted(E, key=lambda e: (-W[e], e)),
        "ebc_binarized": _rank_desc(nx.edge_betweenness_centrality(G0).items()),
        "ebc_weight_as_cost": _rank_desc(
            nx.edge_betweenness_centrality(G0, weight="weight").items()),
        "ebc_random_walk": _rank_desc(nx.edge_betweenness_centrality(Grw, weight="rwl").items()),
    }
    br = set()
    for comp in nx.connected_components(G0):
        br |= {tuple(sorted((u, v))) for u, v in nx.bridges(G0.subgraph(comp))}
    _W["bridges"] = br
    rgf = random.Random(SEED + 12345)
    _W["rank"]["bridge_first"] = (sorted(br) +
                                  rgf.sample([e for e in E if e not in br], len(E) - len(br)))
    for _k, _v in _W["rank"].items():
        assert len(set(_v)) == len(_v), "duplicate edge in ranking " + _k
        assert set(_v) <= set(E), "ranking holds a non-edge: " + _k
    if not strong_only:
        deg = dict(G0.degree())
        bt = nx.betweenness_centrality(G0)
        kc = nx.core_number(G0)
        pr = nx.pagerank(B["Gd"], weight="weight")
        _W["node_rank"] = {
            "degree": sorted(G0.nodes(), key=lambda x: (-deg[x], x)),
            "betweenness": sorted(G0.nodes(), key=lambda x: (-bt[x], x)),
            "kcore": sorted(G0.nodes(), key=lambda x: (-kc[x], -deg[x], x)),
            "pagerank_directed_weighted": sorted(G0.nodes(), key=lambda x: (-pr.get(x, 0.0), x)),
        }


def _measure(G, ps):
    e = C.eff_three(G, ps)
    n, m = G.number_of_nodes(), G.number_of_edges()
    sizes = sorted((len(c) for c in nx.connected_components(G)), reverse=True) if n else []
    c = len(sizes)
    if n:
        cc = max(nx.connected_components(G), key=len)
        S = G.subgraph(cc)
        n_g, m_g = S.number_of_nodes(), S.number_of_edges()
        nbr = sum(sum(1 for _ in nx.bridges(G.subgraph(x))) for x in nx.connected_components(G))
    else:
        n_g = m_g = nbr = 0
    return {"E_full": e["E_full"], "E_gcc": e["E_gcc"], "E_fixed": e["E_fixed"],
            "n_alive": n, "m_alive": m, "n_components": c,
            "gcc": n_g, "edges_in_gcc": m_g,
            "largest_share": n_g / n if n else 0.0,
            "cyclomatic_gcc": m_g - n_g + (1 if n_g else 0),
            "feedback_all": m - n + c,
            "n_bridges": nbr,
            "bridge_share": nbr / m if m else 0.0,
            "isolated": sum(1 for _x, _d in G.degree() if _d == 0),
            "second_cc": sizes[1] if len(sizes) > 1 else 0}


def _run(task):
    kind, q, seed = task
    G0, ps = _W["G0"], _W["ps"]
    H = G0.copy()
    if kind == "random_edges":
        rg = random.Random(seed)
        cut = rg.sample(_W["E"], q)
    elif kind == "random_nodes":
        rg = random.Random(seed)
        cut = None
        H.remove_nodes_from(rg.sample(_W["npool"], q))
    elif kind.startswith("node_"):
        cut = None
        H.remove_nodes_from(_W["node_rank"][kind[len("node_"):]][:q])
    else:
        cut = _W["rank"][kind][:q]
    if cut is not None:
        H.remove_edges_from(cut)
    r = _measure(H, ps)
    r.update({"strategy": kind, "q_removed": q, "seed": seed})
    return r


def _pool(tasks, init_kwargs=None):
    from multiprocessing import Pool
    kw = init_kwargs or {}
    with Pool(processes=N_JOBS, initializer=_init_worker, initargs=(kw.get("strong_only", False),)) as p:
        return p.map(_run, tasks, chunksize=2)


def _tasks(edge_universe, node_universe=0, include_nodes=True):
    out = []
    for s in EDGE_STRATS:
        for fr in FRACS:
            q = max(1, int(round(edge_universe * fr)))
            if s == "random_edges":
                out += [(s, q, SEED + 700000 + rep) for rep in range(N_RANDOM)]
            else:
                out.append((s, q, SEED))
    if not include_nodes:
        return out
    for s in NODE_STRATS:
        for fr in FRACS:
            q = max(1, int(round(node_universe * fr)))
            if s == "random_nodes":
                out += [(s, q, SEED + 800000 + rep) for rep in range(N_RANDOM)]
            else:
                out.append((s, q, SEED))
    return out


AGG = {
    "n_removed": ("q_removed", "first"), "reps": ("seed", "count"),
    "E_full": ("E_full", "mean"), "E_full_sd": ("E_full", "std"),
    "E_gcc": ("E_gcc", "mean"), "E_gcc_sd": ("E_gcc", "std"),
    "E_fixed": ("E_fixed", "mean"), "E_fixed_sd": ("E_fixed", "std"),
    "gcc": ("gcc", "mean"), "gcc_sd": ("gcc", "std"),
    "n_components": ("n_components", "mean"), "largest_share": ("largest_share", "mean"),
    "edges_in_gcc": ("edges_in_gcc", "mean"), "m_alive": ("m_alive", "mean"),
    "cyclomatic_gcc": ("cyclomatic_gcc", "mean"), "feedback_all": ("feedback_all", "mean"),
    "n_bridges": ("n_bridges", "mean"), "bridge_share": ("bridge_share", "mean"),
    "second_cc": ("second_cc", "mean"),
}
SD_COLS = ["E_full_sd", "E_gcc_sd", "E_fixed_sd", "gcc_sd"]


def _summarise(raw):
    """Mean/sd per (strategy, q) without DataFrame.apply (pandas-3 safe)."""
    out = raw.groupby(["strategy", "q_removed"], sort=True).agg(**AGG).reset_index()
    out = out.drop(columns=["q_removed"])
    out[SD_COLS] = out[SD_COLS].fillna(0.0)
    return out.sort_values(["strategy", "n_removed"]).reset_index(drop=True)


def main():
    wall = []

    def tic(lbl):
        wall.append({"stage": lbl, "t0": time.time()})

    def toc():
        now = time.time()
        for r in wall:
            if "wall_s" not in r:
                r["wall_s"] = round(now - r["t0"], 3)

    t_all = time.time()
    print("E4 - edge-level attacks on the USTC subdomain network")
    print("=" * 104)

    # ------------------------------------------------------- 00 baseline census
    tic("00 build + baseline edge census")
    B = C.build_all()
    C.assert_anchors(B)                                   # CONVENTIONS SS1
    bad = C.check_anchors(B)
    print("  anchors: " + str(len(bad)) + " checked, " +
          str(int((~bad["ok"].astype(bool)).sum())) + " failed")
    assert int((~bad["ok"].astype(bool)).sum()) == 0, "SS1 anchor mismatch"
    Gu, Gd = B["Gu"], B["Gd"]
    G0 = C.induced(Gu, B["sets"]["gcc"])
    ps = B["ps"]
    N0, M0 = G0.number_of_nodes(), G0.number_of_edges()
    assert (N0, M0) == (674, 2497), "baseline must be the 674/2497 main component"
    W = {tuple(sorted((u, v))): d.get("weight", 1) for u, v, d in G0.edges(data=True)}
    Wsum = sum(W.values())
    _init_worker()
    base = _measure(G0, ps)
    nbr0 = len(_W["bridges"])
    print("  baseline n=%d m=%d  total observed link count W=%d  max edge weight=%d (%.2f%% of W)"
          % (N0, M0, Wsum, max(W.values()), 100 * max(W.values()) / Wsum))
    print("  baseline E_full=%.6f E_gcc=%.6f E_fixed=%.6f" % (base["E_full"], base["E_gcc"], base["E_fixed"]))
    print("  baseline components=%d  bridges=%d (%.2f%% of m)  cyclomatic m-n+1=%d"
          % (base["n_components"], nbr0, 100 * nbr0 / M0, M0 - N0 + 1))
    assert abs(base["E_full"] - 0.3315) < 5e-3, "baseline efficiency drifted from the SS2.3 table"
    vc = pd.Series(list(W.values())).value_counts().sort_index()
    wd = pd.DataFrame({"weight": vc.index, "n_edges": vc.values})
    wd["edge_share_pct"] = (100 * wd.n_edges / M0).round(3)
    wd["weight_share_pct"] = (100 * wd.weight * wd.n_edges / Wsum).round(3)
    wd["cum_edges_ge"] = wd.n_edges[::-1].cumsum()[::-1].values
    C.save_csv(wd, MODULE, "baseline_edge_weight_distribution.csv")
    print(wd.head(10).to_string(index=False))
    print("  edges with w>=10: %d (%.2f%% of m) carrying %.2f%% of W"
          % (sum(1 for v in W.values() if v >= 10), 100 * sum(1 for v in W.values() if v >= 10) / M0,
             100 * sum(v for v in W.values() if v >= 10) / Wsum))
    toc()

    # ------------------------------ 01 the three edge-betweenness 口径 disagree?
    tic("01 edge-betweenness variant agreement")
    ebc_bin = {tuple(sorted(k)): v for k, v in nx.edge_betweenness_centrality(G0).items()}
    ebc_cost = {tuple(sorted(k)): v for k, v in
                nx.edge_betweenness_centrality(G0, weight="weight").items()}
    Grw = _rwl_graph(G0)
    ebc_rw = {tuple(sorted(k)): v for k, v in
              nx.edge_betweenness_centrality(Grw, weight="rwl").items()}
    assert len(ebc_bin) == len(ebc_cost) == len(ebc_rw) == M0, "EBC did not cover every edge"
    ser = pd.DataFrame({"ebc_binarized": pd.Series(ebc_bin), "ebc_weight_as_cost": pd.Series(ebc_cost),
                        "ebc_random_walk": pd.Series(ebc_rw), "weight": pd.Series(W)}).reindex(_W["E"])
    ser["is_bridge"] = [int(e in _W["bridges"]) for e in ser.index]
    agr = []
    for a, b in [("ebc_binarized", "ebc_weight_as_cost"), ("ebc_binarized", "ebc_random_walk"),
                 ("ebc_weight_as_cost", "ebc_random_walk"), ("ebc_binarized", "weight"),
                 ("ebc_random_walk", "weight")]:
        top = 125
        ra = _rank_desc(ser[a].items())[:top]
        rb = _rank_desc(ser[b].items())[:top]
        agr.append({"variant_A": a, "variant_B": b,
                    "spearman_rho": round(float(spearmanr(ser[a], ser[b]).statistic), 4),
                    "kendall_tau": round(float(kendalltau(ser[a], ser[b]).statistic), 4),
                    "top5pct_overlap": round(len(set(ra) & set(rb)) / top, 4)})
    agdf = pd.DataFrame(agr)
    C.save_csv(agdf, MODULE, "edge_metric_agreement.csv")
    print(agdf.to_string(index=False))
    top = pd.DataFrame({
        "rank_binarized": range(1, 21),
        "edge": _W["rank"]["ebc_binarized"][:20],
        "weight": [W[e] for e in _W["rank"]["ebc_binarized"][:20]],
        "ebc_binarized": [ebc_bin[e] for e in _W["rank"]["ebc_binarized"][:20]],
        "rank_if_weight_as_cost": [_W["rank"]["ebc_weight_as_cost"].index(e) + 1
                                   for e in _W["rank"]["ebc_binarized"][:20]],
        "rank_if_random_walk": [_W["rank"]["ebc_random_walk"].index(e) + 1
                                for e in _W["rank"]["ebc_binarized"][:20]],
        "is_bridge": [int(e in _W["bridges"]) for e in _W["rank"]["ebc_binarized"][:20]],
    })
    C.save_csv(top, MODULE, "top_edges_by_binarized_ebc.csv")
    print(top.to_string(index=False))
    toc()

    # --------------------------------------------- 02 static attack curves
    tic("02 static edge+node curves (" + str(N_RANDOM) + " reps for random)")
    tasks = _tasks(M0, N0)
    print("  " + str(len(tasks)) + " removal evaluations")
    t0 = time.time()
    rows = _pool(tasks)
    raw = pd.DataFrame(rows)
    print("  done in " + str(round(time.time() - t0, 1)) + "s")
    C.save_csv(raw, MODULE, "attack_replicates.csv")
    curve = _summarise(raw)
    curve["is_edge_attack"] = curve.strategy.isin(EDGE_STRATS)
    curve["universe"] = np.where(curve.is_edge_attack, "edges m=2497", "nodes n=674")
    curve["removed_pct_of_universe"] = np.where(
        curve.is_edge_attack, 100 * curve.n_removed / M0, 100 * curve.n_removed / N0)
    for c, ref in [("E_fixed", base["E_fixed"]), ("E_full", base["E_full"]),
                   ("E_gcc", base["E_gcc"]), ("gcc", float(N0)), ("edges_in_gcc", float(M0))]:
        curve[c + "_ratio"] = curve[c] / ref
    curve = curve.sort_values(["strategy", "n_removed"]).reset_index(drop=True)
    for c in curve.columns:
        if curve[c].dtype.kind == "f":
            curve[c] = curve[c].round(6)
    C.save_csv(curve, MODULE, "edge_attack_curves.csv")
    piv = curve.pivot_table(index="removed_pct_of_universe", columns="strategy", values="E_fixed")
    print(piv.round(4).to_string())
    toc()

    # ------------------------------------- 03 tree-before-fragmentation check
    tic("03 breakdown checkpoints")
    ck = []
    for strat in EDGE_STRATS + NODE_STRATS:
        sub = curve[curve.strategy == strat].sort_values("n_removed")
        if not len(sub):
            continue
        row = {"strategy": strat, "attack_on": "edges" if strat in EDGE_STRATS else "nodes"}
        for lbl, thr in [("p10", 0.90), ("p50", 0.50), ("p90", 0.10)]:
            hit = sub[sub.gcc_ratio < thr]
            row[lbl + "_pct_removed"] = float(hit.removed_pct_of_universe.iloc[0]) if len(hit) else np.nan
            row[lbl + "_n_removed"] = float(hit.n_removed.iloc[0]) if len(hit) else np.nan
        for met in ("E_fixed", "gcc"):
            hit = sub[sub[met] <= 0.5 * (base["E_fixed"] if met == "E_fixed" else N0)]
            row["half_" + met + "_pct"] = float(hit.removed_pct_of_universe.iloc[0]) if len(hit) else np.nan
        conn = sub[sub.n_components <= 1]
        if len(conn):
            row["last_connected_pct"] = float(conn.removed_pct_of_universe.iloc[-1])
            row["cyclomatic_while_still_connected"] = float(conn.cyclomatic_gcc.iloc[-1])
            row["edges_in_gcc_when_first_disconnected"] = float(
                sub[sub.n_components > 1].edges_in_gcc.iloc[0]) if len(sub[sub.n_components > 1]) else np.nan
        hit50 = sub[sub.gcc_ratio < 0.50]
        row["cyclomatic_at_gcc50"] = float(hit50.cyclomatic_gcc.iloc[0]) if len(hit50) else np.nan
        row["is_tree_at_gcc50"] = bool(len(hit50) and hit50.cyclomatic_gcc.iloc[0] <= 0)
        row["E_fixed_at_gcc50"] = float(hit50.E_fixed.iloc[0]) if len(hit50) else np.nan
        row["n_components_at_last_checkpoint"] = float(sub.n_components.iloc[-1])
        row["E_fixed_at_last_checkpoint"] = float(sub.E_fixed.iloc[-1])
        ck.append(row)
    ckd = pd.DataFrame(ck)
    for c in ckd.columns:
        if ckd[c].dtype.kind == "f":
            ckd[c] = ckd[c].round(4)
    C.save_csv(ckd, MODULE, "breakdown_checkpoints.csv")
    print(ckd.to_string(index=False))
    toc()

    # -------------------------------- 04 strict per-edge dynamic Girvan-Newman
    tic("04 dynamic EBC (strict to " + str(STRICT_DYNAMIC_UPTO) + " edges, then block)")
    dyn = []
    H = G0.copy()
    removed = 0
    targets = sorted({max(1, int(round(M0 * f))) for f in FRACS})
    ti = 0
    tstart = time.time()
    for tgt in targets:
        while removed < tgt:
            if removed <= STRICT_DYNAMIC_UPTO:           # exact: re-rank after EVERY edge
                eb = nx.edge_betweenness_centrality(H)
                if not eb:
                    break
                u, v = max(eb.items(), key=lambda kv: kv[1])[0]
                H.remove_edge(u, v)
                removed += 1
            else:                                        # block re-ranking (cost cap)
                eb = nx.edge_betweenness_centrality(H)
                if not eb:
                    break
                order = _rank_desc(eb.items())
                take = min(tgt - removed, max(1, len(order)))
                H.remove_edges_from(order[:take])
                removed += take
        m = _measure(H, ps)
        m.update({"strategy": "dynamic_ebc_binarized", "q_removed": removed, "seed": SEED,
                  "mode": "strict_per_edge" if tgt <= STRICT_DYNAMIC_UPTO else "block_rerank"})
        dyn.append(m)
        print("    q=%-5d (%.1f%%) E_fixed=%.6f gcc=%-4d comps=%-4d cyc=%-6d %.0fs"
              % (removed, 100 * removed / M0, m["E_fixed"], m["gcc"], m["n_components"],
                 m["cyclomatic_gcc"], time.time() - tstart))
    dyndf = pd.DataFrame(dyn)
    for c in dyndf.columns:
        if dyndf[c].dtype.kind == "f":
            dyndf[c] = dyndf[c].round(6)
    C.save_csv(dyndf, MODULE, "dynamic_ebc_curve.csv")
    # FIX 2026-11-23 (lead): _summarise() 在 L238 丢弃了 q_removed（改由 n_removed 承担），
    # 这里却仍按 q_removed 取列 → KeyError: "['q_removed'] not in index"，整个阶段 03 挂掉。
    # 取 n_removed 再改名，保持已写出的 edge_attack_curves.csv 列结构不变（避免下游 schema 漂移）。
    stat = (curve[curve.strategy == "ebc_binarized"]
            [["n_removed", "E_fixed", "gcc", "n_components"]]
            .rename(columns={"n_removed": "q_removed"}))
    cmpd = dyndf.merge(stat, on="q_removed", how="inner", suffixes=("_dynamic", "_static"))
    cmpd["E_fixed_dynamic_minus_static"] = cmpd.E_fixed_dynamic - cmpd.E_fixed_static
    C.save_csv(cmpd, MODULE, "dynamic_vs_static_ebc.csv")
    print(cmpd[["q_removed", "E_fixed_dynamic", "E_fixed_static", "E_fixed_dynamic_minus_static",
                "gcc_dynamic", "gcc_static"]].to_string(index=False))
    toc()

    # ------------------------------- 05 equal cost: edge attack vs node attack
    tic("05 equal-cost comparison")
    def nearest(strat, pct, col="removed_pct_of_universe"):
        sub = curve[curve.strategy == strat]
        if not len(sub):
            return None
        return sub.iloc[(sub[col] - pct).abs().argsort().iloc[0]]

    def node_equiv(metric, value, strat):
        """smallest node-removal % of strategy `strat` whose `metric` is at least as bad"""
        sub = curve[curve.strategy == strat].sort_values("n_removed")
        ref = base["E_fixed"] if metric == "E_fixed" else float(N0)
        hit = sub[sub[metric] <= value]
        return float(hit.removed_pct_of_universe.iloc[0]) if len(hit) else np.nan

    eq = []
    for pct in (1, 5, 10, 20, 30, 50, 70):
        for estrat in ("random_edges", "weight_descending", "weight_ascending",
                       "ebc_binarized", "ebc_weight_as_cost", "ebc_random_walk", "bridge_first"):
            r = nearest(estrat, pct)
            if r is None:
                continue
            eq.append({
                "edge_strategy": estrat, "edge_pct_of_m": round(float(r.removed_pct_of_universe), 3),
                "edges_removed": int(r.n_removed),
                "E_fixed": round(float(r.E_fixed), 6), "gcc": round(float(r.gcc), 1),
                "n_components": round(float(r.n_components), 1),
                "nodes_pct_for_same_Efixed_degree": node_equiv("E_fixed", r.E_fixed, "node_degree"),
                "nodes_pct_for_same_Efixed_betweenness": node_equiv("E_fixed", r.E_fixed, "node_betweenness"),
                "nodes_pct_for_same_Efixed_pagerank": node_equiv("E_fixed", r.E_fixed, "node_pagerank_directed_weighted"),
                "nodes_pct_for_same_Efixed_random": node_equiv("E_fixed", r.E_fixed, "random_nodes"),
                "nodes_pct_for_same_gcc_degree": node_equiv("gcc", r.gcc, "node_degree"),
                "nodes_pct_for_same_gcc_betweenness": node_equiv("gcc", r.gcc, "node_betweenness"),
                "nodes_pct_for_same_gcc_random": node_equiv("gcc", r.gcc, "random_nodes"),
            })
    eqdf = pd.DataFrame(eq)
    eqdf["edges_cheaper_than_degree_on_Efixed"] = (
        eqdf.nodes_pct_for_same_Efixed_degree > eqdf.edge_pct_of_m)
    eqdf["object_ratio_m_over_n"] = M0 / N0
    C.save_csv(eqdf, MODULE, "equal_cost_edge_vs_node.csv")
    print(eqdf[eqdf.edge_pct_of_m.isin([1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 70.0])].to_string(index=False))
    toc()

    # --------------------------- 06 strong-edge skeleton (is it just big weights?)
    tic("06 w>=%d skeleton replication" % W_STRONG)
    sb_all = C.strong_backbone(G0, W_STRONG)
    sb = C.induced(sb_all, [x for x in sb_all if sb_all.degree(x) > 0])
    Nsb, Msb = sb.number_of_nodes(), sb.number_of_edges()
    sb_base = _measure(sb, B["ps"])
    sizes = sorted((len(c) for c in nx.connected_components(sb)), reverse=True)
    print("  skeleton w>=%d: non-isolated nodes %d (isolated %d), edges %d, components %d, top sizes %s"
          % (W_STRONG, Nsb, N0 - Nsb, Msb, len(sizes), sizes[:6]))
    print("  skeleton E_full=%.6f E_gcc=%.6f E_fixed=%.6f (denominator still the 674-node baseline)"
          % (sb_base["E_full"], sb_base["E_gcc"], sb_base["E_fixed"]))
    ts = _tasks(Msb, include_nodes=False)          # edge attacks only on the skeleton
    rows_sb = _pool(ts, {"strong_only": True})
    raw_sb = pd.DataFrame(rows_sb)
    C.save_csv(raw_sb, MODULE, "skeleton_attack_replicates.csv")
    csb = _summarise(raw_sb)
    csb["is_edge_attack"] = csb.strategy.isin(EDGE_STRATS)
    csb["removed_pct_of_universe"] = 100 * csb.n_removed / Msb
    csb["E_fixed_ratio_vs_skeleton"] = csb.E_fixed / sb_base["E_fixed"]
    csb["gcc_ratio_vs_skeleton_nodes"] = csb.gcc / Nsb
    for c in csb.columns:
        if csb[c].dtype.kind == "f":
            csb[c] = csb[c].round(6)
    C.save_csv(csb, MODULE, "skeleton_attack_curves.csv")
    print(csb.pivot_table(index="removed_pct_of_universe", columns="strategy",
                          values="E_fixed_ratio_vs_skeleton").round(4).to_string())
    toc()

    # ---------------------------------------- 07 navigation-hub edge sensitivity
    tic("07 hub-edge (template) sensitivity")
    hubs = C.hub_nodes(G0)
    hn = [e for e in G0.edges() if set(e) & hubs]
    H1 = G0.copy()
    H1.remove_edges_from(hn)
    r1 = _measure(H1, ps)
    print("  removing all %d edges (%.2f%% of m) that touch the 6 navigation hubs"
          % (len(hn), 100 * len(hn) / M0))
    print("  before E_full=%.6f E_gcc=%.6f E_fixed=%.6f comps=%d gcc=%d bridges=%d"
          % (base["E_full"], base["E_gcc"], base["E_fixed"], base["n_components"], base["gcc"], base["n_bridges"]))
    print("  after  E_full=%.6f E_gcc=%.6f E_fixed=%.6f comps=%d gcc=%d bridges=%d"
          % (r1["E_full"], r1["E_gcc"], r1["E_fixed"], r1["n_components"], r1["gcc"], r1["n_bridges"]))
    hubdf = pd.DataFrame([{"variant": "baseline", "n_edges_removed": 0, **base},
                          {"variant": "strip_all_hub_edges", "n_edges_removed": len(hn), **r1}])
    hubdf["pct_of_m"] = (100 * hubdf.n_edges_removed / M0).round(3)
    C.save_csv(hubdf, MODULE, "hub_edge_sensitivity.csv")
    toc()

    el = time.time() - t_all
    print("")
    print("=" * 104)
    print("ELAPSED %.2fs  seed=%d  reps(random)=%d" % (el, SEED, N_RANDOM))
    for r in wall:
        print("  " + r["stage"].ljust(58) + str(r["wall_s"]).rjust(9) + "s")
    lines = ["# RUN.txt - written by run_e4.py",
             "command: PYTHONHASHSEED=0 python out/E4_edge_attacks/run_e4.py",
             "seed: " + str(SEED) + " (every random draw gets SEED + a stage offset)",
             "random replicates per checkpoint: " + str(N_RANDOM),
             "baseline graph: 674 nodes / 2497 edges = C.induced(Gu, sets['gcc'])",
             "E_fixed denominator: %d ordered pairs (C.PairSet on the baseline main component)" % int(ps.denom),
             "main edge-betweenness口径: nx.edge_betweenness_centrality(G0) with weight=None"
             " i.e. the skeleton is BINARIZED before ranking",
             "elapsed_s: " + str(round(el, 2)),
             "versions: " + str(C.versions())]
    for r in wall:
        lines.append("stage | " + r["stage"] + " | wall_s=" + str(r["wall_s"]))
    with open(os.path.join(OUT, "RUN.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    C.save_csv(pd.DataFrame(wall), MODULE, "stage_walltimes.csv")
    print("RUN.txt written")


if __name__ == "__main__":
    main()
