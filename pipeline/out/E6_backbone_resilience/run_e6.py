# -*- coding: utf-8 -*-
"""
E6 — DISPARITY-FILTER BACKBONE AND ITS RESILIENCE
=================================================
Contract: pipeline/CONVENTIONS.md — clauses that bind this module:
  SS1   baseline = 674-node / 2497-edge main component, C.induced(B["Gu"], B["sets"]["gcc"]).
        Weight = "observed link count", never traffic / importance.
  SS2.3 three efficiency口径 together; conclusions from E_fixed only, at EQUAL k.
        Full-graph runs use B["ps"] (denominator 453,602 = 674*673 ordered pairs);
        backbone runs use a C.PairSet built on the BACKBONE NODE SET, so "equal k"
        always means equal k inside the graph actually under attack. Both denominators
        are printed in every table header line and in RUN.txt.
  SS3   component counts given BOTH ways: including isolated nodes, and non-singleton
        only.  Main tables use the non-singleton value.
  SS4   random strategies: >=100 seeded replicates, mean +- sd.
  SS5   PageRank main口径 = directed + weighted: nx.pagerank(Gd, weight="weight"),
        here restricted to the node set of the graph under attack (C.induced).
  SS6   static ranking (rank once, delete top-k) is the main口径; betweenness also gets
        a dynamic re-ranking variant (stage 06).
  SS7   no causal wording.
  SS9   run_e6.py / *.csv (UTF-8-sig) / result.md / RUN.txt / DISSENT.md.

THE FILTER (Serrano, Boguñá, Vespignani & Corbetta, PNAS 106:6483, 2009)
networkx 3.6.1 has NO disparity_filter, so it is implemented here.  For node i with
strength s_i = sum_j w_ij and k_i neighbours, the null hypothesis is that i's strength
is split proportionally to a_ij = w_ij / s_i.  Two closed forms of the resulting upper
tail are in circulation and they are NOT equivalent, so both are computed and reported:

  A "allocation"  p_ij = 1 - BinomCDF(w_ij - 1;  s_i - w_ij,  a_ij)
      (the s_i - w_ij strength units NOT on the focal link are allocated with prob a_ij)
  B "paper count" p_ij = 1 - BinomCDF(w_ij - 1;  k_i - 2,     a_ij)
      (the printed form of eq.[4], which counts LINKS, valid only for integer weights)

Main口径 = A, because our weights are strength units and A is the form the allocation
null implies.  B is kept as a sensitivity.  Both are compared with an explicit
MONTE-CARLO realisation of the null: draw n_ij* ~ Multinomial(s_i; a_i.) at both
endpoints and take p = P[max_j-cell >= w_ij] (stage 03).

Per-node multiple-comparison corrections compared (an edge is kept only if BOTH
endpoints pass their own corrected level):
  sidak_power : alpha'_i = 1 - (1 - alpha)^(1/(k_i - 1))   <- the paper's correction
  bonferroni  : alpha'_i = alpha / (k_i - 1)
  none        : alpha'_i = alpha                           <- SS-warning case

Run:  PYTHONHASHSEED=0 python out/E6_backbone_resilience/run_e6.py
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
from scipy.stats import binom, spearmanr

import csa_common as C

SEED = 20261113
N_RANDOM = 100
N_JOBS = 8
W_STRONG = 10
ALPHAS = (0.05, 0.01)
FORMS = ("allocation_s_minus_w", "paper_count_k_minus_2", "paper_closed_form")
FORM_IDX = {f: i for i, f in enumerate(FORMS)}
CORRS = ("sidak_power", "bonferroni", "none")
MAIN_FORM, MAIN_ALPHA, MAIN_CORR = "paper_closed_form", 0.05, "sidak_power"
N_BOOT_EDGES = 250
BOOT_DRAWS = 20000
MODULE = "E6_backbone_resilience"
OUT = SCRIPT_DIR
FRACS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
         0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

_W = {}


# --------------------------------------------------------------- the filter
def directional_p(G):
    """{(i,j): (pA, pB, pC)} for every ordered adjacency pair with k_i >= 2.

    pC is the closed form Serrano et al. (2009) print as their eq.(4): with
    p_ij = w_ij / s_i,  1 - (k_i-1) * ∫_0^{p_ij} (1-x)^{k_i-2} dx  =  (1 - p_ij)^{k_i-1}.
    That is the canonical implementation (and what the widely used snippets compute);
    A and B are kept because they were used earlier in this pipeline and the census
    difference is itself a result.
    """
    out = {}
    for i, nbrs in G.adj.items():
        k_i = len(nbrs)
        s_i = sum(d.get("weight", 1) for d in nbrs.values())
        for j, d in nbrs.items():
            w = int(d.get("weight", 1))
            a = w / s_i
            if k_i < 2:
                out[(i, j)] = (1.0, 1.0, 1.0)
                continue
            nA = int(s_i - w)
            nB = k_i - 2
            pA = float(binom.sf(w - 1, nA, a)) if nA > 0 else 0.0
            # literal reading of the printed form: nB = 0 trials => X == 0 => sf = 0 for w>=1,
            # i.e. EVERY link of a degree-2 node is declared significant.  Degenerate, kept
            # as printed and flagged in result.md / DISSENT.md.
            pB = float(binom.sf(w - 1, nB, a))
            pC = float((1.0 - a) ** (k_i - 1))
            out[(i, j)] = (pA, pB, pC)
    return out


def alpha_of(alpha, k_i, corr):
    if corr == "none" or k_i <= 1:
        return alpha
    if corr == "bonferroni":
        return min(1.0, alpha / (k_i - 1))
    return 1.0 - (1.0 - alpha) ** (1.0 / (k_i - 1))


def make_backbone(G, dp, alpha, form, corr):
    """Keep (i,j) iff p_ij <= alpha'_i AND p_ji <= alpha'_j."""
    idx = FORM_IDX[form]
    deg = dict(G.degree())
    S = nx.Graph()
    S.add_nodes_from(G.nodes())
    kept = 0
    for (i, j), ps in dp.items():
        if i < j:                      # look at each undirected edge once
            if ps[idx] <= alpha_of(alpha, deg[i], corr) and \
               dp[(j, i)][idx] <= alpha_of(alpha, deg[j], corr):
                S.add_edge(i, j, **G[i][j])
                kept += 1
    assert kept == S.number_of_edges()
    return S


def census(S, G):
    iso = [n for n in S.nodes() if S.degree(n) == 0]
    non = C.induced(S, [n for n in S.nodes() if S.degree(n) > 0])
    sizes = sorted((len(c) for c in nx.connected_components(non)), reverse=True) \
        if non.number_of_nodes() else []
    w_k = sum(d.get("weight", 1) for _, _, d in S.edges(data=True))
    w_t = sum(d.get("weight", 1) for _, _, d in G.edges(data=True))
    return {"edges": S.number_of_edges(),
            "edge_share_pct": 100.0 * S.number_of_edges() / G.number_of_edges(),
            "weight_kept": w_k, "weight_share_pct": 100.0 * w_k / w_t,
            "nodes_touched": non.number_of_nodes(), "isolated_nodes": len(iso),
            "comps_excl_isolated": len(sizes),
            "comps_incl_isolated": len(sizes) + len(iso),
            "nonsingleton_comps": int(sum(1 for x in sizes if x > 1)),
            # 孤立点在建图里各自就是一个单点分量。把"非孤立点内部的单点分量"和
            # "孤立点当成的单点分量"分成两列，避免 isolated_nodes=548 与
            # singleton_comps=0 被读成自相矛盾（non 已剔除孤立点，前者恒为 0）。
            "singleton_comps_among_touched": int(sum(1 for x in sizes if x == 1)),
            "singleton_comps_incl_isolated": int(len(iso)),
            "largest_comp": sizes[0] if sizes else 0,
            "second_comp": sizes[1] if len(sizes) > 1 else 0,
            "largest_share_of_touched_pct": 100.0 * (sizes[0] / non.number_of_nodes())
            if non.number_of_nodes() else 0.0,
            "top_comp_sizes": sizes[:15],
            "mean_degree_of_touched": (2 * S.number_of_edges() / non.number_of_nodes())
            if non.number_of_nodes() else 0.0}


def bootstrap_p(G, edges, n_draws=BOOT_DRAWS, seed=SEED):
    """Explicit MC of the allocation null at BOTH endpoints; p = max over endpoints."""
    rng = np.random.default_rng(seed)
    rows = []
    for (u, v) in edges:
        w = int(G[u][v]["weight"])
        pv = []
        for i, j in ((u, v), (v, u)):
            nbrs = G.adj[i]
            k_i = len(nbrs)
            s_i = int(sum(d.get("weight", 1) for d in nbrs.values()))
            if k_i < 2:
                pv.append(1.0)
                continue
            names = list(nbrs)
            pr = np.array([nbrs[x].get("weight", 1) for x in names], dtype=float)
            pr /= pr.sum()
            jj = names.index(j)
            draw = rng.multinomial(s_i, pr, size=n_draws)
            pv.append(float((draw[:, jj] >= w).mean()))
        rows.append({"edge_u": u, "edge_v": v, "weight": w,
                     "deg_u": len(G.adj[u]), "deg_v": len(G.adj[v]),
                     "strength_u": int(sum(d["weight"] for _, _, d in G.edges(u, data=True))),
                     "p_boot_u": pv[0], "p_boot_v": pv[1], "p_boot_edge": max(pv)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------- attack harness
def ac_fiedler(H, normalized=True):
    """lambda_2 and the Fiedler vector of the combinatorial (normalized=True) Laplacian.

    networkx 3.6.1 has NO `algebraic_connectivity_sequence` (the removed `networkx.gnp`
    module), so Newman's (2007) greedy rule is implemented directly here with a DENSE
    symmetric eigensolve, which is numerically identical for graphs this small and
    cannot fail to converge the way an ARPACK power iteration can.  See DISSENT.md.
    """
    nodes = sorted(H.nodes())
    A = nx.to_numpy_array(H, nodelist=nodes, dtype=float)
    d = A.sum(1)
    if normalized:
        dinv = np.where(d > 0, 1.0 / np.sqrt(np.maximum(d, 1e-12)), 0.0)
        L = np.eye(len(nodes)) - dinv[:, None] * A * dinv[None, :]
    else:
        L = np.diag(d) - A
    w, v = np.linalg.eigh(L)
    if len(w) < 2:
        return 0.0, np.zeros(len(nodes)), nodes
    return float(w[1]), v[:, 1], nodes


def ac_removal_order(H, max_steps, normalized=True, log=None):
    """Greedy Fiedler ordering: drop the node with the SMALLEST |v_i| at each step
    (the removal that least decreases the algebraic connectivity)."""
    Gc = H.copy()
    order = []
    for step in range(max_steps):
        if Gc.number_of_nodes() < 2:
            break
        a2, x, nodes = ac_fiedler(Gc, normalized)
        idx = int(np.argmin(np.abs(x)))
        victim = nodes[idx]
        order.append(victim)
        if log is not None:
            log.append({"step": step + 1, "node": victim, "ac_before_removal": round(a2, 6),
                        "fiedler_component": round(float(x[idx]), 6),
                        "degree_before_removal": int(Gc.degree(victim)),
                        "n_alive_before": len(nodes)})
        Gc.remove_node(victim)
    return order


def _init_worker():
    B = C.build_all()
    _W["Gd"] = B["Gd"]
    _W["G0"] = C.induced(B["Gu"], B["sets"]["gcc"])
    _W["ps_full"] = B["ps"]


def orderings(G, Gd, ac_steps=None, ac_log=None):
    deg = dict(G.degree())
    st = dict(G.degree(weight="weight"))
    bet = nx.betweenness_centrality(G)
    pr = nx.pagerank(C.induced(Gd, G.nodes()), weight="weight")
    n = G.number_of_nodes()
    k_ac = min(ac_steps if ac_steps is not None else 0, max(0, n - 1))
    ac = ac_removal_order(G, k_ac, normalized=True, log=ac_log) if k_ac else []
    return {"degree": sorted(G.nodes(), key=lambda x: (-deg[x], x)),
            "strength": sorted(G.nodes(), key=lambda x: (-st[x], x)),
            "betweenness": sorted(G.nodes(), key=lambda x: (-bet[x], x)),
            "pagerank_directed_weighted": sorted(G.nodes(), key=lambda x: (-pr.get(x, 0.0), x)),
            "ac_fiedler_greedy": ac}


def measure(H, ps):
    e = C.eff_three(H, ps)
    sizes = sorted((len(c) for c in nx.connected_components(H)), reverse=True) \
        if H.number_of_nodes() else []
    return {"E_full": e["E_full"], "E_gcc": e["E_gcc"], "E_fixed": e["E_fixed"],
            "gcc": sizes[0] if sizes else 0, "n_components": len(sizes),
            "largest_share": (sizes[0] / H.number_of_nodes()) if H.number_of_nodes() else 0.0,
            "edges_left": H.number_of_edges(), "alive": H.number_of_nodes()}


def _task(t):
    label, strategy, k, seed = t
    G, ps, order = _W["graphs"][label]
    H = G.copy()
    if strategy == "random":
        H.remove_nodes_from(random.Random(seed).sample(sorted(G.nodes()), k))
    else:
        H.remove_nodes_from(order[strategy][:k])
    r = measure(H, ps)
    r.update({"graph": label, "strategy": strategy, "k_removed": k, "seed": seed})
    return r


def _init_graphs(graphs):
    """Module-level Pool initializer (Windows/spawn cannot pickle a nested closure)."""
    _init_worker()
    _W["graphs"] = graphs


def _pool(tasks, graphs):
    from multiprocessing import Pool
    with Pool(processes=N_JOBS, initializer=_init_graphs, initargs=(graphs,)) as p:
        return pd.DataFrame(p.map(_task, tasks, chunksize=4))


def main():
    wall = []

    def tic(l):
        wall.append({"stage": l, "t0": time.time()})

    def toc():
        now = time.time()
        for r in wall:
            if "wall_s" not in r:
                r["wall_s"] = round(now - r["t0"], 3)

    t_all = time.time()
    print("E6 - disparity backbone and its resilience")
    print("=" * 104)
    tic("00 build + anchors")
    Ba = C.build_all()
    C.assert_anchors(Ba)                                  # CONVENTIONS SS1
    bad = C.check_anchors(Ba)
    print("  anchors: %d checked, %d failed" % (len(bad), int((~bad["ok"].astype(bool)).sum())))
    assert int((~bad["ok"].astype(bool)).sum()) == 0, "SS1 anchor mismatch"
    _init_worker()
    G0, Gd, ps_full = _W["G0"], _W["Gd"], _W["ps_full"]
    N0, M0 = G0.number_of_nodes(), G0.number_of_edges()
    assert (N0, M0) == (674, 2497)
    deg0 = dict(G0.degree())
    toc()

    # ------------------------------------------------ 01 directional p-values
    tic("01 disparity p-values (both closed forms)")
    t0 = time.time()
    dp = directional_p(G0)
    print("  %d directional tests in %.2fs" % (len(dp), time.time() - t0))
    print("  nodes with k_i<2 (no test possible): %d  -> all their edges are dropped"
          % sum(1 for v in deg0.values() if v < 2))
    print("  nodes with k_i==2 (form B has 0 trials, degenerate): %d"
          % sum(1 for v in deg0.values() if v == 2))
    print("  edges with a single strength unit (w=1): %d"
          % sum(1 for _, _, d in G0.edges(data=True) if d["weight"] == 1))
    toc()

    # ------------------------------------ 02 survival + fragmentation grid
    tic("02 survival grid: form x alpha x correction")
    rows, backs = [], {}
    for form in FORMS:
        for a in ALPHAS:
            for corr in CORRS:
                S = make_backbone(G0, dp, a, form, corr)
                c = census(S, G0)
                c.update({"closed_form": form, "alpha": a, "correction": corr})
                backs[(form, a, corr)] = S
                rows.append(c)
    naive = C.strong_backbone(G0, W_STRONG)
    cn = census(naive, G0)
    cn.update({"closed_form": "NAIVE_THRESHOLD", "alpha": np.nan,
               "correction": "w>=%d" % W_STRONG})
    rows.append(cn)
    bt = pd.DataFrame(rows)
    bt["top_comp_sizes"] = bt.top_comp_sizes.map(str)
    for c in bt.columns:
        if bt[c].dtype.kind == "f":
            bt[c] = bt[c].round(4)
    C.save_csv(bt, MODULE, "backbone_survival_fragments.csv")
    print(bt[["closed_form", "alpha", "correction", "edges", "edge_share_pct",
              "weight_share_pct", "nodes_touched", "isolated_nodes", "comps_excl_isolated",
              "nonsingleton_comps", "comps_incl_isolated", "largest_comp", "second_comp",
              "top_comp_sizes"]].to_string(index=False))
    toc()

    # ------------------------------------------- 03 bootstrap cross-check
    tic("03 bootstrap null: " + str(N_BOOT_EDGES) + " edges x " + str(BOOT_DRAWS) + " draws")
    all_e = list({tuple(sorted((i, j))) for (i, j) in dp})
    sel = random.Random(SEED + 616).sample(all_e, min(N_BOOT_EDGES, len(all_e)))
    t0 = time.time()
    boot = bootstrap_p(G0, sel)
    print("  %.2fs" % (time.time() - t0))
    boot["p_form_A_allocation"] = [dp[(r.edge_u, r.edge_v)][0] if (r.edge_u, r.edge_v) in dp
                                   else dp[(r.edge_v, r.edge_u)][0] for r in boot.itertuples()]
    boot["p_form_B_count"] = [dp[(r.edge_u, r.edge_v)][1] if (r.edge_u, r.edge_v) in dp
                              else dp[(r.edge_v, r.edge_u)][1] for r in boot.itertuples()]
    thr_u = [alpha_of(MAIN_ALPHA, r.deg_u, MAIN_CORR) for r in boot.itertuples()]
    thr_v = [alpha_of(MAIN_ALPHA, r.deg_v, MAIN_CORR) for r in boot.itertuples()]
    boot["threshold_min_endpoint"] = np.minimum(thr_u, thr_v)
    boot["keep_boot"] = boot.p_boot_edge <= boot.threshold_min_endpoint
    # form-specific keep needs both endpoint tests; approximate with the max over
    # endpoints for the cross-check table (conservative, stated in result.md)
    pmax_A, pmax_B, pmax_C = [], [], []
    for r in boot.itertuples():
        pa = max(dp[(r.edge_u, r.edge_v)][0], dp[(r.edge_v, r.edge_u)][0])
        pb = max(dp[(r.edge_u, r.edge_v)][1], dp[(r.edge_v, r.edge_u)][1])
        pc = max(dp[(r.edge_u, r.edge_v)][2], dp[(r.edge_v, r.edge_u)][2])
        pmax_A.append(pa)
        pmax_B.append(pb)
        pmax_C.append(pc)
    boot["p_max_endpoint_A"] = pmax_A
    boot["p_max_endpoint_B"] = pmax_B
    boot["p_max_endpoint_C"] = pmax_C
    boot["keep_A"] = boot.p_max_endpoint_A <= boot.threshold_min_endpoint
    boot["keep_B"] = boot.p_max_endpoint_B <= boot.threshold_min_endpoint
    boot["keep_C"] = boot.p_max_endpoint_C <= boot.threshold_min_endpoint
    boot["abs_diff_A_boot"] = (boot.p_max_endpoint_A - boot.p_boot_edge).abs()
    boot["abs_diff_B_boot"] = (boot.p_max_endpoint_B - boot.p_boot_edge).abs()
    boot["abs_diff_C_boot"] = (boot.p_max_endpoint_C - boot.p_boot_edge).abs()
    keep = boot[["edge_u", "edge_v", "weight", "deg_u", "deg_v", "strength_u",
                 "p_boot_u", "p_boot_v", "p_boot_edge", "p_max_endpoint_A",
                 "p_max_endpoint_B", "p_max_endpoint_C", "threshold_min_endpoint",
                 "keep_boot", "keep_A", "keep_B", "keep_C", "abs_diff_A_boot",
                 "abs_diff_B_boot", "abs_diff_C_boot"]].copy()
    for c in keep.columns:
        if keep[c].dtype.kind == "f":
            keep[c] = keep[c].round(8)
    C.save_csv(keep, MODULE, "bootstrap_vs_analytic_edges.csv")
    summ = pd.DataFrame([{
        "edges_checked": len(boot), "draws_per_edge": BOOT_DRAWS,
        "alpha": MAIN_ALPHA, "correction": MAIN_CORR,
        "mean_abs_diff_A_vs_boot": round(float(boot.abs_diff_A_boot.mean()), 6),
        "max_abs_diff_A_vs_boot": round(float(boot.abs_diff_A_boot.max()), 6),
        "mean_abs_diff_B_vs_boot": round(float(boot.abs_diff_B_boot.mean()), 6),
        "max_abs_diff_B_vs_boot": round(float(boot.abs_diff_B_boot.max()), 6),
        "mean_abs_diff_C_vs_boot": round(float(boot.abs_diff_C_boot.mean()), 6),
        "max_abs_diff_C_vs_boot": round(float(boot.abs_diff_C_boot.max()), 6),
        "spearman_A_vs_boot": round(float(spearmanr(boot.p_max_endpoint_A, boot.p_boot_edge).statistic), 4),
        "spearman_B_vs_boot": round(float(spearmanr(boot.p_max_endpoint_B, boot.p_boot_edge).statistic), 4),
        "spearman_C_vs_boot": round(float(spearmanr(boot.p_max_endpoint_C, boot.p_boot_edge).statistic), 4),
        "keep_A": int(boot.keep_A.sum()), "keep_B": int(boot.keep_B.sum()),
        "keep_C": int(boot.keep_C.sum()), "keep_boot": int(boot.keep_boot.sum()),
        "agreement_A_vs_boot": round(float((boot.keep_A == boot.keep_boot).mean()), 4),
        "agreement_B_vs_boot": round(float((boot.keep_B == boot.keep_boot).mean()), 4),
        "agreement_C_vs_boot": round(float((boot.keep_C == boot.keep_boot).mean()), 4)}])
    C.save_csv(summ, MODULE, "bootstrap_crosscheck_summary.csv")
    print(summ.to_string(index=False))
    toc()

    # ------------------------------- 04 backbone vs naive big-weight skeleton
    tic("04 overlap with the naive w>=10 skeleton")
    en = {tuple(sorted((u, v))) for u, v in naive.edges()}
    ov = []
    for (form, a, corr), S in backs.items():
        eb = {tuple(sorted((u, v))) for u, v in S.edges()}
        it = eb & en
        ov.append({"backbone": "%s | a=%s | %s" % (form, a, corr), "edges": len(eb),
                   "naive_w10_edges": len(en), "shared": len(it),
                   "jaccard": round(len(it) / max(1, len(eb | en)), 4),
                   "only_disparity": len(eb - en), "only_naive": len(en - eb),
                   "pct_of_naive_recovered": round(100.0 * len(it) / max(1, len(en)), 2)})
    odf = pd.DataFrame(ov)
    C.save_csv(odf, MODULE, "backbone_vs_naive_threshold.csv")
    print(odf.to_string(index=False))
    toc()

    # ------------------------------ 05 attacks on the backbone vs full graph
    tic("05 attack curves: full vs disparity backbone vs naive w>=10 skeleton")
    S_main = backs[(MAIN_FORM, MAIN_ALPHA, MAIN_CORR)]
    Sb = C.induced(S_main, [n for n in S_main.nodes() if S_main.degree(n) > 0])
    NB, MB = Sb.number_of_nodes(), Sb.number_of_edges()
    ps_b = C.PairSet(sorted(Sb.nodes()))
    # naive w>=10 skeleton: NOT a disparity backbone, only the "usable sparsified graph"
    # comparator, because the disparity filter leaves no connected component to attack.
    Sn = C.induced(naive, [n for n in naive.nodes() if naive.degree(n) > 0])
    NN, MN = Sn.number_of_nodes(), Sn.number_of_edges()
    ps_n = C.PairSet(sorted(Sn.nodes()))
    meas = {}
    for lbl, G, ps in (("full", G0, ps_full), ("backbone", Sb, ps_b),
                       ("naive_w10", Sn, ps_n)):
        meas[lbl] = measure(G, ps)
        print("  %-10s %4d nodes %4d edges | E_full=%.6f E_gcc=%.6f E_fixed=%.6f (denom %d)"
              % (lbl, G.number_of_nodes(), G.number_of_edges(), meas[lbl]["E_full"],
                 meas[lbl]["E_gcc"], meas[lbl]["E_fixed"], ps.denom))
    bf, bb, bn = meas["full"], meas["backbone"], meas["naive_w10"]
    AC_MAX_FRAC = 0.50          # the greedy Fiedler order is only built this far
    ac_logs = {}
    gr = {}
    for lbl, G, ps in (("full", G0, ps_full), ("backbone", Sb, ps_b),
                       ("naive_w10", Sn, ps_n)):
        ac_logs[lbl] = []
        gr[lbl] = (G, ps, orderings(G, Gd, int(round(G.number_of_nodes() * AC_MAX_FRAC)),
                                    ac_logs[lbl]))
    graphs = gr
    _W["graphs"] = graphs
    STRATS = ["random", "degree", "strength", "betweenness", "pagerank_directed_weighted",
              "ac_fiedler_greedy"]
    tasks = []
    for lbl, (G, ps, _) in graphs.items():
        u = G.number_of_nodes()
        for st in STRATS:
            for fr in FRACS:
                if st == "ac_fiedler_greedy" and fr > AC_MAX_FRAC:
                    continue
                k = max(1, int(round(u * fr)))
                if st == "random":
                    tasks += [(lbl, st, k, SEED + 313000 + rep) for rep in range(N_RANDOM)]
                else:
                    tasks.append((lbl, st, k, SEED))
    t0 = time.time()
    ac = _pool(tasks, graphs)
    print("  %d removals in %.1fs" % (len(tasks), time.time() - t0))
    C.save_csv(ac, MODULE, "attack_replicates.csv")

    AGG = {"nodes_removed": ("k_removed", "first"), "reps": ("seed", "count"),
           "E_full": ("E_full", "mean"), "E_gcc": ("E_gcc", "mean"),
           "E_fixed": ("E_fixed", "mean"), "E_fixed_sd": ("E_fixed", "std"),
           "E_full_sd": ("E_full", "std"),
           "gcc": ("gcc", "mean"), "gcc_sd": ("gcc", "std"),
           "n_components": ("n_components", "mean"),
           "largest_share": ("largest_share", "mean"),
           "edges_left": ("edges_left", "mean")}
    act = ac.groupby(["graph", "strategy", "k_removed"], sort=True).agg(**AGG).reset_index()
    act = act.drop(columns=["k_removed"])
    act[["E_fixed_sd", "E_full_sd", "gcc_sd"]] = act[["E_fixed_sd", "E_full_sd", "gcc_sd"]].fillna(0.0)
    UN = {"full": float(N0), "backbone": float(NB), "naive_w10": float(NN)}
    REF = {"full": bf["E_fixed"], "backbone": bb["E_fixed"], "naive_w10": bn["E_fixed"]}
    act["removed_pct_of_universe"] = 100.0 * act.nodes_removed / act.graph.map(UN)
    act["E_fixed_ratio"] = act.E_fixed / act.graph.map(REF)
    act["gcc_ratio"] = act.gcc / act.graph.map(UN)
    act["backbone_denominator"] = act.graph.map({"full": float(ps_full.denom),
                                                 "backbone": float(ps_b.denom),
                                                 "naive_w10": float(ps_n.denom)})
    act = act.sort_values(["graph", "strategy", "nodes_removed"]).reset_index(drop=True)
    for c in act.columns:
        if act[c].dtype.kind == "f":
            act[c] = act[c].round(6)
    C.save_csv(act, MODULE, "attack_curves_backbone_vs_full.csv")
    for lbl in ("full", "backbone", "naive_w10"):
        print("\n  E_fixed / own baseline  [" + lbl + "]")
        print(act[act.graph == lbl].pivot_table(index="removed_pct_of_universe",
              columns="strategy", values="E_fixed_ratio").round(4).to_string())

    pts = []
    for lbl in ("full", "backbone", "naive_w10"):
        for st in STRATS:
            sub = act[(act.graph == lbl) & (act.strategy == st)].sort_values("nodes_removed")
            row = {"graph": lbl, "strategy": st,
                   "monotone_E_fixed_decreasing": bool((sub.E_fixed.diff().dropna() <= 1e-9).all())}
            for nm, thr in [("lt90", 0.90), ("lt50", 0.50), ("lt10", 0.10)]:
                hit = sub[sub.gcc_ratio < thr]
                row["pct_when_gcc" + nm] = float(hit.removed_pct_of_universe.iloc[0]) if len(hit) else np.nan
            for nm, thr in [("Ehalf", 0.50), ("Etenth", 0.10)]:
                hit = sub[sub.E_fixed_ratio <= thr]
                row["pct_when_" + nm] = float(hit.removed_pct_of_universe.iloc[0]) if len(hit) else np.nan
            for p in (10, 30, 50):
                row["Efix_ratio_at" + str(p)] = float(
                    sub.iloc[(sub.removed_pct_of_universe - p).abs().values.argmin()].E_fixed_ratio)
            pts.append(row)
    pdf = pd.DataFrame(pts)
    for c in pdf.columns:
        if pdf[c].dtype.kind == "f":
            pdf[c] = pdf[c].round(4)
    C.save_csv(pdf, MODULE, "breakdown_points_backbone_vs_full.csv")
    print("")
    print(pdf.to_string(index=False))

    rev = []
    for st in STRATS:
        if st == "random":
            continue
        of = graphs["full"][2][st]
        rf = {n: i for i, n in enumerate(of)}
        for other in ("backbone", "naive_w10"):
            ob = graphs[other][2][st]
            rb = {n: i for i, n in enumerate(ob)}
            common = sorted(set(ob) & set(rf))
            rev.append({"strategy": st, "comparison": "full_vs_" + other,
                        "nodes_in_both": len(common),
                        "spearman_full_vs_subgraph": round(float(spearmanr(
                            [rf[n] for n in common], [rb[n] for n in common]).statistic), 4),
                        "top20_full_within_subgraph": len(set(of[:20]) & set(ob[:20])),
                        "top20_subgraph_also_top20_full": len(set(ob[:20]) & set(of[:20]))})
    rdf = pd.DataFrame(rev)
    C.save_csv(rdf, MODULE, "ranking_shift_full_vs_backbone.csv")
    print("")
    print(rdf.to_string(index=False))
    toc()

    # ---------------------------------------- 06 dynamic betweenness re-ranking
    tic("06 dynamic betweenness (SS6 robustness)")
    dyn = []
    for lbl, (G, ps, _) in graphs.items():
        u = G.number_of_nodes()
        targets = sorted({max(1, int(round(u * f))) for f in FRACS if f <= 0.50})
        H = G.copy()
        removed = 0
        t0 = time.time()
        for tgt in targets:
            while removed < tgt:
                b = nx.betweenness_centrality(H)
                if not b:
                    break
                H.remove_node(max(b.items(), key=lambda kv: kv[1])[0])
                removed += 1
            r = measure(H, ps)
            r.update({"graph": lbl, "strategy": "betweenness_DYNAMIC",
                      "k_removed": removed, "seed": SEED})
            dyn.append(r)
        print("  %s dynamic done in %.1fs" % (lbl, time.time() - t0))
    ddf = pd.DataFrame(dyn)
    ddf = ddf.rename(columns={"E_fixed": "E_fixed_dynamic", "gcc": "gcc_dynamic",
                              "n_components": "n_components_dynamic"})
    for c in ddf.columns:
        if ddf[c].dtype.kind == "f":
            ddf[c] = ddf[c].round(6)
    C.save_csv(ddf, MODULE, "dynamic_betweenness_curves.csv")
    st_cmp = act[act.strategy == "betweenness"][
        ["graph", "nodes_removed", "E_fixed", "gcc", "n_components"]].rename(
        columns={"nodes_removed": "k_removed", "E_fixed": "E_fixed_static",
                 "gcc": "gcc_static", "n_components": "n_components_static"})
    mrg = ddf.merge(st_cmp, on=["graph", "k_removed"], how="inner")
    mrg["E_fixed_static_minus_dynamic"] = mrg.E_fixed_static - mrg.E_fixed_dynamic
    mrg["gcc_static_minus_dynamic"] = mrg.gcc_static - mrg.gcc_dynamic
    C.save_csv(mrg, MODULE, "dynamic_vs_static_betweenness.csv")
    print(ddf.pivot_table(index="k_removed", columns="graph",
                          values="E_fixed_dynamic").round(4).to_string())
    toc()

    # -------------------------- 07 GN hierarchy + algebraic-connectivity order
    tic("07 girvan_newman + algebraic connectivity on the naive w>=10 skeleton")
    gn = []
    t0 = time.time()
    # 在 naive w>=10 骨架上做（disparity 主骨架只有几十条边、最大分量 4，看不出层级）
    for lvl, parts in zip(range(1, 26), nx.community.girvan_newman(Sn)):
        parts = [set(p) for p in parts]
        sizes = sorted((len(p) for p in parts), reverse=True)
        lou = C.louvain(Sn, seed=SEED + 100 + lvl)
        gn.append({"gn_level": lvl, "n_parts": len(parts), "largest_part": sizes[0],
                   "Q_of_gn_cut": round(float(nx.community.modularity(Sn, parts, weight="weight")), 4),
                   "louvain_Q": round(lou["Q"], 4),
                   "louvain_comms_total": lou["n_comms_total"],
                   "louvain_comms_nonsingleton": lou["n_comms_nonsingleton"],
                   "louvain_largest": lou["largest"]})
    gndf = pd.DataFrame(gn)
    C.save_csv(gndf, MODULE, "gn_levels_on_backbone.csv")
    print("  girvan_newman 25 levels in %.1fs" % (time.time() - t0))
    print(gndf.head(10).to_string(index=False))

    t0 = time.time()
    st_b = dict(Sn.degree(weight="weight"))
    deg_b = dict(Sn.degree())
    bet_b = nx.betweenness_centrality(Sn)
    acdf = pd.DataFrame([{"graph": lbl, **r,
                          "degree_backbone": deg_b.get(r["node"], 0),
                          "strength_backbone": st_b.get(r["node"], 0),
                          "betweenness_backbone": round(float(bet_b.get(r["node"], np.nan)), 6),
                          "degree_in_full_graph": deg0.get(r["node"], 0),
                          "still_in_backbone": r["node"] in deg_b}
                         for lbl, rows in ac_logs.items() for r in rows])
    acdf["graph_node"] = acdf.graph + "|" + acdf.node
    print("  greedy Fiedler orders: %d rows in %.1fs" % (len(acdf), time.time() - t0))
    for c in acdf.columns:
        if acdf[c].dtype.kind == "f":
            acdf[c] = acdf[c].round(6)
    C.save_csv(acdf, MODULE, "algebraic_connectivity_order.csv")
    rho = []
    for lbl, rows in ac_logs.items():
        d = acdf[acdf.graph == lbl]
        rho.append({"graph": lbl, "n_ordered": len(d),
                    "ac_first": round(float(d.ac_before_removal.iloc[0]), 6),
                    "ac_last_recorded": round(float(d.ac_before_removal.iloc[-1]), 6),
                    "spearman_acrank_vs_minus_degree": round(float(spearmanr(
                        d.step, -d.degree_in_full_graph if lbl == "full" else -d.degree_backbone
                    ).statistic), 4),
                    "spearman_acrank_vs_minus_strength": round(float(spearmanr(
                        d.step, -d.strength_backbone).statistic), 4),
                    "spearman_acrank_vs_minus_betweenness": round(float(spearmanr(
                        d.step, -d.betweenness_backbone).statistic), 4)})
    rdf2 = pd.DataFrame(rho)
    rdf2["interpretation_note"] = ("nx.gnp.algebraic_connectivity_sequence does not exist in "
                                   "networkx 3.6.1; the greedy Fiedler rule of Newman (2007) is "
                                   "re-implemented in this file (dense symmetric eigensolve, "
                                   "normalised Laplacian, remove min |v_i| each step). It is a "
                                   "NODE removal order, so it is the algebraic-connectivity "
                                   "analogue of the GN edge-cut hierarchy, not identical to it.")
    C.save_csv(rdf2, MODULE, "algebraic_connectivity_correlations.csv")
    print(rdf2.to_string(index=False))
    toc()

    el = time.time() - t_all
    print("")
    print("=" * 104)
    print("ELAPSED %.2fs seed=%d" % (el, SEED))
    for r in wall:
        print("  " + r["stage"].ljust(56) + str(r["wall_s"]).rjust(9) + "s")
    lines = ["# RUN.txt - written by run_e6.py",
             "command: cd /home/baifeiyang2026/csa_run_20261123/pipeline && "
             "PYTHONHASHSEED=0 ~/miniconda3/envs/csa/bin/python out/E6_backbone_resilience/run_e6.py",
             "seed: " + str(SEED), "random replicates per checkpoint: " + str(N_RANDOM),
             "bootstrap: " + str(N_BOOT_EDGES) + " edges x " + str(BOOT_DRAWS) + " multinomial draws",
             "networkx 3.6.1 provides no disparity_filter -> implemented in this file",
             "three closed forms compared for the disparity p-value:",
             "  A allocation_s_minus_w  = 1 - BinomCDF(w-1; s_i - w, w/s_i)",
             "  B paper_count_k_minus_2 = 1 - BinomCDF(w-1; k_i - 2, w/s_i)   [literal printed form;",
             "      degenerate for k_i=2: 0 trials => every link of a degree-2 node is significant]",
             "  C paper_closed_form     = (1 - w/s_i)^(k_i - 1)   [Serrano et al. 2009 eq.(4) after",
             "      evaluating the integral; this is the canonical implementation] <- MAIN",
             "main backbone: form=%s alpha=%s correction=%s -> %d edges / %d non-isolated nodes "
             "/ %d isolated / %d components / largest %d"
             % (MAIN_FORM, MAIN_ALPHA, MAIN_CORR, MB, NB, N0 - NB,
                meas["backbone"]["n_components"], meas["backbone"]["gcc"]),
             "NOTE the disparity backbone is NOT a connected network: it cannot be attacked the "
             "E1 way, so the E1-style curves are also run on the naive w>=%d skeleton (%d edges, "
             "%d non-isolated nodes, largest component %d) as the usable comparator; the disparity "
             "census itself is the result for 'what happens if only significant edges are kept'."
             % (W_STRONG, MN, NN, bn["gcc"]),
             "column definitions in backbone_survival_fragments.csv — the two component counts are "
             "NOT contradictory:",
             "  isolated_nodes = nodes with degree 0; in graph theory each is its own singleton "
             "component",
             "  comps_excl_isolated = components of the induced subgraph on NON-isolated nodes only; "
             "nonsingleton_comps == that same set (a degree>0 node can never be a singleton "
             "component), so both are the same count",
             "  singleton_comps_among_touched = singleton components inside that non-isolated "
             "subgraph, which is 0 by construction",
             "  singleton_comps_incl_isolated = isolated_nodes (each isolated node one component)",
             "  comps_incl_isolated = comps_excl_isolated + isolated_nodes",
             "revision history (why this file was rebuilt more than once):",
             "  run #1 (14:47-14:52): disparity filter only had forms A/B, MAIN=A -> 74 edges / "
             "548 isolated / largest component 5; E1-style curves were degenerate on that graph.",
             "  run #2 (16:40-16:46): added form C (paper eq.(4) closed form) as MAIN, added the "
             "naive w>=10 comparator and the three-graph attack curves.",
             "  run #3 (this run): the census column named singleton_comps was ambiguous against "
             "isolated_nodes; split into singleton_comps_among_touched + "
             "singleton_comps_incl_isolated. No numeric result changed except those two columns.",
             "E_fixed denominators: full %d / disparity %d / naive %d"
             % (int(ps_full.denom), int(ps_b.denom), int(ps_n.denom)),
             "elapsed_s: " + str(round(el, 2)), "versions: " + str(C.versions())]
    for r in wall:
        lines.append("stage | " + r["stage"] + " | wall_s=" + str(r["wall_s"]))
    with open(os.path.join(OUT, "RUN.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    C.save_csv(pd.DataFrame(wall), MODULE, "stage_walltimes.csv")
    print("RUN.txt written")


if __name__ == "__main__":
    main()
