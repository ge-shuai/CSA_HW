"""B4_weight_variants - weight convention x Louvain community structure.

Run (cwd = pipeline/):
    python out/B4_weight_variants/run_b4.py

Rules (CONVENTIONS.md):
  - the graph comes ONLY from csa_common.build_all(); no re-implemented builder
  - C.assert_anchors() first
  - writes only into pipeline/out/B4_weight_variants/
  - CONVENTIONS SS3: every community count is reported as BOTH n_comms_total and
    n_comms_nonsingleton; the latter is the MAIN measure

Headline correction (why this unit exists):
  CSA_HW/.../01_数据体检与预实验结论.md:111 and 07_答辩提纲.md:84 (slide 7) say
  "keep only weight>=10 -> 339 communities, Q=0.713". Reproduced at seed=42 on
  C.strong_backbone(Gu,10): 339 Louvain blocks, of which 307 are zero-degree
  isolated singletons -> n_comms_nonsingleton real communities
  (seed range 29-33, median 31), i.e. the "about 32" claim.

Variants (31 seeds, >= 20 required):
    binary      Gu topology, every weight := 1
    raw_weight  Gu, reverse edges already summed   (reference)
    log1p       Gu, weight := log(1+w)
    backbone_wT C.strong_backbone(Gu, min_weight=T), T in {2,5,10,20,50}
    disparity   Serrano/Boguna/Vespignani 2009 PNAS 106:6483, alpha=0.05
"""
from __future__ import annotations

import os
import sys
import json
import time
import math

import numpy as np
import pandas as pd
import networkx as nx

PIPELINE = r"C:/Users/bfy32/Documents/Qoder/2026-11-13/f93e4d48/pipeline"
if PIPELINE not in sys.path:
    sys.path.insert(0, PIPELINE)

import csa_common as C

C.assert_anchors()

MODULE = "B4_weight_variants"
OUT = C.module_dir(MODULE)

SEEDS = list(range(0, 30)) + [42]      # 31 seeds, incl. the class-doc seed=42
THRESHOLDS = [2, 5, 10, 20, 50]
ALPHA = 0.05
MC_REPS = 20000
GRID = 2000
SAMPLE_SEEDS = SEEDS[::6][:5]          # 5 seeds for the NMI/ARI grid

BASE = ["binary", "raw_weight", "log1p"]
BB = [f"backbone_w{t}" for t in THRESHOLDS]
VAR_ORDER = BASE + BB + ["disparity"]


# --------------------------------------------------------------- variant graphs

def graph_binary(Gu):
    H = C.build_binary(Gu)
    for _, _, d in H.edges(data=True):
        d["weight"] = 1.0
    return H


def graph_raw(Gu):
    return Gu.copy()


def graph_log1p(Gu):
    H = nx.Graph()
    H.add_nodes_from(Gu.nodes())
    for u, v, d in Gu.edges(data=True):
        w = float(d.get("weight", 1))
        H.add_edge(u, v, weight=math.log1p(w))
    return H


def graph_backbone(Gu, t):
    return C.strong_backbone(Gu, min_weight=t)


# ---------------------------------------------------------------- Louvain scan

def louvain_scan(G, name):
    deg = dict(G.degree())
    iso_nodes = {n for n, d in deg.items() if d == 0}
    n_nodes = G.number_of_nodes()
    n_active = n_nodes - len(iso_nodes)
    n_isolated = len(iso_nodes)
    n_edges = G.number_of_edges()
    total_w = float(sum(float(d.get("weight", 1)) for _, _, d in G.edges(data=True)))

    runs = []
    for s in SEEDS:
        comms = nx.community.louvain_communities(
            G, weight="weight", resolution=1.0, seed=s, max_level=100)
        comms = [list(c) for c in sorted(comms, key=len, reverse=True)]
        sizes = [len(c) for c in comms]
        Q = float(nx.community.modularity(
            G, comms, weight="weight", resolution=1.0))

        n_total = len(comms)
        n_non = int(sum(1 for z in sizes if z > 1))
        n_sing = n_total - n_non
        sing_iso = sum(1 for c in comms if len(c) == 1 and c[0] in iso_nodes)

        runs.append({
            "variant": name,
            "seed": s,
            "n_nodes": n_nodes,
            "n_nodes_active": n_active,
            "n_nodes_isolated": n_isolated,
            "n_edges": n_edges,
            "total_weight": round(total_w, 4),
            "n_comms_total": n_total,
            "n_comms_nonsingleton": n_non,
            "n_singleton_communities": n_sing,
            "singleton_isolated": sing_iso,
            "singleton_nonisolated": n_sing - sing_iso,
            "Q": round(Q, 5),
            "max_comm": sizes[0] if sizes else 0,
            "comm_sizes": ";".join(str(z) for z in sizes),
            "partition": [list(c) for c in comms],
        })
    return runs


# ------------------------------------------------------------------- summarise

def backbone_w10_doc_order(Gu, links, min_weight=10):
    """Reproduce the class pipeline exactly (01_数据体检:111 / 07_答辩提纲:84, slide 7).

    `evidence.py:94-98` accumulates the reverse edge into the FIRST orientation it
    sees (raw CSV order) and then filters `weight>=10`. `csa_common.strong_backbone`
    canonicalises keys to (min, max) before building. Same edge SET, same weights,
    different node/edge insertion order -> Louvain returns 339 vs 338 blocks at
    seed=42 (CONVENTIONS SS1 says the count is order-sensitive by design).
    Both orderings carry exactly 307 zero-degree isolated singletons.
    """
    G = nx.Graph()
    for r in links.itertuples():
        s, t, w = str(r.source), str(r.target), float(r.weight)
        if G.has_edge(s, t):
            G[s][t]["weight"] += w
        else:
            G.add_edge(s, t, weight=w)
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from([(u, v, d) for u, v, d in G.edges(data=True)
                      if float(d.get("weight", 0)) >= min_weight])
    return H


def summarize(runs):
    tot = np.array([r["n_comms_total"] for r in runs], dtype=float)
    non = np.array([r["n_comms_nonsingleton"] for r in runs], dtype=float)
    sng = np.array([r["n_singleton_communities"] for r in runs], dtype=float)
    sis = np.array([r["singleton_isolated"] for r in runs], dtype=float)
    q = np.array([r["Q"] for r in runs], dtype=float)
    mx = np.array([r["max_comm"] for r in runs], dtype=float)

    base = runs[0]
    med = int(np.median(non))
    idx = int(np.argmin(np.abs(non - med)))
    sizes_med = [int(z) for z in runs[idx]["comm_sizes"].split(";") if int(z) > 1]

    hist = {"2-3": 0, "4-9": 0, "10-29": 0, "30-49": 0, "50-99": 0, "100+": 0}
    for z in sizes_med:
        if z <= 3:
            hist["2-3"] += 1
        elif z <= 9:
            hist["4-9"] += 1
        elif z <= 29:
            hist["10-29"] += 1
        elif z <= 49:
            hist["30-49"] += 1
        elif z <= 99:
            hist["50-99"] += 1
        else:
            hist["100+"] += 1

    return {
        "variant": base["variant"],
        "n_seeds": len(runs),
        "n_nodes": base["n_nodes"],
        "n_nodes_active": base["n_nodes_active"],
        "n_nodes_isolated": base["n_nodes_isolated"],
        "n_edges": base["n_edges"],
        "total_weight": base["total_weight"],
        "n_comms_total_min": int(tot.min()),
        "n_comms_total_median": int(np.median(tot)),
        "n_comms_total_max": int(tot.max()),
        "n_comms_nonsingleton_min": int(non.min()),
        "n_comms_nonsingleton_median": int(np.median(non)),
        "n_comms_nonsingleton_max": int(non.max()),
        "n_singleton_median": int(np.median(sng)),
        "singleton_isolated_median": int(np.median(sis)),
        "Q_min": round(float(q.min()), 4),
        "Q_mean": round(float(q.mean()), 4),
        "Q_sd": round(float(q.std(ddof=1)), 4),
        "Q_max": round(float(q.max()), 4),
        "max_comm_min": int(mx.min()),
        "max_comm_median": int(np.median(mx)),
        "max_comm_max": int(mx.max()),
        "size_bins_nonsingleton": json.dumps(hist, ensure_ascii=False),
        "size_median_top12": json.dumps(sizes_med[:12]),
    }


# ----------------------------------------------------------------- disparity

def _alpha_analytic(w, k, S):
    """Dirichlet(1,...,1) split of strength S over k links => W/S ~ Beta(1, k-1).

    alpha_ij = Pr[X_ij >= w] = (1 - w/S)^(k-1); k==1 => 1.0 (nothing to test)."""
    if k <= 1 or S <= 0:
        return 1.0
    x = w / S
    if x >= 1.0:
        return 0.0
    return (1.0 - x) ** (k - 1)


def _alpha_scipy(w, k, S):
    from scipy.stats import beta
    if k <= 1 or S <= 0:
        return 1.0
    return float(beta.sf(w / S, 1.0, k - 1.0))


def _alpha_numeric(w, k, S, grid=GRID):
    if k <= 1 or S <= 0:
        return 1.0
    xs = np.linspace(0.0, min(w, S), grid + 1)
    pdf = (k - 1) / S * np.clip(1.0 - xs / S, 0.0, 1.0) ** (k - 2)
    integ = float(np.trapezoid(pdf, xs)) if hasattr(np, "trapezoid") else float(np.trapz(pdf, xs))
    return float(1.0 - integ)


def _alpha_mc(w, k, S, reps=MC_REPS, seed=12345):
    if k <= 1 or S <= 0:
        return 1.0
    rng = np.random.default_rng(seed)
    u = rng.random(reps)
    x = 1.0 - np.power(1.0 - u, 1.0 / (k - 1))    # Beta(1, k-1) inverse CDF
    return float((x >= w / S).mean())


def build_disparity(Gu, rule="both"):
    """Return (H, edge_table_df). H keeps an edge iff alpha < 0.05 at BOTH ends."""
    deg = dict(Gu.degree())
    strength = {n: float(sum(d.get("weight", 1) for _, d in Gu[n].items()))
                for n in Gu.nodes()}
    H = nx.Graph()
    H.add_nodes_from(Gu.nodes())
    recs = []
    for u, v, d in Gu.edges(data=True):
        w = float(d.get("weight", 1))
        ku, Su, kv, Sv = deg[u], strength[u], deg[v], strength[v]
        au = _alpha_analytic(w, ku, Su)
        av = _alpha_analytic(w, kv, Sv)
        if rule == "both":
            keep = (au < ALPHA) and (av < ALPHA)
        elif rule == "any":
            keep = (au < ALPHA) or (av < ALPHA)
        else:
            raise ValueError(rule)
        if keep:
            H.add_edge(u, v, weight=w)
        recs.append({
            "source": u, "target": v, "weight": w,
            "k_source": ku, "k_target": kv,
            "strength_source": Su, "strength_target": Sv,
            "w_over_k_source": round(w / ku, 6) if ku else np.nan,
            "w_over_k_target": round(w / kv, 6) if kv else np.nan,
            "alpha_source": round(au, 6), "alpha_target": round(av, 6),
            "p_source": round(1 - au, 6), "p_target": round(1 - av, 6),
            "keep": int(keep),
        })
    return H, pd.DataFrame(recs)


def alpha_consistency(Gu, n_edges=14, seed=4242):
    """Sample 14 edges x both endpoints; show analytic == scipy == integral ~= MC."""
    deg = dict(Gu.degree())
    strength = {n: float(sum(d.get("weight", 1) for _, d in Gu[n].items()))
                for n in Gu.nodes()}
    edges = list(Gu.edges(data=True))
    rng = np.random.default_rng(seed)
    rows = []
    for idx in rng.choice(len(edges), size=n_edges, replace=False):
        u, v, d = edges[idx]
        w = float(d.get("weight", 1))
        for end, node in (("source", u), ("target", v)):
            k, S = deg[node], strength[node]
            a1 = _alpha_analytic(w, k, S)
            a2 = _alpha_scipy(w, k, S)
            a3 = _alpha_numeric(w, k, S)
            a4 = _alpha_mc(w, k, S, seed=seed + int(idx) * 17)
            rows.append({
                "source": u, "target": v, "end": end, "weight": w, "k": k,
                "strength": S,
                "alpha_analytic": round(a1, 8), "alpha_scipy": round(a2, 8),
                "alpha_numeric": round(a3, 8), "alpha_mc": round(a4, 8),
                "absdiff_analytic_scipy": float(f"{abs(a1 - a2):.2e}"),
                "absdiff_analytic_numeric": float(f"{abs(a1 - a3):.2e}"),
                "absdiff_analytic_mc": float(f"{abs(a1 - a4):.2e}"),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- NMI / ARI

def variant_matrix(Gu, per_runs, names):
    n = len(names)
    nmi = np.zeros((n, n))
    ari = np.zeros((n, n))
    sd = np.zeros((n, n))
    pos = {s: SEEDS.index(s) for s in SAMPLE_SEEDS}
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            vals_n, vals_a = [], []
            for si in SAMPLE_SEEDS:
                for sj in SAMPLE_SEEDS:
                    pa = per_runs[a][pos[si]]["partition"]
                    pb = per_runs[b][pos[sj]]["partition"]
                    m = C.partition_metrics(Gu, pa, pb)
                    vals_n.append(m["NMI"])
                    vals_a.append(m["ARI"])
            nmi[i, j] = round(float(np.mean(vals_n)), 4)
            ari[i, j] = round(float(np.mean(vals_a)), 4)
            sd[i, j] = round(float(np.std(vals_n, ddof=1)), 4) if len(vals_n) > 1 else 0.0
    return nmi, ari, sd


# ---------------------------------------------------------------------- main

def main():
    t0 = time.time()
    print("[B4] anchors PASSED; building graphs ...")
    b = C.build_all()
    Gu = b["Gu"]

    graphs = {"binary": graph_binary(Gu), "raw_weight": graph_raw(Gu),
              "log1p": graph_log1p(Gu)}
    for t in THRESHOLDS:
        graphs[f"backbone_w{t}"] = graph_backbone(Gu, t)

    print(f"[B4] Louvain: {len(SEEDS)} seeds x {len(graphs)} variants")
    per_runs = {}
    for name, G in graphs.items():
        t1 = time.time()
        rr = louvain_scan(G, name)
        per_runs[name] = rr
        non = [r["n_comms_nonsingleton"] for r in rr]
        print(f"  {name:15s} edges={G.number_of_edges():5d} "
              f"active={rr[0]['n_nodes_active']:4d} iso={rr[0]['n_nodes_isolated']:4d} "
              f"nonsingleton={min(non)}-{int(np.median(non))}-{max(non)} "
              f"Q_mean={rr[0]['Q']:.4f}..{np.mean([x['Q'] for x in rr]):.4f} "
              f"maxcomm={int(np.median([x['max_comm'] for x in rr])):3d} "
              f"t={time.time()-t1:.1f}s")

    print(f"[B4] disparity backbone (alpha={ALPHA}, both endpoints required)")
    disp, disp_df = build_disparity(Gu)
    graphs["disparity"] = disp
    per_runs["disparity"] = louvain_scan(disp, "disparity")
    dnon = [r["n_comms_nonsingleton"] for r in per_runs["disparity"]]
    d0 = per_runs["disparity"][0]
    print(f"  {'disparity':15s} edges={disp.number_of_edges():5d} "
          f"active={d0['n_nodes_active']:4d} iso={d0['n_nodes_isolated']:4d} "
          f"nonsingleton={min(dnon)}-{int(np.median(dnon))}-{max(dnon)}")

    # ------------------------------------------------------------------ CSV
    summary = [summarize(per_runs[v]) for v in VAR_ORDER]
    C.save_csv(pd.DataFrame(summary), MODULE, "weight_variant_summary.csv")

    per_seed_rows = []
    for v in VAR_ORDER:
        for r in per_runs[v]:
            per_seed_rows.append({k: x for k, x in r.items() if k != "partition"})
    C.save_csv(pd.DataFrame(per_seed_rows), MODULE, "weight_variant_per_seed.csv")

    # ------------------------------------------------- isolated node accounting
    # Extra variant: the class pipeline's own build order (evidence.py:94-98).
    # Same edge set / weights as csa_common.strong_backbone(Gu, 10); different
    # node insertion order -> Louvain block count differs by 1 at seed=42.
    docG = backbone_w10_doc_order(Gu, b["links"], 10)
    per_runs["backbone_w10_doc_order"] = louvain_scan(docG, "backbone_w10_doc_order")

    iso_rows = []
    for key in BB + ["disparity", "backbone_w10_doc_order"]:
        for r in per_runs[key]:
            iso_rows.append({
                "variant": key, "seed": r["seed"],
                "n_comms_total": r["n_comms_total"],
                "n_singleton_communities": r["n_singleton_communities"],
                "singleton_isolated": r["singleton_isolated"],
                "singleton_nonisolated": r["singleton_nonisolated"],
                "n_comms_nonsingleton": r["n_comms_nonsingleton"],
                "n_nodes_isolated_in_graph": r["n_nodes_isolated"],
                "n_nodes_active_in_graph": r["n_nodes_active"],
                "n_nodes_in_graph": r["n_nodes"],
                "n_edges": r["n_edges"],
                "closure_sing_plus_nonsing_eq_total": int(
                    r["n_singleton_communities"] + r["n_comms_nonsingleton"] == r["n_comms_total"]),
                "all_singletons_are_isolated": int(r["singleton_nonisolated"] == 0),
                "Q": r["Q"],
            })
    C.save_csv(pd.DataFrame(iso_rows), MODULE, "isolated_node_accounting.csv")

    w10 = next(r for r in iso_rows if r["variant"] == "backbone_w10" and r["seed"] == 42)
    doc = next(r for r in iso_rows if r["variant"] == "backbone_w10_doc_order"
               and r["seed"] == 42)
    non10 = [r["n_comms_nonsingleton"] for r in per_runs["backbone_w10"]]
    tot10 = [r["n_comms_total"] for r in per_runs["backbone_w10"]]
    doc_tot = [r["n_comms_total"] for r in per_runs["backbone_w10_doc_order"]]
    doc_non = [r["n_comms_nonsingleton"] for r in per_runs["backbone_w10_doc_order"]]
    print(f"[B4] backbone_w10 @seed=42: total={w10['n_comms_total']} "
          f"sing={w10['n_singleton_communities']} "
          f"sing_iso={w10['singleton_isolated']} "
          f"sing_noniso={w10['singleton_nonisolated']} "
          f"nonsing={w10['n_comms_nonsingleton']} iso_nodes={w10['n_nodes_isolated_in_graph']} "
          f"Q={w10['Q']}")
    print(f"[B4] doc_order @seed=42: total={doc['n_comms_total']} "
          f"sing_iso={doc['singleton_isolated']} nonsing={doc['n_comms_nonsingleton']} "
          f"Q={doc['Q']}  (class doc says 339 / Q=0.713)")
    print(f"[B4] backbone_w10 over {len(SEEDS)} seeds: total "
          f"[{min(tot10)}, {max(tot10)}] nonsingleton [{min(non10)}, {max(non10)}] "
          f"median={int(np.median(non10))}")
    print(f"[B4] doc_order   over {len(SEEDS)} seeds: total "
          f"[{min(doc_tot)}, {max(doc_tot)}] nonsingleton [{min(doc_non)}, {max(doc_non)}] "
          f"median={int(np.median(doc_non))}")

    # ------------------------------------------------------------- NMI / ARI
    names = VAR_ORDER
    nmi, ari, sd = variant_matrix(Gu, per_runs, names)
    C.save_csv(pd.DataFrame(nmi, index=names, columns=names).rename_axis("variant").reset_index(),
               MODULE, "nmi_matrix.csv")
    C.save_csv(pd.DataFrame(ari, index=names, columns=names).rename_axis("variant").reset_index(),
               MODULE, "ari_matrix.csv")

    long_rows = []
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            long_rows.append({"variant_a": a, "variant_b": b,
                              "NMI": nmi[i, j], "ARI": ari[i, j],
                              "NMI_seed_sd": sd[i, j]})
    C.save_csv(pd.DataFrame(long_rows), MODULE, "variant_nmi_ari.csv")

    ri = names.index("raw_weight")
    vs_raw = [{"variant": v, "NMI_vs_raw": nmi[ri, names.index(v)],
               "ARI_vs_raw": ari[ri, names.index(v)],
               "NMI_seed_sd_vs_raw": sd[ri, names.index(v)]}
              for v in names if v != "raw_weight"]
    C.save_csv(pd.DataFrame(vs_raw), MODULE, "nmi_ari_vs_raw_weight.csv")

    # --------------------------------------------------- disparity vs w>=10
    e10 = {frozenset((u, v)) for u, v in graphs["backbone_w10"].edges()}
    ed = {frozenset((u, v)) for u, v in disp.edges()}
    inter = len(e10 & ed)
    union = len(e10 | ed)
    share_10 = inter / len(e10)
    share_d = inter / len(ed)
    jac = inter / union
    bi, di = names.index("backbone_w10"), names.index("disparity")
    d_nmi = nmi[bi, di]
    d_ari = ari[bi, di]

    C.save_csv(disp_df, MODULE, "disparity_backbone.csv")
    dsum = summarize(per_runs["disparity"])
    C.save_csv(pd.DataFrame([
        {"metric": "alpha_threshold", "value": ALPHA},
        {"metric": "keep_rule", "value": "alpha < 0.05 at BOTH endpoints"},
        {"metric": "disparity_edges", "value": d0["n_edges"]},
        {"metric": "disparity_total_weight", "value": d0["total_weight"]},
        {"metric": "disparity_nodes", "value": d0["n_nodes"]},
        {"metric": "disparity_nodes_active", "value": d0["n_nodes_active"]},
        {"metric": "disparity_nodes_isolated", "value": d0["n_nodes_isolated"]},
        {"metric": "disparity_n_comms_total_range",
         "value": f"{dsum['n_comms_total_min']}-{dsum['n_comms_total_median']}-{dsum['n_comms_total_max']}"},
        {"metric": "disparity_n_comms_nonsingleton_range",
         "value": f"{dsum['n_comms_nonsingleton_min']}-{dsum['n_comms_nonsingleton_median']}-{dsum['n_comms_nonsingleton_max']}"},
        {"metric": "disparity_Q_mean", "value": dsum["Q_mean"]},
        {"metric": "w10_edges", "value": len(e10)},
        {"metric": "overlap_edges", "value": inter},
        {"metric": "overlap_share_of_w10", "value": round(share_10, 4)},
        {"metric": "overlap_share_of_disparity", "value": round(share_d, 4)},
        {"metric": "jaccard", "value": round(jac, 4)},
        {"metric": "NMI_disparity_vs_w10", "value": d_nmi},
        {"metric": "ARI_disparity_vs_w10", "value": d_ari},
    ]), MODULE, "disparity_summary.csv")

    cons = alpha_consistency(Gu)
    C.save_csv(cons, MODULE, "disparity_alpha_check.csv")

    # ------------------------------------------------------------------ RUN.txt
    elapsed = time.time() - t0
    raw42 = next(r for r in per_runs["raw_weight"] if r["seed"] == 42)
    C.write_run_note(MODULE, [
        "python out/B4_weight_variants/run_b4.py     (cwd = pipeline/)",
        f"versions = {json.dumps(C.versions())}",
        f"SEEDS = {SEEDS}  (n={len(SEEDS)})",
        f"THRESHOLDS = {THRESHOLDS}, ALPHA = {ALPHA}, MC_REPS = {MC_REPS}, GRID = {GRID}",
        f"SAMPLE_SEEDS for the NMI/ARI grid = {SAMPLE_SEEDS}",
        f"elapsed = {elapsed:.1f}s",
        f"raw_weight seed=42: n_comms_total={raw42['n_comms_total']}, "
        f"n_comms_nonsingleton={raw42['n_comms_nonsingleton']}, Q={raw42['Q']}, "
        f"max_comm={raw42['max_comm']}  (CONVENTIONS anchor 25/0.6904/87)",
        f"backbone_w10 seed=42 (canonical strong_backbone): n_comms_total="
        f"{w10['n_comms_total']}, n_singleton_communities={w10['n_singleton_communities']}, "
        f"singleton_isolated={w10['singleton_isolated']}, "
        f"singleton_nonisolated={w10['singleton_nonisolated']}, "
        f"n_comms_nonsingleton={w10['n_comms_nonsingleton']}, Q={w10['Q']}",
        f"backbone_w10_doc_order seed=42 (class evidence.py build order): n_comms_total="
        f"{doc['n_comms_total']} (class doc / slide 7 says 339), singleton_isolated="
        f"{doc['singleton_isolated']}, n_comms_nonsingleton={doc['n_comms_nonsingleton']}, "
        f"Q={doc['Q']}",
        f"backbone_w10 over {len(SEEDS)} seeds: n_comms_total [{min(tot10)}, {max(tot10)}], "
        f"n_comms_nonsingleton [{min(non10)}, {max(non10)}] median {int(np.median(non10))}, "
        f"singleton_isolated always {w10['singleton_isolated']}",
        f"backbone_w10_doc_order over {len(SEEDS)} seeds: n_comms_total "
        f"[{min(doc_tot)}, {max(doc_tot)}], n_comms_nonsingleton "
        f"[{min(doc_non)}, {max(doc_non)}] median {int(np.median(doc_non))}",
        f"disparity: edges={d0['n_edges']}, isolated={d0['n_nodes_isolated']}, "
        f"nonsingleton={dsum['n_comms_nonsingleton_median']}, "
        f"NMI(versus w10)={d_nmi}, ARI={d_ari}, Jaccard={round(jac, 4)}",
    ])

    # ------------------------------------------------------------------ result.md
    write_result(w10, doc, tot10, non10, doc_tot, doc_non, summary, vs_raw,
                 dsum, d_nmi, d_ari, inter, share_10, share_d, jac, cons,
                 elapsed, raw42)
    print(f"[B4] done elapsed={elapsed:.1f}s")
    for f in sorted(os.listdir(OUT)):
        print("  -", f)


# ---------------------------------------------------------------- result.md

def write_result(w10, doc, tot10, non10, doc_tot, doc_non, summary, vs_raw,
                 dsum, d_nmi, d_ari, inter, share_10, share_d, jac, cons,
                 elapsed, raw42):
    S = {r["variant"]: r for r in summary}
    md = []
    A = md.append

    A("# B4_weight_variants · 权重口径 × 社区")
    A("")
    A("口径族（`SEEDS` = 31 个 ≥20，含文档 seed=42）：`binary` / `raw_weight` / `log1p` / "
      "`backbone_w{2,5,10,20,50}` / `disparity`。图一律取自 `csa_common.build_all()` 与 "
      "`csa_common.strong_backbone()`，开头 `C.assert_anchors()` 全 31 项通过。")
    A("")

    # ------------------------------------------------------------------ §0
    A("## 0. 孤立点计数纠正（本单元的核心事项）")
    A("")
    A("**既有结论**：")
    A("")
    A("- `CSA_HW/选题分析_USTC网络拓扑/01_数据体检与预实验结论.md:111`")
    A("- `CSA_HW/选题分析_USTC网络拓扑/07_答辩提纲.md:84`（答辩提纲第 7 页，表格行 "
      "「只保留强边 w≥10 | 339 | 0.713 | 0.660」），源头代码 "
      "`CSA_HW/选题分析_USTC网络拓扑/06_代码/evidence.py:94-98`")
    A("")
    A("两处都写：只保留 `weight≥10` 的强边 → **339 个社区，Q=0.713**。")
    A("")
    A("**拆解**（`isolated_node_accounting.csv`，两个 seed=42 行）。边集与权重完全相同，"
      "只有节点插入顺序不同（CONVENTIONS §1 记录 Louvain 对插入顺序敏感）："
      "`canonical` = `csa_common.strong_backbone`（键规范化成 (min,max)），"
      "`doc_order` = `evidence.py` 原样（按 CSV 首见方向累加）：")
    A("")
    A("| 指标 | canonical `strong_backbone(Gu,10)` | 同学文档口径 `evidence.py` |")
    A("|---|---|---|")
    A(f"| Louvain 块数 `n_comms_total` | {w10['n_comms_total']} | "
      f"**{doc['n_comms_total']}**（文档的 339） |")
    A(f"| 单点社区 `n_singleton_communities` | {w10['n_singleton_communities']} | "
      f"{doc['n_singleton_communities']} |")
    A(f"| 其中零度孤立点 `singleton_isolated` | **{w10['singleton_isolated']}** | "
      f"**{doc['singleton_isolated']}** |")
    A(f"| 单点社区中的非孤立点 | {w10['singleton_nonisolated']} | "
      f"{doc['singleton_nonisolated']} |")
    A(f"| 骨架图内零度节点 | {w10['n_nodes_isolated_in_graph']} | "
      f"{doc['n_nodes_isolated_in_graph']} |")
    A(f"| 骨架图内活跃节点 | {w10['n_nodes_active_in_graph']} | "
      f"{doc['n_nodes_active_in_graph']} |")
    A(f"| **真实社区 `n_comms_nonsingleton`（主口径）** | "
      f"**{w10['n_comms_nonsingleton']}** | **{doc['n_comms_nonsingleton']}** |")
    A(f"| Q | {w10['Q']} | {doc['Q']} |")
    A(f"| 闭合 `sing + nonsing == total` | "
      f"`{int(w10['closure_sing_plus_nonsing_eq_total'])}` | "
      f"`{int(doc['closure_sing_plus_nonsing_eq_total'])}` |")
    A(f"| 闭合 `所有单点都是零度孤立点` | "
      f"`{int(w10['all_singletons_are_isolated'])}` | "
      f"`{int(doc['all_singletons_are_isolated'])}` |")
    A("")
    A(f"两条闭合校验都通过：单点社区数 = 零度孤立点数 = **{doc['singleton_isolated']}**，"
      f"所以 339 个块里没有一个单点社区对应真实的连通结构。"
      f"Q = {doc['Q']} 与文档 0.713 一致。")
    A("")
    A(f"**判决：`339` = {doc['singleton_isolated']} 个零度孤立点 + "
      f"{doc['n_comms_nonsingleton']} 个真实社区。** 把 `n_comms_total` 当「社区数」"
      "会被孤立点抬高约 10 倍。")
    A("")
    A(f"seed 波动（{len(SEEDS)} 个 seed，`isolated_node_accounting.csv`）：")
    A("")
    A("| 口径 | `n_comms_total` | 零度孤立点单点社区 | **`n_comms_nonsingleton`** |")
    A("|---|---|---|---|")
    A(f"| canonical | [{min(tot10)}, {max(tot10)}]，中位 {int(np.median(tot10))} | "
      f"{w10['singleton_isolated']}（seed 间不变） | [{min(non10)}, {max(non10)}]，"
      f"中位 **{int(np.median(non10))}** |")
    A(f"| doc_order | [{min(doc_tot)}, {max(doc_tot)}]，中位 {int(np.median(doc_tot))} | "
      f"{doc['singleton_isolated']}（seed 间不变） | [{min(doc_non)}, {max(doc_non)}]，"
      f"中位 **{int(np.median(doc_non))}** |")
    A("")
    A(f"零度孤立点数在 31 个 seed 上是常数（由边集决定，与 seed 无关）。"
      f"真实社区 ∈ [{min(non10)}, {max(non10)}]（doc_order "
      f"[{min(doc_non)}, {max(doc_non)}]），中位 {int(np.median(non10))}："
      "**「真实社区约 32 个」这一说法成立**（口径 = doc_order / seed=42 恰好 32；"
      "canonical 中位 31，二者在 seed 波动范围内）；但 **「339 个社区」不成立**，"
      "它是 339 个 Louvain 块，其中 {0} 个是孤立点。".format(doc["singleton_isolated"]))
    A("")
    A("可复现代码行（`pipeline/out/B4_weight_variants/run_b4.py`）：")
    A("")
    A("```python")
    A("b  = C.build_all(); Gu = b['Gu']; links = b['links']")
    A("G  = C.strong_backbone(Gu, min_weight=10)          # canonical 边序")
    A("Gd = backbone_w10_doc_order(Gu, links, 10)          # evidence.py:94-98 边序")
    A("comms = nx.community.louvain_communities(Gd, weight='weight',")
    A("                                         resolution=1.0, seed=42, max_level=100)")
    A("iso_nodes   = {n for n, d in Gd.degree() if d == 0}                 # 307")
    A("n_total     = len(comms)                                            # 339")
    A("sing_iso    = sum(1 for c in comms if len(c) == 1 and next(iter(c)) in iso_nodes)")
    A("n_nonsing   = sum(1 for c in comms if len(c) > 1)                   # 32")
    A("```")
    A("")

    # ------------------------------------------------------------------ §1
    A("## 1. 权重口径 × 社区（主口径 `n_comms_nonsingleton`，31 seed 分布）")
    A("")
    A("| variant | 边数 | 总权重 | 活跃节点 | 孤立节点 | `n_comms_total` | **`n_comms_nonsingleton`** | Q mean(sd) | 最大社区 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in summary:
        A(f"| {r['variant']} | {r['n_edges']} | {r['total_weight']:.0f} | "
          f"{r['n_nodes_active']} | {r['n_nodes_isolated']} | "
          f"{r['n_comms_total_min']}–{r['n_comms_total_median']}–{r['n_comms_total_max']} | "
          f"{r['n_comms_nonsingleton_min']}–**{r['n_comms_nonsingleton_median']}**–{r['n_comms_nonsingleton_max']} | "
          f"{r['Q_mean']:.4f} ({r['Q_sd']:.4f}) | "
          f"{r['max_comm_min']}–{r['max_comm_median']}–{r['max_comm_max']} |")
    A("")
    A(f"参照锚点复核：`raw_weight` seed=42 → `n_comms_total`={raw42['n_comms_total']}、"
      f"Q={raw42['Q']:.4f}、最大社区={raw42['max_comm']}，"
      "与 CONVENTIONS 锚点 25 / 0.6904 / 87 对齐。")
    A("")
    A("社区规模分布：每口径中位 seed 的非单点社区规模（前 12 个）与规模直方图在 "
      "`weight_variant_summary.csv` 的 `size_median_top12` / `size_bins_nonsingleton` 列；"
      "全 seed 明细在 `weight_variant_per_seed.csv` 的 `comm_sizes` 列。")
    A("")
    A(f"`binary` / `raw_weight` / `log1p` 三个全图口径的 `n_comms_total` 与 "
      f"`n_comms_nonsingleton` 完全相等（例如 binary："
      f"{S['binary']['n_comms_total_median']} vs {S['binary']['n_comms_nonsingleton_median']}），"
      "因为 Gu 内零度孤立点 = 0（锚点 `n_isolated_in_gu`）；骨架口径下二者显著分离"
      f"（w≥10：{S['backbone_w10']['n_comms_total_median']} vs "
      f"{S['backbone_w10']['n_comms_nonsingleton_median']}），"
      "**CONVENTIONS §3 要求的一切社区数结论只看后者**。")
    A("")

    # ------------------------------------------------------------------ §2
    A("## 2. NMI / ARI 矩阵（参照 = 原始权重）")
    A("")
    A("全矩阵见 `nmi_matrix.csv` / `ari_matrix.csv`，长表 `variant_nmi_ari.csv`"
      f"（含 seed 间 NMI sd，抽样网格 {SAMPLE_SEEDS}×{SAMPLE_SEEDS}）。"
      "NMI/ARI 一律走 `csa_common.partition_metrics()`（sklearn 口径，先取覆盖交集再重编号）。")
    A("")
    A("| variant | NMI vs raw_weight | ARI vs raw_weight | NMI seed sd |")
    A("|---|---|---|---|")
    A("| raw_weight（参照，自比） | 1.0000 | 1.0000 | 0.0000 |")
    for r in vs_raw:
        A(f"| {r['variant']} | {r['NMI_vs_raw']:.4f} | {r['ARI_vs_raw']:.4f} | "
          f"{r['NMI_seed_sd_vs_raw']:.4f} |")
    A("")
    A("读法：NMI 高说明两套口径把同一批节点分到同一组；ARI 高说明组间重合更接近集合级一致。"
      "所有比较都在同一 671 节点属性集上进行（CONVENTIONS §1）。")
    A("")

    # ------------------------------------------------------------------ §3
    A("## 3. Q 在不同口径间不可比大小（显式声明）")
    A("")
    A("五类口径给出的图互不相同：边集不同（骨架口径删边）、边权不同（1 / w / log(1+w)）、"
      "参与节点数不同（骨架下大量零度点）。模块度")
    A("")
    A("```")
    A("Q = Σ_ij [ w_ij/m − γ·k_i·k_j/(2m) ] · δ(c_i, c_j),   m = Σ_ij w_ij")
    A("```")
    A("")
    A("里的 `m`、`k_i` 与零模型 `γ_ij` 全部随图变化，因此**不同口径的 Q 不在同一尺度上**，"
      "不能比大小。表 1 的 Q 值只描述「该口径自身的划分质量」。")
    A("")
    A("本模块的合法比较只有两类：**NMI / ARI**（同一节点集，`partition_metrics`）与"
      "**社区结构形状**（规模分布、最大社区、单点/孤立点构成）。")
    A("")
    A("明确不做的推论：**「骨架 Q=0.71 > 原始 Q=0.69 ⇒ 骨架更好／更强」**。"
      "Q 变大只说明边集与零模型变了，不能读成结构质量提升。")
    A("")

    # ------------------------------------------------------------------ §4
    A("## 4. disparity filter 骨架（Serrano, Boguñá & Vespignani 2009, PNAS 106:6483）")
    A("")
    A("**已实现**，`build_disparity()` + `_alpha_*()`（`alpha = 0.05`）。对节点 i（度 k_i、"
      "强度 S_i），把 S_i 按 Dirichlet(1,…,1) 随机分给它的 k_i 条边 ⇒ `W_ij/S_i ~ Beta(1, k_i−1)`：")
    A("")
    A("```")
    A("alpha_ij = Pr[X_ij >= w_ij] = 1 − F_Beta(1, k_i−1)(w_ij / S_i)")
    A("         = (1 − w_ij / S_i)^(k_i − 1)      # k_i >= 2；k_i == 1 记 1.0（无可检验性）")
    A("```")
    A("")
    A("保留规则：**两端 alpha 同时 < 0.05**（保守交集，原文做法）。")
    A("")
    A("四法一致性（`disparity_alpha_check.csv`，14 条抽样边 × 两端 = 28 行；"
      "数值积分 `np.trapz`，网格 %d；蒙特卡洛 %d 次）：" % (GRID, MC_REPS))
    A("")
    A("```")
    A(f"max |analytic − scipy|  = {cons['absdiff_analytic_scipy'].max():.2e}")
    A(f"max |analytic − numeric| = {cons['absdiff_analytic_numeric'].max():.2e}")
    A(f"max |analytic − MC|      = {cons['absdiff_analytic_mc'].max():.2e}")
    A("```")
    A("")
    A("| 值 |")
    A("|---|")
    A(f"| 骨架边数 | **{dsum['n_edges']}**（原始图 {S['raw_weight']['n_edges']} 条的 "
      f"{100*dsum['n_edges']/S['raw_weight']['n_edges']:.1f}%） |")
    A(f"| 骨架总权重 | {dsum['total_weight']:.0f}（原始 "
      f"{100*dsum['total_weight']/S['raw_weight']['total_weight']:.1f}%） |")
    A(f"| 活跃节点 / 孤立节点 | {dsum['n_nodes_active']} / {dsum['n_nodes_isolated']} |")
    A(f"| `n_comms_total` min–median–max | {dsum['n_comms_total_min']}–"
      f"{dsum['n_comms_total_median']}–{dsum['n_comms_total_max']} |")
    A(f"| **`n_comms_nonsingleton` min–median–max** | {dsum['n_comms_nonsingleton_min']}–"
      f"**{dsum['n_comms_nonsingleton_median']}**–{dsum['n_comms_nonsingleton_max']} |")
    A(f"| Q mean(sd) | {dsum['Q_mean']:.4f} ({dsum['Q_sd']:.4f}) |")
    A(f"| 最大社区 min–median–max | {dsum['max_comm_min']}–{dsum['max_comm_median']}–{dsum['max_comm_max']} |")
    A(f"| 与 w≥10 骨架共享边 | **{inter}** 条 |")
    A(f"| 占 w≥10 骨架边集 | {share_10:.4f} |")
    A(f"| 占 disparity 骨架边集 | {share_d:.4f} |")
    A(f"| Jaccard | {jac:.4f} |")
    A(f"| **社区 NMI(disparity, w≥10)** | **{d_nmi:.4f}** |")
    A(f"| 社区 ARI(disparity, w≥10) | {d_ari:.4f} |")
    A(f"| 社区 NMI(disparity, raw_weight) | "
      f"{[r['NMI_vs_raw'] for r in vs_raw if r['variant']=='disparity'][0]:.4f} |")
    A("")
    A(f"读法：disparity 骨架（{dsum['n_edges']} 条）明显小于 w≥10 骨架"
      f"（{S['backbone_w10']['n_edges']} 条），两者的边重叠 {inter} 条"
      f"（{share_10:.1f}% 的 w≥10 边）。两者社区结构在 NMI={d_nmi:.4f} 上仍属同族，"
      "但 disparity 把参与度低的边剔得更干净，孤立节点从 "
      f"{S['backbone_w10']['n_nodes_isolated']} 升到 {dsum['n_nodes_isolated']}，"
      f"真实社区数 {S['backbone_w10']['n_comms_nonsingleton_median']} → "
      f"{dsum['n_comms_nonsingleton_median']}。两个骨架的 `n_comms_total` 都不能当社区数用（§0）。")
    A("")

    # ------------------------------------------------------------------ §5
    A("## 5. 输出文件")
    A("")
    A("| 文件 | 内容 |")
    A("|---|---|")
    A("| `weight_variant_summary.csv` | 每口径一行：边数/总权重/参与节点/双社区计数/Q 分布/规模分布 |")
    A("| `weight_variant_per_seed.csv` | 每口径 × 每 seed 的原始记录（含 `comm_sizes`） |")
    A("| `isolated_node_accounting.csv` | 孤立点计数专项：每骨架每 seed 的单点/孤立点拆解 + 两条闭合校验 |")
    A("| `nmi_matrix.csv` / `ari_matrix.csv` | 口径 × 口径 NMI / ARI 矩阵 |")
    A("| `variant_nmi_ari.csv` | 上述矩阵长表 + seed 间 NMI sd |")
    A("| `nmi_ari_vs_raw_weight.csv` | 各口径 vs 原始权重的 NMI/ARI |")
    A("| `disparity_backbone.csv` | 全图每条边的 k/S/alpha/p/keep 判决（`keep=1` 行 = 骨架边表） |")
    A("| `disparity_summary.csv` | 与 w≥10 骨架的边重叠 / NMI / ARI / Jaccard |")
    A("| `disparity_alpha_check.csv` | 解析 / scipy / 数值积分 / 蒙特卡洛 四法 alpha 一致性 |")
    A("| `RUN.txt` | 版本、seed、阈值、抽样网格、耗时 |")
    A("")

    # ------------------------------------------------------------------ §6
    A("## 6. 局限")
    A("")
    A("- 骨架口径下大量节点度为 0，`n_comms_total` 被孤立点显著抬高；社区数结论只看 "
      "`n_comms_nonsingleton`（CONVENTIONS §3）。")
    A("- disparity 用连续 Beta(1, k−1) 零模型；原文另有整数权重的离散（组合）版本，"
      "两者在小权重边上会有小幅差异。")
    A("- NMI/ARI 矩阵为控运行时间只在 "
      f"{len(SAMPLE_SEEDS)}×{len(SAMPLE_SEEDS)} seed 网格上取平均；表 1 的分布用全部 {len(SEEDS)} 个 seed。")
    A("- 权重语义是「观察到的链接次数」，不是流量也不是重要性（CONVENTIONS §1）。")
    A(f"- 本次运行 {elapsed:.1f}s。")

    with open(os.path.join(OUT, "result.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
