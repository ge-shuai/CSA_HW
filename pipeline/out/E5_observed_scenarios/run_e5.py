# -*- coding: utf-8 -*-
"""
E5 — OBSERVED FAILURE SCENARIOS vs SAME-SIZE RANDOM CONTROLS
USTC subdomain hyperlink network / 复杂系统分析
================================================================================
Contract: pipeline/CONVENTIONS.md.  Clauses that bind this module:
  SS1   baseline graph = 674-node / 2497-edge main component of the undirected
        weighted graph -> C.induced(B["Gu"], B["sets"]["gcc"]).
  SS2.2 true failures = A_conn_fail + D_not_found + F_server_error (=129);
        false failures = B_auth_or_skip + C_robots_blocked (=32) + G_other_400 (=1,
        ambiguous) are crawl artefacts and are RESTORED in the corrected view.
        Every result is reported before AND after that correction.
  SS2.3 three efficiency口径 at once, conclusions from E_fixed ONLY, and only at
        EQUAL k.  Only C.eff_three / C.PairSet are used — no hand-rolled efficiency.
        E_fixed denominator is frozen at 453,602 ordered pairs (674*673).
  SS4   random nulls: >=100 independent replicates, explicit seed, mean +- sd,
        z score and empirical percentile.
  SS7   no causal wording.
  SS8   SEED = 20261113, PYTHONHASHSEED=0.
  SS9   run_e5.py / *.csv (UTF-8-sig) / result.md / RUN.txt / DISSENT.md.

Scenarios (labels verified in stage 02 against data/dead_categories.csv and the
nodes table, not copied from the brief):
  dead162            every unreachable subdomain          A+B+C+D+F+G = 162
  true129            true failures only                   A+D+F       = 129
  false32            false failures only (SS2.2)          B+C         =  32
  ambiguous1         to be judged by hand                 G           =   1
  partial149         every partially-failing subdomain    B/C/D/F     = 149
  partial_clean80    partial, no -1/-2/403 URL            (SS2.2 clean) = 80
  partial_authlike69 partial with -1/-2/403 URL           complement  =  69
  combo209           true129 | partial_clean80            = 209
  combo311           dead162  | partial149                = 311
Each is scored twice: as observed (uncorrected) and after the SS2.2 correction.

Run:  PYTHONHASHSEED=0 python out/E5_observed_scenarios/run_e5.py
"""
from __future__ import annotations

import os
import random
import sys
import time
from itertools import combinations

# one BLAS thread per worker: 8 workers x 16 threads each thrashed the machine
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

import networkx as nx
import numpy as np
import pandas as pd

import csa_common as C

SEED = 20261113            # CONVENTIONS SS8
N_RANDOM = 100             # CONVENTIONS SS4 (>=100 replicates)
N_JOBS = 8                 # worker processes for the random controls
BATCH = 100                # tasks per resumable batch
OUT = SCRIPT_DIR
MODULE = "E5_observed_scenarios"

_W = {}


def _init_worker():
    """Every worker builds the SAME sanctioned graph through csa_common."""
    B = C.build_all()
    _W["G0"] = C.induced(B["Gu"], B["sets"]["gcc"])
    _W["ps"] = B["ps"]
    _W["pool"] = sorted(_W["G0"].nodes())


def _measure(G0, ps, rm):
    H = G0.copy()
    H.remove_nodes_from(rm)
    e = C.eff_three(H, ps)
    sizes = sorted((len(c) for c in nx.connected_components(H)), reverse=True)
    gcc = sizes[0] if sizes else 0
    alive = H.number_of_nodes()
    return {
        "E_full": e["E_full"], "E_gcc": e["E_gcc"], "E_fixed": e["E_fixed"],
        "gcc": gcc, "n_alive": alive, "n_components": len(sizes),
        "largest_share": gcc / alive if alive else 0.0,
        "second_cc": sizes[1] if len(sizes) > 1 else 0,
        "edges_left": H.number_of_edges(),
        "edges_in_gcc": H.subgraph(max(nx.connected_components(H), key=len)).number_of_edges()
                        if alive else 0,
    }


def _run_task(task):
    """task = (kind, k, rep, seed, tuple(removed nodes))."""
    kind, k, rep, seed, rm = task
    G0, ps = _W["G0"], _W["ps"]
    if rm is None:                       # pure random draw inside the worker
        rg = random.Random(seed)
        rm = set(rg.sample(_W["pool"], k))
    else:
        rm = set(rm)
    r = _measure(G0, ps, rm)
    r.update({"kind": kind, "k_removed": k, "rep": rep, "seed": seed})
    return r


def _parallel(tasks, cache_name=None):
    """Run `tasks` in N_JOBS workers, BATCH tasks at a time.

    cache_name: when given, every finished batch is APPENDED to
    out/E5_observed_scenarios/<cache_name> (plain utf-8, intermediate file) and rows
    already present for the same (kind, k_removed, rep, seed) key are skipped, so an
    interrupted run resumes instead of starting over.  Deleting the cache file forces a
    full recomputation, which is what a fresh checkout does.
    """
    if not tasks:
        return pd.DataFrame()

    def key(t):
        return "%s|%s|%s|%s" % (t[0], t[1], t[2], t[3])

    from multiprocessing import Pool
    cp = os.path.join(OUT, cache_name) if cache_name else None
    done = set()
    if cp and os.path.exists(cp) and os.path.getsize(cp):
        prev = pd.read_csv(cp)
        done = {"%s|%s|%s|%s" % (r.kind, r.k_removed, r.rep, r.seed) for r in prev.itertuples(index=False)}
    todo = [t for t in tasks if key(t) not in done]
    print("  [%s] %d of %d tasks pending (%d cached)"
          % (cache_name or "nocache", len(todo), len(tasks), len(tasks) - len(todo)))
    parts = []
    if todo:
        write_header = not (cp and os.path.exists(cp) and os.path.getsize(cp))
        with Pool(processes=N_JOBS, initializer=_init_worker) as p:
            for i in range(0, len(todo), BATCH):
                chunk = todo[i:i + BATCH]
                t0 = time.time()
                df = pd.DataFrame(p.map(_run_task, chunk, chunksize=2))
                if cp:
                    df.to_csv(cp, mode="a", header=write_header, index=False, encoding="utf-8")
                    write_header = False
                else:
                    parts.append(df)
                print("    batch %4d/%4d  %3d tasks  %6.1fs"
                      % (i // BATCH + 1, (len(todo) + BATCH - 1) // BATCH, len(chunk), time.time() - t0))
    if not cp:
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    alld = pd.read_csv(cp)
    want = {key(t) for t in tasks}
    keep = alld.apply(lambda r: "%s|%s|%s|%s" % (r.kind, r.k_removed, r.rep, r.seed) in want, axis=1)
    return alld[keep].reset_index(drop=True)


def _serial(tasks):
    _init_worker()
    return pd.DataFrame([_run_task(t) for t in tasks])


def run_tasks(tasks, cache_name=None):
    if os.environ.get("CSA_NO_MULTIPROCESS") == "1":
        return _serial(tasks)
    try:
        return _parallel(tasks, cache_name)
    except Exception as exc:                      # pragma: no cover
        print("  [warn] multiprocessing failed (" + repr(exc) + "), falling back to serial")
        return _serial(tasks)


def fmt(x, n=6):
    return str(round(float(x), n))


def main():
    wall = []
    t_all = time.time()

    def tic(label):
        wall.append({"stage": label, "t0": time.time()})

    def toc():
        e = time.time()
        for r in wall:
            if "wall_s" not in r:
                r["wall_s"] = round(e - r["t0"], 3)

    print("E5 - observed scenarios vs same-size random controls")
    print("=" * 100)

    # ------------------------------------------------------------ 00 build
    tic("00 build_all + assert_anchors")
    B = C.build_all()
    bad = C.check_anchors(B)
    n_bad = int((~bad["ok"]).sum())
    print("  assert_anchors: " + str(len(bad)) + " anchors, " + str(n_bad) + " mismatched")
    if n_bad:
        print(bad[~bad["ok"]].to_string(index=False))
    assert n_bad == 0, "CONVENTIONS SS1: anchor mismatch -> stop, write DISSENT.md"

    Gu, Gd = B["Gu"], B["Gd"]
    nodes, fail, ps = B["nodes"], B["fail"], B["ps"]
    G0 = C.induced(Gu, B["sets"]["gcc"])          # THE baseline graph (SS1)
    N0, M0 = G0.number_of_nodes(), G0.number_of_edges()
    print("  Gu " + str(Gu.number_of_nodes()) + "/" + str(Gu.number_of_edges()) +
          "   Gd " + str(Gd.number_of_edges()) + "   baseline main component " +
          str(N0) + "/" + str(M0))
    assert (N0, M0) == (C.ANCHORS["gcc_size"], 2497), "baseline graph is not 674/2497"
    assert ps.denom == N0 * (N0 - 1) == 453602.0, "PairSet denominator drifted"
    base = _measure(G0, ps, set())
    print("  baseline E_full=" + fmt(base["E_full"]) + " E_gcc=" + fmt(base["E_gcc"]) +
          " E_fixed=" + fmt(base["E_fixed"]) + "  components=" + str(base["n_components"]))
    # contract SS2.3 reference table cross-check
    assert abs(base["E_full"] - 0.3315) < 5e-3 and abs(base["E_fixed"] - 0.3315) < 5e-3
    toc()

    # ------------------------------------------------- 01 labelled sets
    tic("01 C.failure_sets (SS2.2 labels)")
    for k in sorted(fail):
        print("  fail[" + k.ljust(18) + "] = " + str(len(fail[k])).rjust(3) +
              "  inside baseline " + str(len(fail[k] & set(G0.nodes()))).rjust(3))
    toc()

    # ------------------------------- 02 independent verification of counts
    tic("02 verify counts from data/dead_categories.csv")
    dc = B["dead_cat"]
    dcs = pd.read_csv(os.path.join(C.CLEAN_DIR, "dead_categories.csv"))
    assert len(dc) == len(dcs) == 162
    cat_counts = dc["cat"].value_counts().to_dict()
    print("  dead_categories rows=" + str(len(dc)) + " by cat: " + str(cat_counts))
    true_set = set(dc.loc[dc["cat"].isin(C.TRUE_FAIL_CATS), "node"])
    false_set = set(dc.loc[dc["cat"].isin(C.FALSE_FAIL_CATS), "node"])
    amb_set = set(dc.loc[dc["cat"].isin(C.AMBIGUOUS_CATS), "node"])
    print("  recomputed true=" + str(len(true_set)) + " false=" + str(len(false_set)) +
          " ambiguous=" + str(len(amb_set)) + "  sum=" +
          str(len(true_set) + len(false_set) + len(amb_set)))
    assert true_set == fail["true_fail"], "true-failure set differs from csa_common"
    assert false_set == fail["false_fail"], "false-failure set differs from csa_common"
    assert amb_set == fail["ambiguous"], "ambiguous set differs from csa_common"
    assert len(true_set | false_set | amb_set) == 162 == len(fail["dead_all"])
    # partial failure classification comes from nodes_full.csv status_codes, re-derived
    pn = nodes[nodes["is_partial"]]
    print("  partial sites recomputed=" + str(len(pn)) + " csa_common=" + str(len(fail["partial_all"])))
    assert set(pn["node"]) == fail["partial_all"]
    pclean = set(nodes.loc[nodes["partial_clean"], "node"])
    pauth = set(nodes.loc[nodes["partial_has_auth_like"], "node"])
    assert pclean | pauth == fail["partial_all"] and not (pclean & pauth)
    print("  partial_clean=" + str(len(pclean)) + "  partial_auth_like=" + str(len(pauth)) +
          "  (partition of partial149: " + str(len(pclean) + len(pauth)) + "=149)")

    sets = {
        "dead162": fail["dead_all"],
        "true129": fail["true_fail"],
        "false32": fail["false_fail"],
        "ambiguous1": fail["ambiguous"],
        "partial149": fail["partial_all"],
        "partial_clean80": pclean,
        "partial_authlike69": pauth,
        "combo209": fail["true_fail"] | pclean,
        "combo311": fail["dead_all"] | fail["partial_all"],
    }
    labels = {"dead162": 162, "true129": 129, "false32": 32, "ambiguous1": 1,
              "partial149": 149, "partial_clean80": 80, "partial_authlike69": 69,
              "combo209": 209, "combo311": 311}
    mismatches = []
    for name, s in sets.items():
        ok = len(s) == labels[name]
        print("  " + name.ljust(20) + " label=" + str(labels[name]).rjust(3) +
              " verified=" + str(len(s)).rjust(3) + " in-baseline=" +
              str(len(s & set(G0.nodes()))).rjust(3) + "  match=" + str(ok))
        if not ok:
            mismatches.append((name, labels[name], len(s)))
    assert all(len(s & set(G0.nodes())) == len(s) for s in sets.values()), \
        "a scenario touches nodes outside the baseline component"
    toc()

    # ------------------------------------------------- 03 summary csv
    tic("03 scenario_summary.csv")
    codes_by_node = nodes.set_index("node")["status_codes"].to_dict()
    cat_by_node = dc.set_index("node")["cat"].to_dict()
    rows_s = []
    for name, s in sets.items():
        nt = len(s & fail["true_fail"])
        nf = len(s & fail["false_fail"])
        na = len(s & fail["ambiguous"])
        npart = len(s & fail["partial_all"])
        rows_s.append({
            "scenario": name, "label_count": labels[name], "verified_count": len(s),
            "count_matches_label": len(s) == labels[name],
            "n_in_baseline": len(s & set(G0.nodes())),
            "pct_of_baseline_nodes": round(100.0 * len(s) / N0, 3),
            "n_true_fail": nt, "n_false_fail": nf, "n_ambiguous": na,
            "n_partial": npart, "n_dead": len(s & fail["dead_all"]),
            "true_share_pct": round(100.0 * nt / len(s), 2),
            "false_share_pct": round(100.0 * nf / len(s), 2),
            "is_all_true_failure": bool(nf == 0 and na == 0),
            "k_after_SS22_correction": len(s - false_set),
        })
    summ = pd.DataFrame(rows_s).sort_values("verified_count").reset_index(drop=True)
    C.save_csv(summ, MODULE, "scenario_summary.csv")
    print(summ.to_string(index=False))
    toc()

    # ------------------------------------------------- 04 random controls
    tic("04 random controls (" + str(N_RANDOM) + " reps x every needed k)")
    ks = sorted({len(s) for s in sets.values()} |
                {len(s - false_set) for s in sets.values()})
    ks = [k for k in ks if k > 0]
    print("  needed k values: " + str(ks))
    tasks = []
    for k in ks:
        rg = random.Random(SEED + 5000 + k)          # one explicit stream per k
        for rep in range(N_RANDOM):
            rm = frozenset(rg.sample(sorted(G0.nodes()), k))
            tasks.append(("random_same_k", k, rep, SEED + 5000 + k, rm))
    # restricted-pool nulls: is the WITHIN-CLASS selection of the observed damage
    # worse than picking the same number of sites at random inside that same class?
    restricted = [
        ("true129_within_dead162", fail["dead_all"], len(fail["true_fail"])),
        ("pclean80_within_partial149", fail["partial_all"], len(pclean)),
        ("pauth69_within_partial149", fail["partial_all"], len(pauth)),
        ("combo209_within_combo311", sets["combo311"], len(sets["combo209"])),
    ]
    for name, pool_s, k in restricted:
        rg = random.Random(SEED + 91000 + k + 7 * len(name))
        for rep in range(N_RANDOM):
            rm = frozenset(rg.sample(sorted(pool_s), k))
            tasks.append(("poolnull_" + name, k, rep, SEED + 91000 + k + 7 * len(name), rm))
    t0 = time.time()
    nulls = run_tasks(tasks, "_cache_random_controls.csv")
    nulls = nulls.sort_values(["kind", "rep", "k_removed"]).reset_index(drop=True)
    C.save_csv(nulls, MODULE, "random_controls.csv")
    print("  " + str(len(tasks)) + " removals evaluated in " +
          str(round(time.time() - t0, 1)) + "s")
    rnd = nulls[nulls["kind"] == "random_same_k"]
    stats = rnd.groupby("k_removed").agg(
        reps=("E_fixed", "size"), ef_mean=("E_fixed", "mean"), ef_sd=("E_fixed", "std"),
        ef_min=("E_fixed", "min"), ef_max=("E_fixed", "max"),
        efull_mean=("E_full", "mean"), efull_sd=("E_full", "std"),
        egcc_mean=("E_gcc", "mean"), gcc_mean=("gcc", "mean"), gcc_sd=("gcc", "std"),
        ncomp_mean=("n_components", "mean"), ncomp_sd=("n_components", "std"),
        eigr_mean=("edges_in_gcc", "mean")).reset_index()
    C.save_csv(stats, MODULE, "random_control_summary.csv")
    print(stats.to_string(index=False))
    assert rnd.groupby("k_removed").size().min() >= 100, "SS4: fewer than 100 replicates"
    toc()

    # ------------------------------------------- 05 scenario + attack curves
    tic("05 scenario metrics + node-attack reference curve")
    scen_tasks, attack_tasks, pair_tasks = [], [], []
    order = sorted(sets, key=lambda n: len(sets[n]))
    for name in order:
        s = sets[name]
        for mode, rm in (("as_observed", s), ("corrected_SS22", s - false_set)):
            scen_tasks.append((name + "|" + mode, len(rm), 0, SEED, frozenset(rm)))
    # E1-style reference: one static attack per checkpoint, all three口径
    deg = dict(G0.degree())
    bet = nx.betweenness_centrality(G0)
    pr_dw = nx.pagerank(Gd, weight="weight")
    pr_uu = nx.pagerank(G0)
    st = dict(G0.degree(weight="weight"))
    ranking = {
        "attack_random": sorted(G0.nodes(), key=lambda x: (random.Random(SEED + 1).random(), x)),
        "attack_degree": sorted(G0.nodes(), key=lambda x: (-deg[x], x)),
        "attack_betweenness": sorted(G0.nodes(), key=lambda x: (-bet[x], x)),
        "attack_pagerank_directed_weighted": sorted(G0.nodes(), key=lambda x: (-pr_dw.get(x, 0.0), x)),
        "attack_pagerank_undirected_binary": sorted(G0.nodes(), key=lambda x: (-pr_uu.get(x, 0.0), x)),
        "attack_strength": sorted(G0.nodes(), key=lambda x: (-st[x], x)),
    }
    checkpoints = sorted({max(1, int(round(N0 * fr))) for fr in
                          [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
                           0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]})
    for meth, order_nodes in ranking.items():
        for ck in checkpoints:
            attack_tasks.append((meth, ck, 0, SEED, frozenset(order_nodes[:ck])))
    # cross-check against the CONVENTIONS SS2.3 reference table (equal-k pairs)
    for k in (162, 129, 311, 209, 149, 80):
        pool = sorted(G0.nodes())
        rg = random.Random(SEED + 4242 + k)
        for rep in range(20):
            pair_tasks.append(("contract_sanity_k" + str(k), k, rep, SEED + 4242 + k,
                               frozenset(rg.sample(pool, k))))
    res = pd.concat([run_tasks(scen_tasks, "_cache_scenario_attacks.csv"),
                     run_tasks(attack_tasks, "_cache_scenario_attacks.csv"),
                     run_tasks(pair_tasks, "_cache_scenario_attacks.csv")], ignore_index=True)
    toc()

    # ---------------------------------------------------- 06 metrics table
    tic("06 scenario_metrics.csv")
    srows = []
    for name in order:
        s = sets[name]
        for mode in ("as_observed", "corrected_SS22"):
            key = name + "|" + mode
            r = res[res["kind"] == key].iloc[0]
            k = int(r["k_removed"])
            nl = rnd[rnd["k_removed"] == k]
            if k == 0:
                nl = pd.DataFrame({"E_fixed": [base["E_fixed"]], "gcc": [N0],
                                   "E_full": [base["E_full"]], "E_gcc": [base["E_gcc"]],
                                   "n_components": [1]})
            mn, sd = nl.E_fixed.mean(), (nl.E_fixed.std(ddof=1) if len(nl) > 1 else 0.0)
            z = (r["E_fixed"] - mn) / sd if sd > 0 else np.nan
            sn = nulls[(nulls["kind"] == "poolnull_" + name)]
            sn = sn[sn["k_removed"] == k] if len(sn) else sn
            srows.append({
                "scenario": name, "view": mode, "k_removed": k,
                "E_full": r["E_full"], "E_gcc": r["E_gcc"], "E_fixed": r["E_fixed"],
                "E_fixed_ratio_vs_baseline": r["E_fixed"] / base["E_fixed"],
                "E_fixed_ratio_vs_random_same_k": (r["E_fixed"] / mn) if mn else np.nan,
                "rel_damage_vs_random_pct": 100.0 * (1.0 - r["E_fixed"] / mn) if mn else np.nan,
                "rand_mean_E_fixed": mn, "rand_sd_E_fixed": sd,
                "z_vs_random_same_k": z,
                "emp_pctl_vs_random_same_k": float((nl.E_fixed <= r["E_fixed"]).mean() * 100.0),
                "within_scenario_rand_mean": (sn.E_fixed.mean() if len(sn) else np.nan),
                "within_scenario_rand_sd": (sn.E_fixed.std(ddof=1) if len(sn) else np.nan),
                "gcc": int(r["gcc"]), "gcc_ratio_vs_N0": r["gcc"] / N0,
                "n_components": int(r["n_components"]),
                "largest_share_pct": 100.0 * r["largest_share"],
                "second_cc": int(r["second_cc"]),
                "edges_in_gcc": int(r["edges_in_gcc"]), "edges_left": int(r["edges_left"]),
                "dE_full_pct": 100.0 * (r["E_full"] - base["E_full"]) / base["E_full"],
                "dE_gcc_pct": 100.0 * (r["E_gcc"] - base["E_gcc"]) / base["E_gcc"],
                "dE_fixed_pct": 100.0 * (r["E_fixed"] - base["E_fixed"]) / base["E_fixed"],
                "E_full_rises": bool(r["E_full"] > base["E_full"]),
                "artifact_gap_Efull_minus_Efixed_pct_points":
                    100.0 * ((r["E_full"] - base["E_full"]) - (r["E_fixed"] - base["E_fixed"]))
                    / base["E_fixed"],
                "n_random_reps_at_k": int(len(nl)),
                "significant_at_0.05": bool(abs(z) > 1.96) if sd > 0 else False,
                "conclusion_read_from": "E_fixed",
            })
    met = pd.DataFrame(srows)
    for c in met.columns:
        if met[c].dtype.kind == "f":
            met[c] = met[c].round(6)
    C.save_csv(met, MODULE, "scenario_metrics.csv")
    print(met[["scenario", "view", "k_removed", "E_full", "E_gcc", "E_fixed",
               "rand_mean_E_fixed", "z_vs_random_same_k", "rel_damage_vs_random_pct",
               "gcc", "n_components", "E_full_rises"]].to_string(index=False))
    toc()

    # ------------------------------------------ 07 before/after SS2.2 pair
    tic("07 before_after_correction.csv")
    brows = []
    for name in order:
        a = met[(met.scenario == name) & (met.view == "as_observed")].iloc[0]
        b = met[(met.scenario == name) & (met.view == "corrected_SS22")].iloc[0]

        def verdict(r):
            if r["k_removed"] == 0:
                return "NOTHING_REMOVED"
            if not np.isfinite(r["z_vs_random_same_k"]) or abs(r["z_vs_random_same_k"]) <= 1.96:
                return "indistinguishable_from_random"
            return "worse_than_random" if r["z_vs_random_same_k"] < 0 else "milder_than_random"

        va, vb = verdict(a), verdict(b)
        brows.append({
            "scenario": name, "k_as_observed": int(a["k_removed"]), "k_corrected": int(b["k_removed"]),
            "n_false_failures_restored": int(a["k_removed"] - b["k_removed"]),
            "E_fixed_before": a["E_fixed"], "E_fixed_after": b["E_fixed"],
            "rand_mean_E_fixed_at_k_before": a["rand_mean_E_fixed"],
            "rand_mean_E_fixed_at_k_after": b["rand_mean_E_fixed"],
            "z_before": a["z_vs_random_same_k"], "z_after": b["z_vs_random_same_k"],
            "pctl_before": a["emp_pctl_vs_random_same_k"], "pctl_after": b["emp_pctl_vs_random_same_k"],
            "rel_damage_vs_random_pct_before": a["rel_damage_vs_random_pct"],
            "rel_damage_vs_random_pct_after": b["rel_damage_vs_random_pct"],
            "E_full_before": a["E_full"], "E_full_after": b["E_full"],
            "E_full_rises_before": bool(a["E_full_rises"]), "E_full_rises_after": bool(b["E_full_rises"]),
            "gcc_before": int(a["gcc"]), "gcc_after": int(b["gcc"]),
            "n_components_before": int(a["n_components"]), "n_components_after": int(b["n_components"]),
            "verdict_before": va, "verdict_after": vb, "verdict_changed": va != vb,
            "note": "k differs before/after, so the two z scores are against DIFFERENT same-k nulls (SS2.3)",
        })
    bdf = pd.DataFrame(brows)
    for c in bdf.columns:
        if bdf[c].dtype.kind == "f":
            bdf[c] = bdf[c].round(6)
    C.save_csv(bdf, MODULE, "before_after_correction.csv")
    print(bdf[["scenario", "k_as_observed", "k_corrected", "E_fixed_before", "E_fixed_after",
               "z_before", "z_after", "rel_damage_vs_random_pct_before",
               "rel_damage_vs_random_pct_after", "verdict_before", "verdict_after",
               "verdict_changed"]].to_string(index=False))
    toc()

    # ------------------------------------------ 08 damage vs selection curve
    tic("08 damage_vs_selection_strength.csv")
    _init_worker()                       # serial evaluation in the parent process
    drows = []
    rg = random.Random(SEED + 31337)
    dead_only = fail["dead_all"] - false_set                  # 130 true+ambiguous
    grad = sorted(G0.nodes(), key=lambda x: (rg.random(), x))
    for frac in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        n_from_dead = int(round(frac * len(dead_only)))
        n_from_rand = 162 - n_from_dead
        picks = []
        if n_from_dead:
            rg2 = random.Random(SEED + 40000 + n_from_dead)
            picks += rg2.sample(sorted(dead_only), n_from_dead)
        rest = [x for x in grad if x not in dead_only]
        if n_from_rand:
            rg3 = random.Random(SEED + 41000 + n_from_rand)
            picks += rg3.sample(sorted(set(rest) - set(picks)), n_from_rand)
        rr = _run_task(("grad", 162, 0, SEED, frozenset(picks)))
        drows.append({"share_true_failure_pct": round(100.0 * frac, 1),
                      "k_total": 162, "k_true_failures": n_from_dead,
                      "k_alive_random": n_from_rand,
                      "E_fixed": rr["E_fixed"], "E_full": rr["E_full"], "E_gcc": rr["E_gcc"],
                      "gcc": rr["gcc"], "n_components": rr["n_components"]})
    dtab = pd.DataFrame(drows)
    dtab["rel_damage_vs_random_same_k_pct"] = 100.0 * (1.0 - dtab.E_fixed / rnd[rnd.k_removed == 162].E_fixed.mean())
    dtab["emp_pctl_vs_random_same_k"] = [float((rnd[rnd.k_removed == 162].E_fixed <= v).mean() * 100) for v in dtab.E_fixed]
    for c in dtab.columns:
        if dtab[c].dtype.kind == "f":
            dtab[c] = dtab[c].round(6)
    C.save_csv(dtab, MODULE, "damage_vs_selection_strength.csv")
    print(dtab.to_string(index=False))
    toc()

    # ------------------------------------- 09 contract table cross-check
    tic("09 CONVENTIONS SS2.3 anchor cross-check")
    ck = []
    ref = {162: 0.1602, 129: None, 311: None, 209: None, 149: None, 80: None}
    for k in (162, 129, 311, 209, 149, 80):
        sub = res[res["kind"] == "contract_sanity_k" + str(k)]
        ck.append({"check": "random_same_k_reproduction", "k": k,
                   "our_mean_E_fixed": round(sub.E_fixed.mean(), 6),
                   "our_sd": round(sub.E_fixed.std(ddof=1), 6),
                   "contract_value": ref.get(k), "n_reps": len(sub)})
    obs = {n: float(met[(met.scenario == n) & (met.view == "as_observed")].E_fixed.iloc[0])
           for n in ("dead162", "combo311", "combo209", "partial149", "partial_clean80")}
    contract_obs = {"dead162": 0.2039, "combo311": 0.0901, "combo209": 0.1533}
    for n, v in contract_obs.items():
        ck.append({"check": "scenario_E_fixed_vs_contract", "k": len(sets[n]),
                   "our_mean_E_fixed": round(obs[n], 6), "contract_value": v,
                   "abs_diff": round(abs(obs[n] - v), 6), "n_reps": 1})
    ckd = pd.DataFrame(ck)
    C.save_csv(ckd, MODULE, "contract_crosscheck.csv")
    print(ckd.to_string(index=False))
    toc()

    # ------------------------------------------------------ 10 attack table
    tic("10 attack_reference_curve.csv")
    arows = []
    for meth in ranking:
        sub = res[res["kind"] == meth].sort_values("k_removed")
        for _, r in sub.iterrows():
            k = int(r["k_removed"])
            nl = rnd[rnd["k_removed"] == k]
            arows.append({"strategy": meth, "k_removed": k, "k_pct": 100.0 * k / N0,
                          "E_full": r["E_full"], "E_gcc": r["E_gcc"], "E_fixed": r["E_fixed"],
                          "gcc": int(r["gcc"]), "n_components": int(r["n_components"]),
                          "edges_in_gcc": int(r["edges_in_gcc"]),
                          "rand_mean_E_fixed_same_k": nl.E_fixed.mean() if len(nl) else np.nan,
                          "rand_sd_E_fixed_same_k": nl.E_fixed.std(ddof=1) if len(nl) else np.nan,
                          "z_vs_random_same_k": ((r["E_fixed"] - nl.E_fixed.mean()) / nl.E_fixed.std(ddof=1))
                          if len(nl) > 1 and nl.E_fixed.std(ddof=1) > 0 else np.nan})
    at = pd.DataFrame(arows)
    for c in at.columns:
        if at[c].dtype.kind == "f":
            at[c] = at[c].round(6)
    C.save_csv(at, MODULE, "attack_reference_curve.csv")
    print(at.pivot(index="k_pct", columns="strategy", values="E_fixed").round(4).to_string())
    toc()

    elapsed = time.time() - t_all
    print("")
    print("=" * 100)
    print("ELAPSED " + str(round(elapsed, 2)) + "s   seed=" + str(SEED) +
          "   reps/k=" + str(N_RANDOM))
    for r in wall:
        print("  " + r["stage"].ljust(56) + str(r["wall_s"]).rjust(9) + "s")
    lines = ["# RUN.txt written by run_e5.py", "command: PYTHONHASHSEED=0 python out/E5_observed_scenarios/run_e5.py",
             "seed: " + str(SEED), "n_random_reps_per_k: " + str(N_RANDOM),
             "elapsed_s: " + str(round(elapsed, 2)), "versions: " + str(C.versions())]
    for r in wall:
        lines.append("stage " + r["stage"] + " wall_s=" + str(r["wall_s"]))
    C.save_csv(pd.DataFrame(wall), MODULE, "stage_walltimes.csv")
    with open(os.path.join(OUT, "RUN.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("RUN.txt written")
    return {"mismatches": mismatches}


if __name__ == "__main__":
    main()
