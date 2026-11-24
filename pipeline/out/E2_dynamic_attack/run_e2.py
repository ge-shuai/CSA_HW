"""E2_dynamic_attack — 动态移除（每删一个点重算中心性）与静态口径的对照。

契约依据：
  §6  介数必须补一版动态（每删一个重算），动态更狠，作为稳健性对照
  §2.3 效率主口径 E_fixed（固定点对分母），只在同一 k 下跨策略比
  §5  PageRank 两个变体分列；动态 PageRank 只有在预算内才跑，否则显式声明超预算
  §1  基准图 = 674 节点 / 2497 边主分量
效率只用 C.eff_three / C.PairSet。静态参照由本脚本自行重算并与 E1 的 CSV 交叉校验。

重跑：cd pipeline && PYTHONHASHSEED=0 python out/E2_dynamic_attack/run_e2.py
"""
from __future__ import annotations

import os
import sys
import time
import json
import warnings
from multiprocessing import Pool

import networkx as nx
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEED = 20261113

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE = os.path.dirname(os.path.dirname(HERE))
E1_DIR = os.path.join(PIPELINE, "out", "E1_attack_curves")
if PIPELINE not in sys.path:
    sys.path.insert(0, PIPELINE)
import csa_common as C  # noqa: E402

F_GRID = [round(0.02 * i, 2) for i in range(1, 31)]
SUMMARY_F = [0.05, 0.10, 0.20, 0.30, 0.50]
CHECK_F = sorted(set(F_GRID) | set(SUMMARY_F))
MAX_F = F_GRID[-1]
DYN_STRATEGIES = ["betweenness", "degree", "kcore", "pagerank_dw", "pagerank_uu"]
LABEL = {
    "betweenness": "介数（每步重算）",
    "degree": "度（每步重算）",
    "kcore": "k-core（每步重算）",
    "pagerank_dw": "PageRank 有向加权（每步重算）【§5 主口径】",
    "pagerank_uu": "PageRank 无权无向（每步重算）【§5 对照】",
}
# 预算：单个动态策略超过 BUDGET_S 秒则截断并**显式声明**（§6 要求，不得静默跳过）。
# 与葛帅那套"每删一步重算"的算法同复杂度；674 节点上纯重算 CPU 只有几十~几百秒，
# 这里给 1800 s 墙钟是因为本机同时有其它模块的进程在抢核（RUN.txt 记录实测）。
BUDGET_S = 1800.0
_WS: dict = {}


# ------------------------------------------------------------------ 基准图
def base_graphs() -> dict:
    b = C.build_all()
    C.assert_anchors(b)
    gcc = set(b["sets"]["gcc"])
    return {
        "H0": C.induced(b["Gb"], gcc),
        "Hd0": C.induced(b["Gd"], gcc),
        "ps": b["ps"],
        "n0": len(gcc),
        "nodes": sorted(gcc),
    }


def check_grid(n0: int):
    cs = {int(round(f * n0)) for f in CHECK_F}
    nominal = {}
    for f in CHECK_F:
        nominal.setdefault(int(round(f * n0)), f)
    return cs, nominal


# ------------------------------------------------------------------ 打分函数
def safe_pagerank(G, **kw):
    """图删到碎片化后幂迭代可能不收敛 -> 先试 200 轮，再退到 2000 轮 / 松容差。"""
    try:
        return nx.pagerank(G, max_iter=200, **kw)
    except nx.PowerIterationFailedConvergence:
        return nx.pagerank(G, max_iter=2000, tol=1e-4, **kw)


def scorer(strategy: str):
    if strategy == "betweenness":
        return lambda H, Hd: nx.betweenness_centrality(H)
    if strategy == "degree":
        return lambda H, Hd: dict(H.degree())
    if strategy == "kcore":
        return lambda H, Hd: {k: float(v) for k, v in nx.core_number(H).items()}
    if strategy == "pagerank_dw":
        return lambda H, Hd: safe_pagerank(Hd, weight="weight")     # §5 主口径
    if strategy == "pagerank_uu":
        return lambda H, Hd: safe_pagerank(H)                        # §5 对照
    raise ValueError(strategy)


def static_order(strategy: str, B) -> list:
    """一次性排序（静态口径，§6 主口径），平手按 FQDN 升序。"""
    val = scorer(strategy)(B["H0"], B["Hd0"])
    return sorted(val, key=lambda n: (-val[n], n))


# ------------------------------------------------------------------ 度量快照
def checkpoint(H, comps_desc, ps, k, f_nominal):
    g = len(comps_desc[0]) if comps_desc else 0
    e = C.eff_three(H, ps)
    if g >= 2:
        apl = float(nx.average_shortest_path_length(C.largest_component(H)))
    else:
        apl = float("nan")
    return {"step": k, "f_nominal": f_nominal, "f_actual": k / ps.n0,
            "n_remaining": H.number_of_nodes(), "gcc_size": g,
            "gcc_pct": 100.0 * g / ps.n0, "n_components": len(comps_desc),
            "E_full": e["E_full"], "E_gcc": e["E_gcc"], "E_fixed": e["E_fixed"],
            "avg_pathlen_gcc": apl}


def run_dynamic(strategy, H0, Hd0, cs, nominal, ps, n0):
    """动态：每删 1 个点重算中心性，取当前分数最大的点（平手按 FQDN 升序）。"""
    H = H0.copy()
    Hd = Hd0.copy() if Hd0 is not None else None
    sf = scorer(strategy)
    maxk = int(round(MAX_F * n0))
    steps, checks = [], []
    t_score = 0.0
    comps = sorted(nx.connected_components(H), key=len, reverse=True)
    checks.append(checkpoint(H, comps, ps, 0, 0.0))
    t0 = time.perf_counter()
    for k in range(1, maxk + 1):
        t = time.perf_counter()
        val = sf(H, Hd)
        t_score += time.perf_counter() - t
        victim = min(val, key=lambda n: (-val[n], n))
        H.remove_node(victim)
        if Hd is not None:
            Hd.remove_node(victim)
        comps = sorted(nx.connected_components(H), key=len, reverse=True)
        g = len(comps[0]) if comps else 0
        steps.append({"step": k, "n_remaining": H.number_of_nodes(), "gcc_size": g,
                      "gcc_pct": 100.0 * g / n0, "n_components": len(comps)})
        if k in cs:
            checks.append(checkpoint(H, comps, ps, k, nominal.get(k, k / n0)))
        if time.perf_counter() - t0 > BUDGET_S:
            checks.append(checkpoint(H, comps, ps, k, nominal.get(k, k / n0)))
            return steps, checks, t_score, time.perf_counter() - t0, k, "OVER_BUDGET"
    return steps, checks, t_score, time.perf_counter() - t0, maxk, "complete"


def run_static(strategy, H0, Hd0, cs, nominal, ps, n0):
    """静态参照：只排一次序（§6 主口径），随后按前缀逐个删。"""
    H = H0.copy()
    order = static_order(strategy, {"H0": H0, "Hd0": Hd0})
    maxk = int(round(MAX_F * n0))
    steps, checks = [], []
    comps = sorted(nx.connected_components(H), key=len, reverse=True)
    checks.append(checkpoint(H, comps, ps, 0, 0.0))
    t0 = time.perf_counter()
    for k, victim in zip(range(1, maxk + 1), order):
        H.remove_node(victim)
        comps = sorted(nx.connected_components(H), key=len, reverse=True)
        g = len(comps[0]) if comps else 0
        steps.append({"step": k, "n_remaining": H.number_of_nodes(), "gcc_size": g,
                      "gcc_pct": 100.0 * g / n0, "n_components": len(comps)})
        if k in cs:
            checks.append(checkpoint(H, comps, ps, k, nominal.get(k, k / n0)))
    return steps, checks, time.perf_counter() - t0


# ------------------------------------------------------------------ worker
def _init_worker():
    _WS.clear()
    _WS.update(base_graphs())


def _job(arg):
    kind, strategy = arg
    B = _WS
    n0, ps = B["n0"], B["ps"]
    cs, nominal = check_grid(n0)
    if kind == "dynamic":
        steps, checks, t_score, wall, lastk, status = run_dynamic(
            strategy, B["H0"], B["Hd0"] if strategy.startswith("pagerank") else None,
            cs, nominal, ps, n0)
        for c in checks:
            c["strategy"], c["mode"] = strategy, "dynamic"
        for s in steps:
            s["strategy"], s["mode"] = strategy, "dynamic"
        return kind, strategy, steps, checks, {
            "wall_s": round(wall, 1), "score_only_s": round(t_score, 1),
            "n_recomputations": lastk, "last_step": lastk, "status": status}
    steps, checks, wall = run_static(strategy, B["H0"], B["Hd0"], cs, nominal, ps, n0)
    for c in checks:
        c["strategy"], c["mode"] = strategy, "static"
    for s in steps:
        s["strategy"], s["mode"] = strategy, "static"
    return kind, strategy, steps, checks, {"wall_s": round(wall, 1),
                                           "rank_s": round(0.0, 1), "status": "complete"}


# ------------------------------------------------------------------ 对照表
def gap_table(all_checks, n0):
    """同 f 静态 vs 动态：GCC% 差、E_fixed 差（§2.3 只允许同 k 比较）。"""
    idx = {}
    for c in all_checks:
        idx[(c["strategy"], c["mode"], c["step"])] = c
    rows = []
    for s in DYN_STRATEGIES:
        steps = sorted({c["step"] for c in all_checks
                        if c["strategy"] == s and c["mode"] == "dynamic"})
        for k in steps:
            a = idx.get((s, "static", k))
            d = idx.get((s, "dynamic", k))
            if a is None or d is None:
                continue
            rows.append({
                "strategy": s, "step": k, "f_actual": k / n0, "f_nominal": d["f_nominal"],
                "gcc_pct_static": a["gcc_pct"], "gcc_pct_dynamic": d["gcc_pct"],
                "gcc_gap_pp": a["gcc_pct"] - d["gcc_pct"],
                "gcc_ratio_dyn_over_stat": (d["gcc_pct"] / a["gcc_pct"]) if a["gcc_pct"] > 0 else np.nan,
                "n_comp_static": a["n_components"], "n_comp_dynamic": d["n_components"],
                "E_fixed_static": a["E_fixed"], "E_fixed_dynamic": d["E_fixed"],
                "E_fixed_gap": a["E_fixed"] - d["E_fixed"],
                "E_fixed_gap_rel_pct": (100.0 * (a["E_fixed"] - d["E_fixed"]) / a["E_fixed"]
                                        if a["E_fixed"] > 0 else np.nan),
                "E_gcc_static": a["E_gcc"], "E_gcc_dynamic": d["E_gcc"],
                "E_full_static": a["E_full"], "E_full_dynamic": d["E_full"],
                "avg_pathlen_static": a["avg_pathlen_gcc"], "avg_pathlen_dynamic": d["avg_pathlen_gcc"],
            })
    return pd.DataFrame(rows)


def main():
    t_all = time.perf_counter()
    print("[E2] anchors + build ...", flush=True)
    B = base_graphs()
    n0, ps = B["n0"], B["ps"]
    if n0 != C.ANCHORS["gcc_size"]:
        raise AssertionError("基准图不是 674 主分量")
    cs, nominal = check_grid(n0)
    print(f"[E2] n0={n0} pairset={ps!r} checkpoints={len(cs)} max_step={max(cs)}", flush=True)

    jobs = [("dynamic", s) for s in DYN_STRATEGIES] + [("static", s) for s in DYN_STRATEGIES]
    t = time.perf_counter()
    results, stage = [], {}
    with Pool(min(10, len(jobs)), initializer=_init_worker) as pool:
        for res in pool.imap_unordered(_job, jobs, chunksize=1):
            results.append(res)
            stage[f"{res[0]}_{res[1]}"] = res[4]
            print(f"[E2] {res[0]:7s} {res[1]:12s} done in {res[4]['wall_s']}s {res[4].get('status','')}",
                  flush=True)
    wall_pool = time.perf_counter() - t

    all_checks, all_steps = [], []
    for kind, strategy, steps, checks, info in sorted(results, key=lambda r: (r[0], r[1])):
        all_checks += checks
        if kind == "dynamic":
            all_steps += steps

    # ---- 与 E1 静态曲线交叉校验
    xcheck = {"file": None}
    e1p = os.path.join(E1_DIR, "e1_curve_gcc_per_step_deterministic.csv")
    if os.path.exists(e1p):
        e1 = pd.read_csv(e1p)
        diffs = []
        for s in DYN_STRATEGIES:
            a = e1[e1["strategy"] == s].set_index("step")["gcc_size"]
            b_ = pd.DataFrame([c for c in all_checks if c["strategy"] == s and c["mode"] == "static"]
                              ).set_index("step")["gcc_size"]
            j = a.index.intersection(b_.index)
            if len(j):
                diffs.append({"strategy": s,
                              "max_abs_gcc_diff": float((a.loc[j] - b_.loc[j]).abs().max())})
        xcheck = {"file": e1p, "max_abs_gcc_diff_nodes": int(max(d["max_abs_gcc_diff"] for d in diffs)),
                  "per_strategy": diffs}
    print("[E2] static cross-check vs E1:", json.dumps(xcheck), flush=True)

    gap = gap_table(all_checks, n0)
    # 最大差距及其出现档位（同 f 比较，§2.3 硬规则 2）
    heads = []
    for s in DYN_STRATEGIES:
        sub = gap[gap["strategy"] == s]
        if not len(sub):
            continue
        i = sub["gcc_gap_pp"].idxmax()
        j = sub["E_fixed_gap"].idxmax()
        heads.append({
            "strategy": s,
            "max_gcc_gap_pp": round(float(sub.loc[i, "gcc_gap_pp"]), 2),
            "at_f": round(float(sub.loc[i, "f_actual"]), 4),
            "gcc_static_pct_at_max_gap": round(float(sub.loc[i, "gcc_pct_static"]), 2),
            "gcc_dynamic_pct_at_max_gap": round(float(sub.loc[i, "gcc_pct_dynamic"]), 2),
            "max_E_fixed_gap": round(float(sub.loc[j, "E_fixed_gap"]), 4),
            "E_fixed_gap_at_f": round(float(sub.loc[j, "f_actual"]), 4),
            "E_fixed_static": round(float(sub.loc[j, "E_fixed_static"]), 4),
            "E_fixed_dynamic": round(float(sub.loc[j, "E_fixed_dynamic"]), 4),
            "mean_gcc_gap_pp_over_grid": round(float(sub["gcc_gap_pp"].mean()), 2),
        })

    # ---- 落盘
    eff_cols = ["strategy", "mode", "step", "f_nominal", "f_actual", "n_remaining", "gcc_size",
                "gcc_pct", "n_components", "E_full", "E_gcc", "E_fixed", "avg_pathlen_gcc"]
    pd.DataFrame(all_checks)[eff_cols].round(6).to_csv(
        os.path.join(HERE, "e2_curve_efficiency.csv"), index=False, encoding="utf-8-sig")
    stp = pd.DataFrame(all_steps)
    stp["f_actual"] = stp["step"] / n0
    stp[["strategy", "mode", "step", "f_actual", "n_remaining", "gcc_size", "gcc_pct",
         "n_components"]].round(6).to_csv(os.path.join(HERE, "e2_gcc_per_step.csv"),
                                          index=False, encoding="utf-8-sig")
    gap.round(6).to_csv(os.path.join(HERE, "e2_static_vs_dynamic_gap.csv"), index=False,
                        encoding="utf-8-sig")
    pd.DataFrame(heads).to_csv(os.path.join(HERE, "e2_headline_gap.csv"), index=False,
                               encoding="utf-8-sig")

    meta = {
        "seed": SEED, "n0": n0, "pairset": repr(ps), "check_f": CHECK_F, "max_step": max(cs),
        "strategies": DYN_STRATEGIES, "labels": LABEL, "budget_s": BUDGET_S,
        "pool_wall_s": round(wall_pool, 1), "total_wall_s": round(time.perf_counter() - t_all, 1),
        "stage_timings": stage, "static_vs_e1_crosscheck": xcheck, "headline_gap": heads,
        "versions": C.versions(), "anchor_check": "csa_common.assert_anchors() 31/31 PASS",
        "dynamic_pagerank_status": {s: stage.get(f"dynamic_{s}", {}).get("status")
                                    for s in ("pagerank_dw", "pagerank_uu")},
    }
    with open(os.path.join(HERE, "E2_summary.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=float)
    C.write_run_note("E2_dynamic_attack", run_lines(meta))
    print("[E2] headline gap:\n" + pd.DataFrame(heads).to_string(index=False), flush=True)
    print(f"[E2] pool wall {wall_pool:.1f}s total {time.perf_counter() - t_all:.1f}s", flush=True)


def run_lines(meta):
    v = meta["versions"]
    lines = [
        "E2_dynamic_attack RUN.txt",
        "command: cd pipeline && PYTHONHASHSEED=0 python out/E2_dynamic_attack/run_e2.py",
        f"python={v['python']} networkx={v['networkx']} pandas={v['pandas']} "
        f"numpy={v['numpy']} PYTHONHASHSEED={v['PYTHONHASHSEED']} SEED={meta['seed']}",
        f"anchors: {meta['anchor_check']}; baseline {meta['n0']} nodes (undirected main component, §1); "
        f"{meta['pairset']}",
        f"f grid {meta['check_f'][0]}..{meta['check_f'][-1]} (+0.05); efficiency at "
        f"{len(meta['check_f'])} checkpoints, GCC tracked every step 1..{meta['max_step']}",
        "dynamic = recompute the ranking after EVERY single deletion; tie-break = FQDN asc; "
        "static reference recomputed in the same process and cross-checked against E1 CSV",
        f"budget per dynamic strategy: {meta['budget_s']} s (over-budget runs are truncated AND reported, "
        "never silently skipped)",
        f"pool wall: {meta['pool_wall_s']} s; total wall: {meta['total_wall_s']} s",
        "per-stage:",
    ]
    for k, s in meta["stage_timings"].items():
        lines.append("  " + k + ": " + json.dumps(s, default=float))
    lines += [
        "dynamic PageRank status: " + json.dumps(meta["dynamic_pagerank_status"]),
        "static cross-check vs E1: " + json.dumps(meta["static_vs_e1_crosscheck"], default=float),
        "outputs: e2_curve_efficiency.csv e2_gcc_per_step.csv e2_static_vs_dynamic_gap.csv "
        "e2_headline_gap.csv E2_summary.json",
    ]
    return lines


if __name__ == "__main__":
    main()
