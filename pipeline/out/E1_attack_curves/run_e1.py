"""E1_attack_curves — 静态移除（rank once on the pristine graph, delete top-k）攻击曲线。

口径全部来自 pipeline/CONVENTIONS.md：
  §1   基准图 = 674 节点 / 2497 边的无向主分量（鲁棒性实验唯一基准图）
  §2.3 效率三口径同时报；**主口径 E_fixed**（分母恒为基线 GCC 的 453,602 个有序点对）；
       只允许"同一 k 跨情景"比较，禁止把 E_full/E_gcc 相对基线上升当结论
  §5   PageRank 变体分列：主口径 = 有向加权，无权无向只作对照
  §6   主口径 = 静态；确定性策略每档 1 次，随机策略 >=100 次独立重复并报 sd
效率只用 C.eff_three / C.PairSet / C.largest_component，禁止自己实现。

重跑：cd pipeline && PYTHONHASHSEED=0 python out/E1_attack_curves/run_e1.py
"""
from __future__ import annotations

import os
import sys
import time
import json
from multiprocessing import Pool

import networkx as nx
import numpy as np
import pandas as pd

SEED = 20261113

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE = os.path.dirname(os.path.dirname(HERE))
if PIPELINE not in sys.path:
    sys.path.insert(0, PIPELINE)
import csa_common as C  # noqa: E402

# ------------------------------------------------------------------ 配置
N_RANDOM_REPS = int(os.environ.get("E1_REPS", "120"))     # >=100（CONVENTIONS §6）；RUN.txt 记录实际值
F_GRID = [round(0.02 * i, 2) for i in range(1, 31)]        # 0.02 -> 0.60，30 档
SUMMARY_F = [0.05, 0.10, 0.20, 0.30, 0.50]
CHECK_F = sorted(set(F_GRID) | set(SUMMARY_F))             # 实际记效率的档位
MAX_F = F_GRID[-1]
DETERMINISTIC = ["degree", "betweenness", "pagerank_dw", "pagerank_uu", "kcore"]
STRATEGIES = ["random"] + DETERMINISTIC
LABEL = {
    "random": "随机（独立重复，取均值±sd）",
    "degree": "度（无向无权，降序）",
    "betweenness": "介数（无向无权，降序）",
    "pagerank_dw": "PageRank 有向加权【§5 主口径】",
    "pagerank_uu": "PageRank 无权无向【§5 对照】",
    "kcore": "k-core 数（降序）",
}
WORKERS = min(14, max(1, (os.cpu_count() or 4) - 2))

_WS: dict = {}


# ------------------------------------------------------------------ 基准图
def base_graphs() -> dict:
    """674 节点主分量上的三张图 + 固定点对集。"""
    b = C.build_all()
    C.assert_anchors(b)
    gcc = set(b["sets"]["gcc"])
    att = set(b["nodes"]["node"])
    H0 = C.induced(b["Gb"], gcc)          # 工作图：无权无向（连通性/效率按跳数）
    return {
        "H0": H0,
        "Hw": C.induced(b["Gu"], gcc),    # 加权无向（权重口径排名用）
        "Hd": C.induced(b["Gd"], gcc),    # 有向加权（PageRank 主口径）
        # 仅供 §5 PageRank 口径诊断使用的两套对照节点集
        "alt": {"676": (C.induced(b["Gb"], set(b["Gu"].nodes())),
                        C.induced(b["Gd"], set(b["Gd"].nodes()))),
                "671": (C.induced(b["Gb"], att), C.induced(b["Gd"], att))},
        "ps": b["ps"],
        "n0": H0.number_of_nodes(),
        "m0": H0.number_of_edges(),
        "nodes": sorted(H0.nodes()),
    }


# ------------------------------------------------------------------ 排名
def ranking(strategy: str, B: dict) -> list:
    """确定性策略的移除顺序：值降序，同值按 FQDN 升序（可复现的平手裁决）。"""
    H0, Hw, Hd = B["H0"], B["Hw"], B["Hd"]
    if strategy == "degree":
        val = dict(H0.degree())
    elif strategy == "betweenness":
        val = nx.betweenness_centrality(H0)
    elif strategy == "pagerank_dw":
        val = nx.pagerank(Hd, weight="weight")                 # §5 主口径
    elif strategy == "pagerank_uu":
        val = nx.pagerank(H0)                                  # §5 对照
    elif strategy == "kcore":
        val = {k: float(v) for k, v in nx.core_number(H0).items()}
    else:
        raise ValueError(strategy)
    return sorted(val, key=lambda n: (-val[n], n))


def random_order(B: dict, rep: int) -> list:
    """一次独立随机重复：整置乱（各 f 档相互嵌套，等价于每档独立抽样但方差略小）。"""
    rng = np.random.default_rng([SEED, rep])
    nodes = list(B["nodes"])
    rng.shuffle(nodes)
    return nodes


# ------------------------------------------------------------------ 跑一条序列
def nominal_map(n0: int) -> dict:
    """删除数 k -> 名义 f（一个 k 可对应多个名义 f，取列表）。"""
    m: dict = {}
    for f in CHECK_F:
        m.setdefault(int(round(f * n0)), []).append(f)
    return m


def run_sequence(H0, order, check_steps, ps, nominal):
    """按 order 逐个删点：每步记 GCC/分量数，check_steps 档记三效率 + 平均路径。

    返回 (per_step[list of dict], checks[list of dict], step_seconds[float])
    """
    H = H0.copy()
    n0 = H0.number_of_nodes()
    steps, checks = [], []
    maxk = int(round(MAX_F * n0))
    t_step_total = 0.0

    comps = sorted(nx.connected_components(H), key=len, reverse=True)
    rec = checkpoint(H, comps, ps, 0, nominal.get(0, [0.0]) or [0.0])
    checks.append(rec)

    for k, victim in enumerate(order[:maxk], start=1):
        t = time.perf_counter()
        H.remove_node(victim)
        comps = sorted(nx.connected_components(H), key=len, reverse=True)
        g = len(comps[0]) if comps else 0
        dt = time.perf_counter() - t
        t_step_total += dt
        steps.append({"step": k, "n_remaining": H.number_of_nodes(), "gcc_size": g,
                      "gcc_pct": 100.0 * g / n0, "n_components": len(comps), "secs": dt})
        if k in check_steps:
            checks.append(checkpoint(H, comps, ps, k, nominal.get(k, [k / n0])))
    return steps, checks, t_step_total


def checkpoint(H, comps_desc, ps, k, nominal_f):
    """在删除 k 个点后取三个效率口径 + 存活最大分量平均路径长度。"""
    g = len(comps_desc[0]) if comps_desc else 0
    e = C.eff_three(H, ps)
    if g >= 2:
        gcc = C.largest_component(H)
        apl = float(nx.average_shortest_path_length(gcc))
    else:
        apl = float("nan")
    return {
        "step": k, "f_nominal": nominal_f[0], "f_actual": k / ps.n0,
        "n_remaining": H.number_of_nodes(), "gcc_size": g,
        "gcc_pct": 100.0 * g / ps.n0, "n_components": len(comps_desc),
        "E_full": e["E_full"], "E_gcc": e["E_gcc"], "E_fixed": e["E_fixed"],
        "avg_pathlen_gcc": apl,
    }


def steps_to_checkpoint_grid(n0: int) -> set:
    return {int(round(f * n0)) for f in CHECK_F}


# ------------------------------------------------------------------ 进程池 worker
def _init_worker():
    _WS.clear()
    _WS.update(base_graphs())


def _work_random(rep):
    B = _WS
    t = time.perf_counter()
    order = random_order(B, rep)
    steps, checks, secs = run_sequence(B["H0"], order, steps_to_checkpoint_grid(B["n0"]),
                                       B["ps"], nominal_map(B["n0"]))
    return rep, steps, checks, secs, time.perf_counter() - t


def _work_tiebreak(arg):
    """平手裁决敏感性：同值节点换序后曲线是否移动（k-core 值只有 11 档，必须查）。"""
    strategy, mode, seed = arg
    B = _WS
    H0 = B["H0"]
    if strategy == "degree":
        val = dict(H0.degree())
    elif strategy == "kcore":
        val = {k: float(v) for k, v in nx.core_number(H0).items()}
    else:
        val = nx.betweenness_centrality(H0)
    if mode == "name_asc":
        order = sorted(val, key=lambda n: (-val[n], n))
    elif mode == "name_desc":
        order = sorted(val, key=lambda n: (-val[n], n[::-1]))
    elif mode == "shuffled":
        rng = np.random.default_rng([SEED, 7000 + seed])
        keys = list(val)
        rng.shuffle(keys)
        order = sorted(keys, key=lambda n: -val[n])           # 稳定排序：只在平手内部换序
    else:
        raise ValueError(mode)
    steps, checks, secs = run_sequence(H0, order, steps_to_checkpoint_grid(B["n0"]),
                                       B["ps"], nominal_map(B["n0"]))
    return strategy, mode, seed, checks


# ------------------------------------------------------------------ 随机策略汇总
def agg_random(per_rep, n0):
    """逐档 mean/sd（跨独立重复）。效率按 nan-aware 均值。"""
    step_by_k, chk_by_step = {}, {}
    for rep, steps, checks, secs, cpu in per_rep:
        for s in steps:
            step_by_k.setdefault(s["step"], []).append(s)
        for c in checks:
            chk_by_step.setdefault(c["step"], []).append(c)

    rows_step = []
    for k in sorted(step_by_k):
        recs = step_by_k[k]
        g = np.array([r["gcc_size"] for r in recs], float)
        nc = np.array([r["n_components"] for r in recs], float)
        rows_step.append({
            "strategy": "random", "step": k, "f_actual": k / n0,
            "gcc_size": g.mean(), "gcc_size_sd": g.std(ddof=1),
            "gcc_pct": 100.0 * g.mean() / n0, "gcc_pct_sd": 100.0 * g.std(ddof=1) / n0,
            "n_components": nc.mean(), "n_components_sd": nc.std(ddof=1),
            "n_reps": len(recs),
        })

    rows_chk = []
    for k in sorted(chk_by_step):
        df = pd.DataFrame(chk_by_step[k])
        row = {"strategy": "random", "step": int(k), "f_nominal": float(df["f_nominal"].iloc[0]),
               "f_actual": k / n0, "n_reps": len(df)}
        for col in ["n_remaining", "gcc_size", "gcc_pct", "n_components",
                    "E_full", "E_gcc", "E_fixed", "avg_pathlen_gcc"]:
            v = df[col].to_numpy(dtype=float)
            row[col] = float(np.nanmean(v))
            row[col + "_sd"] = float(np.nanstd(v, ddof=1))
        rows_chk.append(row)
    return rows_step, rows_chk


# ------------------------------------------------------------------ 摘要表
def pr_variant_diagnostic(B) -> pd.DataFrame:
    """§5 契约复现：把"加权有向 / 无权无向 / 加权无向"以及若干备选 PageRank 口径
    的静态 GCC 存活曲线一次算完（只算 GCC，不算效率，成本 ~秒级）。"""
    H0, Hw, Hd = B["H0"], B["Hw"], B["Hd"]
    n0 = B["n0"]
    fr = (0.10, 0.20, 0.30)
    ks = [int(round(x * n0)) for x in fr]

    def curve(val, base=None):
        order = sorted(val, key=lambda x: (-val[x], x))
        H = (base if base is not None else H0).copy()
        got = {}
        for k, v in zip(range(1, max(ks) + 1), order):
            if v in H:
                H.remove_node(v)
            if k in ks:
                got[k] = 100.0 * max(len(c) for c in nx.connected_components(H)) / n0
        return {f"gcc_pct_at_f{int(x * 100)}": round(got[int(round(x * n0))], 2) for x in fr}

    cands = {
        "A_pagerank_directed_weighted_[§5主口径]": lambda: nx.pagerank(Hd, weight="weight"),
        "B_pagerank_undirected_binary_[§5'无权无向']": lambda: nx.pagerank(H0),
        "C_pagerank_undirected_weighted": lambda: nx.pagerank(Hw, weight="weight"),
        "D_pagerank_directed_unweighted": lambda: nx.pagerank(Hd),
        "E_strength_undirected_weighted": lambda: dict(Hw.degree(weight="weight")),
        "F_degree_undirected_binary": lambda: dict(H0.degree()),
    }
    rows = []
    for name, f in cands.items():
        rows.append({"variant": name, **curve(f())})
    # §5 的"无权无向"列复现不出来 -> 把 PageRank 的两个自由参数扫一遍，证明不是参数选择问题
    for a in (0.5, 0.7, 0.85, 0.9, 0.95):
        rows.append({"variant": f"B_sweep_alpha={a:.2f} undirected binary (rank on 674)",
                     **curve(nx.pagerank(H0, alpha=a))})
        rows.append({"variant": f"A_sweep_alpha={a:.2f} directed weighted (rank on 674)",
                     **curve(nx.pagerank(Hd, weight="weight", alpha=a, max_iter=500))})
    for tag, (Gbin, Gdir) in B["alt"].items():
        rows.append({"variant": f"B_sweep_rank_on_{tag} undirected binary",
                     **curve(nx.pagerank(Gbin))})
        rows.append({"variant": f"A_sweep_rank_on_{tag} directed weighted",
                     **curve(nx.pagerank(Gdir, weight="weight"))})
    contract = {"A_pagerank_directed_weighted_[§5主口径]": (76.3, 55.8, 38.3),
                "B_pagerank_undirected_binary_[§5'无权无向']": (66.2, 41.5, 12.5)}
    for r in rows:
        c = contract.get(r["variant"])
        if c:
            r["contract_pct"] = "/".join(str(x) for x in c)
            r["max_abs_diff_pp"] = round(max(abs(c[i] - r[f"gcc_pct_at_f{int(x * 100)}"])
                                          for i, x in enumerate((0.10, 0.20, 0.30))), 2)
    return pd.DataFrame(rows)


def build_summary(det_checks, r_checks, n0):
    """f = {0.05,0.10,0.20,0.30,0.50}：同 k 跨策略比较，E_fixed 给出相对随机基线的百分比差。"""
    det = {}
    for c in det_checks:
        det.setdefault(c["strategy"], {})[round(c["f_actual"], 6)] = c
    rnd = {round(c["f_actual"], 6): c for c in r_checks}
    rows = []
    for s in STRATEGIES:
        for f in SUMMARY_F:
            key = round(int(round(f * n0)) / n0, 6)
            if s == "random":
                c = rnd.get(key)
                if c is None:
                    continue
                row = dict(c)
                row["dE_fixed_vs_random_pct"] = 0.0
                rows.append(row)
            else:
                c = det[s].get(key)
                if c is None:
                    continue
                rb = rnd.get(key)
                row = dict(c)
                row["n_reps"] = 1
                for col in ["gcc_size", "n_components", "E_full", "E_gcc", "E_fixed", "avg_pathlen_gcc"]:
                    row[col + "_sd"] = np.nan
                row["gcc_pct_sd"] = np.nan
                row["dE_fixed_vs_random_pct"] = (
                    100.0 * (c["E_fixed"] - rb["E_fixed"]) / rb["E_fixed"] if rb else np.nan)
                rows.append(row)
    cols = ["strategy", "f_nominal", "f_actual", "step", "n_reps", "gcc_pct", "gcc_pct_sd",
            "n_components", "n_components_sd", "E_full", "E_full_sd", "E_gcc", "E_gcc_sd",
            "E_fixed", "E_fixed_sd", "avg_pathlen_gcc", "avg_pathlen_gcc_sd",
            "dE_fixed_vs_random_pct"]
    return pd.DataFrame(rows)[cols].round(6)


# ------------------------------------------------------------------ 主流程
def main():
    t_all = time.perf_counter()
    timing = {}
    print("[E1] anchors + build ...", flush=True)
    t = time.perf_counter()
    B = base_graphs()
    timing["build_and_anchors"] = time.perf_counter() - t
    n0, ps = B["n0"], B["ps"]
    print(f"[E1] baseline n0={n0} m0={B['m0']} pairset={ps!r}", flush=True)
    if n0 != C.ANCHORS["gcc_size"]:
        raise AssertionError("基准图不是 674 主分量")
    cs = steps_to_checkpoint_grid(n0)
    nominal = nominal_map(n0)
    maxk = max(cs)

    det_checks, det_steps = [], []
    for s in DETERMINISTIC:
        t = time.perf_counter()
        order = ranking(s, B)
        trank = time.perf_counter() - t
        t = time.perf_counter()
        steps, checks, secs = run_sequence(B["H0"], order, cs, ps, nominal)
        tc = time.perf_counter() - t
        timing[f"rank_{s}"] = trank
        timing[f"run_{s}"] = tc
        for c in checks:
            c["strategy"] = s
        det_checks += checks
        for r in steps:
            r2 = dict(r)
            r2["strategy"] = s
            r2["f_actual"] = r["step"] / n0
            r2["n_reps"] = 1
            for sd in ["gcc_size_sd", "gcc_pct_sd", "n_components_sd"]:
                r2[sd] = np.nan
            det_steps.append(r2)
        prof = {c["step"]: c["gcc_pct"] for c in checks}
        print(f"[E1] {s:12s} rank={trank:6.2f}s run={tc:6.1f}s  GCC%@10/20/30% = "
              + " ".join(f"{prof[int(round(f*n0))]:5.1f}" for f in (0.10, 0.20, 0.30)), flush=True)

    t = time.perf_counter()
    prdiag = pr_variant_diagnostic(B)
    timing["pr_variant_diagnostic"] = time.perf_counter() - t
    prdiag.to_csv(os.path.join(HERE, "e1_pagerank_variant_diagnostic.csv"), index=False,
                  encoding="utf-8-sig")
    print("[E1] PageRank variant diagnostic (static, GCC%% of 674):\n"
          + prdiag.to_string(index=False).encode("utf-8", "replace").decode("utf-8", "replace"),
          flush=True)

    # ---------- 随机（>=100 独立重复，进程池并行）
    t = time.perf_counter()
    per_rep = []
    with Pool(WORKERS, initializer=_init_worker) as pool:
        for res in pool.imap_unordered(_work_random, range(N_RANDOM_REPS), chunksize=1):
            per_rep.append(res)
    per_rep.sort(key=lambda x: x[0])
    timing["random_pool_wall"] = time.perf_counter() - t
    timing["random_step_secs_mean"] = float(np.mean([r[3] for r in per_rep]))
    timing["random_step_secs_sum"] = float(np.sum([r[3] for r in per_rep]))
    timing["random_rep_cpu_mean"] = float(np.mean([r[4] for r in per_rep]))
    timing["random_rep_cpu_sum"] = float(np.sum([r[4] for r in per_rep]))
    print(f"[E1] random reps={N_RANDOM_REPS} pool_wall={timing['random_pool_wall']:.1f}s "
          f"rep_cpu_mean={timing['random_rep_cpu_mean']:.1f}s "
          f"rep_cpu_sum={timing['random_rep_cpu_sum']:.1f}s", flush=True)
    r_steps, r_checks = agg_random(per_rep, n0)

    # ---------- 平手裁决敏感性
    t = time.perf_counter()
    jobs = [(s, m, 0) for s in ("kcore", "degree", "betweenness") for m in ("name_asc", "name_desc")]
    jobs += [("kcore", "shuffled", i) for i in range(3)]
    with Pool(min(6, WORKERS), initializer=_init_worker) as pool:
        tb = pool.map(_work_tiebreak, jobs)
    timing["tiebreak_pool_wall"] = time.perf_counter() - t
    tb_rows = []
    for strategy, mode, seed, checks in tb:
        for c in checks:
            tb_rows.append({"strategy": strategy, "tiebreak": mode, "seed": seed,
                            "step": c["step"], "f_nominal": c["f_nominal"], "f_actual": c["f_actual"],
                            "gcc_pct": c["gcc_pct"], "n_components": c["n_components"],
                            "E_fixed": c["E_fixed"], "E_gcc": c["E_gcc"]})
    print(f"[E1] tiebreak runs={len(tb)} wall={timing['tiebreak_pool_wall']:.1f}s", flush=True)

    # 平手裁决：同一排名、同值组内换序造成的 GCC% 极差（单位：百分点）
    tb_spread = {}
    for s in ("kcore", "degree", "betweenness"):
        by_f: dict = {}
        n_ord = len({(m, sd) for (st, m, sd, _c) in tb if st == s})
        for st, m, sd, checks in tb:
            if st == s:
                for c in checks:
                    by_f.setdefault(c["step"], []).append(c["gcc_pct"])
        worst = max(by_f, key=lambda k: max(by_f[k]) - min(by_f[k]))
        tb_spread[s] = {
            "n_orderings": n_ord,
            "max_spread_pp": round(max(max(v) - min(v) for v in by_f.values()), 2),
            "max_spread_at_step": int(worst),
            "spread_at_summary_f": {
                int(round(f * n0)): round(max(by_f[int(round(f * n0))]) - min(by_f[int(round(f * n0))]), 2)
                for f in SUMMARY_F if int(round(f * n0)) in by_f},
        }
    print("[E1] tie-break spread:", json.dumps(tb_spread), flush=True)

    # ---------- 落盘
    t = time.perf_counter()
    eff_cols = ["strategy", "step", "f_nominal", "f_actual", "n_reps", "n_remaining",
                "gcc_size", "gcc_size_sd", "gcc_pct", "gcc_pct_sd", "n_components",
                "n_components_sd", "E_full", "E_full_sd", "E_gcc", "E_gcc_sd", "E_fixed",
                "E_fixed_sd", "avg_pathlen_gcc", "avg_pathlen_gcc_sd"]
    det_eff = []
    for c in det_checks:
        r = {k: v for k, v in c.items()}
        r["n_reps"] = 1
        for col in ["gcc_size", "gcc_pct", "n_components", "E_full", "E_gcc", "E_fixed",
                    "avg_pathlen_gcc"]:
            r[col + "_sd"] = np.nan
        det_eff.append(r)
    eff = pd.concat([pd.DataFrame(det_eff), pd.DataFrame(r_checks)], ignore_index=True)[eff_cols]
    eff.round(6).to_csv(os.path.join(HERE, "e1_curve_efficiency.csv"), index=False,
                        encoding="utf-8-sig")

    step_cols = ["strategy", "step", "f_actual", "gcc_size", "gcc_size_sd", "gcc_pct",
                 "gcc_pct_sd", "n_components", "n_components_sd", "n_reps"]
    stp = pd.DataFrame(det_steps + r_steps)
    stp[stp["strategy"] != "random"].drop(columns=["secs"]).pipe(
        lambda d: d[step_cols]).to_csv(os.path.join(HERE, "e1_curve_gcc_per_step_deterministic.csv"),
                                       index=False, encoding="utf-8-sig")
    pd.DataFrame(r_steps)[step_cols].round(4).to_csv(
        os.path.join(HERE, "e1_curve_gcc_per_step_random.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(tb_rows).round(6).to_csv(os.path.join(HERE, "e1_tiebreak_sensitivity.csv"),
                                          index=False, encoding="utf-8-sig")
    summary = build_summary(det_checks, r_checks, n0)
    summary.to_csv(os.path.join(HERE, "e1_summary_f5.csv"), index=False, encoding="utf-8-sig")
    timing["write_csv"] = time.perf_counter() - t

    # ---------- PageRank §5 契约锚
    pr_anchor = {}
    by_s = {s: [c for c in det_checks if c["strategy"] == s] for s in DETERMINISTIC}
    for s, want in (("pagerank_dw", [76.3, 55.8, 38.3]), ("pagerank_uu", [66.2, 41.5, 12.5])):
        got = [[c["gcc_pct"] for c in by_s[s] if c["step"] == int(round(f * n0))][0]
               for f in (0.10, 0.20, 0.30)]
        pr_anchor[s] = {"contract_pct": want, "got_pct": [round(x, 2) for x in got],
                        "max_abs_diff_pp": round(max(abs(a - b) for a, b in zip(want, got)), 2)}
    print("[E1] PageRank variant anchor:", json.dumps(pr_anchor), flush=True)

    # ---------- 单调性自检：随机 E_fixed 必须基本单调下降（升了就是分母/点对 bug）
    ef = [c["E_fixed"] for c in sorted(r_checks, key=lambda d: d["step"])]
    n_incr = sum(1 for a, b in zip(ef, ef[1:]) if b > a)
    base_fixed = float([c for c in det_checks if c["step"] == 0][0]["E_fixed"])

    meta = {
        "seed": SEED, "n_random_reps": N_RANDOM_REPS, "workers": WORKERS,
        "f_grid": F_GRID, "summary_f": SUMMARY_F, "check_f": CHECK_F,
        "n0": n0, "m0": B["m0"], "pairset": repr(ps), "max_step": maxk,
        "strategies": STRATEGIES, "labels": LABEL,
        "timing_s": {k: round(float(v), 3) for k, v in timing.items()},
        "total_wall_s": round(time.perf_counter() - t_all, 1),
        "versions": C.versions(),
        "anchor_check": "csa_common.assert_anchors() 31/31 PASS",
        "pagerank_contract_anchor": pr_anchor,
        "baseline_E_fixed": round(base_fixed, 6),
        "baseline_E_full": round(float([c for c in det_checks if c["step"] == 0][0]["E_full"]), 6),
        "random_E_fixed_monotone_violations": int(n_incr),
        "random_E_fixed_checkpoints": int(len(ef)),
        "random_E_fixed_first_and_last": [round(ef[0], 6), round(ef[-1], 6)],
        "tiebreak_note": "spreads in percentage points of surviving GCC, same ranking values, "
                         "different order inside equal-value groups",
        "tiebreak_spread_pp": tb_spread,
    }
    with open(os.path.join(HERE, "E1_summary.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=float)
    C.write_run_note("E1_attack_curves", run_lines(meta))
    print(f"[E1] random E_fixed increased at {n_incr}/{len(ef) - 1} checkpoint steps  "
          f"(baseline E_fixed={base_fixed:.4f})", flush=True)
    print(f"[E1] total wall {time.perf_counter() - t_all:.1f} s", flush=True)


def run_lines(meta):
    v = meta["versions"]
    return [
        "E1_attack_curves RUN.txt",
        "command: cd pipeline && PYTHONHASHSEED=0 python out/E1_attack_curves/run_e1.py",
        f"python={v['python']} networkx={v['networkx']} pandas={v['pandas']} "
        f"numpy={v['numpy']} PYTHONHASHSEED={v['PYTHONHASHSEED']} SEED={meta['seed']}",
        f"anchors: {meta['anchor_check']}",
        f"baseline graph: {meta['n0']} nodes / {meta['m0']} edges (undirected main component, §1); "
        f"pairset {meta['pairset']} (E_fixed denominator)",
        f"f grid: {meta['f_grid'][0]}..{meta['f_grid'][-1]} step 0.02 (30 points) + 0.05 for the summary; "
        f"GCC / n_components tracked at EVERY step 1..{meta['max_step']}",
        f"random: {meta['n_random_reps']} independent replicates (contract >=100), "
        f"seed per rep = np.random.default_rng([{meta['seed']}, rep]), {meta['workers']} processes",
        "deterministic: 1 run per f (§6); removal order = value desc, ties by FQDN asc; "
        "tie-break sensitivity: kcore/degree/betweenness x {name_asc, name_desc} + kcore x 3 shuffled",
        "efficiency: C.eff_three (E_full / E_gcc / E_fixed), hop-count distances",
        "ranking bases: degree=binary undirected deg; betweenness=binary undirected BC; "
        "pagerank_dw=nx.pagerank(Gd induced on 674, weight='weight') [§5 primary]; "
        "pagerank_uu=nx.pagerank(binary undirected) [§5 control]; kcore=core_number(binary)",
        f"wall time total: {meta['total_wall_s']} s",
        "per-stage wall (s): " + json.dumps(meta["timing_s"]),
        "random pool CPU: per-replicate (removal loop + 31 efficiency checkpoints) mean "
        f"{round(meta['timing_s'].get('random_rep_cpu_mean', 0), 1)} s, "
        f"sum {round(meta['timing_s'].get('random_rep_cpu_sum', 0), 1)} s over "
        f"{meta['n_random_reps']} reps; the remove_node+GCC inner loop alone is only "
        f"{round(meta['timing_s'].get('random_step_secs_mean', 0), 2)} s per replicate, i.e. the "
        "cost is entirely C.eff_three",
        "tie-break sensitivity (same ranking, different order inside equal-value groups): "
        + json.dumps(meta["tiebreak_spread_pp"]),
        f"baseline E_full={meta['baseline_E_full']} E_fixed={meta['baseline_E_fixed']} "
        "(§2.3 table says E_gcc=E_fixed=0.3315 on the 674 GCC; E_full=0.3295 there is the 676-node graph)",
        f"sanity: random E_fixed increases at {meta['random_E_fixed_monotone_violations']} of "
        f"{meta['random_E_fixed_checkpoints'] - 1} checkpoint steps (denominator-bug check); "
        f"first/last {meta['random_E_fixed_first_and_last']}",
        "PageRank §5 contract anchor: " + json.dumps(meta["pagerank_contract_anchor"]),
        "outputs: e1_curve_efficiency.csv e1_curve_gcc_per_step_deterministic.csv "
        "e1_curve_gcc_per_step_random.csv e1_tiebreak_sensitivity.csv e1_summary_f5.csv E1_summary.json",
    ]


if __name__ == "__main__":
    main()
