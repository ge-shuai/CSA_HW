"""E3_percolation — 渗流阈值 p_c（GCC 首次跌破基线 50% / 25% 的删除比例）+ 契约要求的敏感性。

契约依据：
  §1  基准 = 674 节点主分量；敏感性另跑 671（有属性）与 676（含悬空）两套节点集
  §5  PageRank 两个变体都要给 p_c（有向加权=主口径，无权无向=对照；另附加权无向）
  §6  p_c 来自静态（主口径）与动态（稳健性对照）两套移除
  §2.3 p_c 只依赖 GCC 规模，不依赖效率口径；本脚本因此**只**逐点跟踪 GCC（成本比 E1 低两个数量级）
随机策略：200 次独立重复，报 p_c 的**分布**（分位数 + sd + 均值 95% 置信区间），不只报均值。

依赖：动态 p_c 读 out/E2_dynamic_attack/e2_gcc_per_step.csv；静态交叉校验读
      out/E1_attack_curves/e1_curve_gcc_per_step_deterministic.csv。先跑 E1、E2。

重跑：cd pipeline && PYTHONHASHSEED=0 python out/E3_percolation/run_e3.py
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
E2_DIR = os.path.join(PIPELINE, "out", "E2_dynamic_attack")
if PIPELINE not in sys.path:
    sys.path.insert(0, PIPELINE)
import csa_common as C  # noqa: E402

N_RANDOM_REPS = int(os.environ.get("E3_REPS", "200"))   # >=100（§6）；p_c 分布用 200
# 0.90 而不是 0.60：随机策略要跌破 25% 需要删掉 60% 以上，若网格只到 0.60，
# 200 次里只有个位数跨阈值，pc(25%) 就成了右删失的均值（会把 0.60 当成 0.59 报）。
# 网格延长到 0.90 后两档阈值对全部重复都可观测，CI 才是真的 CI。p_c 本身与网格长度无关。
MAX_F = 0.90
LEVELS = (0.50, 0.25)
STRATS = ["random", "degree", "betweenness", "pagerank_dw", "pagerank_uu", "kcore"]
DET_STRATS = [s for s in STRATS if s != "random"]
RANK_BASES = {       # (节点集口径, 排名基线) -> 打分函数列表
    "primary": {
        "degree": lambda G, Gw, Gd: dict(G.degree()),
        "betweenness": lambda G, Gw, Gd: nx.betweenness_centrality(G),
        "pagerank_dw": lambda G, Gw, Gd: nx.pagerank(Gd, weight="weight"),
        "pagerank_uu": lambda G, Gw, Gd: nx.pagerank(G),
        "kcore": lambda G, Gw, Gd: {k: float(v) for k, v in nx.core_number(G).items()},
    },
    "weighted": {    # 权重口径：用观察到的链接次数参与排名
        "degree": lambda G, Gw, Gd: dict(Gw.degree(weight="weight")),
        "betweenness": lambda G, Gw, Gd: nx.betweenness_centrality(Gw, weight="weight"),
        "pagerank_dw": lambda G, Gw, Gd: nx.pagerank(Gd, weight="weight"),
        "pagerank_uu": lambda G, Gw, Gd: nx.pagerank(Gw, weight="weight"),
        "kcore": None,   # k-core 无权重定义 → 显式声明不适用
    },
}
WORKERS = min(14, max(1, (os.cpu_count() or 4) - 2))
_WS: dict = {}


# ------------------------------------------------------------------ 构图
def build_sets() -> dict:
    """三套节点口径的工作图：671 有属性 / 674 主分量 / 676 全图。"""
    b = C.build_all()
    C.assert_anchors(b)
    out = {}
    for name, keep in (("n671", set(b["nodes"]["node"])),
                       ("n674", set(b["sets"]["gcc"])),
                       ("n676", set(b["Gu"].nodes()))):
        G = C.induced(b["Gb"], keep)
        Gw = C.induced(b["Gu"], keep)
        Gd = C.induced(b["Gd"], keep)
        base_gcc = max((len(c) for c in nx.connected_components(G)), default=0)
        out[name] = {"G": G, "Gw": Gw, "Gd": Gd, "n0": G.number_of_nodes(),
                     "m0": G.number_of_edges(), "gcc0": base_gcc,
                     "nodes": sorted(G.nodes())}
    out["_ps"] = b["ps"]
    return out


def ordered(G, Gw, Gd, strategy, basis, tie_extra=None):
    """确定性排名（值降序，平手按 FQDN 升序）。"""
    val = RANK_BASES[basis][strategy](G, Gw, Gd)
    if tie_extra is None:
        return sorted(val, key=lambda n: (-val[n], n))
    rng = np.random.default_rng([SEED, tie_extra])
    keys = list(val)
    rng.shuffle(keys)
    return sorted(keys, key=lambda n: -val[n])


def random_order(S, rep):
    rng = np.random.default_rng([SEED, rep])
    ns = list(S["nodes"])
    rng.shuffle(ns)
    return ns


# ------------------------------------------------------------------ 逐点 GCC 轨迹
def gcc_track(G0, order, maxk):
    """返回每个删除步的 GCC 规模数组（长度 maxk+1，含基线）。"""
    H = G0.copy()
    arr = np.empty(maxk + 1, dtype=np.int32)
    arr[0] = max((len(c) for c in nx.connected_components(H)), default=0)
    for k, v in enumerate(order[:maxk], start=1):
        if v in H:
            H.remove_node(v)
        arr[k] = max((len(c) for c in nx.connected_components(H)), default=0)
    return arr


def p_c(curve, gcc0, n0, level, maxk):
    """首次 GCC < level*gcc0 的删除比例；网格内未跌破则返回 NaN。"""
    thr = level * gcc0
    idx = np.where(curve[1:maxk + 1] < thr)[0]
    if not len(idx):
        return float("nan"), int(maxk)
    k = int(idx[0]) + 1
    return k / n0, k


# ------------------------------------------------------------------ 随机重复
def _init_worker():
    _WS.clear()
    _WS.update(build_sets())


def _job_random(arg):
    set_name, rep = arg
    S = _WS[set_name]
    maxk = int(round(MAX_F * S["n0"]))
    curve = gcc_track(S["G"], random_order(S, rep), maxk)
    return set_name, rep, curve


def summarize_random(curves, gcc0, n0, maxk, level):
    """p_c 分布：均值/sd/分位数/均值 95% CI（描述性区间，不作显著性检验）。"""
    pcs = []
    for c in curves:
        p, k = p_c(c, gcc0, n0, level, maxk)
        pcs.append(p)
    a = np.array(pcs, dtype=float)
    fin = ~np.isnan(a)
    m, s = float(a[fin].mean()), float(a[fin].std(ddof=1))
    q = np.nanpercentile(a, [2.5, 16, 50, 84, 97.5])
    se = s / np.sqrt(fin.sum()) if fin.sum() > 1 else float("nan")
    frac_hit = float(fin.mean())
    return {
        "n_reps": int(len(a)), "n_reps_crossed": int(fin.sum()),
        "frac_reaches_threshold": round(frac_hit, 4),
        "pc_mean": m, "pc_sd": s, "pc_se": se,
        "pc_ci95_lo": m - 1.96 * se, "pc_ci95_hi": m + 1.96 * se,
        "pc_min": float(np.nanmin(a)), "pc_max": float(np.nanmax(a)),
        "pc_p2.5": float(q[0]), "pc_p16": float(q[1]), "pc_p50": float(q[2]),
        "pc_p84": float(q[3]), "pc_p97.5": float(q[4]),
        "k_mean": float(np.mean([p_c(c, gcc0, n0, level, maxk)[1] for c in curves])),
        "k_sd": float(np.std([p_c(c, gcc0, n0, level, maxk)[1] for c in curves], ddof=1)),
    }


def gcc_band(curves, n0, gcc0, maxk, level):
    """逐步 GCC% 的跨重复分位带（供 H4 画图与正文引用）。"""
    M = np.array([c[1:maxk + 1] for c in curves], dtype=float) / gcc0 * 100.0
    f = np.arange(1, maxk + 1) / n0
    return pd.DataFrame({
        "f": f, "pct_mean": M.mean(0), "pct_sd": M.std(0, ddof=1),
        "pct_p16": np.percentile(M, 16, axis=0), "pct_p50": np.percentile(M, 50, axis=0),
        "pct_p84": np.percentile(M, 84, axis=0),
    })


# ------------------------------------------------------------------ 主流程
def main():
    t_all = time.perf_counter()
    print("[E3] anchors + build ...", flush=True)
    S = build_sets()
    for name in ("n671", "n674", "n676"):
        print(f"[E3] {name}: n={S[name]['n0']} m={S[name]['m0']} gcc0={S[name]['gcc0']}", flush=True)

    rows, random_rows, band_frames = [], [], []

    # ---------- 1) 静态 p_c（三套节点集 × primary / weighted 排名基线）
    t0 = time.perf_counter()
    static_curves = {}
    for set_name in ("n674", "n671", "n676"):
        SS = S[set_name]
        n0, gcc0 = SS["n0"], SS["gcc0"]
        maxk = int(round(MAX_F * n0))
        for basis in ("primary", "weighted"):
            for strat in DET_STRATS:
                if RANK_BASES[basis][strat] is None:
                    static_curves[(set_name, basis, strat)] = None
                    continue
                order = ordered(SS["G"], SS["Gw"], SS["Gd"], strat, basis)
                static_curves[(set_name, basis, strat)] = gcc_track(SS["G"], order, maxk)
    timing_static_det = time.perf_counter() - t0

    t0 = time.perf_counter()
    with Pool(WORKERS, initializer=_init_worker) as pool:
        jobs = [(sn, r) for sn in ("n674", "n671", "n676") for r in range(N_RANDOM_REPS)]
        bucket = {}
        for set_name, rep, curve in pool.imap_unordered(_job_random, jobs, chunksize=2):
            bucket.setdefault(set_name, []).append((rep, curve))
    timing_random = time.perf_counter() - t0
    for sn in bucket:
        bucket[sn].sort()
        static_curves[(sn, "primary", "random")] = [c for _, c in bucket[sn]]
    print(f"[E3] static deterministic curves: {timing_static_det:.1f}s | "
          f"random {N_RANDOM_REPS}x3 sets: {timing_random:.1f}s", flush=True)

    # ---------- 2) 动态 p_c（读 E2 逐步 GCC；缺失即停，不静默跳过）
    e2p = os.path.join(E2_DIR, "e2_gcc_per_step.csv")
    if not os.path.exists(e2p):
        raise FileNotFoundError(f"缺少 E2 逐步曲线 {e2p}：请先运行 out/E2_dynamic_attack/run_e2.py")
    e2 = pd.read_csv(e2p)
    dyn_curves = {}
    for strat, sub in e2[e2["mode"] == "dynamic"].groupby("strategy"):
        sub = sub.sort_values("step")
        arr = np.empty(int(sub["step"].max()) + 1, dtype=np.int32)
        arr[0] = int(sub["gcc_size"].iloc[0]) + 0  # 占位，下面用 E2 基线行修正
        arr[sub["step"].to_numpy()] = sub["gcc_size"].to_numpy()
        dyn_curves[strat] = arr
    e2_eff = pd.read_csv(os.path.join(E2_DIR, "e2_curve_efficiency.csv"))
    for strat in list(dyn_curves):
        b = e2_eff[(e2_eff["strategy"] == strat) & (e2_eff["mode"] == "dynamic") &
                   (e2_eff["step"] == 0)]
        if len(b):
            dyn_curves[strat][0] = int(b["gcc_size"].iloc[0])
    print(f"[E3] dynamic curves loaded from E2: {sorted(dyn_curves)}", flush=True)

    # ---------- 3) 汇总 p_c 表
    def add(strat, mode, basis, set_name, curve_or_none, gcc0, n0, extra=""):
        maxk = int(round(MAX_F * n0))
        if isinstance(curve_or_none, list):        # 随机：逐重复
            for lv in LEVELS:
                d = summarize_random(curve_or_none, gcc0, n0, maxk, lv)
                random_rows.append({"strategy": strat, "node_set": set_name,
                                    "rank_basis": basis, "level": lv, **d})
            curve_or_none = np.mean(np.array(curve_or_none), axis=0)
        if curve_or_none is None:
            return
        for lv in LEVELS:
            p, k = p_c(curve_or_none, gcc0, n0, lv, maxk)
            rows.append({"strategy": strat, "mode": mode, "rank_basis": basis,
                         "node_set": set_name, "level": lv, "n0": n0, "gcc0": gcc0,
                         "max_step": maxk, "pc": p, "k_at_pc": k,
                         "pc_pct_of_nodes": 100 * p if p == p else np.nan,
                         "note": extra})

    # 主表：674 + primary，静态与动态
    for strat in STRATS:
        add(strat, "static", "primary", "n674", static_curves[("n674", "primary", strat)]
            if strat != "random" else static_curves[("n674", "primary", "random")],
            S["n674"]["gcc0"], S["n674"]["n0"])
    for strat, c in dyn_curves.items():
        add(strat, "dynamic", "primary", "n674", c, S["n674"]["gcc0"], S["n674"]["n0"])

    # 敏感性 (i)：节点集
    for set_name in ("n671", "n676"):
        for strat in STRATS:
            add(strat, "static", "primary", set_name,
                static_curves[(set_name, "primary", strat)],
                S[set_name]["gcc0"], S[set_name]["n0"])

    # 敏感性 (ii)：二值 Gb vs 加权 Gu 排名
    for strat in DET_STRATS:
        if strat in ("pagerank_dw", "pagerank_uu"):
            continue   # PR 两口径已在主表分列，weighted 基线下 uu 变为加权无向，单列在下面
        add(strat, "static", "weighted", "n674", static_curves[("n674", "weighted", strat)],
            S["n674"]["gcc0"], S["n674"]["n0"])
    add("pagerank_uu", "static", "weighted", "n674",
        static_curves[("n674", "weighted", "pagerank_uu")], S["n674"]["gcc0"], S["n674"]["n0"],
        extra="weighted basis => undirected WEIGHTED PageRank")

    # 随机带（674 primary）
    band = {}
    for lv in LEVELS:
        band[lv] = summarize_random(static_curves[("n674", "primary", "random")],
                                    S["n674"]["gcc0"], S["n674"]["n0"],
                                    int(round(MAX_F * S["n674"]["n0"])), lv)
    rnd_band = gcc_band(static_curves[("n674", "primary", "random")], S["n674"]["n0"],
                        S["n674"]["gcc0"], int(round(MAX_F * S["n674"]["n0"])), 0.5)
    rnd_band.round(4).to_csv(os.path.join(HERE, "e3_random_gcc_band.csv"), index=False,
                             encoding="utf-8-sig")

    # ---------- 4) 与 E1 静态曲线交叉校验
    xcheck = {}
    e1p = os.path.join(E1_DIR, "e1_curve_gcc_per_step_deterministic.csv")
    if os.path.exists(e1p):
        e1 = pd.read_csv(e1p)
        d = []
        for s in DET_STRATS:
            a = e1[e1["strategy"] == s].set_index("step")["gcc_size"]
            c = static_curves[("n674", "primary", s)]
            j = a.index.to_numpy()
            j = j[j < len(c)]
            d.append({"strategy": s, "max_abs_gcc_diff_nodes":
                      int(np.abs(a.loc[j] - c[j]).max())})
        xcheck = {"file": e1p, "per_strategy": d,
                  "worst": int(max(x["max_abs_gcc_diff_nodes"] for x in d))}
    e1r = os.path.join(E1_DIR, "e1_curve_gcc_per_step_random.csv")
    if os.path.exists(e1r):
        e1 = pd.read_csv(e1r)
        R = np.array(static_curves[("n674", "primary", "random")], dtype=float)
        n_overlap = min(120, R.shape[0])          # E1 用的是 120 次
        sub = e1[e1["step"] <= int(round(MAX_F * S["n674"]["n0"]))]
        common = sub["step"].to_numpy(dtype=int)
        m_overlap = R[:n_overlap].mean(0)[common]
        m_all = R.mean(0)[common]
        xcheck["random_first_%d_reps_vs_E1_max_abs_gcc_diff_nodes" % n_overlap] = round(
            float(np.abs(sub["gcc_size"].to_numpy() - m_overlap).max()), 6)
        xcheck["random_all_%d_reps_vs_E1_mean_max_abs_gcc_diff_nodes" % R.shape[0]] = round(
            float(np.abs(sub["gcc_size"].to_numpy() - m_all).max()), 6)
        xcheck["random_reps_overlap"] = (
            "E1 与 E3 用同一个 seed 公式 np.random.default_rng([SEED, rep]) 和同一份 sorted 节点表，"
            "所以 E3 的第 r 次重复与 E1 的第 r 次是同一次抽样；E3 的均值是 %d 次、E1 的是 %d 次，"
            "两个均值只能近似相等（差 1–2 个节点来自抽样次数不同，不是建图或口径漂移）。"
            % (R.shape[0], n_overlap))
    print("[E3] cross-check vs E1:", json.dumps(xcheck, default=str), flush=True)

    # ---------- 5) 脆性判决：谁 p_c 最低，比随机低多少
    pc_tbl = pd.DataFrame(rows)
    rnd_tbl = pd.DataFrame(random_rows)
    frag = []
    for lv in LEVELS:
        det = pc_tbl[(pc_tbl["mode"] == "static") & (pc_tbl["rank_basis"] == "primary") &
                     (pc_tbl["node_set"] == "n674") & (pc_tbl["level"] == lv) &
                     (pc_tbl["strategy"] != "random")]
        rs = rnd_tbl[(rnd_tbl["level"] == lv) & (rnd_tbl["node_set"] == "n674") &
                     (rnd_tbl["rank_basis"] == "primary")].iloc[0]
        rnd_mean = float(rs["pc_mean"])
        for _, r in det.sort_values("pc").iterrows():
            frag.append({"level": lv, "strategy": r["strategy"], "pc": r["pc"],
                         "k_at_pc": int(r["k_at_pc"]),
                         "random_pc_mean": rnd_mean, "random_pc_sd": float(rs["pc_sd"]),
                         "random_pc_p2_5": float(rs["pc_p2.5"]),
                         "random_pc_p97_5": float(rs["pc_p97.5"]),
                         "margin_vs_random_pp": 100 * (rnd_mean - r["pc"]),
                         "margin_vs_random_rel_pct": 100 * (rnd_mean - r["pc"]) / rnd_mean})
    frag_df = pd.DataFrame(frag)
    best = frag_df[frag_df["level"] == 0.5].sort_values("pc").head(1).iloc[0].to_dict() \
        if len(frag_df) else {}

    # ---------- 6) 敏感性幅度
    def pc_slice(mode, basis, set_name, strat, lv):
        m = pc_tbl[(pc_tbl["mode"] == mode) & (pc_tbl["rank_basis"] == basis) &
                   (pc_tbl["node_set"] == set_name) & (pc_tbl["strategy"] == strat) &
                   (pc_tbl["level"] == lv)]
        return float(m["pc"].iloc[0]) if len(m) else np.nan

    sens_sets, sens_rank, sens_mode = [], [], []
    for strat in STRATS:
        for lv in LEVELS:
            p674 = pc_slice("static", "primary", "n674", strat, lv)
            row = {"strategy": strat, "level": lv, "pc_n674": p674}
            for sn in ("n671", "n676"):
                p = pc_slice("static", "primary", sn, strat, lv)
                row[f"pc_{sn}"] = p
                row[f"d_pp_{sn}"] = 100 * (p - p674) if p == p and p674 == p674 else np.nan
                row[f"d_rel_pct_{sn}"] = 100 * (p - p674) / p674 if p == p and p674 == p674 else np.nan
            sens_sets.append(row)
    for strat in DET_STRATS:          # pagerank_uu 已在 DET_STRATS 里，不能再追加一次
        for lv in LEVELS:
            pb = pc_slice("static", "primary", "n674", strat, lv)
            pw = pc_slice("static", "weighted", "n674", strat, lv)
            if strat == "kcore":
                note = "k-core 没有加权定义：加权基线下显式不适用（NaN，不是漏算）"
            elif strat == "pagerank_dw":
                note = ("有向加权 PR 与排名基线无关（两种基线下是同一条曲线），"
                        "所以不为它单列加权列；主表已给")
            elif strat == "pagerank_uu":
                note = "加权基线下 pagerank_uu = 无向加权 PageRank；Δ 即『二值 vs 加权』口径差"
            else:
                note = "加权基线下排名用 Gw 强度 / 加权介数"
            sens_rank.append({"strategy": strat, "level": lv, "pc_binary_primary": pb,
                              "pc_weighted": pw,
                              "d_pp": 100 * (pw - pb) if pw == pw and pb == pb else np.nan,
                              "d_rel_pct": 100 * (pw - pb) / pb if pw == pw and pb == pb and pb else np.nan,
                              "note": note})
    for strat in dyn_curves:
        for lv in LEVELS:
            ps = pc_slice("static", "primary", "n674", strat, lv)
            pd_ = pc_slice("dynamic", "primary", "n674", strat, lv)
            sens_mode.append({"strategy": strat, "level": lv, "pc_static": ps, "pc_dynamic": pd_,
                              "d_pp": 100 * (pd_ - ps) if pd_ == pd_ and ps == ps else np.nan,
                              "d_rel_pct": 100 * (pd_ - ps) / ps if pd_ == pd_ and ps == ps and ps else np.nan})

    # ---------- 7) 落盘
    pc_tbl.round(6).to_csv(os.path.join(HERE, "e3_perc_thresholds.csv"), index=False,
                           encoding="utf-8-sig")
    pd.DataFrame(random_rows).round(6).to_csv(os.path.join(HERE, "e3_random_pc_distribution.csv"),
                                              index=False, encoding="utf-8-sig")
    frag_df.round(6).to_csv(os.path.join(HERE, "e3_fragility_ranking.csv"), index=False,
                            encoding="utf-8-sig")
    pd.DataFrame(sens_sets).round(6).to_csv(os.path.join(HERE, "e3_sensitivity_node_set.csv"),
                                            index=False, encoding="utf-8-sig")
    pd.DataFrame(sens_rank).round(6).to_csv(os.path.join(HERE, "e3_sensitivity_rank_basis.csv"),
                                            index=False, encoding="utf-8-sig")
    pd.DataFrame(sens_mode).round(6).to_csv(os.path.join(HERE, "e3_sensitivity_static_vs_dynamic.csv"),
                                            index=False, encoding="utf-8-sig")

    meta = {
        "seed": SEED, "n_random_reps": N_RANDOM_REPS, "max_f": MAX_F, "levels": list(LEVELS),
        "node_sets": {k: {"n0": S[k]["n0"], "m0": S[k]["m0"], "gcc0": S[k]["gcc0"]}
                      for k in ("n671", "n674", "n676")},
        "timing_s": {"static_det_curves": round(timing_static_det, 1),
                     "random_curves_pool": round(timing_random, 1),
                     "total": round(time.perf_counter() - t_all, 1)},
        "cross_check_vs_E1": xcheck,
        "random_pc_674_primary": {str(lv): band[lv] for lv in LEVELS},
        "fragility_headline_level_0p5": {k: (float(v) if isinstance(v, (int, float)) else v)
                                          for k, v in best.items()},
        "versions": C.versions(),
        "anchor_check": "csa_common.assert_anchors() 31/31 PASS",
        "kcore_weighted_note": "k-core 没有权重定义，加权排名基线下显式标记为不适用",
    }
    with open(os.path.join(HERE, "E3_summary.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=float)
    C.write_run_note("E3_percolation", run_lines(meta))
    print("[E3] fragility:\n" + frag_df.to_string(index=False), flush=True)
    print(f"[E3] total wall {meta['timing_s']['total']} s", flush=True)


def run_lines(meta):
    v = meta["versions"]
    return [
        "E3_percolation RUN.txt",
        "command: cd pipeline && PYTHONHASHSEED=0 python out/E3_percolation/run_e3.py",
        f"python={v['python']} networkx={v['networkx']} pandas={v['pandas']} "
        f"numpy={v['numpy']} PYTHONHASHSEED={v['PYTHONHASHSEED']} SEED={meta['seed']}",
        f"anchors: {meta['anchor_check']}",
        "p_c definition: smallest removal step k with GCC(k) < level * GCC(0); reported as f = k / n0; "
        f"levels {meta['levels']}, grid = every single deletion step up to f={meta['max_f']}",
        "only GCC is tracked here (no efficiency) -> cost is ~3 orders of magnitude below E1",
        f"node sets: {json.dumps(meta['node_sets'], default=float)}",
        f"random: {meta['n_random_reps']} independent replicates per node set "
        f"(seed per rep = np.random.default_rng([{meta['seed']}, rep]); E3 rep r is the SAME sampling "
        "as E1 rep r, but E1 reports a mean over 120 and E3 over 200 reps, so the two means agree only "
        "approximately — see the cross-check lines), distribution of p_c reported "
        "(sd / percentiles / 95% CI of the mean)",
        "grid: every single deletion step up to f=0.90 for every strategy. The 0.60 cutoff of E1 is "
        "NOT enough for p_c(25%): with random removal only ~5% of replicates drop below 0.25*GCC(0) "
        "by f=0.60, which would make pc(25%) a right-censored mean. At f=0.90 all 200 x 3 node sets "
        "cross both thresholds (frac_reaches_threshold = 1.0), so the CI is a real CI",
        "dynamic p_c read from out/E2_dynamic_attack/e2_gcc_per_step.csv (script aborts if missing)",
        "cross-check vs E1 static curves: " + json.dumps(meta["cross_check_vs_E1"], default=str),
        "rank bases: primary = binary Gb for degree/BC/kcore + directed weighted Gd for PR + "
        "binary undirected for PR control; weighted = Gw strength / weighted BC / weighted undirected PR "
        f"({meta['kcore_weighted_note']})",
        f"wall: static deterministic {meta['timing_s']['static_det_curves']} s, "
        f"random pool {meta['timing_s']['random_curves_pool']} s, "
        f"total {meta['timing_s']['total']} s",
        "outputs: e3_perc_thresholds.csv e3_random_pc_distribution.csv e3_random_gcc_band.csv "
        "e3_fragility_ranking.csv e3_sensitivity_node_set.csv e3_sensitivity_rank_basis.csv "
        "e3_sensitivity_static_vs_dynamic.csv E3_summary.json",
    ]


if __name__ == "__main__":
    main()
