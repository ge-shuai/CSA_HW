"""B3_resolution_scan — 加权 Louvain 的分辨率（γ）扫描。

背景：Fortunato & Barthélemy (2007, PNAS 104:36) 的分辨率限制。
    Q(γ) = Σ_c [ e_c − γ·a_c² ] ,  e_c = I_c/m , a_c = K_c/(2m) , 2m = 总边权
γ 只乘在零模型项上 → **不同 γ 下的 Q 不可比大小**（零模型换了）。本单元因此
不把 Q(γ) 的峰值当判据，改用四个与零模型无关的判据：
  (1) 与 γ=1 参照划分的 NMI（20×20 seed 全交叉，报 mean±sd）
  (2) 同一 γ 内 seed 间 NMI（= 该 γ 的可复现性，也是 (1) 的天花板）
  (3) 共识矩阵熵（seed 间共属关系的逐对二分熵均值，越小越稳）
  (4) 成对合并判据的解析阈值 γ* = (W_AB/m)/(2 a_A a_B)：γ<γ* 时把 A、B 合并使 Q 升高
      —— 这是分辨率限制可直接检验的形式，用它在事件表里给出"预测相变点 vs 实测相变点"。

硬口径（CONVENTIONS §1/§3/§7/§8）：
  - 图只从 `csa_common.build_all()` 取；开跑先 `C.assert_anchors()`。
  - 社区数同时报 `n_comms_total` / `n_comms_nonsingleton`；主口径 = 671 有属性节点，
    676（含 5 悬空）作对照列。
  - 两套划分的比较只用 `C.partition_metrics()`。
  - 每个 γ 跑 20 个 seed，报 mean±sd，不报单值。
  - 不使用因果措辞。

输出（只写 pipeline/out/B3_resolution_scan/）：
  resolution_scan.csv / resolution_summary.csv / merge_split_events.csv
  pair_merge_threshold.csv / resolution_stability.csv / verdict.json / RUN.txt
"""
from __future__ import annotations

import os
import sys
import time
import math
import platform

import numpy as np
import pandas as pd
import networkx as nx

PIPELINE_DIR = r"C:/Users/bfy32/Documents/Qoder/2026-11-13/f93e4d48/pipeline"
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

import csa_common as C  # noqa: E402

MODULE = "B3_resolution_scan"
OUT_DIR = os.path.join(C.OUT_ROOT, MODULE)
SEED = C.SEED                      # 20261113（CONVENTIONS §8）
N_SEEDS = 20                       # 判据要求 ≥20
SEEDS = [SEED + i for i in range(N_SEEDS)]
REF_GAMMA = 1.0

LOG: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    LOG.append(str(msg))


# --------------------------------------------------------------------- γ 网格

def coarse_grid() -> list[float]:
    main = [round(0.1 * i, 2) for i in range(1, 31)]      # 0.1 ... 3.0, 步长 0.1
    extra = [0.05, 0.25, 0.75]                             # 关键点（0.5/1.5/2.5/3.0 已在网格内）
    return sorted(set(main) | set(extra) | {REF_GAMMA})


# ---------------------------------------------------------------- 图上的权工具

class WTools:
    """边权小工具（只做统计，不重新建图）。"""

    # 记号（与 networkx / Newman 一致）：S = Σ_无向边 w =「总边权」= 加权版的边数 m；
    # Σ_i k_i = 2S。Q(γ) = Σ_c [ I_c/S − γ (K_c/2S)² ]，I_c = 社区内部边权、K_c = 社区强度和。
    def __init__(self, G: nx.Graph):
        self.G = G
        self.S = float(sum(float(d.get("weight", 1)) for _, _, d in G.edges(data=True)))
        self.twoS = 2.0 * self.S
        self.deg = {n: float(d) for n, d in G.degree(weight="weight")}
        self.adj = {n: [(nbr, float(dd.get("weight", 1))) for nbr, dd in G[n].items()]
                    for n in G.nodes()}

    def between(self, A: set, B: set) -> tuple[float, int]:
        """A、B 之间的边权与边条数（A、B 需不相交；用邻接表，成本 O(Σdeg(小集合))）。"""
        small, big = (A, B) if len(A) <= len(B) else (B, A)
        w, k = 0.0, 0
        for u in small:
            for v, ww in self.adj[u]:
                if v in big:
                    w += ww
                    k += 1
        return w, k

    def internal(self, A: set) -> tuple[float, int]:
        w, k = 0.0, 0
        for u in A:
            for v, ww in self.adj[u]:
                if v in A:
                    w += ww
                    k += 1
        return w / 2.0, k // 2

    def strength(self, A: set) -> float:
        return float(sum(self.deg.get(n, 0.0) for n in A))

    def a(self, A: set) -> float:
        return self.strength(A) / self.twoS

    def gamma_star(self, A: set, B: set) -> dict:
        """合并 A、B 使 Q 升高的临界分辨率 γ*。ΔQ_merge(γ) = W/S − 2γ a_A a_B。"""
        W, n_e = self.between(A, B)
        IA, _ = self.internal(A)
        IB, _ = self.internal(B)
        aA, aB = self.a(A), self.a(B)
        gs = (W / self.S) / (2 * aA * aB) if (aA > 0 and aB > 0 and W > 0) else 0.0
        return {
            "gamma_pred_merge": round(float(gs), 4),
            "cross_weight": round(W, 1), "cross_edges": int(n_e),
            "internal_w_A": round(IA, 1), "internal_w_B": round(IB, 1),
            "coupling_ratio": round(W / (W + IA + IB), 4) if (W + IA + IB) > 0 else 0.0,
            "dQ_merge_at_gamma1": round(W / self.S - 2.0 * aA * aB, 6),
        }

    def dQ_formula(self, A: set, B: set, gamma: float) -> float:
        W, _ = self.between(A, B)
        return W / self.S - 2 * gamma * self.a(A) * self.a(B)

    def nbr_jaccard(self, A: set, B: set) -> float:
        na = set()
        for x in A:
            na |= set(self.G.neighbors(x))
        nb = set()
        for x in B:
            nb |= set(self.G.neighbors(x))
        na -= A
        nb -= B
        u = len(na | nb)
        return round(len(na & nb) / u, 4) if u else 0.0


# ------------------------------------------------------------------ 单次运行

def run_one(G: nx.Graph, attr: set, wt: WTools, gamma: float, seed: int) -> dict:
    r = C.louvain(G, seed=seed, resolution=gamma)
    part = r["partition"]
    sizes = [len(c) for c in part]
    ns = sorted([s for s in sizes if s > 1], reverse=True)
    sizes671 = [len(c & attr) for c in part]
    ns671 = sorted([s for s in sizes671 if s > 1], reverse=True)
    Q_ref = nx.community.modularity(G, part, weight="weight", resolution=REF_GAMMA)
    return {
        "gamma": gamma,
        "seed": seed,
        "n_comms_total": r["n_comms_total"],
        "n_comms_nonsingleton_676": r["n_comms_nonsingleton"],
        "n_singleton": r["n_singleton"],
        "n_covered": r["n_covered"],
        "n_comms_nonsingleton": len(ns671),          # 主口径 = 671
        "largest_676": r["largest"],
        "largest": ns671[0] if ns671 else 0,          # 主口径 = 671
        "second_largest": ns671[1] if len(ns671) > 1 else 0,
        "mean_size_nonsingleton": round(float(np.mean(ns671)), 3) if ns671 else float("nan"),
        "median_size_nonsingleton": float(np.median(ns671)) if ns671 else float("nan"),
        "min_size_nonsingleton": int(min(ns671)) if ns671 else 0,
        "Q_same_null": round(r["Q"], 6),              # 零模型系数 = 本行 γ
        "Q_rescored_at_gamma1": round(float(Q_ref), 6),  # 同一划分换成 γ=1 零模型（仅对照）
        "_partition": part,
    }


# ------------------------------------------------------------ 标签 / 共识矩阵

def label_matrix(runs: list[dict], pos: dict) -> np.ndarray:
    M = np.zeros((len(runs), len(pos)), dtype=np.int16)
    for k, r in enumerate(runs):
        for ci, c in enumerate(r["_partition"]):
            for n in c:
                M[k, pos[n]] = ci
    return M


def consensus_stats(M: np.ndarray, cols: np.ndarray) -> dict:
    """cols = 主口径（671）节点的列下标。共识矩阵 = seed 间同社区比例。"""
    A = M[:, cols].astype(np.int32)
    n = A.shape[1]
    co = np.zeros((n, n), dtype=np.float64)
    for k in range(A.shape[0]):
        co += (A[k][:, None] == A[k][None, :])
    co /= A.shape[0]
    iu = np.triu_indices(n, k=1)
    p = co[iu]
    eps = 1e-12
    h = -(p * np.log2(np.clip(p, eps, 1)) + (1 - p) * np.log2(np.clip(1 - p, eps, 1)))
    return {
        "consensus_entropy_bits": round(float(h.mean()), 5),
        "consensus_mean": round(float(p.mean()), 4),
        "consensus_frac_pairs_ambiguous": round(float(((p > 0.0) & (p < 1.0)).mean()), 4),
    }


def medoid_index(Wn: np.ndarray) -> int:
    """同一 γ 内与其他 run 的 NMI 均值最高者。Wn = 该 γ 内 20×20 NMI 矩阵。"""
    n = Wn.shape[0]
    s = (Wn.sum(axis=1) - np.diag(Wn)) / max(n - 1, 1)
    return int(np.argmax(s))


# ------------------------------------------------------- 子社区重合工具

def contained_pieces(parent: set, comms: list[frozenset], min_frac: float = 0.8,
                     min_nodes: int = 3) -> list[frozenset]:
    """在划分 comms 里找「基本落在 parent 内部」的子社区（≥min_nodes 且 ≥min_frac 重合）。"""
    hits = []
    for x in comms:
        inter = len(parent & x)
        if len(x) >= min_nodes and inter >= min_nodes and inter >= min_frac * len(x):
            hits.append(x)
    return hits


def comm_sets(r: dict) -> list[frozenset]:
    return [frozenset(c) for c in r["_partition"]]


def main():
    t_start = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    log(f"# {MODULE} · started {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"$ python {' '.join(sys.argv)}")
    log(f"cwd={os.getcwd()}")

    b = C.build_all()
    Gu = b["Gu"]
    attr = set(b["nodes"]["node"])
    anchors = C.check_anchors(b)
    C.assert_anchors(b)
    log(f"anchors: {int(anchors['ok'].sum())}/{len(anchors)} PASS ; C.assert_anchors() OK")
    log(f"Gu(676) = {Gu.number_of_nodes()}n/{Gu.number_of_edges()}e ; attributed = {len(attr)}")

    wt = WTools(Gu)
    log(f"total edge weight S = {wt.S:.0f} ; sum of strengths 2S = {wt.twoS:.0f} ; "
        f"FB scale sqrt(2S) = {math.sqrt(wt.twoS):.1f} nodes  (无权对照 sqrt(2|E|) = "
        f"{math.sqrt(2*Gu.number_of_edges()):.1f})")

    indeg = dict(Gu.degree())
    short = {n: n.removesuffix(".ustc.edu.cn") for n in Gu.nodes()}
    nodemeta = {n: (indeg.get(n, 0), short[n]) for n in Gu.nodes()}

    def names(s, k=8):
        return "|".join(x for _, x in sorted((nodemeta[n] for n in s), reverse=True)[:k])

    nodes_all = list(Gu.nodes())
    pos = {n: i for i, n in enumerate(nodes_all)}
    cols671 = np.array([pos[n] for n in nodes_all if n in attr], dtype=int)

    gammas_c = coarse_grid()
    log(f"coarse γ ({len(gammas_c)}): {gammas_c}")
    log(f"seeds ({N_SEEDS}): {SEEDS[0]}..{SEEDS[-1]}")

    def scan(glist, tag, into):
        for g in glist:
            t0 = time.time()
            R = [run_one(Gu, attr, wt, g, s) for s in SEEDS]
            into[g] = R
            a = lambda k: np.array([r[k] for r in R], float)  # noqa: E731
            log(f"  [{tag}] γ={g:<5} {time.time()-t0:5.1f}s | n_ns671 {int(a('n_comms_nonsingleton').min())}-"
                f"{int(a('n_comms_nonsingleton').max())} | n_total {int(a('n_comms_total').min())}-"
                f"{int(a('n_comms_total').max())} | largest {int(a('largest').min())}-"
                f"{int(a('largest').max())} | Q(γ) {a('Q_same_null').min():.4f}-"
                f"{a('Q_same_null').max():.4f}")

    runs: dict[float, list[dict]] = {}
    t0 = time.time()
    scan(gammas_c, "coarse", runs)
    log(f"coarse scan {time.time()-t0:.1f}s")

    # ------------------------------------------------- NMI（只用 partition_metrics）
    Gcmp = C.induced(Gu, sorted(attr))     # 主比较口径 = 671

    def pm_matrices(RA, RB):
        """一次 C.partition_metrics 调用同时取 NMI 与 ARI（ARI 已作机会校正，跨 γ 可比性更好）。"""
        n, k = len(RA), len(RB)
        nmi = np.zeros((n, k))
        ari = np.zeros((n, k))
        for i, x in enumerate(RA):
            for j, y in enumerate(RB):
                d = C.partition_metrics(Gcmp, x["_partition"], y["_partition"])
                nmi[i, j] = d["NMI"]
                ari[i, j] = d["ARI"]
        return nmi, ari

    t0 = time.time()
    within_nmi: dict[float, np.ndarray] = {}
    nmi_vs_ref: dict[float, np.ndarray] = {}
    within_ari: dict[float, np.ndarray] = {}
    ari_vs_ref: dict[float, np.ndarray] = {}
    ref_runs = runs[REF_GAMMA]
    n_pm_calls = 0
    for g in sorted(runs):
        within_nmi[g], within_ari[g] = pm_matrices(runs[g], runs[g])
        n_pm_calls += len(runs[g]) ** 2
        if abs(g - REF_GAMMA) < 1e-9:
            nmi_vs_ref[g], ari_vs_ref[g] = within_nmi[g], within_ari[g]
        else:
            nmi_vs_ref[g], ari_vs_ref[g] = pm_matrices(runs[g], ref_runs)
            n_pm_calls += len(runs[g]) * len(ref_runs)
    log(f"NMI+ARI coarse {time.time()-t0:.1f}s ({n_pm_calls:,} partition_metrics calls)")

    def curve(res):
        ns, nm = {}, {}
        for g in sorted(res):
            mi = medoid_index(within_nmi[g])
            ns[g] = res[g][mi]["n_comms_nonsingleton"]
            M = nmi_vs_ref[g]
            v = M[~np.eye(M.shape[0], dtype=bool)] if abs(g - REF_GAMMA) < 1e-9 else M.ravel()
            nm[g] = float(v.mean())
        return ns, nm

    med_ns, mean_nmi = curve(runs)

    # ---------------------------------------------------- 精化：相变窗口加密
    windows = []
    gs = sorted(runs)
    for x, y in zip(gs, gs[1:]):
        dn = abs(med_ns[y] - med_ns[x])
        dq = abs(mean_nmi[y] - mean_nmi[x])
        if dn >= 3 or dq >= 0.06:
            windows.append((dn + 20.0 * dq, x, y, dn, round(dq, 4)))
    windows.sort(reverse=True)
    picked = windows[:12]
    fine: list[float] = []
    for _, x, y, *_ in picked:
        added = 0
        v = round(x + 0.025, 3)
        while v < y - 1e-9 and added < 3:      # 每个窗口最多插 3 个（真实网格窗口宽 0.1）
            if all(abs(v - z) > 1e-9 for z in gs + fine):
                fine.append(round(v, 3))
                added += 1
            v = round(v + 0.025, 3)
    log(f"refine windows (score,γlo,γhi,Δn,ΔNMI): {[(round(w[0],2), w[1], w[2], w[3], w[4]) for w in picked]}")
    log(f"fine γ ({len(fine)}): {fine}")
    if fine:
        t0 = time.time()
        scan(fine, "fine", runs)
        for g in fine:
            within_nmi[g], within_ari[g] = pm_matrices(runs[g], runs[g])
            nmi_vs_ref[g], ari_vs_ref[g] = pm_matrices(runs[g], ref_runs)
            n_pm_calls += len(runs[g]) ** 2 + len(runs[g]) * len(ref_runs)
        log(f"fine pass {time.time()-t0:.1f}s")

    # ------------------------------------------------------------ scan.csv
    rows = []
    for g in sorted(runs):
        off = nmi_vs_ref[g]
        eye = np.eye(off.shape[0], dtype=bool)
        offa = ari_vs_ref[g]
        vals = np.array([off[i][~eye[i]].mean() if abs(g - REF_GAMMA) < 1e-9 else off[i].mean()
                         for i in range(len(runs[g]))])
        valsa = np.array([offa[i][~eye[i]].mean() if abs(g - REF_GAMMA) < 1e-9 else offa[i].mean()
                          for i in range(len(runs[g]))])
        for i, r in enumerate(runs[g]):
            d = {k: v for k, v in r.items() if not k.startswith("_")}
            d["pass"] = "coarse" if g in set(gammas_c) else "fine"
            d["n_seeds_at_gamma"] = len(runs[g])
            d["nmi_vs_ref_mean_over_ref_seeds"] = round(float(vals[i]), 4)
            d["ari_vs_ref_mean_over_ref_seeds"] = round(float(valsa[i]), 4)
            d["node_set_for_nmi"] = "attributed_671"
            rows.append(d)
    scan_df = pd.DataFrame(rows)
    C.save_csv(scan_df, MODULE, "resolution_scan.csv")
    log(f"wrote resolution_scan.csv rows={len(scan_df)}")

    # ------------------------------------------------------- summary.csv
    medoid_of: dict[float, dict] = {}
    for g in sorted(runs):
        medoid_of[g] = runs[g][medoid_index(within_nmi[g])]
    ref_med_i = medoid_index(within_nmi[REF_GAMMA])

    ref_med_sets = comm_sets(medoid_of[REF_GAMMA])
    sum_rows = []
    for g in sorted(runs):
        R = runs[g]
        a = lambda k: np.array([r[k] for r in R], float)  # noqa: E731
        M = nmi_vs_ref[g]
        v = M[~np.eye(M.shape[0], dtype=bool)] if abs(g - REF_GAMMA) < 1e-9 else M.ravel()
        Wn = within_nmi[g]
        wv = Wn[~np.eye(Wn.shape[0], dtype=bool)]
        Wa = ari_vs_ref[g]
        av = Wa[~np.eye(Wa.shape[0], dtype=bool)] if abs(g - REF_GAMMA) < 1e-9 else Wa.ravel()
        Wi = within_ari[g]
        wiv = Wi[~np.eye(Wi.shape[0], dtype=bool)]
        P = comm_sets(medoid_of[g])
        n_split = n_merged = 0
        if abs(g - REF_GAMMA) > 1e-9:
            n_split = sum(1 for c in ref_med_sets if len(c) >= 3 and len(contained_pieces(set(c), P)) >= 2)
            n_merged = sum(1 for x in P if len(x) >= 3 and len(contained_pieces(set(x), ref_med_sets)) >= 2)
        _m2m = ({"NMI": 1.0, "ARI": 1.0} if abs(g - REF_GAMMA) < 1e-9 else
                C.partition_metrics(Gcmp, medoid_of[g]["_partition"],
                                    medoid_of[REF_GAMMA]["_partition"]))
        med_to_med = round(float(_m2m["NMI"]), 4)
        med_to_med_ari = round(float(_m2m["ARI"]), 4)
        sum_rows.append({
            "pass": "coarse" if g in set(gammas_c) else "fine",
            "gamma": g,
            "n_seeds": len(R),
            "n_comms_total_mean": round(a("n_comms_total").mean(), 2),
            "n_comms_total_sd": round(a("n_comms_total").std(ddof=1), 2),
            "n_comms_nonsingleton_mean": round(a("n_comms_nonsingleton").mean(), 2),
            "n_comms_nonsingleton_sd": round(a("n_comms_nonsingleton").std(ddof=1), 2),
            "n_comms_nonsingleton_min": int(a("n_comms_nonsingleton").min()),
            "n_comms_nonsingleton_max": int(a("n_comms_nonsingleton").max()),
            "n_comms_nonsingleton_676_mean": round(a("n_comms_nonsingleton_676").mean(), 2),
            "n_singleton_mean": round(a("n_singleton").mean(), 2),
            "largest_mean": round(a("largest").mean(), 2),
            "largest_sd": round(a("largest").std(ddof=1), 2),
            "largest_min": int(a("largest").min()),
            "largest_max": int(a("largest").max()),
            "largest_676_mean": round(a("largest_676").mean(), 2),
            "second_largest_mean": round(a("second_largest").mean(), 2),
            "mean_size_nonsingleton_mean": round(float(np.nanmean(a("mean_size_nonsingleton"))), 3),
            "min_size_nonsingleton_mean": round(a("min_size_nonsingleton").mean(), 2),
            "Q_same_null_mean": round(a("Q_same_null").mean(), 4),
            "Q_same_null_sd": round(a("Q_same_null").std(ddof=1), 4),
            "Q_rescored_at_gamma1_mean": round(a("Q_rescored_at_gamma1").mean(), 4),
            "ari_vs_gamma1_mean": round(float(av.mean()), 4),
            "ari_vs_gamma1_sd": round(float(av.std(ddof=1)), 4),
            "within_seed_ARI_mean": round(float(wiv.mean()), 4),
            "within_seed_NMI_mean": round(float(wv.mean()), 4),
            "within_seed_NMI_sd": round(float(wv.std(ddof=1)), 4),
            "nmi_vs_gamma1_mean": round(float(v.mean()), 4),
            "nmi_vs_gamma1_sd": round(float(v.std(ddof=1)), 4),
            "nmi_vs_gamma1_min": round(float(v.min()), 4),
            "nmi_vs_gamma1_max": round(float(v.max()), 4),
            "medoid_nmi_vs_ref_medoid": med_to_med,
            "medoid_ari_vs_ref_medoid": med_to_med_ari,
            "medoid_seed": R[medoid_index(Wn)]["seed"],
            "n_ref_comms_split_here": n_split,
            "n_comms_here_merged_from_ref": n_merged,
            **consensus_stats(label_matrix(R, pos), cols671),
        })
    summary = pd.DataFrame(sum_rows)
    # 派生列：相对 seed 噪声天花板的一致性 + 相邻阶梯的社区数跳变（相变标记）
    _ref_off = within_nmi[REF_GAMMA]
    _ref_off_a = within_ari[REF_GAMMA]
    ref_ceiling_early = float(_ref_off[~np.eye(_ref_off.shape[0], dtype=bool)].mean())
    ref_ari_ceiling_early = float(_ref_off_a[~np.eye(_ref_off_a.shape[0], dtype=bool)].mean())
    summary = summary.sort_values("gamma").reset_index(drop=True)
    summary["nmi_vs_gamma1_rel_ceiling"] = (summary["nmi_vs_gamma1_mean"] / ref_ceiling_early).round(4)
    summary["ari_vs_gamma1_rel_ceiling"] = (summary["ari_vs_gamma1_mean"] / ref_ari_ceiling_early).round(4)
    summary["within_seed_NMI_rel_max"] = (
        summary["within_seed_NMI_mean"] / float(summary["within_seed_NMI_mean"].max())).round(4)
    d_ns = summary["n_comms_nonsingleton_mean"].diff()
    d_g = summary["gamma"].diff()
    summary["dn_ns_vs_prev_ladder_step"] = d_ns.round(2)
    summary["dn_ns_per_dgamma"] = (d_ns / d_g).round(1)
    _pooled_sd = float(summary["n_comms_nonsingleton_sd"].median())
    _thr_ns = max(3.0, 3.0 * _pooled_sd)          # ≈3 倍 seed 间 sd
    summary["phase_transition_flag"] = (d_ns.abs() >= _thr_ns).astype(int)
    summary["pooled_seed_sd_n_ns"] = round(_pooled_sd, 3)
    summary["phase_transition_threshold_dN"] = round(_thr_ns, 2)
    C.save_csv(summary, MODULE, "resolution_summary.csv")
    log(f"wrote resolution_summary.csv rows={len(summary)}")

    # --------------------------------- γ=1 medoid 的成对合并阈值（分辨率限制判据）
    def self_check():
        R = medoid_of[REF_GAMMA]["_partition"]
        myQ = 0.0
        for c in R:
            I, _ = wt.internal(set(c))
            myQ += I / wt.S - 1.0 * wt.a(set(c)) ** 2
        nxQ = nx.community.modularity(Gu, R, weight="weight", resolution=1.0)
        A, B = set(R[0]), set(R[1])
        out = []
        for gt in (0.3, 1.0, 2.7):
            merged = [A | B] + [set(c) for c in R[2:]]
            act = (nx.community.modularity(Gu, merged, weight="weight", resolution=gt)
                   - nx.community.modularity(Gu, R, weight="weight", resolution=gt))
            out.append((gt, wt.dQ_formula(A, B, gt), act))
        return myQ, nxQ, out

    myQ, nxQ, dqchk = self_check()
    log(f"self-check Q(γ=1, medoid): formula {myQ:.6f} vs networkx {nxQ:.6f} |Δ|={abs(myQ-nxQ):.2e}")
    for gt, f, act in dqchk:
        log(f"self-check ΔQ_merge @γ={gt}: formula {f:+.6f} vs recomputed {act:+.6f} "
            f"|Δ|={abs(f-act):.2e}")

    ref_sets = [set(c) for c in ref_med_sets if len(c) >= 2]
    pair_rows = []
    for i in range(len(ref_sets)):
        for j in range(i + 1, len(ref_sets)):
            d = wt.gamma_star(ref_sets[i], ref_sets[j])
            pair_rows.append({
                "comm_i": i, "n_i": len(ref_sets[i]), "names_i": names(ref_sets[i]),
                "comm_j": j, "n_j": len(ref_sets[j]), "names_j": names(ref_sets[j]),
                **d, "nbr_jaccard": wt.nbr_jaccard(ref_sets[i], ref_sets[j]),
                "merge_attractive_at_gamma1": bool(d["dQ_merge_at_gamma1"] > 0),
                "margin_to_gamma1": round(1.0 - d["gamma_pred_merge"], 4),
            })
    pairs = pd.DataFrame(pair_rows).sort_values("gamma_pred_merge", ascending=False)
    C.save_csv(pairs, MODULE, "pair_merge_threshold.csv")
    log(f"wrote pair_merge_threshold.csv pairs={len(pairs)} | γ*>1（γ=1 下按判据本应合并）: "
        f"{int(pairs['merge_attractive_at_gamma1'].sum())} | 0.5<γ*≤1（临界可分）: "
        f"{int(((pairs['gamma_pred_merge']>0.5)&(pairs['gamma_pred_merge']<=1)).sum())} | "
        f"max γ*={pairs['gamma_pred_merge'].max():.3f}")

    # ------------------------------------------------- 合并/分裂事件（沿 γ 阶梯）
    LAB = {g: label_matrix(runs[g], pos) for g in runs}

    def frac_split(g, A: set, B: set, row=None) -> float:
        idxA = [pos[n] for n in A if n in pos]
        idxB = [pos[n] for n in B if n in pos]
        if not idxA or not idxB:
            return float("nan")
        M = LAB[g] if row is None else LAB[g][[row]]
        c = 0
        for k in range(M.shape[0]):
            la, lb = M[k, idxA], M[k, idxB]
            if int(np.bincount(la).argmax()) != int(np.bincount(lb).argmax()):
                c += 1
        return round(c / M.shape[0], 3)

    ladder = sorted(runs)
    ev_rows = []
    ev_nodes: dict[int, tuple[set, set]] = {}
    ev_id = 0
    for x, y in zip(ladder, ladder[1:]):
        Px, Py = comm_sets(medoid_of[x]), comm_sets(medoid_of[y])
        seen: set = set()
        # γ 向上：γ_x 的一个社区在 γ_y 里散成 ≥2 块 → split
        for c in Px:
            if len(c) < 6:
                continue
            kids = contained_pieces(set(c), Py)
            if len(kids) < 2:
                continue
            kids = sorted(kids, key=len, reverse=True)[:4]
            for p in range(len(kids)):
                for q in range(p + 1, len(kids)):
                    key = (kids[p], kids[q])
                    if key in seen:
                        continue
                    seen.add(key)
                    ev_id += 1
                    A, B = set(kids[p]), set(kids[q])
                    d = wt.gamma_star(A, B)
                    ev_nodes[ev_id] = (A, B)
                    ev_rows.append({
                        "event_id": ev_id, "type": "split_when_gamma_up",
                        "gamma_low": x, "gamma_high": y, "parent_size_low": len(c), "merged_size_high": None,
                        "n_pieces": len(kids), "pieceA_size": len(A), "pieceB_size": len(B),
                        "pieceA_names": names(A), "pieceB_names": names(B),
                        "split_frac_at_gamma_low": frac_split(x, A, B),
                        "split_frac_at_gamma_high": frac_split(y, A, B),
                        "gamma_pred_split": d["gamma_pred_merge"],
                        "cross_weight": d["cross_weight"], "cross_edges": d["cross_edges"],
                        "coupling_ratio": d["coupling_ratio"],
                        "nbr_jaccard": wt.nbr_jaccard(A, B),
                        "involves_ref_gamma1": bool(x <= REF_GAMMA <= y),
                    })
        # γ 向上：γ_x 的两个社区在 γ_y 里合成一个 → merge
        for cc in Py:
            if len(cc) < 6:
                continue
            par = contained_pieces(set(cc), Px)
            if len(par) < 2:
                continue
            par = sorted(par, key=len, reverse=True)[:4]
            for p in range(len(par)):
                for q in range(p + 1, len(par)):
                    key = (par[q], par[p])
                    if key in seen:
                        continue
                    seen.add(key)
                    ev_id += 1
                    A, B = set(par[p]), set(par[q])
                    d = wt.gamma_star(A, B)
                    ev_nodes[ev_id] = (A, B)
                    ev_rows.append({
                        "event_id": ev_id, "type": "merge_when_gamma_up",
                        "gamma_low": x, "gamma_high": y, "parent_size_low": None, "merged_size_high": len(cc),
                        "n_pieces": 2, "pieceA_size": len(A), "pieceB_size": len(B),
                        "pieceA_names": names(A), "pieceB_names": names(B),
                        "split_frac_at_gamma_low": frac_split(x, A, B),
                        "split_frac_at_gamma_high": frac_split(y, A, B),
                        "gamma_pred_split": d["gamma_pred_merge"],
                        "cross_weight": d["cross_weight"], "cross_edges": d["cross_edges"],
                        "coupling_ratio": d["coupling_ratio"],
                        "nbr_jaccard": wt.nbr_jaccard(A, B),
                        "involves_ref_gamma1": bool(x <= REF_GAMMA <= y),
                    })
    events = pd.DataFrame(ev_rows)
    if len(events):
        tail_split, tail_all = [], []
        for r in events.to_dict("records"):
            A, B = ev_nodes[int(r["event_id"])]
            tail = [g for g in ladder if g > r["gamma_high"] + 1e-9]
            if not tail:
                tail_split.append(float("nan"))
                tail_all.append(float("nan"))
                continue
            hi_only = [g for g in tail if g <= 3.0]
            tail_split.append(round(float(np.mean([frac_split(g, A, B) for g in hi_only])), 3))
            tail_all.append(len(hi_only))
        events["split_frac_above_ladder"] = tail_split
        events["n_ladder_steps_above"] = tail_all
    events["scheme"] = "ladder_step"

    # --------- 事件表 B：以 γ=1 参照社区为单位的「经验相变点 vs 解析阈值 γ*」
    # 对象集合 = ①γ=1 medoid 的两两社区（看往下降到多少 γ 才并起来）
    #           ②γ=1 medoid 每个社区在 γ=max 里散出的子块两两（看往上到多少 γ 才分开）
    ref_sets_sorted = sorted([set(c) for c in ref_med_sets if len(c) >= 3], key=len, reverse=True)
    cand: dict[str, set] = {f"ref{i:02d}": S for i, S in enumerate(ref_sets_sorted)}
    Pmax = comm_sets(medoid_of[max(ladder)])
    intra_trip = []
    for pi, S in enumerate(ref_sets_sorted):
        if len(S) < 6:
            continue
        kids = sorted(contained_pieces(S, Pmax), key=len, reverse=True)[:4]
        kids = [k for k in kids if len(k) >= 3]
        if len(kids) < 2:
            continue
        ks = []
        for ki, k in enumerate(kids):
            key = f"p{pi:02d}k{ki}"
            cand[key] = set(k)
            ks.append(key)
        for xx in range(len(ks)):
            for yy in range(xx + 1, len(ks)):
                intra_trip.append((pi, ks[xx], ks[yy], len(S)))
    keys = list(cand)
    kidx = {k: i for i, k in enumerate(keys)}
    idxs = [np.array([pos[n] for n in cand[k] if n in pos], dtype=int) for k in keys]
    MAJ, INT = {}, {}
    t0 = time.time()
    for g in ladder:
        M = LAB[g]
        nl = int(M.max()) + 1
        mj = np.zeros((len(keys), M.shape[0]), dtype=np.int16)
        it = np.zeros((len(keys), M.shape[0]), dtype=bool)
        for si, ix in enumerate(idxs):
            if len(ix) == 0:
                continue
            for ri in range(M.shape[0]):
                cnt = np.bincount(M[ri, ix], minlength=nl)
                top = int(cnt.argmax())
                mj[si, ri] = top
                it[si, ri] = cnt[top] >= 0.8 * len(ix)
        MAJ[g], INT[g] = mj, it
    log(f"pair-curve majority tables {time.time()-t0:.1f}s ({len(ladder)}γ × {len(keys)} sets)")

    def sf_curve(ka: str, kb: str) -> list[float]:
        ia, ib = kidx[ka], kidx[kb]
        return [float(np.mean(MAJ[g][ia] != MAJ[g][ib])) for g in ladder]

    probe = [0.05, 0.3, 0.6, 0.9, 1.0, 1.1, 1.5, 2.0, 3.0]
    probe_g = [min(ladder, key=lambda z: abs(z - p)) for p in probe]

    def cross50(sf: list[float]) -> tuple[float, float, str]:
        """返回 (合并点=最后一次 sf≤0.5 的 γ, 分离点=第一次 sf≥0.5 的 γ, 状态)。"""
        if sf[0] >= 0.5:
            return float("nan"), float("nan"), "separated_already_at_grid_min"
        if sf[-1] < 0.5:
            return float("nan"), float("nan"), "together_up_to_grid_max"
        up = dn = float("nan")
        for t in range(1, len(sf)):
            if sf[t] >= 0.5 and up != up:
                w = (0.5 - sf[t - 1]) / (sf[t] - sf[t - 1]) if sf[t] > sf[t - 1] else 0.0
                up = round(float(ladder[t - 1] + w * (ladder[t] - ladder[t - 1])), 4)
        for t in range(len(sf) - 1, -1, -1):
            if sf[t] <= 0.5:
                dn = round(float(ladder[t]), 4)
                break
        return dn, up, "ok"

    pc_rows = []
    for i in range(len(ref_sets_sorted)):
        for j in range(i + 1, len(ref_sets_sorted)):
            ka, kb = f"ref{i:02d}", f"ref{j:02d}"
            A, B = cand[ka], cand[kb]
            d = wt.gamma_star(A, B)
            sf = sf_curve(ka, kb)
            g_dn, gc, gs_note = cross50(sf)
            pc_rows.append({
                "event_id": f"B{len(pc_rows)+1}", "scheme": "pair_curve", "type": "ref_pair",
                "gamma_low": ladder[0], "gamma_high": ladder[-1], "parent_size_low": None,
                "merged_size_high": None,
                "n_pieces": 2, "pieceA_size": len(A), "pieceB_size": len(B),
                "pieceA_names": names(A), "pieceB_names": names(B),
                "ref_parent_size": None,
                "gamma_pred_split": d["gamma_pred_merge"], "gamma_emp_split_cross50": gc,
                "gamma_emp_merge_cross50": g_dn, "gamma_emp_status": gs_note,
                "split_frac_at_gamma_low": round(sf[0], 3), "split_frac_at_gamma_high": round(sf[-1], 3),
                "split_frac_at_gamma1": round(float(np.mean(MAJ[1.0][kidx[ka]] != MAJ[1.0][kidx[kb]])), 3),
                "intact_frac_A_at_gamma1": round(float(INT[1.0][kidx[ka]].mean()), 3),
                "intact_frac_B_at_gamma1": round(float(INT[1.0][kidx[kb]].mean()), 3),
                "split_frac_above_ladder": round(float(np.mean(sf[-5:])), 3),
                "n_ladder_steps_above": None,
                "cross_weight": d["cross_weight"], "cross_edges": d["cross_edges"],
                "internal_w_A": d["internal_w_A"], "internal_w_B": d["internal_w_B"],
                "coupling_ratio": d["coupling_ratio"], "nbr_jaccard": wt.nbr_jaccard(A, B),
                "involves_ref_gamma1": bool((gc == gc and 0.9 <= gc <= 1.1)),
                **{f"sf_at_gamma_{g_}": round(float(sf[probe_g.index(g_)]), 3)
                   for p, g_ in zip(probe, probe_g)},
            })
    for pi, ka, kb, psz in intra_trip:
        A, B = cand[ka], cand[kb]
        d = wt.gamma_star(A, B)
        sf = sf_curve(ka, kb)
        g_dn, gc, gs_note = cross50(sf)
        pc_rows.append({
            "event_id": f"B{len(pc_rows)+1}", "scheme": "pair_curve", "type": "intra_ref_split",
            "gamma_low": ladder[0], "gamma_high": ladder[-1], "parent_size_low": psz,
            "merged_size_high": None,
            "n_pieces": 2, "pieceA_size": len(A), "pieceB_size": len(B),
            "pieceA_names": names(A), "pieceB_names": names(B), "ref_parent_size": psz,
            "gamma_pred_split": d["gamma_pred_merge"], "gamma_emp_cross50": gc,
            "gamma_emp_status": gs_note,
            "split_frac_at_gamma_low": round(1 - sf[0], 3), "split_frac_at_gamma_high": round(sf[-1], 3),
            "split_frac_at_gamma1": round(float(np.mean(MAJ[1.0][kidx[ka]] != MAJ[1.0][kidx[kb]])), 3),
            "intact_frac_A_at_gamma1": round(float(INT[1.0][kidx[ka]].mean()), 3),
            "intact_frac_B_at_gamma1": round(float(INT[1.0][kidx[kb]].mean()), 3),
            "split_frac_above_ladder": round(float(np.mean(sf[-5:])), 3),
            "n_ladder_steps_above": None,
            "cross_weight": d["cross_weight"], "cross_edges": d["cross_edges"],
            "internal_w_A": d["internal_w_A"], "internal_w_B": d["internal_w_B"],
            "coupling_ratio": d["coupling_ratio"], "nbr_jaccard": wt.nbr_jaccard(A, B),
            "involves_ref_gamma1": bool((gc == gc and 0.9 <= gc <= 1.1)),
            **{f"sf_at_{p}": round(float(sf[probe_g.index(g_)]), 3)
               for p, g_ in zip(probe, probe_g)},
        })
    paircurve = pd.DataFrame(pc_rows)
    all_events = pd.concat([events, paircurve], ignore_index=True, sort=False)
    all_events["grid_gamma_min"] = ladder[0]
    all_events["grid_gamma_max"] = ladder[-1]
    all_events["n_seeds_per_gamma"] = N_SEEDS
    C.save_csv(all_events, MODULE, "merge_split_events.csv")
    log(f"wrote merge_split_events.csv rows={len(all_events)} | ladder_step {len(events)} + "
        f"pair_curve {len(paircurve)} (ref_pair {int((paircurve.type=='ref_pair').sum())} / "
        f"intra_ref_split {int((paircurve.type=='intra_ref_split').sum())})")

    # 解析阈值 γ* vs 经验相变点的一致性
    pc = paircurve[(paircurve.gamma_emp_status == "ok") & paircurve.gamma_pred_split.gt(0)]
    if len(pc) >= 2:
        dev = (pc["gamma_pred_split"] - pc["gamma_emp_split_cross50"])
        log(f"pair_curve: γ*>0 且经验分离点落在网格内 n={len(pc)} | "
            f"中位 |γ*−γ_emp|={dev.abs().median():.3f} | γ*−γ_emp 中位={dev.median():+.3f} | "
            f"Pearson r={np.corrcoef(pc.gamma_pred_split, pc.gamma_emp_split_cross50)[0,1]:.3f} | "
            f"γ_emp∈[0.9,1.1] 的件数="
            f"{int(((pc.gamma_emp_split_cross50>=0.9)&(pc.gamma_emp_split_cross50<=1.1)).sum())}")
    else:
        log(f"pair_curve: 可比件数 n={len(pc)} < 2 —— 不报相关系数（样本不足）")
    log(f"pair_curve status counts: {paircurve.gamma_emp_status.value_counts().to_dict()}")
    ip = paircurve[paircurve.type == "intra_ref_split"]
    log(f"intra_ref_split: γ=1 时同处一社区的子块对 n={len(ip)} | 其中在 γ≤3.0 内分开的 "
        f"{int((ip.gamma_emp_status=='ok').sum())} | "
        f"分离点 >1.0 的 {int((ip.gamma_emp_split_cross50>1.0).sum())} | 判据 γ*≤1 的 {int((ip.gamma_pred_split<=1.0).sum())}")

    # ---------------------------------------------- 替代判据表（选 γ 用）
    stab = summary[["gamma", "pass", "n_comms_nonsingleton_mean", "n_comms_nonsingleton_sd",
                    "dn_ns_vs_prev_ladder_step", "dn_ns_per_dgamma", "phase_transition_flag",
                    "largest_mean", "largest_sd", "Q_same_null_mean",
                    "Q_rescored_at_gamma1_mean", "nmi_vs_gamma1_mean", "nmi_vs_gamma1_sd",
                    "nmi_vs_gamma1_rel_ceiling", "ari_vs_gamma1_mean", "ari_vs_gamma1_sd",
                    "ari_vs_gamma1_rel_ceiling", "within_seed_NMI_mean", "within_seed_NMI_rel_max",
                    "within_seed_ARI_mean",
                    "consensus_entropy_bits", "consensus_frac_pairs_ambiguous"]].sort_values("gamma")
    C.save_csv(stab, MODULE, "resolution_stability.csv")
    log(f"wrote resolution_stability.csv rows={len(stab)}")

    # -------------------------------------------------------------- 判决数字
    from scipy.stats import spearmanr
    coarse_only = summary[summary["pass"] == "coarse"].sort_values("gamma").reset_index(drop=True)
    allg = summary.sort_values("gamma").reset_index(drop=True)
    ref_ceiling = float(coarse_only.loc[coarse_only.gamma == REF_GAMMA, "within_seed_NMI_mean"].iloc[0])

    def plateau(df, col, thr):
        ok = (df[col].values >= thr)
        gg = df["gamma"].values
        i = int(np.where(np.isclose(gg, REF_GAMMA))[0][0])
        lo = i
        while lo - 1 >= 0 and ok[lo - 1]:
            lo -= 1
        hi = i
        while hi + 1 < len(gg) and ok[hi + 1]:
            hi += 1
        return [float(gg[lo]), float(gg[hi])]

    ns_ref = float(coarse_only.loc[coarse_only.gamma == REF_GAMMA, "n_comms_nonsingleton_mean"].iloc[0])
    sd_ref = float(coarse_only.loc[coarse_only.gamma == REF_GAMMA, "n_comms_nonsingleton_sd"].iloc[0])
    verdict = {
        "n_seeds": N_SEEDS, "n_gamma_coarse": int(len(coarse_only)),
        "n_gamma_total": int(len(summary)),
        "ref_seed_noise_ceiling_NMI": round(ref_ceiling, 4),
        "fb_scale_sqrt_2S_weighted": round(math.sqrt(wt.twoS), 1),
        "fb_scale_sqrt_2E_unweighted": round(math.sqrt(2 * Gu.number_of_edges()), 1),
        "n_ns671_at_gamma1_mean": round(ns_ref, 2), "n_ns671_at_gamma1_sd": round(sd_ref, 2),
        "argmax_Q_same_null": float(coarse_only.loc[coarse_only.Q_same_null_mean.idxmax(), "gamma"]),
        "argmax_Q_rescored_at_gamma1": float(coarse_only.loc[coarse_only.Q_rescored_at_gamma1_mean.idxmax(), "gamma"]),
        "argmax_within_seed_NMI": float(coarse_only.loc[coarse_only.within_seed_NMI_mean.idxmax(), "gamma"]),
        "argmin_consensus_entropy": float(coarse_only.loc[coarse_only.consensus_entropy_bits.idxmin(), "gamma"]),
        "spearman_gamma_vs_Q": [round(float(spearmanr(coarse_only.gamma, coarse_only.Q_same_null_mean).statistic), 4),
                                float(spearmanr(coarse_only.gamma, coarse_only.Q_same_null_mean).pvalue)],
        "spearman_gamma_vs_n_ns": [round(float(spearmanr(coarse_only.gamma, coarse_only.n_comms_nonsingleton_mean).statistic), 4),
                                   float(spearmanr(coarse_only.gamma, coarse_only.n_comms_nonsingleton_mean).pvalue)],
        "selfcheck_Q_formula_vs_nx_absdiff": round(abs(myQ - nxQ), 10),
        "selfcheck_dQ_formula_maxabsdiff": round(max(abs(f - a) for _, f, a in dqchk), 10),
    }
    for thr in (0.98, 0.95, 0.90, 0.85, 0.80):
        verdict[f"plateau_nmi_ge_{int(thr*100)}_coarse"] = plateau(coarse_only, "nmi_vs_gamma1_mean", thr)
        verdict[f"plateau_nmi_ge_{int(thr*100)}_all"] = plateau(allg, "nmi_vs_gamma1_mean", thr)
    for thr in (1.00, 0.98, 0.95, 0.90):
        verdict[f"plateau_nmi_rel_ceiling_ge_{int(thr*100)}_coarse"] = plateau(
            coarse_only, "nmi_vs_gamma1_rel_ceiling", thr)
        verdict[f"plateau_nmi_rel_ceiling_ge_{int(thr*100)}_all"] = plateau(
            allg, "nmi_vs_gamma1_rel_ceiling", thr)
        verdict[f"plateau_ari_rel_ceiling_ge_{int(thr*100)}_coarse"] = plateau(
            coarse_only, "ari_vs_gamma1_rel_ceiling", thr)
        verdict[f"plateau_ari_rel_ceiling_ge_{int(thr*100)}_all"] = plateau(
            allg, "ari_vs_gamma1_rel_ceiling", thr)
    verdict["ref_seed_noise_ceiling_ARI"] = round(ref_ari_ceiling_early, 4)
    verdict["argmax_within_seed_ARI_coarse"] = float(
        coarse_only.loc[coarse_only.within_seed_ARI_mean.idxmax(), "gamma"])
    verdict["within_seed_ARI_at_gamma1"] = float(
        coarse_only.loc[coarse_only.gamma == REF_GAMMA, "within_seed_ARI_mean"].iloc[0])
    verdict["within_seed_ARI_max_coarse"] = round(float(coarse_only.within_seed_ARI_mean.max()), 4)
    verdict["within_seed_ARI_min_coarse"] = round(float(coarse_only.within_seed_ARI_mean.min()), 4)
    _sd = max(sd_ref, 1e-9)
    _band = [float(g) for g, k in zip(coarse_only.gamma.values,
                                      coarse_only.n_comms_nonsingleton_mean.values)
             if abs(k - ns_ref) <= 2 * _sd]
    verdict["n_ns_within_2sd_band_gamma_range"] = [min(_band), max(_band)] if _band else []
    verdict["phase_transition_gamma_list_coarse"] = [
        float(g) for g, f in zip(coarse_only.gamma.values, coarse_only.phase_transition_flag.values) if f == 1]
    verdict["phase_transition_gamma_list_all_ladder"] = [
        float(g) for g, f in zip(summary.gamma.values, summary.phase_transition_flag.values) if f == 1]
    verdict["max_gamma_star_ref_pairs"] = float(pairs.gamma_pred_merge.max())
    verdict["n_ref_pairs_by_gamma_star"] = {
        ">1": int((pairs.gamma_pred_merge > 1).sum()),
        "(0.8,1]": int(((pairs.gamma_pred_merge > 0.8) & (pairs.gamma_pred_merge <= 1)).sum()),
        "(0.5,1]": int(((pairs.gamma_pred_merge > 0.5) & (pairs.gamma_pred_merge <= 1)).sum()),
        "(0.25,1]": int(((pairs.gamma_pred_merge > 0.25) & (pairs.gamma_pred_merge <= 1)).sum())}
    _pc = paircurve[(paircurve.gamma_emp_status == "ok") & paircurve.gamma_pred_split.gt(0)]
    verdict["pair_curve_n_ok_pairs"] = int(len(_pc))
    if len(_pc) < 2:
        verdict["pair_curve_pearson_r_pred_vs_emp"] = None
    verdict["pair_curve_median_abs_pred_minus_emp"] = (round(
        float((_pc.gamma_pred_split - _pc.gamma_emp_split_cross50).abs().median()), 4) if len(_pc) else None)
    verdict["pair_curve_median_signed_pred_minus_emp"] = (round(
        float((_pc.gamma_pred_split - _pc.gamma_emp_split_cross50).median()), 4) if len(_pc) else None)
    verdict["pair_curve_pearson_r_pred_vs_emp"] = (round(float(np.corrcoef(
        _pc.gamma_pred_split, _pc.gamma_emp_split_cross50)[0, 1]), 4) if len(_pc) >= 2 else None)
    verdict["pair_curve_status_counts"] = paircurve.gamma_emp_status.value_counts().to_dict()
    _ip = paircurve[paircurve.type == "intra_ref_split"]
    verdict["intra_ref_split_n_pairs"] = int(len(_ip))
    verdict["intra_ref_split_n_emp_split_gt_1"] = int((_ip.gamma_emp_split_cross50 > 1.0).sum())
    verdict["intra_ref_split_n_pred_gamma_le_1"] = int((_ip.gamma_pred_split <= 1.0).sum())
    verdict["intra_ref_split_parent_communities"] = sorted(
        {int(x) for x in _ip.ref_parent_size.values})
    _ws = coarse_only.within_seed_NMI_mean
    _thr = float(_ws.max()) - 0.02
    verdict["within_seed_NMI_max"] = round(float(_ws.max()), 4)
    verdict["within_seed_NMI_min"] = round(float(_ws.min()), 4)
    verdict["plateau_within_seed_NMI_ge_max_minus_0.02_coarse"] = [
        float(coarse_only.loc[_ws >= _thr, "gamma"].min()),
        float(coarse_only.loc[_ws >= _thr, "gamma"].max())]
    verdict["n_gamma_ns_within_2sd_of_ref"] = int(
        ((coarse_only.n_comms_nonsingleton_mean - ns_ref).abs() <= 2 * sd_ref).sum())
    verdict["n_gamma_nmi_ge_0.90"] = int((coarse_only.nmi_vs_gamma1_mean >= 0.90).sum())
    verdict["n_gamma_nmi_lt_0.90"] = int((coarse_only.nmi_vs_gamma1_mean < 0.90).sum())
    if len(events):
        near = events[(events.gamma_pred_split >= events.gamma_low) &
                      (events.gamma_pred_split <= events.gamma_high)]
        verdict["events_pred_inside_interval_frac"] = round(float(len(near)) / float(len(events)), 4)
        strong = events[(events.split_frac_at_gamma_high >= 0.8) &
                        (events.split_frac_at_gamma_low <= 0.2)]
        verdict["n_strong_split_events"] = int(len(strong))
        v1 = strong[strong.involves_ref_gamma1]
        verdict["n_strong_split_events_touching_gamma1"] = int(len(v1))
    C.save_json(verdict, MODULE, "verdict.json")
    log("verdict: " + str(verdict))

    # -------------------------------------------------------- 控制台关键表
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    cols = ["gamma", "n_comms_nonsingleton_mean", "n_comms_nonsingleton_sd", "n_comms_total_mean",
            "largest_mean", "largest_sd", "Q_same_null_mean", "Q_rescored_at_gamma1_mean",
            "nmi_vs_gamma1_mean", "nmi_vs_gamma1_sd", "ari_vs_gamma1_mean",
            "within_seed_NMI_mean", "within_seed_ARI_mean",
            "consensus_entropy_bits", "n_ref_comms_split_here", "n_comms_here_merged_from_ref"]
    log("\n== coarse ladder ==")
    log(coarse_only[cols].to_string(index=False))
    log("\n== 相变标记（相邻阶梯 |Δn_ns| ≥ "
        f"{float(summary.phase_transition_threshold_dN.iloc[0]):.2f}）==")
    log(summary[summary.phase_transition_flag == 1][
        ["gamma", "n_comms_nonsingleton_mean", "dn_ns_vs_prev_ladder_step", "dn_ns_per_dgamma",
         "nmi_vs_gamma1_mean", "nmi_vs_gamma1_rel_ceiling", "largest_mean",
         "Q_same_null_mean", "within_seed_NMI_mean", "consensus_entropy_bits"]].to_string(index=False))
    _tmp = paircurve[paircurve.gamma_emp_status == "ok"].copy()
    _tmp["dist1"] = (_tmp.gamma_emp_split_cross50 - 1.0).abs()
    log("\n== pair_curve: 经验分离点最接近 γ=1 的 12 件 ==")
    log(_tmp.sort_values("dist1").head(12)[
        ["type", "pieceA_size", "pieceB_size", "pieceA_names", "pieceB_names", "gamma_pred_split",
         "gamma_emp_split_cross50", "gamma_emp_merge_cross50", "split_frac_at_gamma1",
         "cross_weight", "coupling_ratio", "nbr_jaccard"]].to_string(index=False))
    log("\n== pair_curve: intra_ref_split（γ=1 同一社区内部的子块对）全部 ==")
    log(paircurve[paircurve.type == "intra_ref_split"].sort_values("gamma_emp_split_cross50")[
        ["ref_parent_size", "pieceA_size", "pieceB_size", "pieceA_names", "pieceB_names",
         "gamma_pred_split", "gamma_emp_split_cross50", "gamma_emp_status", "split_frac_at_gamma1",
         "cross_weight", "coupling_ratio", "nbr_jaccard"]].to_string(index=False))
    log("\n== pair_curve: ref_pair 中 γ* 最高的 10 件 ==")
    log(paircurve[paircurve.type == "ref_pair"].sort_values("gamma_pred_split", ascending=False).head(10)[
        ["pieceA_size", "pieceB_size", "pieceA_names", "pieceB_names", "gamma_pred_split",
         "gamma_emp_merge_cross50", "gamma_emp_split_cross50", "gamma_emp_status",
         "split_frac_at_gamma1", "sf_at_gamma_0.9", "sf_at_gamma_1.0", "sf_at_gamma_1.1",
         "cross_weight", "coupling_ratio", "nbr_jaccard"]].to_string(index=False))
    log("\n== γ=1 medoid: 12 pairs with highest γ* ==")
    log(pairs.head(12)[["n_i", "names_i", "n_j", "names_j", "gamma_pred_merge", "cross_weight",
                        "coupling_ratio", "nbr_jaccard", "merge_attractive_at_gamma1"]].to_string(index=False))
    if len(events):
        log("\n== events crossing γ=1 ==")
        log(events[events.involves_ref_gamma1].head(25)[
            ["event_id", "type", "gamma_low", "gamma_high", "pieceA_size", "pieceB_size",
             "pieceA_names", "pieceB_names", "split_frac_at_gamma_low", "split_frac_at_gamma_high",
             "gamma_pred_split", "cross_weight", "coupling_ratio", "nbr_jaccard"]].to_string(index=False))
        log("\n== strongest transitions (0→1 split_frac across a ladder step) ==")
        strong = events[(events.split_frac_at_gamma_high >= 0.8) & (events.split_frac_at_gamma_low <= 0.2)]
        log(strong.sort_values("gamma_low").head(30)[
            ["event_id", "type", "gamma_low", "gamma_high", "pieceA_size", "pieceB_size",
             "pieceA_names", "pieceB_names", "split_frac_at_gamma_low", "split_frac_at_gamma_high",
             "split_frac_above_ladder", "gamma_pred_split", "cross_weight",
             "coupling_ratio", "nbr_jaccard"]].to_string(index=False))

    dt = time.time() - t_start
    C.write_run_note(MODULE, LOG + [
        "",
        "## environment",
        str(C.versions()),
        f"numpy={np.__version__} networkx={nx.__version__} pandas={pd.__version__} "
        f"platform={platform.platform()}",
        "",
        "## config",
        f"SEED = {SEED} ; seeds ({N_SEEDS}) = {SEEDS}",
        f"coarse γ ({len(gammas_c)}) = {gammas_c}",
        f"fine γ ({len(fine)}) = {fine}",
        "graph = C.build_all()['Gu'] (无向加权, 权重相加, edge_order 建图) — 未自建图",
        "louvain = C.louvain(Gu, seed, resolution=γ) ; NMI = C.partition_metrics(G671, A, B)",
        "NMI/共识矩阵主口径 = 671 有属性节点(C.induced) ; 676 口径以 *_676 列对照",
        f"total edge weight S = {wt.S:.0f} ; sqrt(2S) = {math.sqrt(wt.twoS):.1f}",
        f"total wall time = {dt:.1f}s",
        "",
        "## caveats",
        "Q_same_null 与 Q_rescored_at_gamma1 两列都不可跨 γ 比较大小（零模型不同 / 划分不同）；"
        "后者只是把每份划分放回 γ=1 零模型下的描述量。",
        "γ* 判据是成对局部判据（其余社区不动），不是全局最优性条件；Louvain 是贪心局部最优。",
    ])
    log(f"DONE {dt:.1f}s → {OUT_DIR}")


if __name__ == "__main__":
    main()
