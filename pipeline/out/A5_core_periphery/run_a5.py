# -*- coding: utf-8 -*-
"""A5_core_periphery — 核心—边缘分层与 rich-club（USTC 网站群超链接网络）

判据（任务书）
  §1 k-core 剖面：k_max、各核/壳规模、节点所在核；核心与平均度/介数的关系（描述性）。
  §2 rich-club 系数 phi(k) = 2E_{>k} / [n_{>k}(n_{>k}-1)]，逐 k 扫描；
     与度保持零模型（C.degree_preserving_null，独立重复，nswap=10m, max_tries=100*nswap）
     比较：mean/sd/z/双侧经验 p（按偏离 null mean 的绝对值计，CONVENTIONS §4）+ BH 校正；
     同时报实际交换次数、尝试次数、度序列是否保持不变（前后 sorted(deg) 比对）。
     声明：边交换非严格均匀采样（Viger & Latapy 2005）。
  §3 Borgatti & Everett (2000, Social Networks 21:375) 1-mode core/periphery：
     自实现迭代：c <- rows corr(A_i., B_i.)，B_ij = c_i + c_j - c_i c_j
     （核心-核心=1，核心-边缘=1，边缘-边缘=0 的连续松弛）；
     离散 (k, c)：对收敛 c 的 top-k 前缀做全 k 扫描 + 固定 k 交替精化；
     拟合度 = 观测阵与理想阵的非对角 Pearson 相关（闭式矩计算，另有稠密矩阵复核）。
     拟合度亦与度保持零模型比较（同一过程跑 null 池）。
  §4 三套"核心"口径并排：k-core（coreness = k_max）、rich-club（度 > knee k，
     另报"最小 BH 显著 k"口径）、Borgatti-Everett（精化离散核心）；两两 Jaccard。
  §5 核心成员 × 健康状态：is_dead / is_partial 计数与占比、Fisher 精确检验；
     度口径主用入度（CONVENTIONS §2.1：死站出度 100% 删失，无向度对死站 == 入度）。

口径声明
  - 主图：671 个有属性节点上的无权无向图（C.build_all -> Gu 的 671 诱导子图）。
    对照图：676（全图，含 5 悬空节点）。两口径全部重算，主表用 671。
  - 因果措辞禁用（CONVENTIONS §7）；本脚本输出只做描述与对照。
  - 全部随机过程显式传 seed=SEED(=C.SEED=20261113)。

运行：  python out/A5_core_periphery/run_a5.py   （工作目录 = pipeline/）
产出：  richclub_curve.csv, core_membership.csv, ce_fit.csv, kcore_profile.csv,
        core_overlap.csv, health_cross.csv, health_indegree_deciles.csv,
        swap_diagnostics.csv, ce_iterations.csv, result.json, RUN.txt
"""
from __future__ import annotations

import sys, os, json, time, hashlib
if __package__ in (None, ""):
    _PIPE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _PIPE not in sys.path:
        sys.path.insert(0, _PIPE)

import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
from scipy import stats as sps

import csa_common as C

SEED = C.SEED
MODULE = "A5_core_periphery"
OUT = C.module_dir(MODULE)

N_REPS_RC_671 = 400          # ≥300（任务要求），独立重复（每轮 fresh copy 再 swap）
N_REPS_RC_676 = 300          # 对照口径也满足 ≥300
MIN_NGT_FOR_TEST = 10        # phi(k) 进入显著性扫描的条件：n_{>k} >= 10（预先设定，不看结果）
CONT_CAP = 800               # B-E 连续迭代上限
CONT_TOL = 1e-10
REFINE_ITERS = 60            # 固定 k 交替精化上限（循环检测提前停）

_LOG = []
def log(msg: str):
    line = "[%7.2fs] %s" % (time.perf_counter() - _T0, msg)
    print(line, flush=True)
    _LOG.append(line)

# ===================================================================== 工具函数

def bh_fdr(p):
    """Benjamini–Hochberg 校正（升序 step-up，单调化）。"""
    p = np.asarray(p, float)
    adj = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return adj
    order = idx[np.argsort(p[idx])]
    m = len(order)
    scaled = p[order] * m / np.arange(1, m + 1)
    mono = np.minimum.accumulate(scaled[::-1])[::-1]
    adj[order] = np.minimum(mono, 1.0)
    return adj


def knee_index(y):
    """Kneedle 型拐点：有效（非 nan）段上、归一化后、到首末连线的最大垂直偏移。"""
    y = np.asarray(y, float)
    ok = ~np.isnan(y)
    yy = y[ok]
    if len(yy) < 3:
        return 0
    rng_ = yy.max() - yy.min()
    yn = (yy - yy.min()) / rng_ if rng_ > 0 else np.zeros_like(yy)
    x = np.linspace(0.0, 1.0, len(yn))
    chord = yn[0] + (yn[-1] - yn[0]) * x
    dev = np.abs(yn - chord)
    return int(np.argmax(dev))

# ============================================================ swap 探针（计数）

class SwapProbe:
    """包裹 nx.double_edge_swap：统计真实交换次数/尝试次数、检查度序列不变。

    只观察、不改变算法：内部临时替换 nx.utils.discrete_sequence（计一次=一次尝试）
    与图实例上的 add_edge/remove_edge（计调用次数；每次成功双重交换恰为 2 加 2 删），
    调用结束后原样恢复。C.degree_preserving_null 仍按 CONVENTIONS §4 参数调用。
    """

    def __init__(self):
        self.records = []
        self._orig_de = None
        self._orig_ds = None

    def __enter__(self):
        self._orig_de = nx.double_edge_swap
        self._orig_ds = nx.utils.discrete_sequence
        outer = self

        def wrapper(G, nswap=1, max_tries=100, seed=None):
            return outer._run(G, nswap, max_tries, seed)

        nx.double_edge_swap = wrapper
        return self

    def __exit__(self, *exc):
        nx.double_edge_swap = self._orig_de
        nx.utils.discrete_sequence = self._orig_ds
        return False

    def _run(self, G, nswap, max_tries, seed):
        rec = {"nswap_target": int(nswap), "max_tries": int(max_tries), "error": ""}
        deg_before = np.array(sorted(d for _, d in G.degree()), dtype=np.int64)
        tries = [0]
        orig_ds = self._orig_ds

        def counting_ds(*a, **k):
            tries[0] += 1
            return orig_ds(*a, **k)

        nx.utils.discrete_sequence = counting_ds
        adds = [0]
        rems = [0]
        g_add, g_rem = G.add_edge, G.remove_edge

        def add_wrap(*a, **k):
            adds[0] += 1
            return g_add(*a, **k)

        def rem_wrap(*a, **k):
            rems[0] += 1
            return g_rem(*a, **k)

        G.add_edge, G.remove_edge = add_wrap, rem_wrap
        try:
            self._orig_de(G, nswap=nswap, max_tries=max_tries, seed=seed)
        except nx.NetworkXAlgorithmError as e:
            rec["error"] = "NetworkXAlgorithmError(max_tries exhausted before nswap): %s" % e
        finally:
            nx.utils.discrete_sequence = orig_ds
            del G.add_edge
            del G.remove_edge
        deg_after = np.array(sorted(d for _, d in G.degree()), dtype=np.int64)
        rec.update(
            tries=int(tries[0]),
            add_calls=int(adds[0]),
            remove_calls=int(rems[0]),
            swaps_real=int(adds[0] // 2),
            degseq_preserved=bool(np.array_equal(deg_before, deg_after)),
        )
        self.records.append(rec)
        return G

# ==================================================================== rich-club

def phi_arrays(dd, mind, ks_grid):
    ngt = np.array([(dd > k).sum() for k in ks_grid], dtype=float)
    egt = np.array([(mind > k).sum() for k in ks_grid], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = np.where(ngt >= 2, 2.0 * egt / (ngt * (ngt - 1)), np.nan)
    return ngt, egt, phi


def graph_deg_edge_arrays(G, pos):
    dd = np.zeros(G.number_of_nodes())
    for u, dv in G.degree():
        dd[pos[u]] = dv
    mind = np.array([min(dd[pos[u]], dd[pos[v]]) for u, v in G.edges()], dtype=float)
    return dd, mind

# ============================================================ Borgatti–Everett

class CEModel:
    """1-mode core/periphery（Borgatti & Everett 2000），矩闭式实现。

    理想阵 B_ij = c_i + c_j - c_i c_j（i != j）。
    拟合 = corr(A, B)（非对角元 Pearson）。
    行更新 = c_i' = corr_j(A_ij, B_ij)（B-E 的行相关迭代）。
    所有 n^2 项都化为 S1..S4、d、Ac 的 O(n) 组合（稠密复核见 main §3）。
    """

    def __init__(self, G):
        self.nodes = sorted(G.nodes())
        self.n = len(self.nodes)
        self.pos = {v: i for i, v in enumerate(self.nodes)}
        Ad = nx.to_scipy_sparse_array(G, nodelist=self.nodes, dtype=np.float64).toarray()
        np.fill_diagonal(Ad, 0.0)
        self.A = sp.csr_matrix(Ad)
        self.Adense = Ad
        self.d = Ad.sum(axis=1)
        self.E = float(self.d.sum() / 2.0)
        self.n0 = float(self.n * (self.n - 1))
        self.Nm1 = float(self.n - 1)

    @classmethod
    def from_proto_edgelist(cls, proto, edges):
        """零模型图复用同一节点顺序（swap 逐点保度）。"""
        obj = cls.__new__(cls)
        obj.nodes, obj.n, obj.pos = proto.nodes, proto.n, proto.pos
        rows = np.array([obj.pos[u] for u, v in edges], dtype=np.int64)
        cols = np.array([obj.pos[v] for u, v in edges], dtype=np.int64)
        data = np.ones(len(edges))
        Am = sp.coo_matrix((data, (rows, cols)), shape=(obj.n, obj.n)).toarray()
        Am = Am + Am.T
        np.fill_diagonal(Am, 0.0)
        obj.Adense = Am
        obj.A = sp.csr_matrix(Am)
        obj.d = Am.sum(axis=1)
        obj.E = float(obj.d.sum() / 2.0)
        obj.n0 = proto.n0
        obj.Nm1 = proto.Nm1
        return obj

    # ---- 全局拟合 ----
    def moments(self, c):
        Nm1, n0 = self.Nm1, self.n0
        Ac = self.A.dot(c)
        S1 = c.sum(); S2 = (c * c).sum(); S3 = (c ** 3).sum(); S4 = (c ** 4).sum()
        SA = 2.0 * self.E
        SB = 2.0 * Nm1 * S1 - (S1 * S1 - S2)
        SB2 = (2.0 * Nm1 * S2 + (S2 * S2 - S4) + 2.0 * (S1 * S1 - S2)
               - 4.0 * (S1 * S2 - S3))
        SAB = 2.0 * float(np.dot(c, self.d)) - float(np.dot(c, Ac))
        return SA, SB, SB2, SAB

    def fit_closed(self, c):
        SA, SB, SB2, SAB = self.moments(c)
        n0 = self.n0
        va = n0 * SA - SA * SA
        vb = n0 * SB2 - SB * SB
        if va <= 0 or vb <= 0:
            return np.nan
        return float((n0 * SAB - SA * SB) / np.sqrt(va * vb))

    def fit_dense(self, c):
        """稠密复核用：直接构造 B 并算 Pearson。"""
        B = c[:, None] + c[None, :] - np.outer(c, c)
        m = ~np.eye(self.n, dtype=bool)
        return float(np.corrcoef(self.Adense[m], B[m])[0, 1])

    def relaxed_stats(self, c):
        """离散 c 下：relaxed 模型 = 只统计"至少一端在核心"的格子。"""
        SA, SB, SB2, SAB = self.moments(c)
        dens_spec = SAB / SB if SB > 0 else np.nan                 # 指定格的匹配比例（B=1）
        rest = (SA - SAB) / (self.n0 - SB) if self.n0 - SB > 0 else np.nan
        return float(dens_spec), float(rest)

    # ---- 行相关更新 ----
    def row_scores(self, c):
        Nm1 = self.Nm1
        Ac = self.A.dot(c)
        S1 = c.sum(); S2 = (c * c).sum()
        SAB_i = c * self.d + Ac - c * Ac
        Bsum_i = Nm1 * c + (S1 - c) - c * (S1 - c)
        B2sum_i = Nm1 * c * c + 2.0 * c * (1 - c) * (S1 - c) + (1 - c) ** 2 * (S2 - c * c)
        ra = self.d / Nm1
        rb = Bsum_i / Nm1
        va = np.maximum(ra - ra * ra, 0.0)
        vb = np.maximum(B2sum_i / Nm1 - rb * rb, 0.0)
        cov = SAB_i / Nm1 - ra * rb
        den = np.sqrt(va * vb)
        out = np.where(den > 1e-14, cov / np.where(den > 1e-14, den, 1.0), 0.0)
        return out

    # ---- 连续迭代 ----
    def cont_iter(self, c0, cap=CONT_CAP, tol=CONT_TOL):
        c = np.asarray(c0, float).copy()
        trace = []
        for it in range(cap):
            s = self.row_scores(c)
            delta = float(np.max(np.abs(s - c)))
            c = s
            trace.append((it + 1, delta, self.fit_closed(c)))
            if delta < tol:
                break
        return c, trace

    # ---- 离散：对 c_seed 排序后的 top-k 前缀扫描 + 固定 k 精化 ----
    def scan_topk(self, c_seed):
        order = np.argsort(-c_seed, kind="stable")
        dd = self.d[order]
        Dcum = np.concatenate([[0.0], np.cumsum(dd)])[: self.n]
        adj = self.Adense[order][:, order]
        inside = np.zeros(self.n)
        run = 0.0
        for k in range(1, self.n):
            run += adj[k, :k].sum()
            inside[k] = run
        fits = np.full(self.n // 2 + 1, np.nan)
        SA = 2.0 * self.E
        for k in range(1, self.n // 2 + 1):
            SB = float(2.0 * self.Nm1 * k - (k * k - k))
            SB2 = SB
            SAB = float(2.0 * Dcum[k] - 2.0 * inside[k])
            va = self.n0 * SA - SA * SA
            vb = self.n0 * SB2 - SB * SB
            fits[k] = (self.n0 * SAB - SA * SB) / np.sqrt(va * vb) if vb > 0 else np.nan
        return order, fits, Dcum, inside

    def topk_vec(self, order, k):
        c = np.zeros(self.n)
        c[order[:k]] = 1.0
        return c

    def refine_fixed_k(self, k, c_seed, iters=REFINE_ITERS):
        core = self.topk_vec(np.argsort(-c_seed, kind="stable"), k)
        best = (self.fit_closed(core), core.copy())
        seen = {core.tobytes()}
        stable = False
        used = iters
        for t in range(iters):
            s = self.row_scores(core)
            new = self.topk_vec(np.argsort(-s, kind="stable"), k)
            f = self.fit_closed(new)
            if f > best[0]:
                best = (f, new.copy())
            if new.tobytes() in seen:
                used = t + 1
                stable = True          # 回到历史状态 => 循环/不动点，判"收敛"
                break
            seen.add(new.tobytes())
            core = new
        return best[0], best[1], stable, used

    # ---- 全流程（观测与零模型完全一致）----
    def run_procedure(self, extra_random_starts=0):
        starts = {
            "deg_norm": self.d / self.d.max(),
            "bern05": (np.random.default_rng(SEED + 7).random(self.n) < 0.5).astype(float),
        }
        out = {"runs": []}
        for name, c0 in starts.items():
            out["runs"].append(self._one_start(name, c0))
        for i in range(extra_random_starts):
            c0 = (np.random.default_rng(SEED + 100 + i).random(self.n) < 0.5).astype(float)
            r = self._one_start("sens_rand%d" % i, c0)
            r["sensitivity"] = True
            out["runs"].append(r)
        fixed = [r for r in out["runs"] if not r.get("sensitivity")]
        best = max(fixed, key=lambda r: r["fit_star"])
        out["best"] = best
        return out

    def _one_start(self, name, c0):
        c, trace = self.cont_iter(c0)
        fit_cont = self.fit_closed(c)
        order, fits, Dcum, inside = self.scan_topk(c)
        with np.errstate(all="ignore"):
            kb = int(np.nanargmax(fits[1:])) + 1 if np.any(~np.isnan(fits[1:])) else 1
        fstar, cstar, stable, used = self.refine_fixed_k(kb, c)
        k_p05 = int((c >= 0.5).sum())
        k_p0 = int((c > 0).sum())
        ek = knee_index(np.sort(c)[::-1]) + 1
        rec = dict(
            start=name,
            cont_iters=len(trace),
            cont_delta=trace[-1][1],
            cont_converged=bool(trace[-1][1] < CONT_TOL),
            fit_cont=fit_cont,
            fit_cont_best_iter=int(np.argmax([t[2] for t in trace])) + 1,
            fit_cont_best=float(max(t[2] for t in trace)),
            k_scan=kb,
            fit_scan=float(fits[kb]),
            k_star=int(cstar.sum()),
            fit_star=fstar,
            refine_cycles_stopped=bool(stable),
            refine_iters=used,
            k_p05=k_p05,
            fit_p05=float(fits[k_p05]) if 1 <= k_p05 <= self.n // 2 else np.nan,
            k_p0=k_p0,
            fit_p0=float(fits[k_p0]) if 1 <= k_p0 <= self.n // 2 else np.nan,
            k_elbow=ek,
            fit_elbow=float(fits[ek]) if 1 <= ek <= self.n // 2 else np.nan,
            fit_elbow_binary_c=self.fit_closed(self.topk_vec(order, min(ek, self.n))),
        )
        rd, rr = self.relaxed_stats(cstar)
        rec["fit_relaxed_star"] = rd
        rec["periph_periph_density_star"] = rr
        rec["fit_cont_binary_at_star"] = self.fit_closed(cstar)
        rec.pop("relaxed_density", None)
        rec["_c_star"] = cstar
        rec["_c_cont"] = c
        rec["_trace"] = trace
        return rec

# ==================================================================== 主流程

def main():
    global _T0
    _T0 = time.perf_counter()
    wall0 = time.time()
    log("import ok; SEED=%d" % SEED)

    b = C.build_all()
    C.assert_anchors(b)
    log("anchors 31/31 PASS")

    attributed = sorted(set(b["nodes"]["node"]))
    G671 = C.induced(b["Gu"], attributed)
    Gb671 = C.build_binary(G671)
    Gb676 = C.build_binary(b["Gu"])
    pos671 = {v: i for i, v in enumerate(sorted(Gb671.nodes()))}
    pos676 = {v: i for i, v in enumerate(sorted(Gb676.nodes()))}
    log("G671: n=%d m=%d | G676: n=%d m=%d（676 口径含 5 悬空节点，671 口径丢弃 5 条关联边）"
        % (Gb671.number_of_nodes(), Gb671.number_of_edges(),
           Gb676.number_of_nodes(), Gb676.number_of_edges()))

    nd = b["nodes"].set_index("node")

    # ================================================= 1. k-core 剖面
    bet671 = nx.betweenness_centrality(Gb671)
    bet676 = nx.betweenness_centrality(Gb676)
    cn671 = dict(nx.core_number(Gb671))
    cn676 = dict(nx.core_number(Gb676))
    kmax671 = max(cn671.values())
    kmax676 = max(cn676.values())
    log("k_max(671)=%d  k_max(676)=%d" % (kmax671, kmax676))

    prof_rows = []
    for k in range(1, kmax671 + 1):
        shell = [v for v in Gb671 if cn671[v] == k]
        cum = [v for v in Gb671 if cn671[v] >= k]
        def stat(vals, col, f=lambda x: x):
            return float(np.mean([f(nd.loc[v, col]) for v in vals])) if vals else np.nan
        prof_rows.append({
            "scope": "671", "k": k,
            "n_shell": len(shell), "n_cum": len(cum),
            "mean_degree_u_shell": float(np.mean([Gb671.degree(v) for v in shell])),
            "mean_in_degree_shell": stat(shell, "in_degree"),
            "mean_out_degree_shell": stat(shell, "out_degree"),
            "mean_betweenness_shell": float(np.mean([bet671[v] for v in shell])),
            "mean_ok_ratio_shell": stat(shell, "ok_ratio_pct"),
            "n_dead_shell": int(sum(bool(nd.loc[v, "is_dead"]) for v in shell)),
            "n_partial_shell": int(sum(bool(nd.loc[v, "is_partial"]) for v in shell)),
            "mean_degree_u_cum": float(np.mean([Gb671.degree(v) for v in cum])),
            "mean_betweenness_cum": float(np.mean([bet671[v] for v in cum])),
        })
    prof671 = pd.DataFrame(prof_rows)
    prof676 = pd.DataFrame([{
        "scope": "676", "k": k,
        "n_shell": sum(1 for v in Gb676 if cn676[v] == k),
        "n_cum": sum(1 for v in Gb676 if cn676[v] >= k),
    } for k in range(1, kmax676 + 1)])
    profile = pd.concat([prof671, prof676], ignore_index=True)
    C.save_csv(profile, MODULE, "kcore_profile.csv")

    core_vals = np.array([cn671[v] for v in Gb671])
    deg_vals = np.array([Gb671.degree(v) for v in Gb671], float)
    bet_vals = np.array([bet671[v] for v in Gb671])
    sp_core_deg = sps.spearmanr(core_vals, deg_vals)
    sp_core_bet = sps.spearmanr(core_vals, bet_vals)
    in_top = [v for v in Gb671 if cn671[v] == kmax671]
    rest = [v for v in Gb671 if cn671[v] < kmax671]
    log("Spearman(coreness, degree_u)=%.4f  Spearman(coreness, betweenness)=%.4f"
        % (sp_core_deg.statistic, sp_core_bet.statistic))

    # ================================================= 2. rich-club + 度保持零模型
    rc_curves, rc_stats = {}, {}
    for scope, G, pos, n_reps in (("671", Gb671, pos671, N_REPS_RC_671),
                                  ("676", Gb676, pos676, N_REPS_RC_676)):
        dd, mind = graph_deg_edge_arrays(G, pos)
        maxk = int(dd.max())
        ks_all = np.arange(1, maxk)
        ngt_all, egt_all, phi_all = phi_arrays(dd, mind, ks_all)
        test = ngt_all >= MIN_NGT_FOR_TEST
        ks_grid = ks_all[test]
        phi_obs = phi_all[test]
        knot_pos = knee_index(phi_obs)
        k_knee = int(ks_grid[knot_pos])
        k_ref = k_knee
        ref_idx = int(np.where(ks_grid == k_ref)[0][0])

        cache = {"curves": [], "pool": []}
        def stat(H, _pos=pos, _n=G.number_of_nodes(), _grid=ks_grid, _ri=ref_idx, _c=cache):
            d_h, m_h = graph_deg_edge_arrays(H, _pos)
            _, _, ph = phi_arrays(d_h, m_h, _grid)
            _c["curves"].append(ph)
            _c["pool"].append(list(H.edges()))
            return float(ph[_ri])

        t0 = time.perf_counter()
        probe = SwapProbe()
        with probe:
            nullres = C.degree_preserving_null(G, n_reps=n_reps, seed=SEED, stat=stat)
        rc_time = time.perf_counter() - t0
        curves = np.vstack(cache["curves"][:n_reps])           # 前 n_reps 条为零模型
        obs_curve = np.asarray(cache["curves"][-1], float)     # 最后一条 = stat(G)
        assert np.allclose(obs_curve, phi_obs, equal_nan=True), "obs 曲线与 stat(G) 不一致"
        pool = cache["pool"][:n_reps]                          # 供 B-E 零模型复用

        M = curves
        mu = M.mean(axis=0)
        sd = M.std(axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (phi_obs - mu) / sd
            p = np.array([float(np.mean(np.abs(M[:, j] - mu[j]) >= abs(phi_obs[j] - mu[j])))
                          for j in range(len(ks_grid))])
        p_bh = bh_fdr(p)
        sig = (p_bh < 0.05) & (z > 0)
        sig_neg = (p_bh < 0.05) & (z < 0)
        k_sig_first = int(ks_grid[sig][0]) if sig.any() else -1

        rows = pd.DataFrame({
            "scope": scope, "k": ks_grid,
            "n_gt_k": ngt_all[test].astype(int), "E_gt_k": egt_all[test].astype(int),
            "phi_obs": phi_obs, "phi_null_mean": mu, "phi_null_sd": sd,
            "z": z, "p_two_sided": p, "p_bh": p_bh,
            "is_significant": sig.astype(int),
            "is_knee": (ks_grid == k_knee).astype(int),
            "is_k_sig_first": (ks_grid == k_sig_first).astype(int),
        })
        rc_curves[scope] = rows
        rc_stats[scope] = dict(
            n_reps=n_reps, k_knee=k_knee, phi_at_knee=float(phi_obs[ref_idx]),
            official_obs=nullres["obs"], official_null_mean=nullres["null_mean"],
            official_null_sd=nullres["null_sd"], official_z=nullres["z"],
            official_p=nullres["p_two_sided"], k_sig_first=k_sig_first,
            n_sig_pos=int(sig.sum()), n_sig_neg=int(sig_neg.sum()),
            rc_seconds=rc_time,
            swap=dict(
                swaps_real_min=int(min(r["swaps_real"] for r in probe.records)),
                swaps_real_median=float(np.median([r["swaps_real"] for r in probe.records])),
                swaps_real_max=int(max(r["swaps_real"] for r in probe.records)),
                tries_min=int(min(r["tries"] for r in probe.records)),
                tries_median=float(np.median([r["tries"] for r in probe.records])),
                tries_max=int(max(r["tries"] for r in probe.records)),
                nswap_target=int(probe.records[0]["nswap_target"]),
                degseq_ok_ratio=float(np.mean([r["degseq_preserved"] for r in probe.records])),
                errors=int(sum(1 for r in probe.records if r["error"])),
            ),
            _phi_obs=phi_obs, _ks=ks_grid, _pool=pool, _G=G, _pos=pos,
        )
        rc_stats[scope]["_probe"] = probe.records
        log("rich-club[%s]: reps=%d knee_k=%d phi_obs=%.4f z=%.2f p=%.4f k_sig_first=%d "
            "n_sig_pos=%d n_sig_neg=%d "
            "swaps_real(min/med/max)=%d/%.0f/%d degseq_ok=%.3f errors=%d 用时 %.1fs"
            % (scope, n_reps, k_knee, nullres["obs"], nullres["z"], nullres["p_two_sided"],
               k_sig_first, rc_stats[scope]["n_sig_pos"], rc_stats[scope]["n_sig_neg"],
               rc_stats[scope]["swap"]["swaps_real_min"],
               rc_stats[scope]["swap"]["swaps_real_median"],
               rc_stats[scope]["swap"]["swaps_real_max"],
               rc_stats[scope]["swap"]["degseq_ok_ratio"],
               rc_stats[scope]["swap"]["errors"], rc_time))

    richclub = pd.concat([rc_curves["671"], rc_curves["676"]], ignore_index=True)
    C.save_csv(richclub, MODULE, "richclub_curve.csv")
    swapdiag = pd.concat([pd.DataFrame(rc_stats[s]["_probe"]).assign(scope=s)
                          for s in ("671", "676")], ignore_index=True)
    C.save_csv(swapdiag, MODULE, "swap_diagnostics.csv")

    # ================================================= 3. Borgatti–Everett
    ce_csv_rows, trace_rows = [], []
    ce_results = {}
    for scope in ("671", "676"):
        G = rc_stats[scope]["_G"]
        model = CEModel(G)
        proc = model.run_procedure(extra_random_starts=10)
        best = proc["best"]
        # 稠密复核（主口径 best run 的 fit_cont 与 fit_star）
        dcheck = max(abs(model.fit_dense(best["_c_cont"]) - best["fit_cont"]),
                     abs(model.fit_dense(best["_c_star"]) - best["fit_star"]))
        log("CE[%s]: start=%s k*=%d fit*=%.4f fit_cont=%.4f (dense check diff=%.2e) "
            "cont_iters=%d delta=%.1e conv=%s"
            % (scope, best["start"], best["k_star"], best["fit_star"], best["fit_cont"],
               dcheck, best["cont_iters"], best["cont_delta"], best["cont_converged"]))
        for t in best["_trace"]:
            trace_rows.append({"scope": scope, "start": best["start"], "iter": t[0],
                               "max_abs_delta": t[1], "fit_cont": t[2]})
        for r in proc["runs"]:
            ce_csv_rows.append({
                "scope": scope, "role": "observed", "kind": "run",
                "start": r["start"],
                "n_reps": 1,
                "cont_iters": r["cont_iters"], "cont_delta": r["cont_delta"],
                "cont_converged": int(r["cont_converged"]),
                "fit_cont": r["fit_cont"], "fit_cont_best": r["fit_cont_best"],
                "fit_cont_best_iter": r["fit_cont_best_iter"],
                "k_scan": r["k_scan"], "fit_scan": r["fit_scan"],
                "k_star": r["k_star"], "fit_star": r["fit_star"],
                "fit_relaxed_star": r["fit_relaxed_star"],
                "periph_periph_density_star": r["periph_periph_density_star"],
                "refine_cycles_stopped": int(r["refine_cycles_stopped"]),
                "refine_iters": r["refine_iters"],
                "k_p05": r["k_p05"], "fit_p05": r["fit_p05"],
                "k_p0": r["k_p0"], "fit_p0": r["fit_p0"],
                "k_elbow": r["k_elbow"], "fit_elbow": r["fit_elbow"],
                "sensitivity_start": int(bool(r.get("sensitivity"))),
            })
        # ---- 零模型：与观测完全相同的两步过程，池来自 §2 的独立重复 swap ----
        pool = rc_stats[scope]["_pool"]
        null_fits, null_fits_cont, null_k = [], [], []
        t0 = time.perf_counter()
        for gi, edges in enumerate(pool):
            mm = CEModel.from_proto_edgelist(model, edges)
            pp = mm.run_procedure(extra_random_starts=0)
            bb = pp["best"]
            null_fits.append(bb["fit_star"])
            null_fits_cont.append(bb["fit_cont"])
            null_k.append(bb["k_star"])
            if gi == 0:  # 抽一帧复核：零模型逐点度与观测一致（swap 性质）
                assert np.allclose(mm.d, model.d), "null 图逐点度漂移"
        null_fits = np.asarray(null_fits)
        null_fits_cont = np.asarray(null_fits_cont)
        ce_time = time.perf_counter() - t0
        nf = float(null_fits.std(ddof=1))
        zf = (best["fit_star"] - null_fits.mean()) / nf
        pf = float(np.mean(np.abs(null_fits - null_fits.mean()) >= abs(best["fit_star"] - null_fits.mean())))
        nfc = float(null_fits_cont.std(ddof=1))
        zc = (best["fit_cont"] - null_fits_cont.mean()) / nfc
        pc = float(np.mean(np.abs(null_fits_cont - null_fits_cont.mean())
                           >= abs(best["fit_cont"] - null_fits_cont.mean())))
        ce_csv_rows.append({
            "scope": scope, "role": "null", "kind": "aggregate", "start": "ALL",
            "n_reps": len(pool),
            "cont_iters": np.nan, "cont_delta": np.nan, "cont_converged": np.nan,
            "fit_cont": float(null_fits_cont.mean()), "fit_cont_best": np.nan,
            "fit_cont_best_iter": np.nan, "k_scan": np.nan, "fit_scan": np.nan,
            "k_star": float(np.mean(null_k)), "fit_star": float(null_fits.mean()),
            "fit_relaxed_star": np.nan, "periph_periph_density_star": np.nan,
            "refine_cycles_stopped": np.nan, "refine_iters": np.nan,
            "k_p05": np.nan, "fit_p05": np.nan, "k_p0": np.nan, "fit_p0": np.nan,
            "k_elbow": np.nan, "fit_elbow": np.nan, "sensitivity_start": 0,
        })
        ce_results[scope] = dict(
            model=model, proc=proc, best=best,
            null=dict(n=len(pool), fit_star_mean=float(null_fits.mean()),
                      fit_star_sd=nf, z_fit_star=float(zf), p_fit_star=pf,
                      fit_cont_mean=float(null_fits_cont.mean()), fit_cont_sd=nfc,
                      z_fit_cont=float(zc), p_fit_cont=pc,
                      k_star_mean=float(np.mean(null_k)),
                      k_star_sd=float(np.std(null_k, ddof=1)),
                      seconds=ce_time))
        log("CE[%s] null: reps=%d fit*=%.4f(sd %.4f) obs=%.4f z=%.2f p=%.4f | "
            "fit_cont null %.4f obs %.4f z=%.2f p=%.4f 用时 %.1fs"
            % (scope, len(pool), null_fits.mean(), nf, best["fit_star"], zf, pf,
               null_fits_cont.mean(), best["fit_cont"], zc, pc, ce_time))

    ce_fit = pd.DataFrame(ce_csv_rows)
    C.save_csv(ce_fit, MODULE, "ce_fit.csv")
    C.save_csv(pd.DataFrame(trace_rows), MODULE, "ce_iterations.csv")

    # ================================================= 4. 三套核心口径 + Jaccard
    s671 = rc_stats["671"]
    k_knee, k_sig = s671["k_knee"], s671["k_sig_first"]
    set_defs = {}
    set_defs["kcore_top"] = {v for v in Gb671 if cn671[v] == kmax671}
    set_defs["richclub_knee"] = {v for v in Gb671 if Gb671.degree(v) > k_knee}
    set_defs["richclub_sig"] = ({v for v in Gb671 if Gb671.degree(v) > k_sig}
                                if k_sig > 0 else set())
    best = ce_results["671"]["best"]
    c_star = best["_c_star"]
    set_defs["borgatti_everett"] = {ce_results["671"]["model"].nodes[i]
                                    for i in np.where(c_star > 0)[0]}
    set_defs["be_cont_p05"] = {ce_results["671"]["model"].nodes[i]
                               for i in np.where(best["_c_cont"] >= 0.5)[0]}

    def jac(a, b_):
        u = len(a | b_)
        return (len(a & b_) / u) if u else np.nan

    names = list(set_defs)
    orows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            A_, B_ = set_defs[names[i]], set_defs[names[j]]
            orows.append({"scope": "671", "set_a": names[i], "set_b": names[j],
                          "n_a": len(A_), "n_b": len(B_), "intersection": len(A_ & B_),
                          "union": len(A_ | B_), "jaccard": jac(A_, B_)})
    trio = [set_defs[x] for x in ("kcore_top", "richclub_knee", "borgatti_everett")]
    inter3 = trio[0] & trio[1] & trio[2]
    union3 = trio[0] | trio[1] | trio[2]
    orows.append({"scope": "671", "set_a": "TRIPLE(kcore,rc_knee,be)", "set_b": "-",
                  "n_a": len(inter3), "n_b": len(union3), "intersection": len(inter3),
                  "union": len(union3), "jaccard": len(inter3) / len(union3)})
    C.save_csv(pd.DataFrame(orows), MODULE, "core_overlap.csv")
    log("核心口径规模: kcore_top=%d rc_knee(deg>%d)=%d rc_sig(deg>%d)=%d be(k*=%d)=%d"
        % (len(set_defs["kcore_top"]), k_knee, len(set_defs["richclub_knee"]),
           k_sig, len(set_defs["richclub_sig"]), best["k_star"],
           len(set_defs["borgatti_everett"])))

    # ================================================= 5. 成员表 + 健康交叉
    cm = pd.DataFrame({
        "node": attributed,
    })
    cm["sub"] = cm["node"].map(nd["sub"])
    cm["coreness_671"] = cm["node"].map(cn671)
    cm["coreness_676"] = cm["node"].map(cn676)
    cm["degree_u_671"] = cm["node"].map(dict(Gb671.degree()))
    cm["degree_u_676"] = cm["node"].map(dict(Gb676.degree()))
    cm["in_degree"] = cm["node"].map(nd["in_degree"])
    cm["out_degree"] = cm["node"].map(nd["out_degree"])
    cm["betweenness_671"] = cm["node"].map(bet671)
    cm["ce_c_cont"] = cm["node"].map(pd.Series(best["_c_cont"],
                                               index=ce_results["671"]["model"].nodes))
    cm["in_kcore_top"] = cm["node"].isin(set_defs["kcore_top"]).astype(int)
    cm["in_richclub_knee"] = cm["node"].isin(set_defs["richclub_knee"]).astype(int)
    cm["in_richclub_sig"] = cm["node"].isin(set_defs["richclub_sig"]).astype(int)
    cm["in_borgatti_everett"] = cm["node"].isin(set_defs["borgatti_everett"]).astype(int)
    cm["health_state"] = cm["node"].map(nd["health_state"])
    cm["is_dead"] = cm["node"].map(nd["is_dead"]).astype(bool).astype(int)
    cm["is_partial"] = cm["node"].map(nd["is_partial"]).astype(bool).astype(int)
    cm["is_true_fail"] = cm["node"].map(nd["true_fail"]).astype(bool).astype(int)
    cm["is_false_fail"] = cm["node"].map(nd["false_fail"]).astype(bool).astype(int)
    cm["ok_ratio_pct"] = cm["node"].map(nd["ok_ratio_pct"])
    C.save_csv(cm, MODULE, "core_membership.csv")

    # 健康交叉表（对每套口径；Fisher：核心内外 死/非死 与 部分/非部分）
    hrows = []
    base_dead = int(nd["is_dead"].astype(bool).sum())
    for label, S in list(set_defs.items()) + [("ALL_671", set(attributed))]:
        if not S:
            continue
        sub = nd.loc[sorted(S)]
        out = nd.loc[sorted(set(attributed) - S)]
        def fisher(col):
            if label == "ALL_671" or len(out) == 0:
                return np.nan
            tab = [[int(sub[col].astype(bool).sum()), len(sub) - int(sub[col].astype(bool).sum())],
                   [int(out[col].astype(bool).sum()), len(out) - int(out[col].astype(bool).sum())]]
            return float(sps.fisher_exact(tab)[1])
        hrows.append({
            "scope": "671", "definition": label, "n_core": len(sub),
            "n_dead": int(sub["is_dead"].astype(bool).sum()),
            "n_partial": int(sub["is_partial"].astype(bool).sum()),
            "n_healthy": int(sub["is_healthy"].astype(bool).sum()),
            "n_true_fail": int(sub["true_fail"].astype(bool).sum()),
            "n_false_fail": int(sub["false_fail"].astype(bool).sum()),
            "n_ambiguous": int(sub["ambiguous_fail"].astype(bool).sum()),
            "pct_dead": 100.0 * sub["is_dead"].astype(bool).mean(),
            "pct_partial": 100.0 * sub["is_partial"].astype(bool).mean(),
            "mean_in_degree": float(sub["in_degree"].mean()),
            "mean_out_degree": float(sub["out_degree"].mean()),
            "mean_degree_u_671": float(np.mean([Gb671.degree(v) for v in sub.index])),
            "mean_ok_ratio_pct": float(sub["ok_ratio_pct"].mean()),
            "fisher_p_dead_vs_rest": fisher("is_dead"),
            "fisher_p_partial_vs_rest": fisher("is_partial"),
        })
    health_cross = pd.DataFrame(hrows)
    C.save_csv(health_cross, MODULE, "health_cross.csv")

    # 入度十分位 × 健康（pd.qcut；分位按 rank 打散重尾 ties）
    dec = pd.qcut(nd["in_degree"].rank(method="first"), 10, labels=False)
    drows = []
    for q in range(10):
        m_ = dec == q
        drows.append({
            "scope": "671", "indegree_decile": q + 1, "n": int(m_.sum()),
            "in_degree_min": int(nd.loc[m_, "in_degree"].min()),
            "in_degree_max": int(nd.loc[m_, "in_degree"].max()),
            "pct_dead": 100.0 * nd.loc[m_, "is_dead"].astype(bool).mean(),
            "pct_partial": 100.0 * nd.loc[m_, "is_partial"].astype(bool).mean(),
            "pct_healthy": 100.0 * nd.loc[m_, "is_healthy"].astype(bool).mean(),
            "mean_coreness": float(np.mean([cn671[v] for v in nd.index[m_]])),
        })
    C.save_csv(pd.DataFrame(drows), MODULE, "health_indegree_deciles.csv")

    # 锚点口径对照（CONVENTIONS §2.1 的 1.83/9.49 vs 1.83/4.94，本模块自算复核）
    indeg_by = nd.groupby("health_state")[["in_degree", "out_degree", "degree_u"]].mean()
    anchor_cmp = {
        "mean_in_degree_dead": float(indeg_by.loc["dead", "in_degree"]),
        "mean_in_degree_healthy": float(indeg_by.loc["healthy", "in_degree"]),
        "ratio_in_degree": float(indeg_by.loc["healthy", "in_degree"] / indeg_by.loc["dead", "in_degree"]),
        "mean_degree_u_dead": float(indeg_by.loc["dead", "degree_u"]),
        "mean_degree_u_healthy": float(indeg_by.loc["healthy", "degree_u"]),
        "ratio_degree_u": float(indeg_by.loc["healthy", "degree_u"] / indeg_by.loc["dead", "degree_u"]),
    }
    log("入度口径 dead/healthy = %.3f / %.3f（%.2fx）；无向度 = %.3f / %.3f（%.2fx）"
        % (anchor_cmp["mean_in_degree_dead"], anchor_cmp["mean_in_degree_healthy"],
           anchor_cmp["ratio_in_degree"], anchor_cmp["mean_degree_u_dead"],
           anchor_cmp["mean_degree_u_healthy"], anchor_cmp["ratio_degree_u"]))

    # ================================================= 汇总落盘
    result = {
        "module": MODULE, "seed": SEED,
        "scope_primary": "671 attributed undirected binary (m=%d)" % Gb671.number_of_edges(),
        "kcore": {
            "k_max_671": kmax671, "k_max_676": kmax676,
            "shell_sizes_671": {int(k): int(sum(1 for v in Gb671 if cn671[v] == k))
                                 for k in range(1, kmax671 + 1)},
            "cum_sizes_671": {int(k): int(sum(1 for v in Gb671 if cn671[v] >= k))
                               for k in range(1, kmax671 + 1)},
            "spearman_coreness_degree": float(sp_core_deg.statistic),
            "spearman_coreness_betweenness": float(sp_core_bet.statistic),
            "top_core_members": sorted(in_top),
            "bet_mwu_top_vs_rest_p": float(sps.mannwhitneyu(
                [bet671[v] for v in in_top], [bet671[v] for v in rest])[1]),
        },
        "richclub": {s: {k_: v for k_, v in rc_stats[s].items()
                         if not k_.startswith("_")} for s in ("671", "676")},
        "borgatti_everett": {
            s: {
                "best_start": ce_results[s]["best"]["start"],
                "k_star": ce_results[s]["best"]["k_star"],
                "fit_star": ce_results[s]["best"]["fit_star"],
                "fit_cont": ce_results[s]["best"]["fit_cont"],
                "fit_cont_best_during_iter": ce_results[s]["best"]["fit_cont_best"],
                "cont_converged": ce_results[s]["best"]["cont_converged"],
                "cont_iters": ce_results[s]["best"]["cont_iters"],
                "cont_delta": ce_results[s]["best"]["cont_delta"],
                "sensitivity_fit_star_range": [
                    min(r["fit_star"] for r in ce_results[s]["proc"]["runs"]),
                    max(r["fit_star"] for r in ce_results[s]["proc"]["runs"])],
                "null": ce_results[s]["null"],
            } for s in ("671", "676")},
        "core_sets_671": {k_: sorted(v) if k_ != "ALL_671" else []
                           for k_, v in set_defs.items()},
        "overlap_671": orows,
        "health_cross": hrows,
        "anchor_degree_cmp": anchor_cmp,
        "swap_probe_summary": {s: rc_stats[s]["swap"] for s in ("671", "676")},
        "versions": C.versions(),
        "wall_seconds": time.time() - wall0,
    }
    C.save_json(result, MODULE, "result.json")

    run_lines = [
        "A5_core_periphery RUN record",
        "command: python out/A5_core_periphery/run_a5.py  (cwd=pipeline)",
        "versions: " + json.dumps(C.versions()),
        "SEED=%d; PYTHONHASHSEED=%s" % (SEED, os.environ.get("PYTHONHASHSEED")),
        "anchors: 31/31 PASS (C.assert_anchors at script start)",
        "n_reps: rich-club null 671=%d, 676=%d (independent fresh-copy swaps, nswap=10m, max_tries=100*nswap)"
        % (N_REPS_RC_671, N_REPS_RC_676),
        "swap diagnostics: " + json.dumps(result["swap_probe_summary"]),
        "note: edge-switching is NOT an exactly uniform sampler (Viger & Latapy 2005)",
        "CE null pool reused from the same independent-swap replicates (671: %d, 676: %d graphs)"
        % (len(rc_stats["671"]["_pool"]), len(rc_stats["676"]["_pool"])),
        "dense-vs-closed-form CE fit check recorded in log lines",
    ] + _LOG
    with open(os.path.join(OUT, "RUN.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(run_lines) + "\n")
    log("saved csv/json/RUN to %s; total %.1fs" % (OUT, time.time() - wall0))


if __name__ == "__main__":
    main()
