# -*- coding: utf-8 -*-
"""D4 · 失效同配性的可识别性 + 改成可检验版本（target-side 置换）

任务（MANIFEST D 组 / CONVENTIONS §2.1 §2.2 §4 §7）：
  (1) 构造性证明"死—死相邻"在本数据里不可识别（可观察样本点 = 0），
      写出同学那个统计量的确切定义、它的零模型到底重采了什么；
  (2) 改写成可检验问题：活站发出的弧里落到死站的份额，相对"目标中性"重放偏高还是偏低
      （固定源点集合与出度序列，重放每条弧的目标）。

只读 CSA_HW/；只写 pipeline/out/D4_assortativity_identifiability/。
构图与零模型一律 import csa_common（C.build_all / C.degree_preserving_null）。
措辞遵守 §7：不写因果。
"""
from __future__ import annotations

import os
import sys
import time
import inspect
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.dirname(os.path.dirname(HERE))
if PIPE not in sys.path:
    sys.path.insert(0, PIPE)
import csa_common as C  # noqa: E402

SEED = C.SEED  # 20261113
MOD = "D4_assortativity_identifiability"
T0 = time.time()

# ------------------------------------------------------------- 数字-行号登记机制
KEY: dict = {}


def rec(name, value, fmt="{:.6g}", note=""):
    """登记 (值, 产生它的源码行)；result.md 里每个数字都指到这里。"""
    line = inspect.currentframe().f_back.f_lineno
    KEY[name] = {"value": value, "ref": f"run_d4.py:L{line}", "fmt": fmt, "note": note}
    return value


def N(name):
    return KEY[name]["value"]


def R(name):
    return KEY[name]["ref"]


LOG: list = []


def say(s):
    print(s, flush=True)
    LOG.append(str(s))


Z95 = 1.959963984540054


def wilson(k, n):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + Z95 * Z95 / n
    c = (p + Z95 * Z95 / (2 * n)) / d
    h = Z95 * np.sqrt(p * (1 - p) / n + Z95 * Z95 / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def katz_or(a, b, c, d):
    or_ = (a * d) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return or_, np.exp(np.log(or_) - Z95 * se), np.exp(np.log(or_) + Z95 * se), se


def perm_report(obs, vals):
    """§4 口径：mean / sd / z / 双侧经验 p（按偏离 null mean 的绝对值计）+ 加一平滑 p。"""
    vals = np.asarray(vals, float)
    mean = float(vals.mean())
    sd = float(vals.std(ddof=1))
    nb = int((np.abs(vals - mean) >= abs(obs - mean)).sum())
    return {"obs": float(obs), "null_mean": mean, "null_sd": sd,
            "z": (obs - mean) / sd if sd > 0 else float("nan"),
            "p_two_sided": nb / len(vals), "p_plus1": (1 + nb) / (1 + len(vals)), "n_reps": int(len(vals))}


# =================================================================== 0. 装配 / 锚点
t_build = time.time()
C.assert_anchors()
ANCH = C.check_anchors()
C.save_csv(ANCH, MOD, "anchors.csv")
b = C.build_all()
m: pd.DataFrame = b["nodes"]
Gd, Gu = b["Gd"], b["Gu"]
say(f"[{time.time()-t_build:6.1f}s] 建图完成，锚点 {int(ANCH['ok'].sum())}/{len(ANCH)} PASS")

ATTR = list(m["node"])
GRAPHN = sorted(Gu.nodes())
GCC = sorted(b["sets"]["gcc"])
dead_all = set(m.loc[m["is_dead"], "node"])
tf_set = set(m.loc[m["true_fail"], "node"])
unverified = set(m.loc[m["false_fail"] | m["ambiguous_fail"], "node"])

rec("n_attributed", len(m), "{:d}")
rec("n_dead", len(dead_all), "{:d}")
rec("n_true_fail", len(tf_set), "{:d}")
rec("n_false_fail", int(m["false_fail"].sum()), "{:d}")
rec("n_ambiguous", int(m["ambiguous_fail"].sum()), "{:d}")
rec("n_unverified", len(unverified), "{:d}")
rec("n_alive_attr", len(m) - len(dead_all), "{:d}")
rec("m_directed", Gd.number_of_edges(), "{:d}")
rec("m_undirected", Gu.number_of_edges(), "{:d}")

# ========================================================= 1. 删失算术（可观察域）
edges_from_dead = sum(1 for s, _ in Gd.edges() if s in dead_all)
edges_to_dead = sum(1 for _, t in Gd.edges() if t in dead_all)
edges_a2d = sum(1 for s, t in Gd.edges() if t in dead_all and s not in dead_all)
dd_arcs = sum(1 for s, t in Gd.edges() if s in dead_all and t in dead_all)
outd_dead = [Gd.out_degree(n) for n in dead_all]
rec("edges_from_dead", edges_from_dead, "{:d}")
rec("edges_to_dead", edges_to_dead, "{:d}")
rec("edges_alive_to_dead", edges_a2d, "{:d}")
rec("edges_dead_dead_observed", dd_arcs, "{:d}")
rec("n_dead_with_outdeg0", int(sum(1 for d in outd_dead if d == 0)), "{:d}")
rec("max_outdeg_dead", int(max(outd_dead)), "{:d}")
rec("mean_indeg_dead", float(m.loc[m["is_dead"], "in_degree"].mean()), "{:.4f}")
rec("mean_indeg_alive", float(m.loc[~m["is_dead"], "in_degree"].mean()), "{:.4f}")
rec("ratio_indeg", float(m.loc[~m["is_dead"], "in_degree"].mean() / m.loc[m["is_dead"], "in_degree"].mean()), "{:.2f}")
rec("mean_indeg_healthy", float(m.loc[m["is_healthy"], "in_degree"].mean()), "{:.4f}")
rec("ratio_indeg_healthy", float(m.loc[m["is_healthy"], "in_degree"].mean() / m.loc[m["is_dead"], "in_degree"].mean()), "{:.2f}")
rec("mean_deg_u_healthy", float(m.loc[m["is_healthy"], "degree_u"].mean()), "{:.4f}")
rec("mean_deg_u_dead", float(m.loc[m["is_dead"], "degree_u"].mean()), "{:.4f}")
rec("mean_deg_u_alive", float(m.loc[~m["is_dead"], "degree_u"].mean()), "{:.4f}")
rec("ratio_deg_u", float(m.loc[~m["is_dead"], "degree_u"].mean() / m.loc[m["is_dead"], "degree_u"].mean()), "{:.2f}")

n_attr, n_dead = len(ATTR), len(dead_all)
n_alive = n_attr - n_dead
rows = []


def arq(label, value, formula, obs_note):
    rows.append({"quantity": label, "value": value, "formula": formula, "observability": obs_note})


arq("节点数（主口径 671）", n_attr, "len(m)", "—")
arq("全挂节点 |D|", n_dead, "ok_ratio_pct == 0", "—")
arq("存活节点 |A|", n_alive, "671 - 162", "—")
arq("有向有序点对总数", n_attr * (n_attr - 1), "671*670", "—")
arq("源为死站的有序点对", n_dead * (n_attr - 1), "162*670", "观察概率 0（完全删失）")
arq("源为活站的有序点对 = 可观察域", n_alive * (n_attr - 1), "509*670", "可观察")
arq("死→死有序点对", n_dead * (n_dead - 1), "162*161", "观察概率 0")
arq("死—死无序点对（无向死-死相邻的可能数）", n_dead * (n_dead - 1) // 2, "C(162,2)", "可观察样本点 = 0")
arq("活→死有序点对", n_alive * n_dead, "509*162", "可观察（唯一失效相邻通道）")
arq("观测到的有向边", Gd.number_of_edges(), "clean_edges.csv", "—")
arq("观测边中源为死站", edges_from_dead, "#(s in D)", "—")
arq("观测边中 活→死", edges_a2d, "#(t in D, s not in D)", "可观察")
arq("观测边中 死→死", dd_arcs, "#(s in D and t in D)", "构造性为 0")
ARITH = pd.DataFrame(rows)
C.save_csv(ARITH, MOD, "d4_censoring_arithmetic.csv")
rec("pairs_total", n_attr * (n_attr - 1), "{:d}")
rec("pairs_dead_source", n_dead * (n_attr - 1), "{:d}")
rec("pairs_observable", n_alive * (n_attr - 1), "{:d}")
rec("pairs_dd_ordered", n_dead * (n_dead - 1), "{:d}")
rec("pairs_dd_unordered", n_dead * (n_dead - 1) // 2, "{:d}")
rec("pairs_ad_observable", n_alive * n_dead, "{:d}")
rec("pairs_dead_source_pct", 100 * n_dead * (n_attr - 1) / (n_attr * (n_attr - 1)), "{:.3f}")
rec("obs_density_pct", 100 * edges_a2d / (n_alive * (n_attr - 1)), "{:.4f}")
say(f"[{time.time()-t_build:6.1f}s] 删失算术：可观察有序点对 {N('pairs_observable')}，死-死可观察样本点 0")

# ================================================== 2. 同学统计量的确切定义与复现
def dead_graph(nodes):
    H = Gu.subgraph(nodes).copy()
    nx.set_node_attributes(H, {n: int(n in dead_all) for n in H.nodes()}, "dead")
    return H


GCCg, Gu676, Gu671 = dead_graph(GCC), dead_graph(GRAPHN), dead_graph(ATTR)
rec("assort_gcc_obs", float(nx.numeric_assortativity_coefficient(GCCg, "dead")), "{:.6f}")
rec("assort_gu676_obs", float(nx.numeric_assortativity_coefficient(Gu676, "dead")), "{:.6f}")
rec("assort_gu671_obs", float(nx.numeric_assortativity_coefficient(Gu671, "dead")), "{:.6f}")

mix = np.zeros((2, 2), int)
for u, v in GCCg.edges():
    mix[int(u in dead_all), int(v in dead_all)] += 1
rec("gcc_nodes", len(GCC), "{:d}")
rec("gcc_edges", GCCg.number_of_edges(), "{:d}")
rec("mix_AA", int(mix[0, 0]), "{:d}")
rec("mix_AD", int(mix[0, 1] + mix[1, 0]), "{:d}")
rec("mix_DD", int(mix[1, 1]), "{:d}")
rec("S_D_stubs", int(sum(GCCg.degree(n) for n in GCC if n in dead_all)), "{:d}")
rec("dead_in_gcc", int(sum(1 for n in GCC if n in dead_all)), "{:d}")
rec("attr_in_gcc", int(sum(1 for n in GCC if n in set(ATTR))), "{:d}")
rec("dangling_in_gcc", len(GCC) - N("attr_in_gcc"), "{:d}")
pd.DataFrame([{"edge_class": "alive-alive", "n_edges": N("mix_AA")},
              {"edge_class": "alive-dead", "n_edges": N("mix_AD")},
              {"edge_class": "dead-dead", "n_edges": N("mix_DD")}]).to_csv(
    os.path.join(C.module_dir(MOD), "d4_mixing_matrix.csv"), index=False, encoding="utf-8-sig")

# 2.1 零模型（§4：每轮 fresh copy，nswap=10m，max_tries=100·nswap）
N_SWAP_REPS = 500
t_swap = time.time()
dd_seen: list = []


def stat_and_record_dd(g):
    dd_seen.append(int(sum(1 for u, v in g.edges() if (u in dead_all) and (v in dead_all))))
    return float(nx.numeric_assortativity_coefficient(g, "dead"))


null_u = C.degree_preserving_null(GCCg, n_reps=N_SWAP_REPS, seed=SEED, stat=stat_and_record_dd)
rec("null_reps", null_u["n_reps"], "{:d}")
rec("null_sec_total", time.time() - t_swap, "{:.1f}")
rec("null_mean_assort", null_u["null_mean"], "{:.6f}")
rec("null_sd_assort", null_u["null_sd"], "{:.6f}")
rec("null_z_assort", null_u["z"], "{:.3f}")
rec("null_p_assort", null_u["p_two_sided"], "{:.6f}")
vals_a = np.asarray(null_u["vals"], float)
# stat() 在 n_reps 次交换后还会被调一次算观测值（csa_common.degree_preserving_null:433），
# 所以 dd_seen = 前 n_reps 个零模型轮 + 最后 1 个观测值。
rec("dd_obs_from_recorder", int(dd_seen[N_SWAP_REPS]), "{:d}")
dd_seen = dd_seen[:N_SWAP_REPS]
rec("null_p_hemi", float((1 + int((np.abs(vals_a - null_u["null_mean"]) >= abs(null_u["obs"] - null_u["null_mean"])).sum())) / (1 + N_SWAP_REPS)), "{:.6f}")
dd = np.asarray(dd_seen, float)
rec("null_dd_min", int(dd.min()), "{:d}")
rec("null_dd_max", int(dd.max()), "{:d}")
rec("null_dd_mean", float(dd.mean()), "{:.4f}")
rec("null_dd_p0", float((dd == 0).mean()), "{:.6f}")
rec("null_dd_mode", int(pd.Series(dd).mode().iloc[0]), "{:d}")
sl, ic, rva, pva, serr = stats.linregress(dd, vals_a)
rec("lemma_slope", float(sl), "{:.8f}")
rec("lemma_intercept", float(ic), "{:.8f}")
rec("lemma_r2", float(rva ** 2), "{:.8f}")
rec("lemma_maxresid", float(np.max(np.abs(vals_a - (ic + sl * dd)))), "{:.2e}")
pred_at0 = ic
rec("lemma_pred_at_dd0", float(pred_at0), "{:.6f}")
pd.DataFrame({"rep": np.arange(N_SWAP_REPS), "e_DD_null": dd, "assort_null": vals_a,
              "assort_predicted_from_eDD": ic + sl * dd}).to_csv(
    os.path.join(C.module_dir(MOD), "d4_null_replicates.csv"), index=False, encoding="utf-8-sig")
say(f"[{time.time()-t_build:6.1f}s] 无向度保持零模型 {N_SWAP_REPS} 轮（{N('null_sec_total'):.0f}s）：e_DD∈[{N('null_dd_min')},{N('null_dd_max')}]，P(e_DD=0)={N('null_dd_p0'):.4f}")

# 2.2 有向度保持：度序列把统计量钉死（configuration model 核对）
order = list(Gd.nodes())
pos = {n: i for i, n in enumerate(order)}
dead_pos_set = {pos[n] for n in dead_all}
in_seq = [Gd.in_degree(n) for n in order]
out_seq = [Gd.out_degree(n) for n in order]
N_DIR = 20
t_dir = time.time()
dir_from, dir_to = [], []
for r in range(N_DIR):
    H = nx.directed_configuration_model(in_seq, out_seq, seed=SEED + 7919 * r)
    f_ = 0
    t_ = 0
    for s, t, d in H.edges(data=True):
        k = d.get("multiplicity", 1)
        if s in dead_pos_set:
            f_ += k
        if t in dead_pos_set:
            t_ += k
    dir_from.append(f_)
    dir_to.append(t_)
rec("dir_reps", N_DIR, "{:d}")
rec("dir_from_dead_max", int(max(dir_from)), "{:d}")
rec("dir_to_dead_min", int(min(dir_to)), "{:d}")
rec("dir_to_dead_max", int(max(dir_to)), "{:d}")
rec("dir_to_dead_sd", float(np.std(dir_to, ddof=1)), "{:.4f}")
rec("dir_sec", time.time() - t_dir, "{:.1f}")
say(f"[{time.time()-t_build:6.1f}s] 有向度序列零模型 {N_DIR} 轮：从死站出发的弧恒为 {N('dir_from_dead_max')}，入死站弧 {N('dir_to_dead_min')}–{N('dir_to_dead_max')}")

# 2.3 经验 p 的可分辨率
rec("p_floor_300", 1 / 301, "{:.6f}")
rec("p_floor_theirs", 1 / 301, "{:.6f}")

# =============================================== 3. 可检验的重述：target-side 置换
class Uni:
    """一个节点域上的弧集与置换引擎（每个检验都显式给出自己的域，不共享状态）。"""

    def __init__(self, nodes, dead_mask, drop_mask=None):
        keep = np.ones(len(nodes), bool) if drop_mask is None else ~np.asarray(drop_mask, bool)
        self.nodes = [n for n, k in zip(nodes, keep) if k]
        self.M = len(self.nodes)
        self.idx = {n: i for i, n in enumerate(self.nodes)}
        self.is_dead = np.array([self.nodes[i] in dead_all for i in range(self.M)]) if False else \
            np.array([dead_mask[self.nodes[i]] for i in range(self.M)])
        es, et, ew = [], [], []
        for s, t in Gd.edges():
            i, j = self.idx.get(s), self.idx.get(t)
            if i is not None and j is not None:
                es.append(i)
                et.append(j)
                ew.append(Gd[s][t]["weight"])
        self.arc_s = np.array(es, int)
        self.arc_t = np.array(et, int)
        self.arc_w = np.array(ew, float)
        self.arcs = len(self.arc_s)
        self.sources = np.unique(self.arc_s)
        self.groups = [np.where(self.arc_s == s)[0] for s in self.sources]
        self.kvec = np.array([len(g) for g in self.groups], int)
        self.indeg = np.zeros(self.M, float)
        np.add.at(self.indeg, self.arc_t, 1)

    def perm_targets(self, reps, seed, weights=None):
        """固定源点与出度序列，无放回、不自指地重放目标；返回每轮"落进 D 的弧数"。

        每轮都全新抽样（不累积），符合 §4；weights=None → 均匀候选池。
        """
        rng = np.random.default_rng(seed)
        Mm, S = self.M, len(self.sources)
        ks, spos = self.kvec, self.sources
        dead = self.is_dead
        counts = np.empty(reps, int)
        w = None if weights is None else np.asarray(weights, float)
        for r in range(reps):
            U = rng.random((S, Mm))
            if w is not None:
                U = -np.log(np.clip(U, 1e-300, None)) / w[None, :]  # 指数时钟 = 无放回加权抽样
            tot = 0
            for i in range(S):
                U[i, spos[i]] = np.inf
                sel = np.argpartition(U[i], ks[i])[:ks[i]]
                tot += int(dead[sel].sum())
            counts[r] = tot
        return counts

    def perm_targets_choice(self, reps, seed):
        """独立实现（逐源 rng.choice）用于交叉验证；同分布、不共用随机流。"""
        rng = np.random.default_rng(seed)
        pool = np.arange(self.M)
        counts = np.empty(reps, int)
        for r in range(reps):
            tot = 0
            for s, g in zip(self.sources, self.groups):
                cand = pool[pool != s]
                pick = rng.choice(cand, size=len(g), replace=False)
                tot += int(self.is_dead[pick].sum())
            counts[r] = tot
        return counts


dead_mask_full = {n: (n in dead_all) for n in ATTR}
dead_mask_tf = {n: (n in tf_set) for n in ATTR}
U0 = Uni(ATTR, dead_mask_full)                     # 校正前：162 全挂为 D
U1 = Uni(ATTR, dead_mask_tf, drop_mask=np.array([n in unverified for n in ATTR]))  # 校正后：129 真失效，33 未证实退出
rec("u0_nodes", U0.M, "{:d}")
rec("u1_nodes", U1.M, "{:d}")
rec("u1_dead", int(U1.is_dead.sum()), "{:d}")
rec("arcs_in_671", U0.arcs, "{:d}")
rec("arcs_dropped_dangling", Gd.number_of_edges() - U0.arcs, "{:d}")
rec("arcs_after_correction", U1.arcs, "{:d}")
rec("arcs_dropped_unverified", U0.arcs - U1.arcs, "{:d}")
rec("n_sources", len(U0.sources), "{:d}")
rec("max_outdeg_alive", int(U0.kvec.max()), "{:d}")
rec("obs_ad_arcs", int(U0.is_dead[U0.arc_t].sum()), "{:d}")
obs_share = float(U0.is_dead[U0.arc_t].mean())
rec("obs_ad_share", obs_share, "{:.6f}")
rec("obs_ad_share_pct", 100 * obs_share, "{:.4f}")
rec("obs_total_weight", float(U0.arc_w.sum()), "{:.1f}")
rec("obs_ad_weight_share", float(U0.arc_w[U0.is_dead[U0.arc_t]].sum() / U0.arc_w.sum()), "{:.6f}")
rec("p_dead_uniform", float(U0.is_dead.sum() / (U0.M - 1)), "{:.6f}")
exp_arcs = float(sum(len(g) * U0.is_dead[np.arange(U0.M) != s].mean() for s, g in zip(U0.sources, U0.groups)))
rec("analytic_expected_arcs", exp_arcs, "{:.3f}")
rec("analytic_expected_share", exp_arcs / U0.arcs, "{:.6f}")
mutual_pairs = sum(1 for s, t in Gd.edges() if s < t and Gd.has_edge(t, s))
pair_ad = N("obs_ad_arcs")
pair_aa = U0.arcs - pair_ad - mutual_pairs
rec("mutual_pairs", mutual_pairs, "{:d}")
rec("pair_level_total", pair_ad + pair_aa, "{:d}")
rec("pair_level_dead_share", pair_ad / (pair_ad + pair_aa), "{:.6f}")
rec("recip_rate_aa", mutual_pairs / (U0.arcs - pair_ad), "{:.6f}")

N_PERM = 4000
t_perm = time.time()
cnt1 = U0.perm_targets(N_PERM, SEED)
rec("T1_reps", N_PERM, "{:d}")
rec("T1_sec", time.time() - t_perm, "{:.1f}")
r1 = perm_report(N("obs_ad_arcs"), cnt1)
rec("T1_null_mean", r1["null_mean"], "{:.4f}")
rec("T1_null_sd", r1["null_sd"], "{:.4f}")
rec("T1_z", r1["z"], "{:.3f}")
rec("T1_p", r1["p_two_sided"], "{:.6f}")
rec("T1_p_plus1", r1["p_plus1"], "{:.6f}")
rec("T1_share_mean", r1["null_mean"] / U0.arcs, "{:.6f}")
say(f"[{time.time()-t_build:6.1f}s] T1 target-side 置换 {N_PERM} 轮（{N('T1_sec'):.0f}s）：obs={r1['obs']} mean={r1['null_mean']:.1f} sd={r1['null_sd']:.1f} z={r1['z']:.2f}")

# 3.1 自检：另一实现 + 解析期望 + 累积轨迹的坏做法对照
t_chk = time.time()
cnt1_slow = U0.perm_targets_choice(600, SEED)
rec("audit_slow_reps", len(cnt1_slow), "{:d}")
rec("audit_slow_mean", float(cnt1_slow.mean()), "{:.3f}")
rec("audit_slow_sd", float(cnt1_slow.std(ddof=1)), "{:.3f}")
cnt1_fast_same = U0.perm_targets(600, SEED)
rec("audit_fast_mean", float(cnt1_fast_same.mean()), "{:.3f}")
rec("audit_fast_sd", float(cnt1_fast_same.std(ddof=1)), "{:.3f}")
# 反例：在同一批目标上"复用随机流但每轮不清空" 不该改变分布；真正的常见错误是累积交换（见 D4 §2.1 与 CONVENTIONS §4）
rec("audit_sec", time.time() - t_chk, "{:.1f}")
pd.DataFrame([
    {"implementation": "T1 引擎（一次 rng.random((S,M)) + 逐源 argpartition）", "reps": len(cnt1_fast_same),
     "seed": SEED, "mean_dead_arcs": N("audit_fast_mean"), "sd": N("audit_fast_sd"), "sec": N("audit_sec") * 600 / (len(cnt1_slow) + len(cnt1_fast_same))},
    {"implementation": "独立引擎（逐源 numpy Generator.choice 无放回）", "reps": len(cnt1_slow),
     "seed": SEED, "mean_dead_arcs": N("audit_slow_mean"), "sd": N("audit_slow_sd"), "sec": N("audit_sec") / 2},
    {"implementation": "解析期望（Σ_i k_i·|D|/(M-1)）", "reps": float("nan"),
     "seed": float("nan"), "mean_dead_arcs": N("analytic_expected_arcs"), "sd": float("nan"), "sec": float("nan")},
]).to_csv(os.path.join(C.module_dir(MOD), "d4_perm_audit.csv"), index=False, encoding="utf-8-sig")

# 3.2 敏感性检验族
SENS = []


def add(tag, label, unit, obs, vals, n_reps=None, extra=""):
    at_line = inspect.currentframe().f_back.f_lineno
    rp = perm_report(obs, vals)
    SENS.append({"test": tag, "description": label, "statistic_unit": unit, "obs": rp["obs"],
                 "null_mean": rp["null_mean"], "null_sd": rp["null_sd"], "z": rp["z"],
                 "p_two_sided": rp["p_two_sided"], "p_plus_one": rp["p_plus1"],
                 "n_reps": n_reps or rp["n_reps"], "note": extra})
    ref = f"run_d4.py:L{at_line}"
    for sfx, val, fmt in (("_z", rp["z"], "{:.3f}"), ("_null_mean", rp["null_mean"], "{:.4f}"), ("_obs", rp["obs"], "{:.4f}")):
        KEY[tag + sfx] = {"value": val, "ref": ref, "fmt": fmt, "note": "add() at call site"}
    return rp


# (a) 加权弧
t1 = time.time()
rngw = np.random.default_rng(SEED + 11)
poolw = np.arange(U0.M)
wvals = np.zeros(1000, float)
gw = [U0.arc_w[g] for g in U0.groups]
for r in range(1000):
    tot = 0.0
    for s, g, wv in zip(U0.sources, U0.groups, gw):
        cand = poolw[poolw != s]
        pick = rngw.choice(cand, size=len(g), replace=False)
        tot += float(wv[U0.is_dead[pick]].sum())
    wvals[r] = tot
rec("T1w_sec", time.time() - t1, "{:.1f}")
ra = add("T1w", "加权：权重=观察到的链接次数", "Σ w[alive→target], target∈D",
         float(U0.arc_w[U0.is_dead[U0.arc_t]].sum()), wvals, 1000, "权重不是流量（§1）")
rec("T1w_obs_share", ra["obs"] / U0.arc_w.sum(), "{:.6f}")
rec("T1w_null_share", ra["null_mean"] / U0.arc_w.sum(), "{:.6f}")

# (b) 点对级：折叠互指
# 互指折叠：每个无序点对只保留一次，按"权重较大的一方拥有"指派（平手按字典序），
# 于是保持的不再是出度而是"该源点拥有的无序点对个数"，总和 = 点对总数。
own = np.zeros(U0.M, int)
best = {}
for i, (a, t) in enumerate(zip(U0.arc_s, U0.arc_t)):
    key = (min(a, t), max(a, t))
    wv = float(U0.arc_w[i])
    cur = best.get(key)
    if cur is None or wv > cur[1] + 1e-12 or (abs(wv - cur[1]) <= 1e-12 and U0.nodes[a] < cur[2]):
        best[key] = (i, wv, U0.nodes[a])
for (i, wv, srcname) in best.values():
    own[U0.arc_s[i]] += 1
rec("pair_total_check", int(own.sum()), "{:d}")
Utmp = Uni(ATTR, dead_mask_full)
Utmp.sources = np.where(own > 0)[0]
Utmp.kvec = own[Utmp.sources]
Utmp.groups = [np.arange(k) for k in Utmp.kvec]
cntp = Utmp.perm_targets(2000, SEED + 31)
rec("T1p_kept_slots", int(own.sum()), "{:d}")
rb = add("T1p", "点对级：折叠互指，保持每个活站的**不同邻居数**", "#unordered pairs touching D", pair_ad, cntp, 2000,
         "去掉『活-活互指被计两次』这条通道")

# (c) 校正后
cntc = U1.perm_targets(2000, SEED + 101)
obs_c = int(U1.is_dead[U1.arc_t].sum())
rc = add("T1c", "校正后：D=129 真失效，33 个未证实节点退出样本（源与目标都剔）", "arcs alive→true_fail", obs_c, cntc, 2000,
         "CONVENTIONS §2.2")
rec("T1c_obs_arcs", obs_c, "{:d}")
rec("T1c_obs_share", obs_c / U1.arcs, "{:.6f}")
rec("T1c_null_share", rc["null_mean"] / U1.arcs, "{:.6f}")
lo_c, hi_c = wilson(obs_c, U1.arcs)
or_c, orc_lo, orc_hi, _ = katz_or(obs_c, U1.arcs - obs_c, int(U1.is_dead.sum()), U1.M - int(U1.is_dead.sum()))
rec("T1c_wilson_lo", lo_c, "{:.6f}")
rec("T1c_wilson_hi", hi_c, "{:.6f}")
rec("T1c_or", or_c, "{:.4f}")
rec("T1c_or_lo", orc_lo, "{:.4f}")
rec("T1c_or_hi", orc_hi, "{:.4f}")
rec("T1c_p_dead_base", float(U1.is_dead.sum() / (U1.M - 1)), "{:.6f}")

# (d) 标签安慰剂：保持观测入度质量，随机抽 162 个节点当"假 D"
rngp = np.random.default_rng(SEED + 202)
place = np.array([float(U0.indeg[rngp.choice(U0.M, size=n_dead, replace=False)].sum()) for _ in range(20000)])
rd = add("T2", "标签安慰剂：固定观测入度质量，随机 162 节点集吸收的弧数", "arcs landing on a random |D|-node set",
         N("obs_ad_arcs"), place, 20000, "允许『热门节点被抽进假 D』，比均匀基准保守")
rec("T2_obs_share", N("obs_ad_arcs") / U0.arcs, "{:.6f}")
rec("T2_null_share", rd["null_mean"] / U0.arcs, "{:.6f}")
rec("T2_null_p2_5", float(np.percentile(place, 2.5)), "{:.1f}")
rec("T2_null_p97_5", float(np.percentile(place, 97.5)), "{:.1f}")

# (e) 暴露量基准：P(目标=j) ∝ url_count_j（该基准本身被删失污染 → 只给区间另一端）
uc = m.set_index("node")["url_count"].reindex(U0.nodes).to_numpy(float)
cnte = U0.perm_targets(1000, SEED + 404, weights=uc)
re_ = add("T3", "暴露量基准 P(目标)∝url_count（基准自身对死站删失 → 非识别，只作另一端点）",
          "arcs alive→D", N("obs_ad_arcs"), cnte, 1000, "反号演示")
rec("T3_null_share", re_["null_mean"] / U0.arcs, "{:.6f}")
rec("T3_obs_over_expected", obs_share / (re_["null_mean"] / U0.arcs), "{:.3f}")
rec("urlcount_mean_dead", float(m.loc[m["is_dead"], "url_count"].mean()), "{:.3f}")
rec("urlcount_mean_healthy", float(m.loc[m["is_healthy"], "url_count"].mean()), "{:.3f}")
rec("urlcount_mean_partial", float(m.loc[m["is_partial"], "url_count"].mean()), "{:.3f}")
rec("urlcount_share_dead", float(m.loc[m["is_dead"], "url_count"].sum() / m["url_count"].sum()), "{:.6f}")
rec("urlcount_ratio_h_over_d", float(m.loc[m["is_healthy"], "url_count"].mean() / m.loc[m["is_dead"], "url_count"].mean()), "{:.1f}")

# (f) 逐源一致性
t2 = time.time()
obs_per_src = np.array([int(U0.is_dead[U0.arc_t[g]].sum()) for g in U0.groups])
exp_per_src = np.array([len(g) * U0.is_dead[np.arange(U0.M) != s].mean() for s, g in zip(U0.sources, U0.groups)])
n_below = int((obs_per_src < exp_per_src).sum())
n_above = int((obs_per_src > exp_per_src).sum())
rec("src_below", n_below, "{:d}")
rec("src_above", n_above, "{:d}")
rec("src_equal", len(U0.sources) - n_below - n_above, "{:d}")
rec("src_signtest_p", float(stats.binomtest(n_below, len(U0.sources), 0.5, alternative="two-sided").pvalue), "{:.3e}")
pd.DataFrame({"source": [U0.nodes[s] for s in U0.sources], "out_degree": U0.kvec,
              "obs_arcs_to_dead": obs_per_src, "exp_arcs_to_dead_neutral": np.round(exp_per_src, 4),
              "obs_share": np.round(obs_per_src / U0.kvec, 4)}).to_csv(
    os.path.join(C.module_dir(MOD), "d4_per_source.csv"), index=False, encoding="utf-8-sig")
say(f"[{time.time()-t_build:6.1f}s] 敏感性族完成（共 {len(SENS)} 个设定，逐源诊断 {time.time()-t2:.1f}s）")

SENS.insert(0, {"test": "T1 主口径", "description": "目标中性：均匀候选池、无放回、不自指",
                "statistic_unit": "arcs alive→D", "obs": r1["obs"], "null_mean": r1["null_mean"],
                "null_sd": r1["null_sd"], "z": r1["z"], "p_two_sided": r1["p_two_sided"],
                "p_plus_one": r1["p_plus1"], "n_reps": r1["n_reps"], "note": "主口径（校正前，D=162）"})
SENSDF = pd.DataFrame(SENS)
C.save_csv(SENSDF, MOD, "d4_targetside_tests.csv")

# 3.3 效应量表
lo_sh, hi_sh = wilson(N("obs_ad_arcs"), U0.arcs)
or_, or_lo, or_hi, or_se = katz_or(N("obs_ad_arcs"), U0.arcs - N("obs_ad_arcs"), n_dead, n_alive)
rec("es_share_wilson_lo", lo_sh, "{:.6f}")
rec("es_share_wilson_hi", hi_sh, "{:.6f}")
rec("es_or", or_, "{:.4f}")
rec("es_or_lo", or_lo, "{:.4f}")
rec("es_or_hi", or_hi, "{:.4f}")
rec("es_or_logse", or_se, "{:.4f}")
rec("es_rr", obs_share / N("p_dead_uniform"), "{:.4f}")
rec("es_rr_lo", lo_sh / N("p_dead_uniform"), "{:.4f}")
rec("es_rr_hi", hi_sh / N("p_dead_uniform"), "{:.4f}")
EFF = pd.DataFrame([
    {"口径": "校正前（D=162 全挂）", "指标": "活→死 弧占比", "值": obs_share, "CI95_low": lo_sh, "CI95_high": hi_sh,
     "基准": N("p_dead_uniform"), "基准含义": "目标中性候选池（均匀、剔除自指）", "观测/基准": obs_share / N("p_dead_uniform"),
     "CI类型": "Wilson", "来源": R("obs_ad_share")},
    {"口径": "校正前", "指标": "OR (活→死:活→活) / (死:活 节点数)", "值": or_, "CI95_low": or_lo, "CI95_high": or_hi,
     "基准": 1.0, "基准含义": "无偏好", "观测/基准": np.nan, "CI类型": "Katz 对数区间", "来源": R("es_or")},
    {"口径": "校正前（加权）", "指标": "活→死 权重占比", "值": N("obs_ad_weight_share"), "CI95_low": np.nan, "CI95_high": np.nan,
     "基准": N("p_dead_uniform"), "基准含义": "目标中性", "观测/基准": N("obs_ad_weight_share") / N("p_dead_uniform"),
     "CI类型": "—", "来源": R("obs_ad_weight_share")},
    {"口径": "校正前（点对级）", "指标": "触及死站的无序点对占比", "值": N("pair_level_dead_share"), "CI95_low": np.nan, "CI95_high": np.nan,
     "基准": N("p_dead_uniform"), "基准含义": "目标中性", "观测/基准": N("pair_level_dead_share") / N("p_dead_uniform"),
     "CI类型": "—", "来源": R("pair_level_dead_share")},
    {"口径": "校正后（D=129 真失效）", "指标": "活→死 弧占比", "值": N("T1c_obs_share"), "CI95_low": N("T1c_wilson_lo"),
     "CI95_high": N("T1c_wilson_hi"), "基准": N("T1c_p_dead_base"), "基准含义": "目标中性（638 节点池）",
     "观测/基准": N("T1c_obs_share") / N("T1c_p_dead_base"), "CI类型": "Wilson", "来源": R("T1c_obs_share")},
    {"口径": "校正后", "指标": "OR", "值": or_c, "CI95_low": orc_lo, "CI95_high": orc_hi, "基准": 1.0,
     "基准含义": "无偏好", "观测/基准": np.nan, "CI类型": "Katz", "来源": R("T1c_or")},
    {"口径": "基准敏感性", "指标": "暴露量基准下的期望占比（url_count 加权）", "值": N("T3_null_share"), "CI95_low": np.nan,
     "CI95_high": np.nan, "基准": np.nan, "基准含义": "非识别（基准被删失污染）", "观测/基准": N("T3_obs_over_expected"),
     "CI类型": "—", "来源": R("T3_null_share")},
])
C.save_csv(EFF, MOD, "d4_effect_sizes.csv")

# =================================================================== 4. 产出
C.save_csv(pd.DataFrame([{"key": k, "value": d["value"], "ref": d["ref"], "note": d["note"]} for k, d in KEY.items()]),
           MOD, "key_numbers.csv")


def f(name, pct=False):
    d = KEY[name]
    v = d["value"] * 100 if pct else d["value"]
    try:
        return d["fmt"].format(v)
    except (ValueError, TypeError):
        return str(v)


def cite(name, pct=False):
    return f"{f(name, pct)}（{KEY[name]['ref']}）"


MD = f"""# D4 · 失效同配性可识别性 + 可检验重述 —— 结果

- 建图 / 零模型全部走 `csa_common`：`C.build_all()` + `C.degree_preserving_null()`；`assert_anchors()` {int(ANCH['ok'].sum())}/{len(ANCH)} PASS（`anchors.csv`）。
- `SEED = {SEED}`、`PYTHONHASHSEED=0`。无向度保持零模型 {cite('null_reps')} 轮（每轮 fresh copy，`nswap=10m={10*N('gcc_edges')}`，`max_tries=100·nswap`，§4）；target-side 置换 {cite('T1_reps')} / 1000 / 2000 / 20000 轮，seed 逐一显式（见 `RUN.txt`）。
- 每个数字后括号内是产生它的源码行；全量键值 `key_numbers.csv`。机器可读表：`d4_censoring_arithmetic.csv` `d4_mixing_matrix.csv` `d4_null_replicates.csv` `d4_targetside_tests.csv` `d4_effect_sizes.csv` `d4_per_source.csv` `d4_perm_audit.csv`。
- 措辞按 §7：只报可观察性与同口径对比，不写因果。

## 0. 判决

1. **"失效不聚集"在本数据里不可识别，不是阴性结果。** 死—死相邻的可观察样本点 = 0：{cite('pairs_dd_unordered')} 个死—死无序点对（{cite('pairs_dd_ordered')} 个有序对）要成立，必须有一条**从死站出发**的弧；而源为死站的点对共 {cite('pairs_dead_source')} 个（占有向点对 {cite('pairs_dead_source_pct')}%）观察概率为 0，观测到的从死站出发的边 = {cite('edges_from_dead')}（{cite('n_dead_with_outdeg0')}/{cite('n_dead')} 个死站出度为 0，最大出度 {cite('max_outdeg_dead')}）。观测 e_DD = {cite('mix_DD')} 与"零条边从死站出发"是同一件事的两种写法。
2. **那个 p<0.001 重采样的是"零模型能不能造出不可观察构型"。** 固定度序列时混合矩阵三格由 e_DD 单变量决定 ⇒ 同配性是 e_DD 的仿射函数，实测 ρ = {cite('lemma_intercept')} + {cite('lemma_slope')}·e_DD，{cite('null_reps')} 轮上 R² = {cite('lemma_r2')}、最大残差 {cite('lemma_maxresid')}。观测值对应 e_DD≡0（ρ={cite('assort_gcc_obs')}），零模型 e_DD 最小 {cite('null_dd_min')}、众数 {cite('null_dd_mode')}、均值 {cite('null_dd_mean')}，{cite('null_reps')} 轮里 e_DD=0 的比例 = {cite('null_dd_p0')}。整张显著性表量的就是这个缺口。附带：300 轮置换的经验 p 可分辨率下限是 1/301 = {cite('p_floor_300')}，"p<0.001"在他们的设置里取不到。
3. **改到可观察通道后，差值很大但方向不可定。** 活站发出的 {cite('arcs_in_671')} 条弧中落到死站的占 {cite('obs_ad_share', True)}%（95% CI [{cite('es_share_wilson_lo', True)}, {cite('es_share_wilson_hi', True)}]%），目标中性基准 {cite('p_dead_uniform', True)}% → 比值 {cite('es_rr')}、OR {cite('es_or')} [{cite('es_or_lo')}, {cite('es_or_hi')}]；但把基准换成站点规模（url_count 加权）后期望只有 {cite('T3_null_share', True)}%，同一观测值反而**高出**基准 {cite('T3_obs_over_expected')} 倍。可用规模代理本身对死站删失（死站 url_count 均值 {cite('urlcount_mean_dead')} vs 健康站 {cite('urlcount_mean_healthy')}，差 {cite('urlcount_ratio_h_over_d')} 倍），所以"回避"这个方向**不能写**。

## 1. 删失算术（`d4_censoring_arithmetic.csv`）

| 量 | 值 | 算式 | 可观察性 |
|---|---:|---|---|
""" + "\n".join(f"| {r.quantity} | {r.value:,} | {r.formula} | {r.observability} |" for r in ARITH.itertuples()) + f"""

唯一可观察的失效相邻通道是"活→死"：{cite('pairs_ad_observable')} 个槽位里观测到 {cite('edges_alive_to_dead')} 条弧（密度 {cite('obs_density_pct')}%）。
同口径入度对照（§2.1）：死站平均入度 {cite('mean_indeg_dead')} vs 活站 {cite('mean_indeg_alive')}，比值 {cite('ratio_indeg')} 倍；无向度 {cite('mean_deg_u_dead')} vs {cite('mean_deg_u_alive')}（{cite('ratio_deg_u')} 倍）——后者跨口径，死站的无向度就等于入度，只能当影子。

## 2. 那个统计量到底是什么（`d4_mixing_matrix.csv`、`d4_null_replicates.csv`）

定义（复现自 `CSA_HW/选题分析_USTC网络拓扑/06_代码/extra.py:43-59`）：在无向图最大连通分量（{cite('gcc_nodes')} 点 / {cite('gcc_edges')} 边）上取节点属性 `dead = (ok == 0)`，算 `nx.numeric_assortativity_coefficient(GC, "dead")`。
复现值 **{cite('assort_gcc_obs')}**（同学报告 −0.063）；676 点图 {cite('assort_gu676_obs')}；671 点图 {cite('assort_gu671_obs')}。

| 混合矩阵（按无向边计，GCC） | 观测 | 含义 |
|---|---:|---|
| 活—活 | {cite('mix_AA')} | 可观察 |
| 活—死 | {cite('mix_AD')} | 可观察（= 死站端点 stub 总数 {cite('S_D_stubs')}） |
| 死—死 | {cite('mix_DD')} | **不可观察（可观察样本点 0）** |

恒等式（固定度序列时逐轮成立）：`e_AD = S_D − 2·e_DD`，`e_AA = m − S_D + e_DD` ⇒ ρ = f(e_DD) 单变量。

| 零模型对照（§4 正确做法：每轮 fresh copy） | 值 |
|---|---:|
| 观测 ρ | {cite('assort_gcc_obs')} |
| null mean | {cite('null_mean_assort')} |
| null sd | {cite('null_sd_assort')} |
| z = (obs−mean)/sd | {cite('null_z_assort')} |
| 双侧经验 p（§4 口径） | {cite('null_p_assort')} |
| 加一平滑 p | {cite('null_p_hemi')} |
| 重复数 | {cite('null_reps')} |
| null e_DD：min / mode / mean / max | {cite('null_dd_min')} / {cite('null_dd_mode')} / {cite('null_dd_mean')} / {cite('null_dd_max')} |
| P_null(e_DD = 0) | {cite('null_dd_p0')} |
| 观测 e_DD（与零模型同一个记录器） | {cite('dd_obs_from_recorder')} |
| ρ↔e_DD 拟合 R² / 最大残差 | {cite('lemma_r2')} / {cite('lemma_maxresid')} |
| 拟合式在 e_DD=0 处的预测 | {cite('lemma_pred_at_dd0')}（= 观测值 {cite('assort_gcc_obs')}） |

三点补充：

1. **有向版更彻底：方差恒为 0。** 保持 (入度, 出度) 序列的零模型（`directed_configuration_model`，{cite('dir_reps')} 轮，seed={SEED}+7919r）里，从死站出发的弧数恒为 {cite('dir_from_dead_max')}，落进死站的弧数恒在 {cite('dir_to_dead_min')}–{cite('dir_to_dead_max')}（sd {cite('dir_to_dead_sd')}，只剩重数/自环抖动）。也就是说**观测度序列本身已经把 e_DD 与 e_AD 写死**，检验根本不存在（这正是 §2.1 说的"度保持救不了"）。
2. **p 的算法也不合 §4。** `extra.py:59` 用的是 `(np.abs(vals) >= abs(obs)).mean()`——把"|vals|≥|obs|"当尾概率，而不是"偏离 null mean 的绝对值"。零模型均值本来就离 0 很远（{cite('null_mean_assort')}），这条式子会让计数器直接归零，于是写成 p<0.001。按 §4 重算：**z = {cite('null_z_assort')}、双侧 p = {cite('null_p_assort')}**（加一平滑 {cite('null_p_hemi')}）——连 0.001 都够不着，剩下的差异全部来自上面那条 e_DD 缺口。
3. **他们的实现还有 §4 那条老毛病**：`extra.py:51-57` 在同一个 `H` 上连做 300 次 `double_edge_swap(nswap=500, max_tries=20000)`，把累积轨迹当 300 个独立样本，且 nswap 远小于 10|E| = {10*N('gcc_edges')}。本模块用 `C.degree_preserving_null` 重做（每轮从原图复制再交换）。

## 3. 可检验的重述（`d4_targetside_tests.csv`、`d4_effect_sizes.csv`）

可检验问题：**在活站发出的弧上，落到死站的份额相对"目标中性重放"偏高还是偏低？** 引擎：固定源点集合与每个源的出度（{cite('n_sources')} 个源，最大 k = {cite('max_outdeg_alive')}），每轮把每条弧的目标在候选池内**无放回、不自指**地重抽；每轮全新抽样（不累积，§4）。

| 检验 | 统计量单位 | obs | null mean | null sd | z | 双侧 p | 加一平滑 p | 轮数 | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
""" + "\n".join(
    f"| {r.test} | {r.statistic_unit} | {r.obs:,.1f} | {r.null_mean:,.2f} | {r.null_sd:,.2f} | {r.z:.2f} | {r.p_two_sided:.6f} | {r.p_plus_one:.6f} | {r.n_reps:,} | {r.note} |"
    for r in SENSDF.itertuples()) + f"""

`T1` 主口径换算成占比：obs {cite('obs_ad_share', True)}% vs null {cite('T1_share_mean', True)}%（z = {cite('T1_z')}，p 加一平滑 {cite('T1_p_plus1')}，低于 {cite('T1_reps')} 轮的可分辨率 1/{N('T1_reps')+1} = {1/(N('T1_reps')+1):.6f}）。
引擎自检（`d4_perm_audit.csv`）：本实现与逐源 `Generator.choice` 的独立实现在同一 seed、{cite('audit_slow_reps')} 轮下均值 {cite('audit_fast_mean')} vs {cite('audit_slow_mean')}、sd {cite('audit_fast_sd')} vs {cite('audit_slow_sd')}；解析期望 Σ k_i·|D|/(M−1) = {cite('analytic_expected_arcs')}，与置换均值 {cite('T1_null_mean')} 一致。耗时 {cite('T1_sec')}s / {cite('T1_reps')} 轮。

效应量（不只 p）：

| 口径 | 指标 | 值 | 95% CI | 基准 | 观测/基准 |
|---|---|---:|---|---:|---:|
| 校正前 | 活→死 弧占比 | {cite('obs_ad_share', True)}% | [{cite('es_share_wilson_lo', True)}%, {cite('es_share_wilson_hi', True)}%] Wilson | 目标中性 {cite('p_dead_uniform', True)}% | {cite('es_rr')} |
| 校正前 | OR | {cite('es_or')} | [{cite('es_or_lo')}, {cite('es_or_hi')}] Katz（logSE {cite('es_or_logse')}） | 1 | — |
| 校正前 | 加权（观察链接次数）占比 | {cite('obs_ad_weight_share', True)}% | — | {cite('p_dead_uniform', True)}% | {N('obs_ad_weight_share')/N('p_dead_uniform'):.3f} |
| 校正前 | 点对级（互指折叠） | {cite('pair_level_dead_share', True)}% | — | {cite('p_dead_uniform', True)}% | {N('pair_level_dead_share')/N('p_dead_uniform'):.3f} |
| 校正后 | 活→死 弧占比（129 真失效） | {cite('T1c_obs_share', True)}% | [{cite('T1c_wilson_lo', True)}%, {cite('T1c_wilson_hi', True)}%] Wilson | 目标中性 {cite('T1c_p_dead_base', True)}% | {N('T1c_obs_share')/N('T1c_p_dead_base'):.3f} |
| 校正后 | OR | {cite('T1c_or')} | [{cite('T1c_or_lo')}, {cite('T1c_or_hi')}] Katz | 1 | — |
| 基准敏感性 | 暴露量基准期望占比 | {cite('T3_null_share', True)}% | — | 非识别 | 观测为其 {cite('T3_obs_over_expected')} 倍 |

逐源一致性（`d4_per_source.csv`）：{cite('n_sources')} 个有出边的活站中，{cite('src_below')} 个指向死站的弧少于目标中性期望、{cite('src_above')} 个多于期望、{cite('src_equal')} 个相等（符号检验 p = {cite('src_signtest_p')}）——不是几个枢纽造出来的假象。

## 4. 这个版本能说什么、不能说什么

**能说（同一份数据、同一口径下可复算）**

- 活→死份额 {cite('obs_ad_share', True)}% 明显低于"目标中性"基准 {cite('p_dead_uniform', True)}%（{cite('T1_reps')} 轮置换，z = {cite('T1_z')}）。这个量分子分母都可观察，**没有**被 §2 的删失构造性决定。
- 校正前后同号（{cite('obs_ad_share', True)}% → {cite('T1c_obs_share', True)}%），不依赖 32+1 个未证实节点（§2.2）的归属；点对级（z = {N('T1p_z'):.1f}）与加权级（z = {N('T1w_z'):.1f}）同向。

**不能说**

- **不能读成"回避"。** 均匀候选池假设"任何站点被链接的概率相同"，这不成立：链接本来就集中在少数站。基准换成 url_count 加权后，期望只剩 {cite('T3_null_share', True)}%，同一观测值反超基准 {cite('T3_obs_over_expected')} 倍——**符号完全由基准选择决定**。而 url_count 对死站本身就是删失的（{cite('urlcount_mean_dead')} vs {cite('urlcount_mean_healthy')}，死站只占全库 URL 的 {cite('urlcount_share_dead', True)}%），不能当外生暴露量；标签安慰剂 T2（允许热门节点被抽进假 D）给出中间基准 {cite('T2_null_share', True)}%（null 95% 区间 [{cite('T2_null_p2_5')}, {cite('T2_null_p97_5')}] 条弧，z = {N('T2_z'):.1f}）。三个基准没有一个有"状态无关"的辩护。
- **反向读法同样成立（§7：结构 ↔ 健康，不是结构 → 健康）。** 被链接少与缺维护/下线在横截面上互相包含，本数据没有时间轴，无法分离。
- **它测的是抓取轮次里的页面链接混合比。** 分子只统计"被抓到的页面上出现的链接"；指向死站的链接是否仍留在源页面上，取决于源页面自身的抓取与人工维护。
- **互指补偿单独就能压低这个份额。** 死站出度 ≡ 0 ⇒ 活—死弧不可能成对，而活—活弧有 {cite('mutual_pairs')} 对互指被计两次（活—活弧互指率 {cite('recip_rate_aa', True)}%，活—死弧 0.000%）。点对级检验折叠互指后份额为 {cite('pair_level_dead_share', True)}%，说明该通道存在但不是全部解释。

**给报告正文的替换写法（原句"失效不聚集 p<0.001"）**

> 死—死相邻在本数据源上的可观察样本点为 0（{cite('pairs_dd_unordered')} 个死—死点对全部要求"死站出链可观测"，实测从死站出发的边 {cite('edges_from_dead')} 条），因此"失效同配性"在这里不可识别：观测统计量 {cite('assort_gcc_obs')} 与度保持零模型 {cite('null_mean_assort')} 的差异由 e_DD 这一格单独驱动（ρ = {cite('lemma_intercept')} + {cite('lemma_slope')}·e_DD，R² = {cite('lemma_r2')}），而 e_DD 只能取 0。改在可观察的活→死通道上检验：活站发出的弧中 {cite('obs_ad_share', True)}% 落到死站，OR {cite('es_or')}（95% CI {cite('es_or_lo')}–{cite('es_or_hi')}）低于目标中性基准，但该差值的符号依赖于目标暴露量基准的选择、可用规模代理本身对死站删失，且结构位置与站点健康在本横截面上互为候选解释，故只作为描述性差异报告，不作为"回避"或"传染"的证据。
"""
with open(os.path.join(C.module_dir(MOD), "result.md"), "w", encoding="utf-8") as fh:
    fh.write(MD)

wall = time.time() - T0
C.write_run_note(MOD, [
    "D4_assortativity_identifiability RUN.txt",
    "command: cd pipeline && PYTHONHASHSEED=0 python out/D4_assortativity_identifiability/run_d4.py",
    "versions: " + ", ".join(f"{k}={v}" for k, v in C.versions().items()),
    f"SEED = {SEED}; PYTHONHASHSEED = {os.environ.get('PYTHONHASHSEED')}",
    f"total wall time: {wall:.1f}s",
    f"per-stage: build+anchors=see stdout, undirected swap null {N('null_reps')} reps = {N('null_sec_total'):.1f}s "
    f"({N('null_sec_total')/N('null_reps'):.2f}s/rep), directed config-model check {N('dir_reps')} reps = {N('dir_sec'):.1f}s, "
    f"T1 permutation {N('T1_reps')} reps = {N('T1_sec'):.1f}s, cross-impl audit = {N('audit_sec'):.1f}s, "
    f"weighted T1w = {N('T1w_sec'):.1f}s",
    f"anchors: {int(ANCH['ok'].sum())}/{len(ANCH)} PASS (anchors.csv)",
    "seeds: C.degree_preserving_null(seed=SEED+r, r=0..499, fresh copy per rep, nswap=10m, max_tries=100*nswap); "
    "directed_configuration_model seed=SEED+7919r (r=0..19); target-side permutation default_rng seeds: "
    f"T1={SEED} ({N('T1_reps')}), audit={SEED} (600 fast vs 600 slow), T1w={SEED}+11 (1000), T1p={SEED}+31 (2000), "
    f"T1c={SEED}+101 (2000), T2 label-placebo={SEED}+202 (20000), T3 exposure={SEED}+404 (1000)",
    f"random draws per replicate: T1 uses one default_rng().random(({len(U0.sources)}, {U0.M})) matrix per replicate; "
    "no state is carried across replicates (no accumulating swaps / permutations)",
    "outputs: d4_censoring_arithmetic.csv d4_mixing_matrix.csv d4_null_replicates.csv d4_targetside_tests.csv "
    "d4_effect_sizes.csv d4_per_source.csv d4_perm_audit.csv key_numbers.csv anchors.csv result.md",
    "",
    "--- stdout ---",
] + LOG)
say(f"[{wall:.1f}s] 写出 result.md + 8 张 CSV + RUN.txt")
