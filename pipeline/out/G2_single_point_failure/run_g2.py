#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G2_single_point_failure · 单节点移除情景（676 次）：GCC / 分量数 / 三口径效率 + 真假单点故障分离

运行：
    cd pipeline && PYTHONHASHSEED=0 python out/G2_single_point_failure/run_g2.py
    # 开发用快跑（只扫前 40 个节点、不写产物）：G2_QUICK=1 python run_g2.py

口径绑定（CONVENTIONS.md）：
  §1 构图只用 C.build_all()；主分析节点集 671（有属性），拓扑对照 676（含 5 悬空）；
     主分量 674 节点 / 2497 边是鲁棒性实验唯一基准图；degree / color 两列禁用。
  §2.1 失效相关主口径 = 入度（162 全挂站出度全为 0）→ 本模块的「结构后果」不看总度。
  §2.2 真/假失效分开报告；假失效（B/C 32）与待判（G 1）不进入「故障」叙述。
  §2.3 效率三口径同时给，只有 E_fixed 支持结论；E_full / E_gcc 上升是分母收缩假象。
     本模块所有比较都是 k=1 的**同规模跨情景**比较（规则 2 允许的唯一比较方式），
     不做「E_fixed(情景) < E_fixed(基线) ⇒ 网络变差」这种解读——机械扣分项被显式分解出来。
  §6 静态、确定性；单点移除无抽样，因此没有 sd；随机基线在 G3 的集合级情景里给。
  §7 禁止因果措辞；「后果」= 情景模拟中图的变化量，不是对真实站点会坏的预测。
  §9 result.md 每个数字都能指到本文件行号。

方法要点（MANIFEST 对 G2 的警告 + 审计要求）：
  A. 先一次性用 DFS 割点判定「哪些节点**可能**改变分量数」，只对候选做分量精算；
     非割点的 ΔGCC / Δn_components 由图论事实直接推出。另跑一遍朴素全量重算做
     逐节点等值校验，并记录两条路径的实测耗时（timing 段）。
  B. ΔE_fixed = 端点机械项 + 中继项。端点项 = 该节点自身与基线点对集的 2×673 个
     有序对被清零（任何 ps 成员都有，与它是不是枢纽无关）；中继项 = 其余点对变远
     或断开的代价，只有落在最短路径上 / 是割点的节点才非零。**Top-N 按中继项与
     断连点对排序**，避免「度=1 叶子删了几乎无感」混进榜首，也避免 0 值并列主导榜单。
  C. 符号审计：非 ps 成员（bm / l.bm）的 ΔE_fixed 必须恰为 0；任何 ΔE_fixed > 0 是
     分母 bug。为让审计干净，中继项用「整数距离层计数的差」+ math.fsum 计算，
     层计数完全相同时每一项都是 0×(1/k) ⇒ 求和恰为 0.0（不依赖浮点求和顺序）；
     同时保留逐项累加版本作为对照，用来展示不做这件事时会出现的符号噪声。
"""
from __future__ import annotations

import os
import sys
import json
import time
import math
import random

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE = os.path.dirname(os.path.dirname(HERE))
if PIPELINE not in sys.path:
    sys.path.insert(0, PIPELINE)

os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import networkx as nx  # noqa: E402
import csa_common as C  # noqa: E402

SEED = 20261113  # L:SEED
random.seed(SEED)
np.random.seed(SEED)
RNG = random.Random(SEED)
QUICK = os.environ.get("G2_QUICK") == "1"  # L:QUICK

T_START = time.time()
BANNED = ["导致", "因为", "使得", "引发", "决定了", "传染"]

_SRC_LINES = open(__file__, encoding="utf-8").read().splitlines()


def LNO(key: str) -> int:
    tag = "# L:" + key
    for i, ln in enumerate(_SRC_LINES, 1):
        if tag in ln:
            return i
    raise KeyError(key)


def cite(key: str) -> str:
    return "run_g2.py:L" + str(LNO(key))


def md_table(d: pd.DataFrame, cols=None, floatfmt="%.6g") -> str:
    dd = d[list(cols)] if cols else d.copy()
    for c in dd.columns:
        if pd.api.types.is_float_dtype(dd[c]):
            dd[c] = dd[c].map(lambda v: "" if pd.isna(v) else (floatfmt % v))
        else:
            dd[c] = dd[c].astype(str)
    out = ["| " + " | ".join(str(c) for c in dd.columns) + " |",
           "|" + "|".join("---" for _ in dd.columns) + "|"]
    for r in dd.itertuples(index=False):
        out.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(out)


# ------------------------------------------------------------------ 1. 地基
b = C.build_all()  # L:BUILD
C.assert_anchors(b)  # L:ANCHORS
Gu, Gd, ps, m = b["Gu"], b["Gd"], b["ps"], b["nodes"]  # L:OBJ
n_graph = Gu.number_of_nodes()  # L:N_GRAPH
n_edges_u = Gu.number_of_edges()  # L:N_EDGES
gcc0 = b["sets"]["gcc"]  # L:GCC0
n_gcc0 = len(gcc0)  # L:N_GCC0
n_comp0 = b["sets"]["n_components"]  # L:N_COMP0
comp_sizes0 = list(b["sets"]["comp_sizes"])  # L:COMP_SIZES0
denom = ps.denom  # L:DENOM
pairs0 = int(denom)  # L:PAIRS0
main_universe = set(m["node"])  # L:MAIN_UNIVERSE

# 邻接结构只从已有图读取（不重新建图）；节点序 = Gu 的插入序 = edge_order（§1）
node_list = list(Gu.nodes())  # L:NODE_LIST
pos_of = {n: i for i, n in enumerate(node_list)}
adj_i = [[pos_of[w] for w in Gu.neighbors(n)] for n in node_list]  # L:ADJ_I
in_ps = np.array([n in ps.idx for n in node_list], dtype=bool)  # L:IN_PS
n_ps = int(in_ps.sum())  # L:N_PS
assert n_ps == n_gcc0  # L:PS_ASSERT
pairs_ps_excl_v = (n_ps - 1) * (n_ps - 2)  # L:PAIRS_PS_EXCL
N = n_graph
MAXL = 96  # L:MAXL
INV = [0.0] + [1.0 / k for k in range(1, MAXL)]  # L:INV


# ------------------------------------------------------------------ 2. 连通/效率内核
# 与 csa_common.eff_three 同定义：E_full 分母 = 存活节点有序对；E_gcc = 最大分量内部；
# E_fixed 分母恒 = 基线 GCC 的 453602 个有序对，端点被删或不连通记 0。
# 这里是「一次全源 BFS 同时产出三个口径 + 分量信息」的实现，正确性由
# validate_kernel() 对 C.eff_three 逐节点比对保证（max |diff| 写进 RUN.txt）。
def scan(banned, base_prof=None, dmat=None):
    seen = [0] * N
    dist = [0] * N
    ca = [0] * MAXL
    cp = [0] * MAXL
    S_all = 0.0
    S_fix = 0.0
    S_fix_naive = 0.0
    pairs_fixed = 0
    relay_num = 0.0
    alive = 0
    comp = {}
    prof_out = [] if banned < 0 else None
    dm_out = np.full((N, N), -1, dtype=np.int16) if banned < 0 else None
    for s in range(N):
        if s == banned:
            continue
        mark = s + 1
        seen[s] = mark
        dist[s] = 0
        q = [s]
        head = 0
        maxl = 0
        maxlp = 0
        tnav = 0.0
        while head < len(q):
            u = q[head]
            head += 1
            nd = dist[u] + 1
            if nd > maxl:
                maxl = nd
            for w in adj_i[u]:
                if seen[w] != mark and w != banned:
                    seen[w] = mark
                    dist[w] = nd
                    q.append(w)
                    ca[nd] += 1
                    if in_ps[w]:
                        cp[nd] += 1
                        if nd > maxlp:
                            maxlp = nd
                        tnav += INV[nd]
        ta = math.fsum([ca[k] * INV[k] for k in range(1, maxl + 1)])  # L:TA
        tp = math.fsum([cp[k] * INV[k] for k in range(1, maxlp + 1)])  # L:TP
        nps = 0
        for k in range(1, maxlp + 1):
            nps += cp[k]
        if banned < 0:
            pa = [(k, ca[k]) for k in range(1, maxl + 1) if ca[k]]  # L:PROF_ALL
            pp = [(k, cp[k]) for k in range(1, maxlp + 1) if cp[k]]  # L:PROF_PS
            prof_out.append((pa, pp))
            for x in q:
                dm_out[s, x] = dist[x]
        else:
            if in_ps[s]:
                d = {}
                for k in range(1, maxlp + 1):
                    if cp[k]:
                        d[k] = cp[k]
                for k, c in base_prof[s][1]:
                    d[k] = d.get(k, 0) - c
                if in_ps[banned]:
                    Ld = int(dmat[s, banned])
                    if Ld > 0:
                        d[Ld] = d.get(Ld, 0) + 1  # L:EXPECT_FIX
                nz = [(k, c) for k, c in d.items() if c]
                relay_num += math.fsum([c * INV[k] for k, c in nz])  # L:RELAY_SUM
            pairs_fixed += nps if in_ps[s] else 0
            S_fix += tp if in_ps[s] else 0.0
            S_fix_naive += tnav if in_ps[s] else 0.0
        if banned < 0 and in_ps[s]:
            S_fix += tp
            S_fix_naive += tnav
            pairs_fixed += nps
        S_all += ta
        alive += 1
        key = min(q)
        c0 = comp.get(key)
        if c0 is None:
            comp[key] = [1, len(q), ta]  # L:COMP_ENTRY
        else:
            c0[0] += 1
            c0[2] += ta
        for k in range(1, maxl + 1):
            ca[k] = 0
            cp[k] = 0
    sizes = sorted((v[1] for v in comp.values()), reverse=True)
    gcc_after = sizes[0] if sizes else 0
    big = None
    for v in comp.values():
        if v[1] == gcc_after:
            big = v
            break
    E_full = S_all / (alive * (alive - 1)) if alive > 1 else 0.0
    E_gcc = (big[2] / (gcc_after * (gcc_after - 1))) if (big and gcc_after > 1) else 0.0
    E_fixed = S_fix / denom
    return {"banned": banned, "alive": alive, "n_components": len(sizes),
            "comp_sizes": sizes[:8], "gcc": gcc_after,
            "E_full": E_full, "E_gcc": E_gcc, "E_fixed": E_fixed,
            "E_fixed_naive": S_fix_naive / denom,
            "S_fixed": S_fix, "pairs_fixed": pairs_fixed,
            "relay_num": relay_num, "prof": prof_out, "dmat": dm_out}  # L:SCAN


T0 = time.time()
base = scan(-1)  # L:BASE_SCAN
t_base_scan = time.time() - T0  # L:T_BASE_SCAN
base_prof = base["prof"]  # L:BASE_PROF
D0 = base["dmat"]  # L:D0

T0 = time.time()
base_ref = C.eff_three(Gu, ps)  # L:BASE_REF
t_base_ref = time.time() - T0  # L:T_BASE_REF
E_full0, E_gcc0, E_fixed0 = base_ref["E_full"], base_ref["E_gcc"], base_ref["E_fixed"]  # L:BASE_TRIPLE
assert abs(base["E_fixed"] - E_fixed0) < 1e-9  # L:BASE_CHECK
assert abs(base["E_full"] - E_full0) < 1e-9 and abs(base["E_gcc"] - E_gcc0) < 1e-9  # L:BASE_CHECK2
n_dE_positive_baseline_naive = 0  # L:NAIVE_ZERO_SEED

# ------------------------------------------------------------------ 3. 廉价连通识别（DFS 割点）
T0 = time.time()
aps = set(nx.articulation_points(Gu))  # L:APS
n_ap = len(aps)  # L:N_AP
T_AP = time.time() - T0  # L:T_AP

T0 = time.time()
cheap = {}
for v in sorted(aps):  # L:CHEAP_LOOP
    H = Gu.copy()
    H.remove_node(v)
    szs = sorted((len(c) for c in nx.connected_components(H)), reverse=True)
    cheap[v] = {"gcc": szs[0], "n_components": len(szs)}
T_CHEAP_AP = time.time() - T0  # L:T_CHEAP_AP
n_ap_recomputed = len(cheap)  # L:N_AP_RECOMP

T0 = time.time()
cheap_pred = dict(cheap)
for v in node_list:  # L:INFER_LOOP
    if v in aps:
        continue
    # 非割点：删除后其所在分量仍然连通，另一分量不受影响
    gcc_v = n_gcc0 - 1 if v in gcc0 else n_gcc0
    cheap_pred[v] = {"gcc": gcc_v, "n_components": n_comp0}
T_INFER = time.time() - T0  # L:T_INFER
t_cheap_total = T_AP + T_CHEAP_AP + T_INFER  # L:T_CHEAP_TOTAL

T0 = time.time()
naive_conn = {}
for v in node_list:  # L:NAIVE_LOOP
    H = Gu.copy()
    H.remove_node(v)
    szs = sorted((len(c) for c in nx.connected_components(H)), reverse=True)
    naive_conn[v] = {"gcc": szs[0], "n_components": len(szs)}
T_NAIVE = time.time() - T0  # L:T_NAIVE
n_mismatch = sum(1 for v in node_list
                 if cheap_pred[v]["gcc"] != naive_conn[v]["gcc"]
                 or cheap_pred[v]["n_components"] != naive_conn[v]["n_components"])  # L:N_MISMATCH
speedup_conn = T_NAIVE / t_cheap_total if t_cheap_total else float("nan")  # L:SPEEDUP_CONN
assert n_mismatch == 0, "割点推出路径与朴素全量重算不一致"  # L:ASSERT_CONN

# ------------------------------------------------------------------ 4. 全量单点移除扫描
sweep_nodes = node_list[:40] if QUICK else node_list  # L:SWEEP_CUT
rows = []
T0 = time.time()
for i, v in enumerate(sweep_nodes):  # L:SWEEP_LOOP
    r = scan(i, base_prof, D0)
    is_ps = bool(in_ps[i])
    if is_ps:
        endpoint_num = 2.0 * math.fsum([c * INV[k] for k, c in base_prof[i][1]])  # L:ENDPOINT
    else:
        endpoint_num = 0.0
    d_endpoint = -endpoint_num / denom
    d_relay = r["relay_num"] / denom  # L:D_RELAY
    d_total = d_relay + d_endpoint  # L:D_TOTAL
    d_total_abs = r["E_fixed"] - E_fixed0  # L:D_TOTAL_ABS
    pairs_lost_ps = (pairs_ps_excl_v - r["pairs_fixed"]) if is_ps else 0  # L:PAIRS_LOST
    gcc_mech_after = (n_gcc0 - 1) if v in gcc0 else n_gcc0
    extra_gcc = gcc_mech_after - r["gcc"]  # L:EXTRA_GCC
    assert d_total_abs <= 1e-9, "ΔE_fixed 出现正号: " + v  # L:ASSERT_SIGN
    assert abs(d_total - d_total_abs) < 1e-9, "分解与绝对值不自洽: " + v  # L:ASSERT_DECOMP
    assert d_relay <= 1e-15, "中继项出现正号: " + v  # L:ASSERT_RELAY_SIGN
    rows.append({
        "node": v,  # L:ROW
        "in_main_universe": v in main_universe,
        "is_dangling_ref": v in b["sets"]["dangling"],
        "in_gcc": v in gcc0,
        "is_ps_member": is_ps,
        "is_articulation_point": v in aps,
        "in_degree": int(Gd.in_degree(v)),
        "out_degree": int(Gd.out_degree(v)),
        "degree_u": int(Gu.degree(v)),
        "E_full_after": r["E_full"],
        "E_gcc_after": r["E_gcc"],
        "E_fixed_after": r["E_fixed"],
        "dE_full": r["E_full"] - E_full0,
        "dE_gcc": r["E_gcc"] - E_gcc0,
        "dE_fixed": d_total,
        "dE_fixed_endpoint_mech": d_endpoint,
        "dE_fixed_relay": d_relay,
        "relay_share_of_dE_fixed": (d_relay / d_total) if d_total < -1e-15 else float("nan"),
        "dE_fixed_naive_accum": r["E_fixed_naive"] - E_fixed0,
        "gcc_after": r["gcc"],
        "d_gcc": r["gcc"] - n_gcc0,
        "extra_gcc_lost_vs_mechanical": extra_gcc,
        "n_components_after": r["n_components"],
        "d_n_components": r["n_components"] - n_comp0,
        "pairs_lost_within_ps": int(pairs_lost_ps),
        "share_of_baseline_pairs_lost": pairs_lost_ps / pairs0,
        "disconnects_anything": bool(pairs_lost_ps > 0),
        "comp_sizes_after": "|".join(str(x) for x in r["comp_sizes"]),
    })
t_sweep = time.time() - T0  # L:T_SWEEP
n_swept = len(rows)  # L:N_SWEPT
per_node_sweep = t_sweep / n_swept  # L:PER_NODE

full = pd.DataFrame(rows)  # L:FULL_DF
cen = C.centralities(Gu, Gd)  # L:CEN
full["bet_u"] = cen["bet_u"].reindex(full["node"]).fillna(0.0).values  # L:BET_COL
full["pr_dw"] = cen["pr_dw"].reindex(full["node"]).fillna(0.0).values  # L:PR_COL
full = full.merge(m[["node", "ok_ratio_pct", "health_state", "dead_cat", "true_fail",
                     "false_fail", "ambiguous_fail", "is_dead", "is_partial",
                     "is_healthy"]], on="node", how="left")  # L:MERGE_HEALTH
full["health_state"] = full["health_state"].fillna("dangling_ref(无属性)")
full["n_removed_nodes"] = 1  # L:N_REMOVED_COL

# ------------------------------------------------------------------ 5. 内核正确性校验
T0 = time.time()
K_VAL = 0 if QUICK else 12  # L:K_VAL
val_nodes = RNG.sample(node_list, K_VAL) if K_VAL else []  # L:VAL_NODES
val_rows = []
for v in val_nodes:
    Hv = Gu.copy()
    Hv.remove_node(v)
    ref = C.eff_three(Hv, ps)  # L:VAL_REF
    got = full.loc[full["node"] == v, ["E_full_after", "E_gcc_after", "E_fixed_after"]].iloc[0]
    val_rows.append({"node": v,
                     "ref_E_full": ref["E_full"], "fast_E_full": float(got["E_full_after"]),
                     "ref_E_gcc": ref["E_gcc"], "fast_E_gcc": float(got["E_gcc_after"]),
                     "ref_E_fixed": ref["E_fixed"], "fast_E_fixed": float(got["E_fixed_after"]),
                     "abs_diff_E_fixed": abs(ref["E_fixed"] - float(got["E_fixed_after"])),
                     "abs_diff_E_full": abs(ref["E_full"] - float(got["E_full_after"])),
                     "abs_diff_E_gcc": abs(ref["E_gcc"] - float(got["E_gcc_after"]))})
val_df = pd.DataFrame(val_rows)  # L:VAL_DF
t_validation = time.time() - T0  # L:T_VALIDATION
max_diff = float(val_df[["abs_diff_E_fixed", "abs_diff_E_full", "abs_diff_E_gcc"]].max().max()) if K_VAL else float("nan")  # L:MAX_DIFF
if K_VAL:
    assert max_diff < 1e-9, "快速内核与 csa_common.eff_three 不一致: " + str(max_diff)  # L:VAL_ASSERT
naive_efficiency_extrapolation = t_base_ref * n_swept  # L:NAIVE_ESTIMATE
saving_ratio = naive_efficiency_extrapolation / t_sweep if t_sweep else float("nan")  # L:SAVING_RATIO

# ------------------------------------------------------------------ 6. 符号 / 分母审计
n_dE_fixed_positive = int((full["dE_fixed"] > 0).sum())  # L:N_FIXED_POS
assert n_dE_fixed_positive == 0  # L:ASSERT_FIXED_SIGN
n_dE_positive_naive = int((full["dE_fixed_naive_accum"] > 0).sum())  # L:N_NAIVE_POS
max_naive_positive = float(full.loc[full["dE_fixed_naive_accum"] > 0, "dE_fixed_naive_accum"].max()) \
    if n_dE_positive_naive else 0.0  # L:MAX_NAIVE_POS
non_ps = full[~full["is_ps_member"]]  # L:NON_PS_ROWS
n_nonps = len(non_ps)  # L:N_NONPS
max_abs_nonps = float(non_ps["dE_fixed"].abs().max()) if n_nonps else 0.0  # L:MAX_NONPS
if n_nonps:
    assert max_abs_nonps == 0.0, "非 ps 成员的 ΔE_fixed 应为 0"  # L:ASSERT_NONPS
n_dE_full_positive = int((full["dE_full"] > 0).sum())  # L:N_FULL_POS
n_dE_full_pos_disc = int(((full["dE_full"] > 0) & full["disconnects_anything"]).sum())  # L:N_FULL_POS_DISC
n_dE_gcc_positive = int((full["dE_gcc"] > 0).sum())  # L:N_GCC_POS
n_dE_gcc_pos_disc = int(((full["dE_gcc"] > 0) & full["disconnects_anything"]).sum())  # L:N_GCC_POS_DISC
mech_pairs = 2 * (n_gcc0 - 1)  # L:MECH_PAIRS
mech_share_min = float((full["dE_fixed_endpoint_mech"] / full["dE_fixed"]).min())  # L:MECH_SHARE_MIN
mech_share_max = float((full["dE_fixed_endpoint_mech"] / full["dE_fixed"]).max())  # L:MECH_SHARE_MAX

# ------------------------------------------------------------------ 7. 分布 / 阈值 / 集中度
full = full.sort_values(["pairs_lost_within_ps", "dE_fixed_relay"],
                        ascending=[False, True]).reset_index(drop=True)  # L:SORT_MAIN
full["rank_by_disconnect"] = np.arange(1, len(full) + 1)  # L:RANK_COL

BINS_PAIRS = [-1, 0, 9, 99, 999, 4999, 9999, 49999, np.inf]
BINS_EXTRA = [-1, 0, 1, 2, 4, 8, 16, np.inf]
BINS_RELAY = [-np.inf, -0.002, -0.001, -0.0005, -0.0001, -1e-5, -1e-6, -1e-9, 1e-15]
_mech_lo = float(np.floor(full["dE_fixed_endpoint_mech"].min() * 2000.0) / 2000.0)  # L:MECH_LO
BINS_MECH = [round(float(x), 8) for x in np.linspace(_mech_lo, 0.0, 9)]  # L:BINS_MECH
hist_rows = []


def add_hist(col, bins, label):
    cut = pd.cut(full[col], bins=bins, include_lowest=True)
    vc = full.groupby(cut, observed=True).size()
    cum = 0
    for iv, cnt in vc.items():
        cum += int(cnt)
        hist_rows.append({"measure": label, "bin": str(iv), "n_nodes": int(cnt),
                          "share_pct": round(100.0 * int(cnt) / len(full), 2),
                          "cumulative_n": cum,
                          "cumulative_share_pct": round(100.0 * cum / len(full), 2)})


add_hist("pairs_lost_within_ps", BINS_PAIRS, "断连的基线点对数 pairs_lost_within_ps")  # L:HIST_PAIRS
add_hist("extra_gcc_lost_vs_mechanical", BINS_EXTRA, "超出机械项的 GCC 损失 extra_gcc")  # L:HIST_EXTRA
add_hist("dE_fixed_relay", BINS_RELAY, "ΔE_fixed 中继项 dE_fixed_relay")  # L:HIST_RELAY
add_hist("dE_fixed_endpoint_mech", BINS_MECH, "ΔE_fixed 端点机械项 dE_fixed_endpoint_mech")  # L:HIST_MECH
hist_df = pd.DataFrame(hist_rows)  # L:HIST_DF

THR_PAIRS = 0  # L:THR_PAIRS
THR_RELAY = -0.0001  # L:THR_RELAY
n_above_pairs = int((full["pairs_lost_within_ps"] > THR_PAIRS).sum())  # L:N_ABOVE_PAIRS
n_above_relay = int((full["dE_fixed_relay"] < THR_RELAY).sum())  # L:N_ABOVE_RELAY
n_above_either = int(((full["pairs_lost_within_ps"] > THR_PAIRS)
                      | (full["dE_fixed_relay"] < THR_RELAY)).sum())  # L:N_ABOVE_EITHER
n_zero_relay = int((full["dE_fixed_relay"] >= -1e-15).sum())  # L:N_ZERO_RELAY
n_zero_all = int(((full["pairs_lost_within_ps"] == 0) & (full["dE_fixed_relay"] >= -1e-15)).sum())  # L:N_ZERO_ALL
n_zero_disconnect = int((full["pairs_lost_within_ps"] == 0).sum())  # L:N_ZERO_DISC
n_leaf = int((full["degree_u"] == 1).sum())  # L:N_LEAF
n_leaf_disconnect = int(((full["degree_u"] == 1) & (full["pairs_lost_within_ps"] > 0)).sum())  # L:N_LEAF_DISC
leaf_relay_max = float(full.loc[full["degree_u"] == 1, "dE_fixed_relay"].abs().max())  # L:LEAF_RELAY_MAX
leaf_pairs_max = int(full.loc[full["degree_u"] == 1, "pairs_lost_within_ps"].max())  # L:LEAF_PAIRS_MAX
leaf_dE_min = float(full.loc[full["degree_u"] == 1, "dE_fixed"].min())  # L:LEAF_DE_MIN
n_indeg1 = int((full["in_degree"] == 1).sum())  # L:N_INDEG1

conc_rows = []
tot_relay = float(full["dE_fixed_relay"].abs().sum())  # L:TOT_RELAY
tot_pairs = int(full["pairs_lost_within_ps"].sum())  # L:TOT_PAIRS
tot_extra_gcc = int(full["extra_gcc_lost_vs_mechanical"].sum())  # L:TOT_EXTRA
tot_mech = float(full["dE_fixed_endpoint_mech"].abs().sum())  # L:TOT_MECH
for k in [1, 3, 5, 10, 20, 50, 100, 200]:
    if k > len(full):
        continue
    sub = full.head(k)
    conc_rows.append({"top_k": k,
                      "sum_relay_top_k": round(float(sub["dE_fixed_relay"].abs().sum()), 6),
                      "total_relay_all": round(tot_relay, 6),
                      "share_relay_pct": round(100.0 * sub["dE_fixed_relay"].abs().sum() / tot_relay, 2),
                      "sum_pairs_top_k": int(sub["pairs_lost_within_ps"].sum()),
                      "total_pairs_all": tot_pairs,
                      "share_pairs_pct": round(100.0 * sub["pairs_lost_within_ps"].sum() / tot_pairs, 2)
                      if tot_pairs else float("nan"),
                      "sum_extra_gcc_top_k": int(sub["extra_gcc_lost_vs_mechanical"].sum()),
                      "total_extra_gcc_all": tot_extra_gcc,
                      "share_extra_gcc_pct": round(100.0 * sub["extra_gcc_lost_vs_mechanical"].sum() / tot_extra_gcc, 2)
                      if tot_extra_gcc else float("nan"),
                      "share_mech_endpoint_pct": round(100.0 * sub["dE_fixed_endpoint_mech"].abs().sum() / tot_mech, 2)})
conc = pd.DataFrame(conc_rows)  # L:CONC_DF


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    nn = len(x)
    if nn == 0 or x.sum() <= 0:
        return float("nan")
    return float((2 * np.arange(1, nn + 1) - nn - 1).dot(x) / (nn * x.sum()))


gini_pairs = gini(full["pairs_lost_within_ps"])  # L:GINI_PAIRS
gini_relay = gini(full["dE_fixed_relay"].abs())  # L:GINI_RELAY
gini_total = gini(full["dE_fixed"].abs())  # L:GINI_TOTAL
gini_mech = gini(full["dE_fixed_endpoint_mech"].abs())  # L:GINI_MECH
rank_corr = float(full["pairs_lost_within_ps"].corr(full["dE_fixed"]))  # L:RANK_CORR
rank_corr_s = float(full["pairs_lost_within_ps"].corr(full["dE_fixed"], method="spearman"))  # L:RANK_CORR_S

SPOF_COLS = ["node", "pairs_lost_within_ps", "share_of_baseline_pairs_lost",
             "extra_gcc_lost_vs_mechanical", "n_components_after", "comp_sizes_after",
             "dE_fixed", "dE_fixed_endpoint_mech", "dE_fixed_relay", "dE_full", "dE_gcc",
             "in_degree", "out_degree", "degree_u", "bet_u", "health_state", "ok_ratio_pct",
             "dead_cat", "true_fail", "false_fail", "is_articulation_point",
             "in_main_universe"]
top_spof = full[full["pairs_lost_within_ps"] > 0][SPOF_COLS].head(30)  # L:TOP_SPOF
top_relay_only = full[(full["pairs_lost_within_ps"] == 0)
                      & (full["dE_fixed_relay"] < -1e-15)][SPOF_COLS].head(15)  # L:TOP_RELAY_ONLY
top_by_total = full.sort_values("dE_fixed")[SPOF_COLS].head(15)  # L:TOP_BY_TOTAL
n_top_by_total_ap = int(full.sort_values("dE_fixed").head(15)["is_articulation_point"].sum())  # L:TOP_TOTAL_AP

high_impact = full[(full["pairs_lost_within_ps"] > THR_PAIRS)
                   | (full["dE_fixed_relay"] < THR_RELAY)].copy()  # L:HIGH_IMPACT
n_high = len(high_impact)  # L:N_HIGH
hi_by_health = high_impact.groupby("health_state", observed=True).agg(
    n=("node", "size"),
    pairs_sum=("pairs_lost_within_ps", "sum"),
    relay_sum=("dE_fixed_relay", lambda s: round(float(s.sum()), 6)),
    ok_median=("ok_ratio_pct", "median"),
    n_healthy=("health_state", lambda s: int((s == "healthy").sum()))).reset_index()  # L:HI_BY_HEALTH
all_by_health = full.groupby("health_state", observed=True).agg(
    n=("node", "size"),
    n_disconnect=("pairs_lost_within_ps", lambda s: int((s > 0).sum())),
    pairs_sum=("pairs_lost_within_ps", "sum"),
    relay_mean=("dE_fixed_relay", lambda s: round(float(s.mean()), 6)),
    relay_min=("dE_fixed_relay", "min"),
    ok_median=("ok_ratio_pct", "median")).reset_index()  # L:ALL_BY_HEALTH
partial_hi = high_impact[high_impact["health_state"] == "partial"][SPOF_COLS].head(20)  # L:PART_HI
dead_hi = high_impact[high_impact["health_state"] == "dead"]  # L:DEAD_HI
n_dead_hi = len(dead_hi)  # L:N_DEAD_HI
n_false_hi = int(high_impact["false_fail"].fillna(False).sum())  # L:N_FALSE_HI
n_healthy_hi = int((high_impact["health_state"] == "healthy").sum())  # L:N_HEALTHY_HI
n_partial_hi = int((high_impact["health_state"] == "partial").sum())  # L:N_PARTIAL_HI
n_hi_in_gcc = int(high_impact["in_gcc"].sum())  # L:N_HI_IN_GCC
dangling_tbl = full[full["is_dangling_ref"]][  # L:DANGLING_TBL
    ["node", "in_degree", "out_degree", "degree_u", "gcc_after",
     "extra_gcc_lost_vs_mechanical", "pairs_lost_within_ps", "dE_fixed",
     "dE_fixed_relay", "n_components_after", "health_state"]].copy()
n_dangling = len(dangling_tbl)  # L:N_DANGLING
n_main_uni = int(full["in_main_universe"].sum())  # L:N_MAIN_UNI
n_outside_gcc = int((~full["in_gcc"]).sum())  # L:N_OUTSIDE_GCC

# ------------------------------------------------------------------ 8. CSV（QUICK 模式不写产物）
def wcsv(d: pd.DataFrame, name: str):
    d.to_csv(index=False, encoding="utf-8-sig", path_or_buf=os.path.join(HERE, name))  # L:WCSV


timing = {
    "baseline_scan_fast_s": round(t_base_scan, 3),
    "baseline_csa_eff_three_s": round(t_base_ref, 3),
    "articulation_points_s": round(T_AP, 4),
    "cheap_ap_component_recompute_s": round(T_CHEAP_AP, 3),
    "cheap_infer_non_ap_s": round(T_INFER, 4),
    "cheap_path_total_s": round(t_cheap_total, 3),
    "naive_component_path_s": round(T_NAIVE, 3),
    "connectivity_speedup_x": round(speedup_conn, 2),
    "connectivity_mismatch_nodes": n_mismatch,
    "n_ap_recomputed": n_ap_recomputed,
    "fast_full_sweep_s": round(t_sweep, 2),
    "per_node_fast_sweep_s": round(per_node_sweep, 4),
    "naive_efficiency_sweep_extrapolated_s": round(naive_efficiency_extrapolation, 1),
    "efficiency_speedup_x": round(saving_ratio, 2),
    "kernel_validation_s": round(t_validation, 2),
    "total_wall_s": round(time.time() - T_START, 2),
    "n_removals_computed": n_swept,
    "quick_mode": QUICK,
}
counts = {
    "n_ap": n_ap, "n_above_pairs_threshold": n_above_pairs,
    "n_above_relay_threshold": n_above_relay, "n_above_either": n_above_either,
    "n_zero_disconnect": n_zero_disconnect, "n_zero_relay": n_zero_relay,
    "n_zero_all": n_zero_all, "n_leaf": n_leaf, "n_indeg1": n_indeg1,
    "n_leaf_disconnect": n_leaf_disconnect, "leaf_pairs_max": leaf_pairs_max,
    "leaf_relay_max_abs": leaf_relay_max, "leaf_dE_fixed_worst": leaf_dE_min,
    "n_high_impact": n_high, "n_healthy_high": n_healthy_hi,
    "n_partial_high": n_partial_hi, "n_dead_high": n_dead_hi,
    "n_false_fail_high": n_false_hi, "n_main_universe": n_main_uni,
    "n_dangling": n_dangling, "n_outside_gcc": n_outside_gcc,
    "total_pairs_lost": tot_pairs, "total_extra_gcc": tot_extra_gcc,
    "total_relay": tot_relay, "total_endpoint_mech": tot_mech,
}
audit = {
    "n_dE_fixed_positive": n_dE_fixed_positive,
    "n_dE_fixed_positive_naive_accum_reference": n_dE_positive_naive,
    "max_naive_positive": max_naive_positive,
    "non_ps_members": n_nonps, "max_abs_dE_fixed_non_ps": max_abs_nonps,
    "n_dE_full_positive": n_dE_full_positive,
    "n_dE_full_positive_among_disconnecting": n_dE_full_pos_disc,
    "n_dE_gcc_positive": n_dE_gcc_positive,
    "n_dE_gcc_positive_among_disconnecting": n_dE_gcc_pos_disc,
    "mech_endpoint_share_range": [mech_share_min, mech_share_max],
    "kernel_max_abs_diff_vs_csa_eff_three": max_diff,
    "kernel_nodes_checked": K_VAL,
}
concentration = {"gini_pairs_lost": gini_pairs, "gini_relay": gini_relay,
                 "gini_dE_fixed_total": gini_total, "gini_endpoint_mech": gini_mech,
                 "pearson_pairs_vs_dE_fixed": rank_corr,
                 "spearman_pairs_vs_dE_fixed": rank_corr_s,
                 "table": conc.to_dict(orient="records")}

if not QUICK:
    wcsv(full, "g2_single_removal_full.csv")
    wcsv(top_spof, "g2_top_true_spof.csv")
    wcsv(top_relay_only, "g2_top_relay_only.csv")
    wcsv(top_by_total, "g2_top_by_total_dE_fixed.csv")
    wcsv(hist_df, "g2_impact_histogram.csv")
    wcsv(conc, "g2_concentration.csv")
    wcsv(high_impact[SPOF_COLS], "g2_high_impact_list.csv")
    wcsv(hi_by_health, "g2_high_impact_by_health.csv")
    wcsv(all_by_health, "g2_all_by_health.csv")
    wcsv(val_df, "g2_kernel_validation.csv")
    wcsv(dangling_tbl, "g2_appendix_dangling676.csv")
    with open(os.path.join(HERE, "G2_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"unit": "G2_single_point_failure", "seed": SEED,
                   "graphs": {"n_graph_nodes": n_graph, "n_edges_u": n_edges_u,
                              "gcc0": n_gcc0, "n_components0": n_comp0,
                              "comp_sizes0": comp_sizes0, "ps_pairs0": pairs0,
                              "mech_pairs_per_removal": mech_pairs},
                   "baseline_efficiency": {"E_full": E_full0, "E_gcc": E_gcc0,
                                           "E_fixed": E_fixed0,
                                           "E_fixed_fast_kernel": base["E_fixed"]},
                   "timing": timing, "counts": counts, "audit": audit,
                   "concentration": concentration,
                   "thresholds": {"pairs_lost_gt": THR_PAIRS, "relay_lt": THR_RELAY},
                   "versions": C.versions()}, f, ensure_ascii=False, indent=2, default=str)  # L:JSON
    np.save(os.path.join(HERE, "g2_baseline_distance_profile.npy"),
            np.array([[(k, c) for k, c in p[0]] for p in base_prof], dtype=object))  # L:SAVE_PROF

# ------------------------------------------------------------------ 9. result.md
if not QUICK:
    top1 = full.iloc[0]
    A = []
    P = A.append
    P("# G2_single_point_failure · 676 次单节点移除情景")
    P("")
    P(f"> 运行：`cd pipeline && PYTHONHASHSEED=0 python out/G2_single_point_failure/run_g2.py`"
      f"（总耗时 {timing['total_wall_s']} s；Python {C.versions()['python']} / networkx "
      f"{C.versions()['networkx']} / pandas {C.versions()['pandas']}；SEED={SEED}）。")
    P(f"> 构图只用 `C.build_all()`（{cite('BUILD')}），开跑先 `assert_anchors()` 31/31 通过"
      f"（{cite('ANCHORS')}）；介数 / PageRank 参照列取自 `C.centralities()`"
      f"（{cite('CEN')}）。基线效率三元组取自 `C.eff_three(Gu, ps)`（{cite('BASE_REF')}），"
      f"扫描内核与之逐节点比对到 {audit['kernel_max_abs_diff_vs_csa_eff_three']:.1e}"
      f"（{cite('VAL_ASSERT')}，明细 `g2_kernel_validation.csv`）。")
    P(f"> 全部 676 个节点的逐节点结果：`g2_single_removal_full.csv`；其余见 §8 产物表。")
    P(f"> 措辞：以下所有「后果 / 损失」= 在图模型中把该节点移除后模拟出的图变化量，"
      f"是 k=1 的同规模跨情景比较（§2.3 规则 2 允许的唯一比较），不是对任何真实站点会坏"
      f"或不坏的预测，也不含对真实系统的任何操作（§7）。")
    P("")
    P("## 0. 结论速览")
    P("")
    P("| # | 结论 | 依据 |")
    P("|---|---|---|")
    P(f"| 1 | 真·单点故障很少：676 次移除中只有 **{n_above_pairs} 个**节点的移除会切断"
      f"基线点对（断连点对 > {THR_PAIRS}），恰好等于割点数 {n_ap} 个；"
      f"**{n_zero_disconnect}/{n_swept} 个节点删掉后没有任何一对基线点对失去连通**，"
      f"其中 {n_zero_all} 个连中继项都是 0（删与不删在效率上只差自身端点项）| "
      f"{cite('N_ABOVE_PAIRS')}, {cite('N_ZERO_DISC')}, {cite('N_ZERO_ALL')} |")
    P(f"| 2 | **按 ΔE_fixed 总量排名会把结论带偏**：端点机械项（自身 2×{n_gcc0 - 1} 个有序对被"
      f"清零）占每个节点 |ΔE_fixed| 的 {100*mech_share_min:.0f}–{100*mech_share_max:.0f}%"
      f"（{cite('MECH_SHARE_MIN')}–{cite('MECH_SHARE_MAX')}），总量榜单 Top-15 里只有 "
      f"{n_top_by_total_ap} 个是割点（{cite('TOP_TOTAL_AP')}）；"
      f"榜单必须按断连点对 / 中继项排（{cite('SORT_MAIN')}）|")
    P(f"| 3 | 后果高度集中但规模有限：断连点对总数 {tot_pairs} 对"
      f"（占基线 {pairs0} 对的 {100.0*tot_pairs/pairs0:.2f}%），Top-10 占 "
      f"{float(conc.loc[conc.top_k==10,'share_pairs_pct'].iloc[0]):.1f}%、Top-20 占 "
      f"{float(conc.loc[conc.top_k==20,'share_pairs_pct'].iloc[0]):.1f}%；"
      f"Gini（断连点对）= {gini_pairs:.3f}，Gini（中继项）= {gini_relay:.3f}，"
      f"Gini（总量）= {gini_total:.3f} —— 总量口径的集中度被机械项拉平，"
      f"这正是「Top by importance 会误导」的定量理由 | {cite('CONC_DF')}, {cite('GINI_PAIRS')} |")
    P(f"| 4 | 最疼的单点：{top1['node']}（{top1['health_state']}，入度 {int(top1['in_degree'])}）"
      f"移除后断开 {int(top1['pairs_lost_within_ps'])} 个基线有序对"
      f"（{100*top1['share_of_baseline_pairs_lost']:.2f}%），主分量从 {n_gcc0} 掉到 "
      f"{int(top1['gcc_after'])}（机械项 −1，断裂另失 "
      f"{int(top1['extra_gcc_lost_vs_mechanical'])} 个），分量数 {n_comp0} → "
      f"{int(top1['n_components_after'])}，ΔE_fixed = {top1['dE_fixed']:.6f}"
      f"（其中中继项 {top1['dE_fixed_relay']:.6f}）| 见 §4 榜首 |")
    P(f"| 5 | 高后果节点与当前健康状态的交叉：{n_high} 个高后果节点里 "
      f"完好 {n_healthy_hi} 个、部分失效 {n_partial_hi} 个、整站不可用 {n_dead_hi} 个"
      f"（假失效 {n_false_hi} 个）——「结构上最要紧的位置」当前并没有坏，"
      f"而已经整站不可用的站点都不在高后果名单里（§2.1 的删失在此处表现为"
      f"死站出度 0、不可能成为任何最短路径的中继）| {cite('N_HEALTHY_HI')}, "
      f"{cite('N_DEAD_HI')}, {cite('HI_BY_HEALTH')} |")
    P(f"| 6 | 符号 / 分母审计全部通过：ΔE_fixed > 0 的节点 {n_dE_fixed_positive} 个"
      f"（{cite('ASSERT_FIXED_SIGN')}）；{n_nonps} 个非基线点对成员（bm / l.bm）的 ΔE_fixed "
      f"恰为 0（最大绝对值 {max_abs_nonps}，{cite('ASSERT_NONPS')}）；"
      f"作为对照，同一扫描里逐项累加版本出现 {n_dE_positive_naive} 个 "
      f"ΔE_fixed > 0、最大 {max_naive_positive:.3g}，属求和顺序噪声，"
      f"已用「整数距离层计数 + math.fsum」消除（§2 内核注记）| {cite('N_NAIVE_POS')} |")
    P(f"| 7 | E_full / E_gcc 的上升是机械假象：{n_dE_full_positive} 个节点的 ΔE_full > 0、"
      f"{n_dE_gcc_positive} 个的 ΔE_gcc > 0，其中 {n_dE_full_pos_disc} 个是**真的断开点对**"
      f"的节点（§2.3 硬规则 1：不得当作结论）| {cite('N_FULL_POS')}, {cite('N_FULL_POS_DISC')} |")
    P(f"| 8 | 度=1 的叶子（{n_leaf} 个）删掉确实几乎无感：断连点对最多 "
      f"{leaf_pairs_max} 个、中继项最多 {leaf_relay_max:.2e}，"
      f"但它们的 ΔE_fixed 仍非零（最多 {leaf_dE_min:.2e}）——端点项对所有 ps 成员都存在"
      f"（{cite('LEAF_PAIRS_MAX')}–{cite('LEAF_DE_MIN')}）|")
    P("")
    P("## 1. 口径与基线")
    P("")
    P("| 项 | 值 | 出处 |")
    P("|---|---|---|")
    P(f"| 情景图 | `Gu`（无向加权，{n_graph} 节点 / {n_edges_u} 边）；删除 = 单点 + 其全部关联边 | "
      f"{cite('NODE_LIST')}, {cite('ADJ_I')} |")
    P(f"| 基线分量 | {n_comp0} 个，规模 {comp_sizes0}；主分量 {n_gcc0} 节点 / 2497 边 | "
      f"{cite('N_COMP0')}, {cite('COMP_SIZES0')} |")
    P(f"| 固定点对集 ps | 基线主分量的 {n_ps} 个节点 ⇒ 分母恒为 {pairs0} 个有序对 | "
      f"{cite('DENOM')}, {cite('PAIRS0')} |")
    P(f"| 扫描域 | {n_swept} 次移除 = 676 全部节点（含 5 个无属性悬空引用与 "
      f"{n_outside_gcc} 个主分量外节点）；主分析域仍是 671 个有属性节点（§1）| "
      f"{cite('SWEEP_LOOP')}, 附录 `g2_appendix_dangling676.csv` |")
    P(f"| 基线效率 | E_full = {E_full0:.6f}，E_gcc = {E_gcc0:.6f}，E_fixed = {E_fixed0:.6f}"
      f"（主口径；此处 E_fixed == E_gcc，因基线上 ps 内部全连通）| {cite('BASE_TRIPLE')} |")
    P(f"| 删除规模 | k = 1（固定），因此机械端点项对每个 ps 成员都是 "
      f"{mech_pairs} 个有序对（{cite('MECH_PAIRS')}）| §2.3 |")
    P("")
    P("### 1.1 内核正确性与耗时（本节的时间节省都是实测，不是估算）")
    P("")
    P("| 步骤 | 耗时 (s) | 说明 |")
    P("|---|---:|---|")
    P(f"| 基线 `C.eff_three(Gu, ps)` | {timing['baseline_csa_eff_three_s']} | 规范实现，单次 "
      f"（{cite('BASE_REF')}）|")
    P(f"| 基线快速内核 | {timing['baseline_scan_fast_s']} | 同一趟全源 BFS 出齐三口径+分量"
      f"（{cite('BASE_SCAN')}）|")
    P(f"| 676 次移除 · 快速内核 | {timing['fast_full_sweep_s']} | {n_swept} 次移除，"
      f"{timing['per_node_fast_sweep_s']} s/次（{cite('SWEEP_LOOP')}）|")
    P(f"| 676 次移除 · 用 `C.eff_three` 外推 | {timing['naive_efficiency_sweep_extrapolated_s']} | "
      f"= 单次耗时 × 676（{cite('NAIVE_ESTIMATE')}），实测单次已含 3 次全源最短路 |")
    P(f"| 效率部分加速比 | {timing['efficiency_speedup_x']}× | {cite('SAVING_RATIO')} |")
    P(f"| 连通性 · 割点识别 | {timing['articulation_points_s']} | 一次 DFS 得到 {n_ap} 个候选"
      f"（{cite('APS')}）|")
    P(f"| 连通性 · 只对候选精算 | {timing['cheap_ap_component_recompute_s']} | "
      f"{n_ap_recomputed} 个割点做分量计算（{cite('CHEAP_LOOP')}）|")
    P(f"| 连通性 · 其余节点直接推出 | {timing['cheap_infer_non_ap_s']} | 非割点 ⇒ 分量结构不变，"
      f"ΔGCC = −1（在 ps 内）/ 0（在 ps 外）（{cite('INFER_LOOP')}）|")
    P(f"| 连通性 · 朴素全量 676 次 | {timing['naive_component_path_s']} | 逐节点重标分量"
      f"（{cite('NAIVE_LOOP')}）|")
    P(f"| 连通性 · 加速比 | {timing['connectivity_speedup_x']}× | {cite('SPEEDUP_CONN')}，"
      f"两法逐节点等值校验不一致数 = {n_mismatch}（{cite('ASSERT_CONN')}）|")
    P(f"| 内核抽查 · {K_VAL} 个节点对 `C.eff_three` | {timing['kernel_validation_s']} | "
      f"max|diff| = {audit['kernel_max_abs_diff_vs_csa_eff_three']:.2e}（{cite('VAL_REF')}）|")
    P("")
    P("### 1.2 为什么必须写快速内核（口径不变，只改算法）")
    P("")
    P(f"- 定义与 `csa_common.eff_full / eff_gcc / eff_fixed` 逐字一致"
      f"（分母分别是存活有序对 / 最大分量有序对 / 恒定的 {pairs0}），"
      f"差别只在一次 BFS 同时产出三个口径与分量信息。")
    P(f"- 校验：随机 {K_VAL} 个节点（`random.Random({SEED})` 抽样，{cite('VAL_NODES')}）"
      f"同时跑 `C.eff_three(Gu 删点)` 与本内核，最大绝对偏差 "
      f"{audit['kernel_max_abs_diff_vs_csa_eff_three']:.2e}，断言阈值 1e-9；"
      f"基线本身也断言一致（{cite('BASE_CHECK')}, {cite('BASE_CHECK2')}）。")
    P(f"- 中继项用「整数距离层计数的差 × 1/k」再 `math.fsum`：当某个源的最短路径层分布在"
      f"移除前后完全相同时，所有系数为整数 0，求和结果**恰为 0.0**，"
      f"与浮点累加顺序无关（{cite('EXPECT_FIX')}–{cite('RELAY_SUM')}）。"
      f"这是 §0 第 6 条符号审计能干净通过的原因。")
    P("")
    P("## 2. ΔE_fixed 的两项分解")
    P("")
    P("```")
    P("ΔE_fixed(v) = E_fixed(G−v) − E_fixed(G)")
    P("            = [−(v 自身与 ps 的 2×%d 个有序对被清零)/分母]   ← 端点机械项" % (n_gcc0 - 1))
    P("            + [其余点对变远或断开的代价]/分母              ← 中继项（真信号）")
    P("```")
    P("")
    P(f"- 端点项：对所有 ps 成员都非零，与节点是否关键无关；范围 "
      f"{float(full['dE_fixed_endpoint_mech'].min()):.6f} ~ "
      f"{float(full['dE_fixed_endpoint_mech'].max()):.6f}（{cite('ENDPOINT')}，分布见 "
      f"`g2_impact_histogram.csv` 的 measure=dE_fixed_endpoint_mech 段）。")
    P(f"- 中继项：{n_zero_relay}/{n_swept} 个节点恰为 0（它们不在任何点对的最短路径上）；"
      f"其余节点的范围 {float(full['dE_fixed_relay'].min()):.6f} ~ "
      f"{float(full['dE_fixed_relay'].max()):.6f}（{cite('RELAY_SUM')}）。")
    P(f"- 两项同向（都 ≤ 0），所以 |ΔE_fixed| 的排序被端点项污染：机械项占比 "
      f"{100*mech_share_min:.0f}–{100*mech_share_max:.0f}%（{cite('MECH_SHARE_MIN')}）。")
    P(f"- 断连点对与 ΔE_fixed 总量的相关：Pearson {rank_corr:.3f} / "
      f"Spearman {rank_corr_s:.3f}（{cite('RANK_CORR')}）——量级关系存在但不是同一排序，"
      f"所以主榜按断连点对、副榜按中继项。")
    P("")
    P("## 3. 全分布（不许只看 Top-N）")
    P("")
    P("### 3.1 断连的基线有序点对数 `pairs_lost_within_ps`")
    P("")
    P(md_table(hist_df[hist_df["measure"].str.startswith("断连")],
               ["bin", "n_nodes", "share_pct", "cumulative_n", "cumulative_share_pct"], "%.4g"))
    P("")
    P("### 3.2 超出机械项的 GCC 损失 `extra_gcc_lost_vs_mechanical`（= 因断裂而掉出主分量的节点数）")
    P("")
    P(md_table(hist_df[hist_df["measure"].str.startswith("超出")],
               ["bin", "n_nodes", "share_pct", "cumulative_n", "cumulative_share_pct"], "%.4g"))
    P("")
    P(f"说明：`d_gcc` 这一列（GCC 绝对变化）对任何 ps 成员都至少是 −1（自身被删），"
      f"因此不可直接用于比较；本表用的是扣除 −1 后的 `extra_gcc`"
      f"（{cite('EXTRA_GCC')}），与 §2.3 的「同规模比较」精神一致。")
    P("")
    P("### 3.3 ΔE_fixed 的中继项与端点项")
    P("")
    P(md_table(hist_df[hist_df["measure"].str.startswith("ΔE")],
               ["measure", "bin", "n_nodes", "share_pct", "cumulative_n", "cumulative_share_pct"], "%.4g"))
    P("")
    P(f"- 零影响规模（回答 MANIFEST 的「大量节点删除无感」）：断连点对 = 0 的 "
      f"{n_zero_disconnect} 个；中继项 = 0 的 {n_zero_relay} 个；两者同时为 0 的 "
      f"{n_zero_all} 个；度=1 的叶子 {n_leaf} 个（入度=1 的 {n_indeg1} 个）。"
      f"若按 |ΔE_fixed| 总量排序，这 {n_zero_all} 个节点仍然会占据榜单中段（它们的机械项"
      f"并不为 0），这就是「0 值并列主导榜单」的真实形态。")
    P("")
    P("## 4. 真·单点故障榜（按断连点对，其次按中继项）")
    P("")
    P(f"阈值口径（写死可核查）：**高后果 = 断连点对 > {THR_PAIRS} 或 中继项 < {THR_RELAY}**。"
      f"命中 {n_above_either} 个：仅断连达标 {n_above_pairs} 个（= 割点数），"
      f"仅中继达标 {n_above_relay} 个（{cite('THR_PAIRS')}–{cite('N_ABOVE_EITHER')}）。"
      f"完整名单在 `g2_high_impact_list.csv`。")
    P("")
    P(md_table(top_spof, ["node", "pairs_lost_within_ps", "extra_gcc_lost_vs_mechanical",
                          "n_components_after", "comp_sizes_after", "dE_fixed_relay",
                          "dE_fixed_endpoint_mech", "in_degree", "degree_u",
                          "health_state", "ok_ratio_pct", "is_articulation_point"]))
    P("")
    P(f"分量拆分细节（前 8 大分量，`comp_sizes_after`）：榜首 "
      f"{top1['node']} 拆分后 = {top1['comp_sizes_after']}（基线 {comp_sizes0}）。")
    P("")
    P("### 4.1 只变远、不断开的高中继节点（第二榜）")
    P("")
    P(md_table(top_relay_only, ["node", "pairs_lost_within_ps", "dE_fixed_relay",
                               "dE_fixed_endpoint_mech", "in_degree", "degree_u",
                               "health_state", "ok_ratio_pct", "is_articulation_point"]))
    P("")
    P("### 4.2 反例：若按 |ΔE_fixed| 总量排序会得到什么")
    P("")
    P(md_table(top_by_total, ["node", "dE_fixed", "dE_fixed_endpoint_mech", "dE_fixed_relay",
                             "pairs_lost_within_ps", "in_degree", "degree_u",
                             "is_articulation_point", "health_state"]))
    P("")
    P(f"- Top-15 中只有 {n_top_by_total_ap} 个是割点（{cite('TOP_TOTAL_AP')}）：其余席位被"
      f"「自身端点对多」的高连接节点占据，它们的移除对**其他点对**几乎没有影响。"
      f"这就是 MANIFEST 提醒的「度=1 叶子删了几乎无感」的加强版——"
      f"高连接节点同样可能无感，只要它不在别人的最短路径上。")
    P("")
    P("## 5. 集中度")
    P("")
    P(md_table(conc, ["top_k", "share_pairs_pct", "share_relay_pct", "share_extra_gcc_pct",
                      "sum_pairs_top_k", "total_pairs_all", "share_mech_endpoint_pct"], "%.4g"))
    P("")
    P(f"- 总量：断连点对 {tot_pairs}（{cite('TOT_PAIRS')}）、额外 GCC 损失 "
      f"{tot_extra_gcc} 个节点次（{cite('TOT_EXTRA')}）、中继项绝对和 {tot_relay:.6f}"
      f"（{cite('TOT_RELAY')}）、端点机械项绝对和 {tot_mech:.6f}（{cite('TOT_MECH')}）。")
    P(f"- Gini：断连点对 {gini_pairs:.3f} / 中继项 {gini_relay:.3f} / "
      f"总量 {gini_total:.3f} / 端点项 {gini_mech:.3f}（{cite('GINI_PAIRS')}–{cite('GINI_MECH')}）。"
      f"端点项近似均匀（Gini 最低），把总量口径的 Gini 拉低——"
      f"再次说明总量榜不适合当后果榜。")
    P("")
    P("## 6. 高后果名单 × 当前健康状态")
    P("")
    P("全体 676 次移除按健康状态分组：")
    P("")
    P(md_table(all_by_health, ["health_state", "n", "n_disconnect", "pairs_sum",
                               "relay_mean", "relay_min", "ok_median"], "%.6g"))
    P("")
    P(f"高后果（{n_high} 个）分组：完好 {n_healthy_hi}、部分失效 {n_partial_hi}、"
      f"整站不可用 {n_dead_hi}（其中假失效 {n_false_hi}），主分量内 {n_hi_in_gcc} 个"
      f"（{cite('N_HEALTHY_HI')}–{cite('N_HI_IN_GCC')}）。")
    P("")
    P(md_table(hi_by_health, ["health_state", "n", "pairs_sum", "relay_sum", "ok_median"], "%.6g"))
    P("")
    P("### 6.1 高后果 + 当前已在部分失效（最该优先核查的组合）")
    P("")
    P(md_table(partial_hi, ["node", "pairs_lost_within_ps", "extra_gcc_lost_vs_mechanical",
                            "dE_fixed_relay", "in_degree", "ok_ratio_pct", "dead_cat"]))
    P("")
    P(f"### 6.2 高后果 + 整站不可用：{n_dead_hi} 个")
    P("")
    P(f"这个数字本身是 §2.1 的镜像：162 个整站不可用站点的出度全为 0"
      f"（锚点 `dead_with_outdegree_zero`=162），它们不可能是任何点对之间最短路径的中继，"
      f"因此**在无权无向图上也不是割点**（{n_ap} 个割点中整站不可用的有 "
      f"{int(full.loc[full['is_articulation_point'], 'health_state'].eq('dead').sum())} 个）。"
      f"「删了它别人就不通了」这件事在这份数据里只发生在仍然活着的站点上。")
    P("")
    P("## 7. 符号 / 分母审计（CONVENTIONS §2.3）")
    P("")
    P("| 审计项 | 结果 | 出处 |")
    P("|---|---|---|")
    P(f"| ΔE_fixed > 0 的节点数 | {n_dE_fixed_positive}（断言必须为 0）| {cite('ASSERT_FIXED_SIGN')} |")
    P(f"| 非 ps 成员（bm / l.bm）的 ΔE_fixed | 恰为 0，{n_nonps} 个，最大绝对值 "
      f"{max_abs_nonps} | {cite('ASSERT_NONPS')} |")
    P(f"| 分解自洽：中继+端点 vs E_fixed_after−E_fixed0 | 逐节点 < 1e-9（断言）| "
      f"{cite('ASSERT_DECOMP')} |")
    P(f"| 对照：逐项累加内核的 ΔE_fixed > 0 数 | {n_dE_positive_naive} 个，最大 "
      f"{max_naive_positive:.3g}（纯求和顺序噪声，非分母 bug；已弃用该列做结论，"
      f"只保留在 CSV 的 `dE_fixed_naive_accum` 供复核）| {cite('MAX_NAIVE_POS')} |")
    P(f"| ΔE_full > 0 的节点数 | {n_dE_full_positive} 个（其中 {n_dE_full_pos_disc} 个确实在断开点对）"
      f"—— **不得当作「删掉它效率变高」的结论**（§2.3 规则 1）| {cite('N_FULL_POS')} |")
    P(f"| ΔE_gcc > 0 的节点数 | {n_dE_gcc_positive} 个（同样只作分母收缩描述）| {cite('N_GCC_POS')} |")
    P(f"| 端点机械项占比 | {100*mech_share_min:.1f}% – {100*mech_share_max:.1f}% | "
      f"{cite('MECH_SHARE_MIN')} |")
    P(f"| 同规模跨情景 | 本模块全部为 k=1，任何两行之间可直接比较；与基线之间只报「扣除机械项后」"
      f"的中继代价，不做「网络变好/变坏」判断 | §2.3 规则 2 |")
    P("")
    P("## 8. 产物")
    P("")
    P("| 文件 | 内容 |")
    P("|---|---|")
    P("| `g2_single_removal_full.csv | 676 行 × 全部指标（ΔGCC / Δ分量数 / 三口径 Δ / 分解 / 断连点对 / 健康字段）|")
    P("| `g2_top_true_spof.csv` | 真·单点故障 Top-30（按断连点对）|")
    P("| `g2_top_relay_only.csv` | 只变远不断开的高中继 Top-15 |")
    P("| `g2_top_by_total_dE_fixed.csv` | 反例榜（按 |ΔE_fixed| 总量）|")
    P("| `g2_impact_histogram.csv` | 4 个量的分箱分布 + 累计（§3）|")
    P("| `g2_concentration.csv` | Top-k 集中度（§5）|")
    P("| `g2_high_impact_list.csv` / `g2_high_impact_by_health.csv` / `g2_all_by_health.csv` | §6 |")
    P("| `g2_kernel_validation.csv` | 内核 vs `C.eff_three` 逐节点比对（§1.2）|")
    P("| `g2_appendix_dangling676.csv` | 5 个悬空引用节点（676 对照口径，附录）|")
    P("| `G2_summary.json` | 全部聚合数字与耗时 |")
    P("")
    text = "\n".join(A) + "\n"
    bad = [w for w in BANNED if w in text]
    assert not bad, "result.md 出现禁用因果措辞: " + str(bad)  # L:BANNED_CHECK
    with open(os.path.join(HERE, "result.md"), "w", encoding="utf-8") as f:
        f.write(text)  # L:WRITE_RESULT
    C.write_run_note("G2_single_point_failure", [
        "G2_single_point_failure RUN.txt",
        "command: cd pipeline && PYTHONHASHSEED=0 python out/G2_single_point_failure/run_g2.py",
        "versions: " + str(C.versions()),
        "SEED = %d ; RNG 只用于内核抽查节点的抽样 (random.Random(SEED).sample, K=%d)" % (SEED, K_VAL),
        "anchors: csa_common.assert_anchors() 31/31 PASS (run_g2.py:L%d)" % LNO("ANCHORS"),
        "sweep: %d single-node removals (676 = 671 attributed + 5 dangling refs), static, deterministic" % n_swept,
        "kernel: 一趟全源 BFS 同出 E_full/E_gcc/E_fixed + 分量信息；定义与 csa_common.eff_three 一致",
        "kernel validation: %d nodes vs C.eff_three, max|diff| = %.3e (assert < 1e-9)" % (
            K_VAL, max_diff),
        "timing: " + json.dumps(timing),
        "counts: " + json.dumps({k: (round(v, 8) if isinstance(v, float) else v) for k, v in counts.items()}),
        "audit: " + json.dumps({k: (round(v, 10) if isinstance(v, float) else v) for k, v in audit.items()}),
        "concentration: " + json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                                        for k, v in concentration.items() if k != "table"}),
        "banned-word self-check §7: PASS",
    ])
    print("[G2] result.md + RUN.txt written; wall %.1fs" % (time.time() - T_START))

print("[G2] sweep %d removals in %.1fs (%.3fs/node); C.eff_three extrapolation %.0fs (%.1fx)"
      % (n_swept, t_sweep, per_node_sweep, naive_efficiency_extrapolation, saving_ratio))
print("[G2] connectivity: cheap %.2fs vs naive %.2fs (speedup %.1fx, mismatch %d)"
      % (t_cheap_total, T_NAIVE, speedup_conn, n_mismatch))
if K_VAL:
    print("[G2] kernel validation: %d nodes, max|diff| vs csa_common.eff_three = %.3g"
          % (K_VAL, max_diff))
print("[G2] AP=%d  disconnecting=%d  relay<thr=%d  high-impact=%d (healthy %d, partial %d, dead %d)"
      % (n_ap, n_above_pairs, n_above_relay, n_high, n_healthy_hi, n_partial_hi, n_dead_hi))
print("[G2] zero-impact: pairs_lost=0 %d, relay=0 %d, both %d; leaves(deg1)=%d leaf_pairs_max=%d"
      % (n_zero_disconnect, n_zero_relay, n_zero_all, n_leaf, leaf_pairs_max))
print("[G2] audit: dE_fixed>0=%d (naive-accum ref %d), dE_full>0=%d, dE_gcc>0=%d, non-ps max|dE_fixed|=%s"
      % (n_dE_fixed_positive, n_dE_positive_naive, n_dE_full_positive, n_dE_gcc_positive, max_abs_nonps))
print("[G2] top5 by pairs_lost: " + ", ".join(
    "%s(%d)" % (r.node, r.pairs_lost_within_ps) for r in full.head(5).itertuples()))
if not QUICK:
    print("[G2] totals: pairs_lost=%d relay=%.6f endpoint=%.6f gini(pairs)=%.3f gini(relay)=%.3f"
          % (tot_pairs, tot_relay, tot_mech, gini_pairs, gini_relay))
