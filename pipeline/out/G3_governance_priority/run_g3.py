#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G3_governance_priority · 四类治理建议（保护 / 监控 / 修复 / 下线）+ 结构收益情景模拟

运行：
    cd pipeline && PYTHONHASHSEED=0 python out/G3_governance_priority/run_g3.py
    # 开发快跑（随机基线只跑 5 次、不写产物）：G3_QUICK=1 python run_g3.py

本单元的定位（硬约束）：
    只产出**建议清单**。所有「移除 / 下线 / 修复」都是**图上的情景模拟**，
    是对结构位置的陈述，不代表、也不授权对任何真实站点做任何操作。

口径绑定（CONVENTIONS.md）：
  §1  构图只用 csa_common.build_all()；开跑先 assert_anchors()
  §2.1 结构重要性主口径 = 入度；无向总度只作污染对照，不进判据
  §2.2 真失效(A/D/F) / 假失效(B/C) / 待判(G)：假失效与待判**不进任何治理类**，单列
  §2.3 只有 E_fixed 支持结论；删除节点机械抬高 E_full → 一律配「同 k 随机基线」对照
  §6  静态移除为主口径；随机基线 ≥100 次重复并报 sd（本模块 N_RAND=200）
  §7  禁止因果措辞；措辞只用「与…相关 / 情景模拟显示 / 移除…后模拟的…变化」
  §9  result.md 每个数字都能指到本文件真实行号（run_g3.py:Lxx 由 LNO() 解析）

判据只读**已测字段**：G2 的单点移除全量表 + csa_common 的健康/中心性字段。
不做逐节点人工判断；每条规则都是布尔式，优先级先到先得。
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
RNG = np.random.default_rng(SEED)  # L:RNG

T_START = time.time()
QUICK = os.environ.get("G3_QUICK") == "1"  # L:QUICK
N_RAND = 5 if QUICK else 200  # L:N_RAND  （§6：随机策略 ≥100 次并报 sd）
BANNED_WORDS = ["导致", "因为", "使得", "引发", "决定了", "传染"]  # L:BANNED_LIST

_SRC_LINES = open(__file__, encoding="utf-8").read().splitlines()  # L:SRC


def LNO(key: str) -> int:
    """result.md 里的引用 = 本文件 `# L:key` 注释所在行的真实行号。"""
    tag = "# L:" + key
    for i, ln in enumerate(_SRC_LINES, 1):
        if tag in ln:
            return i
    raise KeyError(key)


def cite(key: str) -> str:
    return "run_g3.py:L" + str(LNO(key))


def md_table(d: pd.DataFrame, cols=None, floatfmt="%.4f") -> str:
    """无 tabulate 依赖的 markdown 表。"""
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
nodes, Gu, Gd, ps = b["nodes"], b["Gu"], b["Gd"], b["ps"]  # L:OBJ
m = nodes.copy()  # L:MFROM
m = m.drop(columns=[c for c in ("degree", "color") if c in m.columns])  # L:DROP_BANNED
assert "degree" not in m.columns and "color" not in m.columns  # L:DROP_CHECK
cen = C.centralities(Gu, Gd)  # L:CEN
m = m.merge(cen, left_on="node", right_index=True, how="left")  # L:MERGE_CEN
n_attr = len(m)  # L:N_ATTR
gcc0 = set(b["sets"]["gcc"])  # L:GCC0
n_gcc0 = len(gcc0)  # L:N_GCC0
denom = ps.denom  # L:DENOM
n_dead = int(m["is_dead"].sum())  # L:N_DEAD
n_true = int(m["true_fail"].sum())  # L:N_TRUE
n_false = int(m["false_fail"].sum())  # L:N_FALSE
n_amb = int(m["ambiguous_fail"].sum())  # L:N_AMB
n_partial = int(m["is_partial"].sum())  # L:N_PARTIAL
n_healthy = int(m["is_healthy"].sum())  # L:N_HEALTHY

# ------------------------------------------------------------------ 2. 读 G2 单点移除全量表
G2DIR = os.path.join(PIPELINE, "out", "G2_single_point_failure")
G2CSV = os.path.join(G2DIR, "g2_single_removal_full.csv")  # L:G2CSV
G2JSON = os.path.join(G2DIR, "G2_summary.json")  # L:G2JSON
assert os.path.exists(G2CSV), "缺少 G2 产物，先跑 ../G2_single_point_failure/run_g2.py"  # L:G2_ASSERT
g2 = pd.read_csv(G2CSV, encoding="utf-8-sig")  # L:G2_READ
g2sum = json.load(open(G2JSON, encoding="utf-8")) if os.path.exists(G2JSON) else {}  # L:G2_SUM
THR_RELAY = float(g2sum.get("thresholds", {}).get("relay_lt", -0.0001))  # L:THR_RELAY
THR_PAIRS = float(g2sum.get("thresholds", {}).get("pairs_lost_gt", 0))  # L:THR_PAIRS
RELAY_EPS = -1e-15  # L:RELAY_EPS
assert len(g2) == Gu.number_of_nodes(), "G2 表行数不等于图节点数"  # L:G2_ROWS
assert set(g2["node"]) == set(Gu.nodes())  # L:G2_NODESET

# ------------------------------------------------------------------ 3. 671 节点判定域 + 布尔特征
G2_USE = ["node", "in_main_universe", "is_dangling_ref", "in_gcc", "is_ps_member",
          "is_articulation_point", "dE_full", "dE_gcc", "dE_fixed",
          "dE_fixed_endpoint_mech", "dE_fixed_relay", "extra_gcc_lost_vs_mechanical",
          "gcc_after", "d_gcc", "n_components_after", "d_n_components",
          "pairs_lost_within_ps", "share_of_baseline_pairs_lost",
          "disconnects_anything"]  # L:G2_USE
df = m.merge(g2[G2_USE], on="node", how="left", validate="one_to_one")  # L:MERGE_G2
assert len(df) == n_attr and df["pairs_lost_within_ps"].notna().all()  # L:MERGE_CHECK
df["in_pct"] = df["in_degree"].rank(method="average", pct=True)  # L:IN_PCT
df["bet_pct"] = df["bet_u"].rank(method="average", pct=True)  # L:BET_PCT
df["pr_pct"] = df["pr_dw"].rank(method="average", pct=True)  # L:PR_PCT
PCT_CUT = 0.95  # L:PCT_CUT  结构位置「高侧」= 三个主口径百分位的并集 top5%
T_ID = 2  # L:T_ID  「有超过一条入边指向它」的修复门槛
T_ID_SENS = [0, 1, 2, 3, 5]  # L:T_ID_SENS

# 逐节点布尔特征（全部是**已测字段**，不是人工判断）
df["f_false_fail"] = df["false_fail"].fillna(False).astype(bool)  # L:F_FF
df["f_ambiguous"] = df["ambiguous_fail"].fillna(False).astype(bool)  # L:F_AMB
df["f_true_fail"] = df["true_fail"].fillna(False).astype(bool)  # L:F_TF
df["f_partial"] = df["is_partial"].fillna(False).astype(bool)  # L:F_PART
df["f_healthy"] = df["is_healthy"].fillna(False).astype(bool)  # L:F_HEALTHY
df["f_ap"] = df["is_articulation_point"].fillna(False).astype(bool)  # L:F_AP
df["f_disc"] = df["disconnects_anything"].fillna(False).astype(bool)  # L:F_DISC
df["f_relay_neg"] = df["dE_fixed_relay"] < THR_RELAY  # L:F_RELAY
df["f_extra_gcc"] = df["extra_gcc_lost_vs_mechanical"] > 0  # L:F_EXTRA
df["f_zero_role"] = (~df["f_ap"]) & (~df["f_disc"]) & (~df["f_relay_neg"]) \
    & (~df["f_extra_gcc"]) & (df["dE_fixed_relay"] >= RELAY_EPS)  # L:F_ZERO
df["f_hi_impact"] = (df["pairs_lost_within_ps"] > THR_PAIRS) | df["f_relay_neg"]  # L:F_HI
df["f_top_cen"] = (df["bet_pct"] >= PCT_CUT) | (df["pr_pct"] >= PCT_CUT) \
    | (df["in_pct"] >= PCT_CUT)  # L:F_TOPCEN
df["f_role_in"] = df["in_degree"] >= T_ID  # L:F_ROLEIN

n_zero_role = int(df["f_zero_role"].sum())  # L:N_ZERO_ROLE
n_hi_impact = int(df["f_hi_impact"].sum())  # L:N_HI
n_top_cen = int(df["f_top_cen"].sum())  # L:N_TOPCEN

# ------------------------------------------------------------------ 4. 四类判据（布尔式 + 优先级先到先得）
RULES = [  # L:RULES
    ("排除_假失效BC", "false_fail（§2.2 B/C：抓取判据不可信，不参与治理排序）",
     df["f_false_fail"]),
    ("排除_待判G", "ambiguous_fail（§2.2 G：证据不足，单列）",
     df["f_ambiguous"]),
    ("下线", "true_fail 且 零实测结构角色：非割点 且 不断连任何点对 且 ΔE_fixed 中继项=0 "
             "且 无超出机械项的 GCC 损失",
     df["f_true_fail"] & df["f_zero_role"]),
    ("修复", "（真失效且不满足零结构角色）或（部分失效 且（高影响 或 入度>=%d））" % T_ID,
     (df["f_true_fail"] & ~df["f_zero_role"])
     | (df["f_partial"] & (df["f_hi_impact"] | df["f_role_in"]))),
    ("保护", "is_healthy 且 高影响（移除它会断连点对 / 中继项 < %.4g）" % THR_RELAY,
     df["f_healthy"] & df["f_hi_impact"]),
    ("监控", "is_healthy 且 未进保护 且 三个主口径百分位并集 top%d%%" % int((1 - PCT_CUT) * 100),
     df["f_healthy"] & ~df["f_hi_impact"] & df["f_top_cen"]),
]

df["class"] = "未分类"  # L:CLASS_INIT
hit_rule = pd.Series("", index=df.index)  # L:HIT_INIT
for name, desc, mask in RULES:  # L:ASSIGN_LOOP
    take = mask & (df["class"] == "未分类")
    df.loc[take, "class"] = name
    hit_rule.loc[take] = desc
hit_rule.loc[df["class"] == "未分类"] = "以上规则都未命中"  # L:HIT_UNC
df["rule_hit"] = hit_rule  # L:RULE_HIT
CLASSES = ["下线", "修复", "保护", "监控", "排除_假失效BC", "排除_待判G", "未分类"]  # L:CLASSES
CLASS_ORDER_PRIO = ["下线", "修复", "保护", "监控"]  # L:PRIO
df["tier"] = np.where(df["class"] == "修复",
                      np.where(df["f_true_fail"], "R1_真失效_有结构角色", "R2_部分失效_有入边"),
                      np.where(df["class"] == "保护", "P_高影响健康", ""))  # L:TIER

cls_counts = {c: int((df["class"] == c).sum()) for c in CLASSES}  # L:CLS_COUNTS
n_unclassified = cls_counts["未分类"]  # L:N_UNC
pct_unclassified = 100.0 * n_unclassified / n_attr  # L:PCT_UNC
n_assigned = n_attr - n_unclassified  # L:N_ASSIGNED
n_excluded = cls_counts["排除_假失效BC"] + cls_counts["排除_待判G"]  # L:N_EXCL

# 规则重叠（不设优先级时的命中数）——公开判据之间的冲突规模
overlap_rows = []  # L:OVERLAP_INIT
for name, desc, mask in RULES:
    overlap_rows.append({
        "class": name, "n_rule_only": int(mask.sum()),
        "n_after_priority": cls_counts[name],
        "n_absorbed_by_higher": int(mask.sum()) - cls_counts[name],
        "rule": desc})
overlap_df = pd.DataFrame(overlap_rows)  # L:OVERLAP_DF

# ------------------------------------------------------------------ 5. 情景内核（批量移除；定义同 C.eff_three）
node_list = list(Gu.nodes())  # L:NODE_LIST
pos_of = {n: i for i, n in enumerate(node_list)}  # L:POS_OF
adj_i = [[pos_of[w] for w in Gu.neighbors(n)] for n in node_list]  # L:ADJ_I
N = len(node_list)  # L:N
in_ps = np.array([n in ps.idx for n in node_list], dtype=bool)  # L:IN_PS
n_ps = int(in_ps.sum())  # L:N_PS
ps_idx = np.flatnonzero(in_ps)  # L:PS_IDX
assert n_ps == n_gcc0  # L:PS_ASSERT


def build_D0():
    """基线全源无权距离矩阵（int16，-1=不连通）。只在 Gu 上 BFS，不重建图。"""
    D0 = np.full((N, N), -1, dtype=np.int16)
    seen = [0] * N
    dist = [0] * N
    for s in range(N):
        mark = s + 1
        seen[s] = mark
        dist[s] = 0
        q = [s]
        head = 0
        while head < len(q):
            u = q[head]
            head += 1
            nd = dist[u] + 1
            for w in adj_i[u]:
                if seen[w] != mark:
                    seen[w] = mark
                    dist[w] = nd
                    q.append(w)
        D0[s, np.asarray(q, dtype=np.int64)] = np.asarray(
            [dist[x] for x in q], dtype=np.int16)
    return D0


T0 = time.time()
D0 = build_D0()  # L:BUILD_D0
t_d0 = time.time() - T0  # L:T_D0
REACH0 = D0 >= 0  # L:REACH0


def e_fixed_pred(banned):
    """端点机械项预测值：若幸存点之间的距离一条都没变、也没有新断连，E_fixed 应为多少。"""
    sub = [i for i in ps_idx if i not in banned]
    if len(sub) < 2:
        return 0.0, 0
    A = D0[np.ix_(sub, sub)]
    R = np.where(A > 0, 1.0 / np.maximum(A, 1), 0.0)
    np.fill_diagonal(R, 0.0)
    return float(R.sum()) / denom, len(sub)  # L:E_PRED


def scenario(banned):
    """一次性批量移除 banned（节点下标集合）后的三个效率口径 + 连通性 + 与基线的逐对距离比对。"""
    banned = frozenset(int(x) for x in banned)
    keep = np.ones(N, dtype=bool)
    for v in banned:
        keep[v] = False
    idx_kept = np.flatnonzero(keep)
    B0 = REACH0[np.ix_(keep, keep)]
    rowsum = B0.sum(axis=1) - 1  # 基线下每个幸存点的可达幸存点数
    seen = [0] * N
    dist = [0] * N
    S_all = 0.0
    S_fix = 0.0
    pairs_contrib = 0
    alive = 0
    n_inc = 0
    max_inc = 0
    n_lost_pairs = 0
    comp = {}
    for i, s in enumerate(idx_kept):
        mark = s + 1
        seen[s] = mark
        dist[s] = 0
        q = [s]
        head = 0
        ta = 0.0
        tp = 0.0
        nps = 0
        while head < len(q):
            u = q[head]
            head += 1
            nd = dist[u] + 1
            for w in adj_i[u]:
                if seen[w] != mark and w not in banned:
                    seen[w] = mark
                    dist[w] = nd
                    q.append(w)
                    inv = 1.0 / nd
                    ta += inv
                    if in_ps[w]:
                        nps += 1
                        tp += inv
                    d0 = D0[s, w]
                    if d0 >= 0 and nd > d0:
                        n_inc += 1
                        if nd - d0 > max_inc:
                            max_inc = int(nd - d0)
        now_reach = len(q) - 1
        br = int(rowsum[i])
        if br > now_reach:
            n_lost_pairs += br - now_reach
        S_all += ta
        alive += 1
        if in_ps[s]:
            S_fix += tp
            pairs_contrib += nps
        key = min(q)
        c = comp.get(key)
        if c is None:
            comp[key] = [len(q), ta]
        else:
            c[1] += ta
    sizes = sorted((v[0] for v in comp.values()), reverse=True)
    gcc = sizes[0] if sizes else 0
    big = None
    for v in comp.values():
        if v[0] == gcc:
            big = v
            break
    k_ps = int(sum(1 for v in banned if in_ps[v]))
    exp_pairs = (n_ps - k_ps) * (n_ps - k_ps - 1)
    pred, n_sub = e_fixed_pred(banned)
    return {"k": len(banned), "k_ps": k_ps, "alive": alive, "gcc": gcc,
            "n_components": len(sizes), "comp_sizes": sizes[:6],
            "E_full": S_all / (alive * (alive - 1)) if alive > 1 else 0.0,
            "E_gcc": (big[1] / (gcc * (gcc - 1))) if (big and gcc > 1) else 0.0,
            "E_fixed": S_fix / denom,
            "E_fixed_pred_mech": pred,
            "pairs_contrib": pairs_contrib, "pairs_expected": exp_pairs,
            "pairs_beyond_mech": exp_pairs - pairs_contrib,
            "n_lost_pairs_all": n_lost_pairs, "n_dist_increased": n_inc,
            "max_dist_increase": max_inc}  # L:SCENARIO


def idx_of(names):
    out = []
    for n in names:
        if n in pos_of:
            out.append(pos_of[n])
    return out


# 基线自检：内核 vs csa_common.eff_three
T0 = time.time()
base = scenario([])  # L:BASE_SCEN
ref0 = C.eff_three(Gu, ps)  # L:REF0
t_base = time.time() - T0  # L:T_BASE
assert abs(base["E_fixed"] - ref0["E_fixed"]) < 1e-12  # L:A_BASE_FIX
assert abs(base["E_full"] - ref0["E_full"]) < 1e-12  # L:A_BASE_FULL
assert abs(base["E_gcc"] - ref0["E_gcc"]) < 1e-12  # L:A_BASE_GCC
E_fixed0 = ref0["E_fixed"]  # L:E0
E_full0 = ref0["E_full"]  # L:EF0
E_gcc0 = ref0["E_gcc"]  # L:EG0

# 内核在**批量移除**图上也对 C.eff_three 负责
val_rows = []  # L:VAL_SEED_SETS
t_val = 0.0
kernel_max_diff = float("nan")  # L:KERNEL_DIFF
if not QUICK:
    T0 = time.time()
    pool = [v for v in range(N)]
    for j, kx in enumerate([7, 33, 129]):
        bs = set(RNG.choice(pool, size=kx, replace=False).tolist())  # L:VAL_BATCH
        sub = Gu.subgraph([node_list[v] for v in range(N) if v not in bs]).copy()
        ref = C.eff_three(sub, ps)  # L:VAL_REF
        got = scenario(bs)
        val_rows.append({"batch": "random_k%d" % kx, "k": kx,
                         "ref_E_fixed": ref["E_fixed"], "fast_E_fixed": got["E_fixed"],
                         "ref_E_gcc": ref["E_gcc"], "fast_E_gcc": got["E_gcc"],
                         "ref_E_full": ref["E_full"], "fast_E_full": got["E_full"],
                         "abs_diff": max(abs(ref["E_fixed"] - got["E_fixed"]),
                                         abs(ref["E_gcc"] - got["E_gcc"]),
                                         abs(ref["E_full"] - got["E_full"]))})
    val_df = pd.DataFrame(val_rows)  # L:VAL_DF
    kernel_max_diff = float(val_df["abs_diff"].max())
    assert kernel_max_diff < 1e-9, "批量内核与 csa_common.eff_three 不一致"  # L:VAL_ASSERT
    t_val = time.time() - T0

# ------------------------------------------------------------------ 6. 各治理类的精确情景
members = {c: df.loc[df["class"] == c, "node"].tolist() for c in CLASSES}  # L:MEMBERS
scen_defs = {  # L:SCEN_DEFS
    "下线": idx_of(members["下线"]),
    "修复": idx_of(members["修复"]),
    "保护": idx_of(members["保护"]),
    "监控": idx_of(members["监控"]),
    "全部真失效": idx_of(df.loc[df["f_true_fail"], "node"]),
    "全部失效_162": idx_of(df.loc[df["is_dead"], "node"]),
}
scen = {}  # L:SCEN_RUN
for tag, bs in scen_defs.items():
    sc = scenario(set(bs))  # L:SCEN_ONE
    sc["k"] = len(bs)
    scen[tag] = sc

# CONVENTIONS §2.3 表交叉校验：删 162 全挂
DOC_162 = {"E_full": 0.3508, "E_gcc": 0.3536, "E_fixed": 0.2039}  # L:DOC_162
diff162 = {k: abs(scen["全部失效_162"][k] - v) for k, v in DOC_162.items()}  # L:DIFF162
assert max(diff162.values()) < 1e-3, "与 §2.3 已测表不一致: " + str(diff162)  # L:A_DOC162

# 同 k 随机基线（§6：≥100 次重复 + sd）
T0 = time.time()
null_reps = {}  # L:NULL_DEFS
for tag in ["下线", "全部真失效"]:  # L:NULL_TAGS
    k = scen_defs[tag]
    kx = len(k)
    rows = []
    for r in range(N_RAND):  # L:NULL_LOOP
        bs = set(RNG.choice(N, size=kx, replace=False).tolist())  # L:NULL_SAMPLE
        sc = scenario(bs)
        rows.append({"rep": r + 1, "k": kx, "E_fixed": sc["E_fixed"],
                     "E_gcc": sc["E_gcc"], "E_full": sc["E_full"], "gcc": sc["gcc"],
                     "pairs_beyond_mech": sc["pairs_beyond_mech"],
                     "n_lost_pairs_all": sc["n_lost_pairs_all"],
                     "n_components": sc["n_components"],
                     "max_dist_increase": sc["max_dist_increase"],
                     "n_dist_increased": sc["n_dist_increased"]})
    null_reps[tag] = pd.DataFrame(rows)  # L:NULL_DF
t_null = time.time() - T0  # L:T_NULL

null_stats = {}  # L:NULL_STATS
for tag, d in null_reps.items():
    obs = scen[tag]["E_fixed"]
    mu = float(d["E_fixed"].mean())
    sd = float(d["E_fixed"].std(ddof=1))
    null_stats[tag] = {
        "k": int(d["k"].iloc[0]), "n_reps": len(d),
        "E_fixed_mean": mu, "E_fixed_sd": sd,
        "E_fixed_min": float(d["E_fixed"].min()), "E_fixed_max": float(d["E_fixed"].max()),
        "E_fixed_p5": float(d["E_fixed"].quantile(0.05)),
        "E_fixed_p95": float(d["E_fixed"].quantile(0.95)),
        "scenario_E_fixed": obs,
        "pct_higher_than_random": 100.0 * (obs - mu) / mu if mu else float("nan"),
        "empirical_p_two_sided": float((d["E_fixed"] - mu).abs().ge(abs(obs - mu)).mean()),
        "z": (obs - mu) / sd if sd else float("nan"),
        "gcc_mean": float(d["gcc"].mean()), "gcc_sd": float(d["gcc"].std(ddof=1)),
        "scenario_gcc": int(scen[tag]["gcc"]),
        "pairs_beyond_mech_mean": float(d["pairs_beyond_mech"].mean()),
        "pairs_beyond_mech_sd": float(d["pairs_beyond_mech"].std(ddof=1)),
        "scenario_pairs_beyond_mech": int(scen[tag]["pairs_beyond_mech"]),
        "n_reps_with_disconnect": int((d["pairs_beyond_mech"] > 0).sum()),
        "n_reps_with_dist_increase": int((d["n_dist_increased"] > 0).sum()),
    }

# 下线无害性的三条硬证据（逐点 + 批量 + 等 k 对照）
off_df = df[df["class"] == "下线"]  # L:OFF_DF
off_ind = {
    "n": len(off_df),
    "n_ap": int(off_df["f_ap"].sum()),
    "n_disc": int(off_df["f_disc"].sum()),
    "n_extra_gcc": int(off_df["f_extra_gcc"].sum()),
    "max_pairs_lost": int(off_df["pairs_lost_within_ps"].max()) if len(off_df) else 0,
    "max_abs_relay": float(off_df["dE_fixed_relay"].abs().max()) if len(off_df) else 0.0,
    "sum_dE_fixed_individual": float(off_df["dE_fixed"].sum()),
    "in_edges_pointing_at_off": int(off_df["in_degree"].sum()),
    "share_of_all_in_edges": 100.0 * float(off_df["in_degree"].sum()) / float(df["in_degree"].sum()),
    "max_in_degree": int(off_df["in_degree"].max()) if len(off_df) else 0,
    "median_in_degree": float(off_df["in_degree"].median()) if len(off_df) else 0.0,
}  # L:OFF_IND
OFF = scen["下线"]  # L:OFF_SCEN
off_batch = {
    "k": OFF["k"], "gcc_before": n_gcc0, "gcc_after": OFF["gcc"],
    "gcc_drop": n_gcc0 - OFF["gcc"], "k_in_gcc": OFF["k_ps"],
    "gcc_drop_equals_k": bool(n_gcc0 - OFF["gcc"] == OFF["k_ps"]),
    "n_components_after": OFF["n_components"],
    "pairs_beyond_mech": OFF["pairs_beyond_mech"],
    "n_lost_pairs_all": OFF["n_lost_pairs_all"],
    "n_dist_increased": OFF["n_dist_increased"],
    "max_dist_increase": OFF["max_dist_increase"],
    "E_fixed_measured": OFF["E_fixed"], "E_fixed_pred_mechanical": OFF["E_fixed_pred_mech"],
    "abs_diff_vs_mechanical": abs(OFF["E_fixed"] - OFF["E_fixed_pred_mech"]),
    "E_full_after": OFF["E_full"], "E_full_before": E_full0,
    "E_gcc_after": OFF["E_gcc"], "E_gcc_before": E_gcc0,
    "E_fixed_drop_total": E_fixed0 - OFF["E_fixed"],
    "E_fixed_drop_mechanical": E_fixed0 - OFF["E_fixed_pred_mech"],
}  # L:OFF_BATCH
assert off_batch["pairs_beyond_mech"] == 0, "下线情景出现超出机械项的断连"  # L:A_OFF_PAIRS
assert off_batch["n_lost_pairs_all"] == 0, "下线情景有幸存点对被断开"  # L:A_OFF_LOST
assert off_batch["max_dist_increase"] == 0, "下线情景抬高了幸存点对距离"  # L:A_OFF_INC
assert off_batch["abs_diff_vs_mechanical"] < 1e-12  # L:A_OFF_MECH
assert off_batch["gcc_drop_equals_k"]  # L:A_OFF_GCC

PROT = scen["保护"]  # L:PROT_SCEN
prot_ind = {
    "n": len(df[df["class"] == "保护"]),
    "sum_pairs_lost_individual": int(df.loc[df["class"] == "保护", "pairs_lost_within_ps"].sum()),
    "joint_pairs_beyond_mech": PROT["pairs_beyond_mech"],
    "joint_lost_pairs_all": PROT["n_lost_pairs_all"],
    "joint_max_dist_increase": PROT["max_dist_increase"],
    "additivity_ratio": (PROT["pairs_beyond_mech"] / df.loc[df["class"] == "保护", "pairs_lost_within_ps"].sum())
    if df.loc[df["class"] == "保护", "pairs_lost_within_ps"].sum() else float("nan"),
    "sum_relay": float(df.loc[df["class"] == "保护", "dE_fixed_relay"].sum()),
    "joint_dE_fixed_beyond_mech": (PROT["E_fixed_pred_mech"] - PROT["E_fixed"]),
}  # L:PROT_IND
MON = scen["监控"]  # L:MON_SCEN
mon_ind = {
    "n": len(df[df["class"] == "监控"]),
    "sum_pairs_lost_individual": int(df.loc[df["class"] == "监控", "pairs_lost_within_ps"].sum()),
    "sum_relay": float(df.loc[df["class"] == "监控", "dE_fixed_relay"].sum()),
    "joint_pairs_beyond_mech": MON["pairs_beyond_mech"],
    "joint_dE_fixed_beyond_mech": (MON["E_fixed_pred_mech"] - MON["E_fixed"]),
    "joint_max_dist_increase": MON["max_dist_increase"],
    "joint_dist_increased": MON["n_dist_increased"],
}  # L:MON_IND

# ------------------------------------------------------------------ 7. 修复类：有界收益 + 假设声明
BUCKETS = [0, 2, 4, 7, 13, 26, np.inf]  # L:BUCKETS
BLABEL = ["0-1", "2-3", "4-6", "7-12", "13-25", "26+"]  # L:BLABEL
df["in_bucket"] = pd.cut(df["in_degree"], bins=BUCKETS, labels=BLABEL, right=False,
                         include_lowest=True)  # L:IN_BUCKET
healthy_pool = df[df["f_healthy"] & ~df["f_false_fail"] & ~df["f_ambiguous"]]  # L:HEALTH_POOL
strata = healthy_pool.groupby("in_bucket", observed=True).agg(
    n_healthy=("node", "size"),
    pairs_median=("pairs_lost_within_ps", "median"),
    pairs_p90=("pairs_lost_within_ps", lambda s: float(s.quantile(0.90))),
    pairs_max=("pairs_lost_within_ps", "max"),
    relay_median=("dE_fixed_relay", "median"),
    relay_min=("dE_fixed_relay", "min")).reset_index()  # L:STRATA
strata["in_bucket"] = strata["in_bucket"].astype(str)
rep_df = df[df["class"] == "修复"].copy()  # L:REP_DF
smap_pairs = dict(zip(strata["in_bucket"], strata["pairs_median"]))
smap_pairs90 = dict(zip(strata["in_bucket"], strata["pairs_p90"]))
smap_relay = dict(zip(strata["in_bucket"], strata["relay_median"]))
rep_df["proxy_pairs_upper"] = rep_df["in_bucket"].astype(str).map(smap_pairs).fillna(0.0)  # L:PROXY_PAIRS
rep_df["proxy_pairs_p90"] = rep_df["in_bucket"].astype(str).map(smap_pairs90).fillna(0.0)  # L:PROXY_P90
rep_df["proxy_relay_upper"] = rep_df["in_bucket"].astype(str).map(smap_relay).fillna(0.0)  # L:PROXY_RELAY
for _c in ["proxy_pairs_upper", "proxy_pairs_p90", "proxy_relay_upper"]:  # L:PROXY_BACK
    df[_c] = rep_df[_c].reindex(df.index)
rep_ind = {
    "n": len(rep_df),
    "n_true_fail_part": int(rep_df["f_true_fail"].sum()),
    "n_partial_part": int(rep_df["f_partial"].sum()),
    "sum_proxy_pairs": float(rep_df["proxy_pairs_upper"].sum()),
    "sum_proxy_pairs_p90": float(rep_df["proxy_pairs_p90"].sum()),
    "sum_proxy_relay": float(rep_df["proxy_relay_upper"].sum()),
    "sum_in_edges": int(rep_df["in_degree"].sum()),
    "share_of_in_edges": 100.0 * float(rep_df["in_degree"].sum()) / float(df["in_degree"].sum()),
    "baseline_pairs": int(denom),
    "proxy_pairs_share_pct": 100.0 * float(rep_df["proxy_pairs_upper"].sum()) / denom,
    "proxy_relay_share_pct": 100.0 * abs(float(rep_df["proxy_relay_upper"].sum())) / E_fixed0,
    "individual_pairs_observed": int(rep_df["pairs_lost_within_ps"].sum()),
    "individual_pairs_observed_tf": int(rep_df.loc[rep_df["f_true_fail"],
                                                   "pairs_lost_within_ps"].sum()),
}  # L:REP_IND
REPS = scen["修复"]  # L:REP_SCEN

# 保护类成员逐点表（清单用）
SPOF_COLS = ["node", "sub", "health_state", "in_degree", "ok_ratio_pct", "pairs_lost_within_ps",
             "share_of_baseline_pairs_lost", "dE_fixed", "dE_fixed_relay",
             "extra_gcc_lost_vs_mechanical", "is_articulation_point", "bet_pct", "pr_pct",
             "in_pct"]  # L:SPOF_COLS

# ------------------------------------------------------------------ 8. 阈值敏感性 + 象限交叉 + 未分类画像
sens_rows = []  # L:SENS_INIT
_taken_above_repair = (df["f_false_fail"] | df["f_ambiguous"]
                       | (df["f_true_fail"] & df["f_zero_role"]))  # L:SENS_TAKEN
for t in T_ID_SENS:  # L:SENS_LOOP
    mk = (df["f_true_fail"] & ~df["f_zero_role"]) \
        | (df["f_partial"] & (df["f_hi_impact"] | (df["in_degree"] >= t)))
    n_rep_t = int((mk & ~_taken_above_repair).sum())
    n_unc_t = n_attr - n_excluded - cls_counts["下线"] - cls_counts["保护"] \
        - cls_counts["监控"] - n_rep_t
    sens_rows.append({"T_id": t, "n_修复": n_rep_t, "n_未分类": n_unc_t,
                      "pct_未分类": round(100.0 * n_unc_t / n_attr, 2),
                      "n_下线": cls_counts["下线"], "n_保护": cls_counts["保护"],
                      "n_监控": cls_counts["监控"]})
sens_df = pd.DataFrame(sens_rows)  # L:SENS_DF
assert int(sens_df.loc[sens_df["T_id"] == T_ID, "n_修复"].iloc[0]) == cls_counts["修复"], \
    "敏感性表在 T_id=%d 处应与主判据一致" % T_ID  # L:A_SENS
assert float(sens_df.loc[sens_df["T_id"] == T_ID, "n_未分类"].iloc[0]) == n_unclassified  # L:A_SENS2
sens_relay_rows = []  # L:SENS_RELAY_INIT
for rr in [-1e-9, -1e-6, -1e-5, THR_RELAY, -0.001, -0.005]:
    hi = (df["pairs_lost_within_ps"] > THR_PAIRS) | (df["dE_fixed_relay"] < rr)
    sens_relay_rows.append({"relay_threshold": rr, "n_hi_impact": int(hi.sum()),
                            "n_保护": int((hi & df["f_healthy"]).sum())})
sens_relay_df = pd.DataFrame(sens_relay_rows)  # L:SENS_RELAY_DF

g1csv = os.path.join(PIPELINE, "out", "G1_risk_matrix", "g1_nodes_scored.csv")  # L:G1CSV
quad_cross = None  # L:QUAD_CROSS
G1_QUAD_COL = "in_degree|corr__quadrant"  # L:G1QCOL  G1 主表：入度 × 校正后健康口径
if os.path.exists(g1csv):
    g1 = pd.read_csv(g1csv, encoding="utf-8-sig")
    if G1_QUAD_COL in g1.columns:
        g1q = g1[["node", G1_QUAD_COL]].rename(columns={G1_QUAD_COL: "quadrant_G1"})
        df = df.merge(g1q, on="node", how="left", validate="one_to_one")  # L:G1_MERGE
        assert (df.loc[df["f_false_fail"] | df["f_ambiguous"], "quadrant_G1"].isna().all()
                ), "G1 校正表里假失效/待判不该有象限标签"  # L:A_G1Q
        df["quadrant_G1"] = df["quadrant_G1"].fillna("NA_未打分")  # L:Q_FILL
    else:
        df["quadrant_G1"] = np.nan
else:
    df["quadrant_G1"] = np.nan
if df["quadrant_G1"].notna().any():
    quad_cross = pd.crosstab(df["class"], df["quadrant_G1"])  # L:QUAD_X
    quad_cross.to_csv(os.path.join(HERE, "g3_class_x_g1quadrant.csv"),
                      encoding="utf-8-sig", index=True)  # L:QUAD_CSV

unc = df[df["class"] == "未分类"]  # L:UNC_DF
unc_profile = unc.groupby("health_state", observed=True).agg(
    n=("node", "size"), in_degree_median=("in_degree", "median"),
    pairs_lost_max=("pairs_lost_within_ps", "max"),
    relay_min=("dE_fixed_relay", "min"),
    bet_pct_median=("bet_pct", "median"), pr_pct_median=("pr_pct", "median")).reset_index()  # L:UNC_PROF
n_unc_healthy = int((unc["f_healthy"]).sum())  # L:UNC_HEALTHY
n_unc_below_med = int((unc["in_pct"] < 0.5).sum())  # L:UNC_BELOWMED
unc_top = unc.nlargest(15, "pairs_lost_within_ps")[
    ["node", "sub", "health_state", "in_degree", "pairs_lost_within_ps",
     "dE_fixed_relay", "bet_pct", "pr_pct"]]  # L:UNC_TOP

# ------------------------------------------------------------------ 9. 类级收益汇总表
def benefit_row(c):
    if c == "下线":
        return {"预期结构收益": "0（情景模拟：幸存点对一条都没断，ΔE_fixed 全部是端点机械项）",
                "收益量级_pairs": 0,
                "收益量级_dE_fixed": 0.0,
                "对照": "同 k 随机基线 E_fixed %.4f±%.4f（%d 次）" % (
                    null_stats["下线"]["E_fixed_mean"], null_stats["下线"]["E_fixed_sd"], N_RAND)}
    if c == "修复":
        return {"预期结构收益": "有界区间 [0, %.1f 个点对]（同入度档健康节点的实测移除断连数作代理上界）"
                % rep_ind["sum_proxy_pairs"],
                "收益量级_pairs": rep_ind["sum_proxy_pairs"],
                "收益量级_dE_fixed": abs(rep_ind["sum_proxy_relay"]),
                "对照": "上界取 p90 档时 %.1f 个点对" % rep_ind["sum_proxy_pairs_p90"]}
    if c == "保护":
        return {"预期结构收益": "避免损失 = 单点移除情景之和 %.0f 点对（联合移除实测 %.0f，可加性 %.2f）"
                % (prot_ind["sum_pairs_lost_individual"], prot_ind["joint_pairs_beyond_mech"],
                   prot_ind["additivity_ratio"]),
                "收益量级_pairs": prot_ind["joint_pairs_beyond_mech"],
                "收益量级_dE_fixed": abs(prot_ind["joint_dE_fixed_beyond_mech"]),
                "对照": "全部落在 E_fixed 上 = %.4f（基线 %.4f 的 %.2f%%）" % (
                    prot_ind["joint_dE_fixed_beyond_mech"], E_fixed0,
                    100.0 * prot_ind["joint_dE_fixed_beyond_mech"] / E_fixed0)}
    if c == "监控":
        return {"预期结构收益": "位置突出但实测移除损伤低于保护阈值：个体断连对数之和 %d，"
                "中继项之和 %.3g；联合移除情景里有 %d 个点对的距离被抬高（无断连）"
                % (mon_ind["sum_pairs_lost_individual"], mon_ind["sum_relay"],
                   mon_ind["joint_dist_increased"]),
                "收益量级_pairs": mon_ind["sum_pairs_lost_individual"],
                "收益量级_dE_fixed": abs(mon_ind["sum_relay"]),
                "对照": "联合移除实测超出机械项 %.0f 点对" % mon_ind["joint_pairs_beyond_mech"]}
    if c == "排除_假失效BC":
        return {"预期结构收益": "不适用：§2.2 B/C 的不可用是抓取判据不可信，不是站点状态",
                "收益量级_pairs": 0, "收益量级_dE_fixed": 0.0,
                "对照": "人工核验清单见 G1 的 g1_false_fail_class.csv"}
    if c == "排除_待判G":
        return {"预期结构收益": "不适用：证据不足", "收益量级_pairs": 0,
                "收益量级_dE_fixed": 0.0, "对照": "单列，不参与排序"}
    return {"预期结构收益": "无可测结构角色（入度/介数/PageRank 三轴都不在 top5%）",
            "收益量级_pairs": 0, "收益量级_dE_fixed": 0.0,
            "对照": "其中 %d 个是当前健康节点" % n_unc_healthy}


summary_rows = []  # L:SUMMARY_INIT
for c in CLASSES:
    sub = df[df["class"] == c]
    rule = dict((n, d) for n, d, _ in RULES).get(c, "以上规则都未命中")
    summary_rows.append({
        "class": c, "rule": rule, "n": len(sub),
        "share_of_671_pct": round(100.0 * len(sub) / n_attr, 2),
        "n_healthy": int(sub["f_healthy"].sum()), "n_partial": int(sub["f_partial"].sum()),
        "n_true_fail": int(sub["f_true_fail"].sum()),
        "in_degree_median": float(sub["in_degree"].median()) if len(sub) else 0.0,
        "ok_ratio_median": float(sub["ok_ratio_pct"].median()) if len(sub) else float("nan"),
        "pairs_lost_sum": int(sub["pairs_lost_within_ps"].sum()),
        "relay_sum": float(sub["dE_fixed_relay"].sum()),
        "n_hi_impact": int(sub["f_hi_impact"].sum()),
        **benefit_row(c)})
class_summary = pd.DataFrame(summary_rows)  # L:CLS_SUM

members_rows = []  # L:MEMBERS_INIT
for c in CLASSES:
    sub = df[df["class"] == c].copy()
    if not len(sub):
        continue
    sub = sub.sort_values(["pairs_lost_within_ps", "in_degree"], ascending=[False, False])
    sub["rank_in_class"] = np.arange(1, len(sub) + 1)
    sub["class"] = c
    keep = ["class", "rank_in_class", "node", "sub", "tier", "health_state", "dead_cat",
            "in_degree", "out_degree", "degree_u", "ok_ratio_pct", "pairs_lost_within_ps",
            "share_of_baseline_pairs_lost", "dE_fixed", "dE_fixed_relay",
            "extra_gcc_lost_vs_mechanical", "is_articulation_point", "bet_pct", "pr_pct",
            "in_pct", "rule_hit"]
    if c == "修复":
        keep = keep + ["in_bucket", "proxy_pairs_upper", "proxy_relay_upper"]
    members_rows.append(sub[keep])
members_df = pd.concat(members_rows, ignore_index=True)  # L:MEMBERS_DF

scen_rows = []  # L:SCEN_ROWS_INIT
for tag in ["下线", "修复", "保护", "监控", "全部真失效", "全部失效_162"]:
    s = scen[tag]
    key = tag if tag in null_stats else None
    scen_rows.append({
        "scenario": "整类批量移除情景：" + tag, "k": s["k"], "k_in_ps": s["k_ps"],
        "gcc_before": n_gcc0, "gcc_after": s["gcc"],
        "n_components_after": s["n_components"],
        "comp_sizes_after": "|".join(str(x) for x in s["comp_sizes"]),
        "E_full_after": s["E_full"], "E_gcc_after": s["E_gcc"], "E_fixed_after": s["E_fixed"],
        "E_fixed_pred_mechanical": s["E_fixed_pred_mech"],
        "beyond_mechanical_dE_fixed": s["E_fixed_pred_mech"] - s["E_fixed"],
        "pairs_beyond_mech": s["pairs_beyond_mech"],
        "n_survivor_pairs_lost": s["n_lost_pairs_all"],
        "n_pairs_distance_increased": s["n_dist_increased"],
        "max_distance_increase": s["max_dist_increase"],
        "null_E_fixed_mean": null_stats[key]["E_fixed_mean"] if key else float("nan"),
        "null_E_fixed_sd": null_stats[key]["E_fixed_sd"] if key else float("nan"),
        "null_pairs_beyond_mech_mean": null_stats[key]["pairs_beyond_mech_mean"] if key else float("nan"),
        "pct_vs_random": null_stats[key]["pct_higher_than_random"] if key else float("nan"),
        "n_reps": N_RAND if key else 0})
scen_df = pd.DataFrame(scen_rows)  # L:SCEN_DF

# ------------------------------------------------------------------ 10. 写产物
def wcsv(d: pd.DataFrame, name: str, index=False):
    d.to_csv(index=index, encoding="utf-8-sig", path_or_buf=os.path.join(HERE, name))  # L:WCSV


if not QUICK:
    wcsv(df[["node", "sub", "class", "tier", "health_state", "dead_cat", "in_degree",
             "out_degree", "degree_u", "ok_ratio_pct", "bet_pct", "pr_pct", "in_pct",
             "pairs_lost_within_ps", "dE_fixed", "dE_fixed_relay",
             "extra_gcc_lost_vs_mechanical", "is_articulation_point", "f_zero_role",
             "f_hi_impact", "f_top_cen", "rule_hit"]], "g3_classified_nodes.csv")
    wcsv(class_summary, "g3_class_summary.csv")
    wcsv(members_df, "g3_class_members.csv")
    wcsv(scen_df, "g3_scenario_summary.csv")
    for tag, d in null_reps.items():
        wcsv(d, "g3_null_random_k%d.csv" % int(d["k"].iloc[0]))
    wcsv(strata, "g3_repair_matched_strata.csv")
    wcsv(rep_df[["node", "sub", "health_state", "dead_cat", "in_degree", "in_bucket",
                 "pairs_lost_within_ps", "dE_fixed_relay", "proxy_pairs_upper",
                 "proxy_pairs_p90", "proxy_relay_upper"]], "g3_repair_benefit_bracket.csv")
    wcsv(overlap_df, "g3_rule_overlap.csv")
    wcsv(sens_df, "g3_threshold_sensitivity_Tid.csv")
    wcsv(sens_relay_df, "g3_threshold_sensitivity_relay.csv")
    wcsv(unc_profile, "g3_unclassified_profile.csv")
    wcsv(unc_top, "g3_unclassified_top15.csv")
    if len(val_rows):
        wcsv(val_df, "g3_kernel_validation.csv")
    with open(os.path.join(HERE, "G3_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"unit": "G3_governance_priority", "seed": SEED, "n_rand": N_RAND,
                   "quick": QUICK,
                   "universe": {"n_attr": n_attr, "n_gcc": n_gcc0, "denom_pairs": int(denom),
                                "n_dead": n_dead, "n_true": n_true, "n_false": n_false,
                                "n_amb": n_amb, "n_partial": n_partial, "n_healthy": n_healthy},
                   "classes": cls_counts, "pct_unclassified": pct_unclassified,
                   "rules": {n: d for n, d, _ in RULES},
                   "thresholds": {"T_id": T_ID, "relay_lt": THR_RELAY,
                                  "pairs_lost_gt": THR_PAIRS, "pct_cut": PCT_CUT},
                   "baseline": {"E_full": E_full0, "E_gcc": E_gcc0, "E_fixed": E_fixed0},
                   "scenarios": scen, "offline_individual": off_ind, "offline_batch": off_batch,
                   "protection": prot_ind, "monitor": mon_ind, "repair": rep_ind,
                   "null": null_stats, "doc162_diff": diff162,
                   "kernel_max_diff_vs_csa_eff_three": kernel_max_diff,
                   "timing": {"d0_s": round(t_d0, 2), "base_s": round(t_base, 2),
                              "val_s": round(t_val, 2), "null_s": round(t_null, 2),
                              "per_scenario_s": round(t_null / (N_RAND * len(null_reps)), 4)
                              if null_reps else float("nan"),
                              "total_s": round(time.time() - T_START, 2)},
                   "versions": C.versions()}, f, ensure_ascii=False, indent=2, default=str)  # L:JSON

# ------------------------------------------------------------------ 11. result.md
if True:  # L:MD_GUARD  QUICK 模式也构建全文（只为跑通格式化与 §7 自检），不写盘
    A = []  # L:MD_INIT

    def P(*xs):
        for x in xs:
            A.append(str(x))

    P("# G3_governance_priority · 四类治理建议与结构收益情景模拟")
    P("")
    P("> **范围声明**：本单元只输出**建议清单**。文中所有「移除 / 下线 / 修复」都指"
      "**图上的情景模拟**（在 `csa_common` 建好的静态快照上做子图计算），"
      "不代表、不授权、也不描述对任何真实站点的操作。数据是单次横截面，"
      "所以「优先保护」是一句**结构位置陈述**，不是对某站点 will fail 的预测（§7）。")
    P("")
    P("## 0. 结论速览")
    P("")
    P("| 类 | 判据（布尔式，先到先得） | n | 占 671 | 预期结构收益（情景模拟） | 依据 |")
    P("|---|---|---:|---:|---|---|")
    for c in CLASSES:
        r = class_summary[class_summary["class"] == c].iloc[0]
        P("| %s | %s | %d | %.2f%% | %s | %s |" % (
            c, r["rule"], r["n"], r["share_of_671_pct"], r["预期结构收益"], cite("RULES")))
    P("")
    P("一句话：**下线在模拟中是无损的**（幸存点对零断连、ΔE_fixed 全部是端点机械项，"
      "%s），而**同 k 随机移除**在 %d 次重复里平均要断 %.0f 个点对（%s）——"
      "两者的差就是本单元唯一能站得住的定量结论。" % (
          cite("A_OFF_LOST"), N_RAND, null_stats["下线"]["pairs_beyond_mech_mean"],
          cite("NULL_LOOP")))
    P("")

    P("## 1. 判据与优先级")
    P("")
    P("优先级 = 表中自上而下，一个节点只进第一个命中的类（%s）。"
      "不设优先级时各规则自身的命中规模与重叠：" % cite("ASSIGN_LOOP"))
    P("")
    P(md_table(overlap_df.rename(columns={"n_rule_only": "规则命中(不看优先级)",
                                          "n_after_priority": "实际入类",
                                          "n_absorbed_by_higher": "被更高优先类吸收",
                                          "class": "类"}),
               ["类", "规则命中(不看优先级)", "实际入类", "被更高优先类吸收", "rule"]))
    P("")
    P("- 判据只读**已测字段**：G2 的 676 行单点移除表（`pairs_lost_within_ps` / "
      "`dE_fixed_relay` / `extra_gcc_lost_vs_mechanical` / `is_articulation_point`，%s）"
      "与 `csa_common` 的健康与中心性字段（%s）。零人工判断。" % (cite("MERGE_G2"), cite("CEN")))
    P("- 「零实测结构角色」= 非割点 且 不断连任何点对 且 中继项为 0 且 无超出机械项的 GCC 损失（%s）。"
      "它是**测出来的**，不是「入度小所以应该无害」的先验假设。" % cite("F_ZERO"))
    P("- 假失效(§2.2 B/C) 与待判(G) 一律**先排除**，不与任何治理类混写（%s）。"
      "假失效 %d 个、待判 %d 个，画像与定位见 G1 的 `g1_false_fail_class.csv`。" % (
          cite("F_FF"), n_false, n_amb))
    P("- 结构重要性轴 = 入度 / 介数 / 有向加权 PageRank 三个主口径的百分位**并集** top%d%%（%s）。"
      "无向总度 `deg_u` 不进任何判据（§2.1，%s）。" % (
          int((1 - PCT_CUT) * 100), cite("F_TOPCEN"), cite("DROP_BANNED")))
    P("")

    P("## 2. 类级汇总")
    P("")
    P(md_table(class_summary.rename(columns={
        "class": "类", "n": "n", "share_of_671_pct": "占671%", "n_healthy": "健康",
        "n_partial": "部分", "n_true_fail": "真失效", "in_degree_median": "入度中位",
        "ok_ratio_median": "ok%中位", "pairs_lost_sum": "单点断连对数之和",
        "relay_sum": "中继项之和", "n_hi_impact": "高影响数"}),
        ["类", "n", "占671%", "健康", "部分", "真失效", "入度中位", "ok%中位",
         "单点断连对数之和", "中继项之和", "高影响数"], floatfmt="%.6g"))
    P("")
    P("基线（%s）：E_full=%.4f / E_gcc=%.4f / **E_fixed=%.4f**，点对分母恒为 %d（§2.3）。" % (
        cite("REF0"), E_full0, E_gcc0, E_fixed0, int(denom)))
    P("")

    P("## 3. 下线：无害性的数值证明")
    P("")
    P("成员规则见 %s；类内 %d 个节点（%s）。" % (cite("RULES"), cls_counts["下线"], cite("MEMBERS")))
    P("")
    P("### 3a. 逐点（读 G2 实测，%s）" % cite("OFF_DF"))
    P("")
    P("| 指标 | 值 | 依据 |")
    P("|---|---|---|")
    P("| 类内是割点的节点数 | %d | %s |" % (off_ind["n_ap"], cite("F_AP")))
    P("| 类内会断连点对的节点数 | %d | %s |" % (off_ind["n_disc"], cite("OFF_IND")))
    P("| 类内单点断连对数最大值 | %d | %s |" % (off_ind["max_pairs_lost"], cite("OFF_IND")))
    P("| 类内 \\|ΔE_fixed 中继项\\| 最大值 | %.3g | %s |" % (off_ind["max_abs_relay"], cite("OFF_IND")))
    P("| 类内超出机械项的 GCC 损失节点数 | %d | %s |" % (off_ind["n_extra_gcc"], cite("F_EXTRA")))
    P("| 类内入度中位 / 最大 | %.1f / %d | %s |" % (
        off_ind["median_in_degree"], off_ind["max_in_degree"], cite("OFF_IND")))
    P("| 指向这些节点的入边条数（**已实现**的断链） | %d（占全网入边 %.2f%%） | %s |" % (
        off_ind["in_edges_pointing_at_off"], off_ind["share_of_all_in_edges"], cite("OFF_IND")))
    P("")
    P("### 3b. 整类批量移除情景（%s）" % cite("SCEN_ONE"))
    P("")
    P("| 量 | 移除前 | 移除后 | 变化 | 判读 |")
    P("|---|---:|---:|---:|---|")
    P("| GCC 节点数 | %d | %d | %+d | 恰好等于被删的 GCC 成员数 %d，**没有连带掉出**（%s） |" % (
        n_gcc0, off_batch["gcc_after"], -off_batch["gcc_drop"], off_batch["k_in_gcc"],
        cite("A_OFF_GCC")))
    P("| 连通分量数 | %d | %d | %+d | 未分裂（%s） |" % (
        b["sets"]["n_components"], off_batch["n_components_after"],
        off_batch["n_components_after"] - b["sets"]["n_components"], cite("SCENARIO")))
    P("| 幸存点对断连数（全图，含非 GCC） | — | — | **%d** | 0 = 一个都没断（%s） |" % (
        off_batch["n_lost_pairs_all"], cite("A_OFF_LOST")))
    P("| 幸存点对距离变大的对数 / 最大增幅 | — | — | **%d / %d** | 0 = 最短路一条都没被拉长（%s） |" % (
        off_batch["n_dist_increased"], off_batch["max_dist_increase"], cite("A_OFF_INC")))
    P("| E_fixed | %.4f | %.4f | %+0.4f | 实测 %.6f vs 端点机械项预测 %.6f，差 %.2g → **全部降幅都是被删点自己不再计入**（%s） |" % (
        E_fixed0, off_batch["E_fixed_measured"], -off_batch["E_fixed_drop_total"],
        off_batch["E_fixed_measured"], off_batch["E_fixed_pred_mechanical"],
        off_batch["abs_diff_vs_mechanical"], cite("A_OFF_MECH")))
    P("| E_full | %.4f | %.4f | %+0.4f | **上升是假象**（分母收缩），只作对照不作结论（§2.3，%s） |" % (
        E_full0, off_batch["E_full_after"], off_batch["E_full_after"] - E_full0, cite("SCENARIO")))
    P("| E_gcc | %.4f | %.4f | %+0.4f | 同上，只作对照 |" % (
        E_gcc0, off_batch["E_gcc_after"], off_batch["E_gcc_after"] - E_gcc0))
    P("")
    P("### 3c. 同 k 随机基线（§6：静态、fresh 抽样、%d 次重复，seed=%d）" % (N_RAND, SEED))
    P("")
    ns = null_stats["下线"]
    P("| 情景 | k | E_fixed | 超出机械项的断连点对 | GCC |")
    P("|---|---:|---:|---:|---:|")
    P("| **下线整类** | %d | %.4f | **%d** | %d |" % (
        off_batch["k"], ns["scenario_E_fixed"], ns["scenario_pairs_beyond_mech"],
        ns["scenario_gcc"]))
    P("| 随机删同 k（mean±sd） | %d | %.4f ± %.4f | %.0f ± %.0f | %.1f ± %.1f |" % (
        ns["k"], ns["E_fixed_mean"], ns["E_fixed_sd"], ns["pairs_beyond_mech_mean"],
        ns["pairs_beyond_mech_sd"], ns["gcc_mean"], ns["gcc_sd"]))
    P("| 随机删同 k 的 5–95 分位 | %d | %.4f – %.4f | — | — |" % (
        ns["k"], ns["E_fixed_p5"], ns["E_fixed_p95"]))
    P("")
    P("- 按 §2.3 的合法说法：*在同等删除规模 k=%d 下，下线情景的 E_fixed 为 %.4f，"
      "随机基线为 %.4f（%d 次重复，sd=%.4f），即下线情景的连通效率水平比随机基线高 %.2f%%"
      "（z=%.1f，双侧经验 p=%.3f）。*" % (
          ns["k"], ns["scenario_E_fixed"], ns["E_fixed_mean"], N_RAND, ns["E_fixed_sd"],
          ns["pct_higher_than_random"], ns["z"], ns["empirical_p_two_sided"]))
    P("- 随机基线里 **%d/%d 次**至少断开 1 个幸存点对、**%d/%d 次**抬高了最短路长度；"
      "下线情景两者都是 0 次（%s）。" % (
          ns["n_reps_with_disconnect"], N_RAND, ns["n_reps_with_dist_increase"], N_RAND,
          cite("NULL_LOOP")))
    P("- 交叉校验 §2.3 已测表（删 162 全挂）：本内核 %.4f / %.4f / %.4f vs 文档 "
      "0.2039 / 0.3536 / 0.3508（E_fixed/E_gcc/E_full），最大偏差 %.2e < 1e-3（%s）。" % (
          scen["全部失效_162"]["E_fixed"], scen["全部失效_162"]["E_gcc"],
          scen["全部失效_162"]["E_full"], max(diff162.values()), cite("A_DOC162")))
    P("")
    P("**必须同时说的反面**：可达性口径看不见「%d 条入边指向已确认消失的页面」这件事，"
      "它在模拟里只表现为端点机械项。下线建议的成本因此是**内容/用户体验**的，"
      "不是**网络可达性**的；本单元只对后者给数字。" % off_ind["in_edges_pointing_at_off"])
    P("")

    P("## 4. 保护 / 监控：位置可承载的损失")
    P("")
    P("| 类 | n | 单点移除断连对数（个体之和） | 联合移除实测超出机械项 | 可加性 | 联合 ΔE_fixed（超出机械项） |")
    P("|---|---:|---:|---:|---:|---:|")
    P("| 保护 | %d | %d | %d | %.2f | %.6f |" % (
        prot_ind["n"], prot_ind["sum_pairs_lost_individual"], prot_ind["joint_pairs_beyond_mech"],
        prot_ind["additivity_ratio"], abs(prot_ind["joint_dE_fixed_beyond_mech"])))
    P("| 监控 | %d | %d | %d | %.2f | %.6f |" % (
        mon_ind["n"], mon_ind["sum_pairs_lost_individual"], mon_ind["joint_pairs_beyond_mech"],
        (mon_ind["joint_pairs_beyond_mech"] / mon_ind["sum_pairs_lost_individual"])
        if mon_ind["sum_pairs_lost_individual"] else float("nan"),
        abs(mon_ind["joint_dE_fixed_beyond_mech"])))
    P("")
    P("- 「预期收益 = 避免的损失」= 该类节点各自的移除情景（%s）。"
      "**个体之和与联合移除不等价**：联合实测 %d 个点对，个体和 %d，可加性 %.2f —— "
      ">1 表示联合更严重（几个中继同时消失时，互为替代的路径也一起消失）；"
      "<1 表示各自的断连集合互相重叠（%s）。" % (
          cite("PROT_IND"), prot_ind["joint_pairs_beyond_mech"],
          prot_ind["sum_pairs_lost_individual"], prot_ind["additivity_ratio"],
          cite("SCEN_ONE")))
    P("- 联合移除情景的幸存点对距离最大增幅 = %d（%s）。" % (
        prot_ind["joint_max_dist_increase"], cite("SCENARIO")))
    P("- 保护类清单（按断连对数降序，前 15）：")
    P("")
    pp = df[df["class"] == "保护"].nlargest(15, "pairs_lost_within_ps")
    P(md_table(pp[["node", "health_state", "ok_ratio_pct", "in_degree",
                   "pairs_lost_within_ps", "share_of_baseline_pairs_lost", "dE_fixed_relay",
                   "bet_pct", "pr_pct"]],
               floatfmt="%.6g"))
    P("")
    P("- 监控类清单（前 15，按入度）：")
    P("")
    mm = df[df["class"] == "监控"].nlargest(15, "in_degree")
    P(md_table(mm[["node", "health_state", "ok_ratio_pct", "in_degree",
                   "pairs_lost_within_ps", "dE_fixed_relay", "bet_pct", "pr_pct", "in_pct"]],
               floatfmt="%.6g"))
    P("")

    P("## 5. 修复：只能给区间，且必须写假设")
    P("")
    P("成员规则见 %s；类内 %d 个（真失效但非零结构角色 %d 个，部分失效 %d 个）（%s）。" % (
        cite("RULES"), rep_ind["n"], rep_ind["n_true_fail_part"], rep_ind["n_partial_part"],
        cite("REP_DF")))
    P("")
    P("**假设声明（不可回避）**：`urls.csv` 没有页面级入链来源字段（见 `data/DATA_DICTIONARY.md`），"
      "被删失的出边无法重建，因此修复带来的**新增**连通在横截面数据上不可观测。"
      "本单元给的收益上界来自一个**明确的情景代理**：")
    P("")
    P("1. 把节点按入度分档（%s）；" % cite("BUCKETS"))
    P("2. 每档取**当前健康**节点在 G2 中的实测单点断连对数中位数，作为「若该节点恢复成同档健康"
      "节点的出边模式，它**至多**能承载多少点对」的代理上界；")
    P("3. 下界取 0：修复 404/500 页面时，若不产生任何新的可抓取边，图上结构**完全不变**。")
    P("")
    P("| 入度档 | 档内健康节点数 | 断连对数中位 | p90 | 最大 | 中继项中位 |")
    P("|---|---:|---:|---:|---:|---:|")
    for r in strata.itertuples(index=False):
        P("| %s | %d | %.1f | %.1f | %d | %.3g |" % (
            r.in_bucket, r.n_healthy, r.pairs_median, r.pairs_p90, r.pairs_max,
            r.relay_median))
    P("")
    P("- 修复类收益区间：**[%d, %.0f] 个有序点对**（代理上界取档内中位；"
      "取 p90 档则为 %.0f），对应 ΔE_fixed 量级 %.6f（基线 E_fixed 的 %.2f%%）。" % (
        0, rep_ind["sum_proxy_pairs"], rep_ind["sum_proxy_pairs_p90"],
        abs(rep_ind["sum_proxy_relay"]), rep_ind["proxy_relay_share_pct"]))
    P("- 该区间是**结构代理**，不是内容收益；逐节点数值在 `g3_repair_benefit_bracket.csv`（%s）。" % cite("REP_DF"))
    P("- 对照：修复类节点**当前**在图上实测的单点断连对数之和 = %d，"
      "其中 %d 个真失效子集贡献 %d —— 真失效节点的入边只指向自己、不承载任何中继"
      "（§2.1 爬虫删失，162/162 出度为 0，%s）。" % (
          rep_ind["individual_pairs_observed"], rep_ind["n_true_fail_part"],
          rep_ind["individual_pairs_observed_tf"], cite("SCENARIO")))
    P("- 整类批量移除情景（把修复类节点从图上拿掉，仅作压力测试）：E_fixed %.4f，"
      "超出机械项断连 %d 对，GCC %d → %d（%s）。" % (
          REPS["E_fixed"], REPS["pairs_beyond_mech"], n_gcc0, REPS["gcc"], cite("SCEN_ONE")))
    P("")

    P("## 6. 未分类比例（必须报）")
    P("")
    P("- 671 个有归属节点中 **%d 个（%.2f%%）未进入四类**；被规则显式排除的假失效+待判 %d 个；"
      "实际给出建议的 %d 个（%.2f%%）（%s）。" % (
          n_unclassified, pct_unclassified, n_excluded, n_assigned,
          100.0 * n_assigned / n_attr, cite("CLS_SUM")))
    P("- 未分类里当前健康 %d 个，入度百分位 < 0.5 的 %d 个（%s）——"
      "绝大多数是「三个结构轴都不靠前、且当前没有可测失效信号」的叶子与小站。" % (
          n_unc_healthy, n_unc_below_med, cite("UNC_PROF")))
    P("")
    P(md_table(unc_profile.rename(columns={
        "health_state": "健康态", "n": "n", "in_degree_median": "入度中位",
        "pairs_lost_max": "单点断连最大", "relay_min": "中继项最小",
        "bet_pct_median": "介数百分位中位", "pr_pct_median": "PR百分位中位"}), floatfmt="%.6g"))
    P("")
    P("- 未分类里结构位置最值得注意的 15 个（按断连对数）在 `g3_unclassified_top15.csv`（%s）。"
      "把它们留在类外是判据的结果，不是判断它们不重要。" % cite("UNC_TOP"))
    P("- 其中 %d 个是**部分失效但既不高影响、入度也 <%d** 的节点：判据故意不把全部失效页面都塞进"
      "「修复」，否则四类会退化成「所有坏页面」。放开这个门槛的规模变化见 §8 的 T_id=0 行"
      "（修复类从 %d 涨到 %d）（%s）。" % (
          int(unc["f_partial"].sum()), T_ID, cls_counts["修复"],
          int(sens_df.loc[sens_df["T_id"] == 0, "n_修复"].iloc[0]), cite("SENS_LOOP")))
    P("")

    if quad_cross is not None:
        P("## 7. 与 G1 四象限的交叉（口径一致性）")
        P("")
        P("G1 主表 = 入度 × 校正后健康口径（`g1_nodes_scored.csv`，%s）。行=治理类，列=G1 象限：" % cite("G1CSV"))
        P("")
        P(md_table(quad_cross.reset_index().rename(columns={"class": "类"}), floatfmt="%.0f"))
        P("")
        P("两套判据是**独立定义**的（G1 用中位数二分，G3 用 G2 的实测移除影响），"
          "所以上表的重叠度就是「结构位置」与「中心性百分位」两种口径的符合程度。")
        P("")

    P("## 8. 阈值敏感性（判据里的两个自由数）")
    P("")
    P("| T_id（修复类的入度门槛） | 修复类 n | 未分类 n | 未分类 % | 下线 | 保护 | 监控 |")
    P("|---:|---:|---:|---:|---:|---:|---:|")
    for r in sens_df.itertuples(index=False):
        P("| %d | %d | %d | %.2f%% | %d | %d | %d |" % (
            r.T_id, r.n_修复, r.n_未分类, r.pct_未分类, r.n_下线, r.n_保护, r.n_监控))
    P("")
    P("| 中继项阈值 | 高影响 n | 其中健康（=保护候选） |")
    P("|---:|---:|---:|")
    for r in sens_relay_df.itertuples(index=False):
        P("| %.3g | %d | %d |" % (r.relay_threshold, r.n_hi_impact, r.n_保护))
    P("")
    P("- 主判据用 %.3g（= G2 的 `THR_RELAY`，%s），是从 G2 的分布里来的数，不是这里挑的。" % (
        THR_RELAY, cite("THR_RELAY")))
    P("")

    P("## 9. 自检与局限")
    P("")
    P("- 地基：`csa_common.build_all()` + `assert_anchors()`（%s，31/31 PASS）；"
      "构图、度数、健康标签一律不自行重建。`nodes_full.csv` 的 `degree` / `color` 列在入口即丢弃（%s）。" % (
          cite("ANCHORS"), cite("DROP_BANNED")))
    P("- 情景内核与 `csa_common.eff_three` 在基线图上逐项相等（%s），"
      "并在 k=7/33/129 的随机批量移除子图上复核，最大 \\|差\\| = %.2e（<1e-9 才继续，%s）。" % (
          cite("A_BASE_FIX"), kernel_max_diff, cite("VAL_ASSERT")))
    P("- 效率口径：只有 E_fixed 用于结论；E_full / E_gcc 与基线比大小一律标注为分母收缩假象（§2.3）。"
      "同 k 对照重复 %d 次（§6 要求 ≥100），显式 seed=%d，PYTHONHASHSEED=0（%s）。" % (
          N_RAND, SEED, cite("SEED")))
    P("- 局限：(1) 单次横截面，无时间序列，一切「收益」都是情景模拟差值；"
      "(2) 死站出边被爬虫删失（162/162 出度为 0），所以修复类收益只能给区间；"
      "(3) 边权是抓取次数，不是流量，「重要」只在结构意义上成立；"
      "(4) 判据里的 T_id / 中继项阈值 / top5% 是显式选择，敏感性见 §8；"
      "(5) 建议清单不等于处置授权，任何处置都需要人工核验页面内容。")
    P("- §7 禁用措辞自检：写文件前全文扫描词表（%s），命中即中断 —— 本次 PASS。" % cite("BANNED_LIST"))
    P("")

    text = "\n".join(A) + "\n"  # L:MD_TEXT
    bad = [w for w in BANNED_WORDS if w in text]
    assert not bad, "result.md 出现禁用因果措辞: " + str(bad)  # L:BANNED_CHECK
    if not QUICK:
        with open(os.path.join(HERE, "result.md"), "w", encoding="utf-8") as f:
            f.write(text)  # L:WRITE_RESULT
    else:
        print("[G3][QUICK] result.md 文本构建通过，%d 行 / %d 字符（未写盘）"
              % (text.count(chr(10)), len(text)))
        open(os.path.join(HERE, "_quick_result_preview.md"), "w",
             encoding="utf-8").write(text)

wall = time.time() - T_START  # L:WALL
C.write_run_note("G3_governance_priority", [
    "G3_governance_priority RUN.txt",
    "command: cd pipeline && PYTHONHASHSEED=0 python out/G3_governance_priority/run_g3.py",
    "versions: " + str(C.versions()),
    "wall_time_s: %.2f" % wall,
    "SEED = %d ; np.random.default_rng(SEED) ; PYTHONHASHSEED=0" % SEED,
    "random equal-k null reps: %d per scenario (CONVENTIONS §6 requires >=100)" % N_RAND,
    "quick_mode: %s" % QUICK,
    "anchors: csa_common.assert_anchors() 31/31 PASS (run_g3.py:L%d)" % LNO("ANCHORS"),
    "inputs: csa_common.build_all() + ../G2_single_point_failure/g2_single_removal_full.csv "
    "(+ G2_summary.json thresholds) + ../G1_risk_matrix/g1_nodes_scored.csv (quadrant cross-tab)",
    "universe: attributed=%d gcc=%d pair_denom=%d (dead=%d true=%d false=%d amb=%d partial=%d healthy=%d)"
    % (n_attr, n_gcc0, int(denom), n_dead, n_true, n_false, n_amb, n_partial, n_healthy),
    "classes: " + str(cls_counts),
    "unclassified: %d (%.2f%% of %d)" % (n_unclassified, pct_unclassified, n_attr),
    "thresholds: T_id=%d relay_lt=%.3g pairs_lost_gt=%.0f pct_cut=%.2f" % (
        T_ID, THR_RELAY, THR_PAIRS, PCT_CUT),
    "kernel: batch scenario() validated vs csa_common.eff_three (baseline exact, "
    "random batches max|diff|=%.3e)" % kernel_max_diff,
    "offline proof: pairs_lost_beyond_mechanical=%d survivor_pairs_lost=%d "
    "max_dist_increase=%d |dE_fixed vs mechanical|=%.3e gcc_drop==k_in_gcc=%s" % (
        off_batch["pairs_beyond_mech"], off_batch["n_lost_pairs_all"],
        off_batch["max_dist_increase"], off_batch["abs_diff_vs_mechanical"],
        off_batch["gcc_drop_equals_k"]),
    "offline vs equal-k random: E_fixed %.4f vs %.4f+-%.4f (n=%d), +%.2f%%" % (
        null_stats["下线"]["scenario_E_fixed"], null_stats["下线"]["E_fixed_mean"],
        null_stats["下线"]["E_fixed_sd"], N_RAND, null_stats["下线"]["pct_higher_than_random"]),
    "true-fail vs equal-k random: E_fixed %.4f vs %.4f+-%.4f (n=%d)" % (
        null_stats["全部真失效"]["scenario_E_fixed"], null_stats["全部真失效"]["E_fixed_mean"],
        null_stats["全部真失效"]["E_fixed_sd"], N_RAND),
    "doc 2.3 cross-check (delete 162 dead): max abs diff %.3e" % max(diff162.values()),
    "timing: D0=%.2fs base=%.2fs val=%.2fs nulls=%.2fs total=%.2fs (per scenario %.3fs)" % (
        t_d0, t_base, t_val, t_null, wall,
        (t_null / (N_RAND * len(null_reps))) if null_reps else float("nan")),
    "banned-word self-check §7: PASS",
    "outputs: g3_classified_nodes.csv g3_class_summary.csv g3_class_members.csv "
    "g3_scenario_summary.csv g3_null_random_k*.csv g3_repair_matched_strata.csv "
    "g3_repair_benefit_bracket.csv g3_rule_overlap.csv g3_threshold_sensitivity_*.csv "
    "g3_unclassified_*.csv g3_class_x_g1quadrant.csv g3_kernel_validation.csv "
    "G3_summary.json result.md RUN.txt",
    "scope note: recommendations only; every removal is a graph scenario simulation on the "
    "static snapshot, never an operation on a live system.",
])  # L:RUNNOTE

print("[G3] done in %.1fs (quick=%s)" % (wall, QUICK))
print("[G3] classes: " + str(cls_counts) + " unclassified=%.2f%%" % pct_unclassified)
print("[G3] offline batch: gcc %d->%d pairs_lost=%d lost=%d max_inc=%d E_fixed=%.4f (mech %.4f)"
      % (n_gcc0, off_batch["gcc_after"], off_batch["pairs_beyond_mech"],
         off_batch["n_lost_pairs_all"], off_batch["max_dist_increase"],
         off_batch["E_fixed_measured"], off_batch["E_fixed_pred_mechanical"]))
print("[G3] offline vs random k=%d: E_fixed %.4f vs %.4f+-%.4f (+%.2f%%) p=%.3f"
      % (null_stats["下线"]["k"], null_stats["下线"]["scenario_E_fixed"],
         null_stats["下线"]["E_fixed_mean"], null_stats["下线"]["E_fixed_sd"],
         null_stats["下线"]["pct_higher_than_random"],
         null_stats["下线"]["empirical_p_two_sided"]))
print("[G3] true-fail vs random: E_fixed %.4f vs %.4f+-%.4f"
      % (null_stats["全部真失效"]["scenario_E_fixed"], null_stats["全部真失效"]["E_fixed_mean"],
         null_stats["全部真失效"]["E_fixed_sd"]))
print("[G3] protection: n=%d sum_indiv_pairs=%d joint=%d additivity=%.2f"
      % (prot_ind["n"], prot_ind["sum_pairs_lost_individual"],
         prot_ind["joint_pairs_beyond_mech"], prot_ind["additivity_ratio"]))
print("[G3] repair bracket: [0, %.0f] pairs over n=%d" % (rep_ind["sum_proxy_pairs"], rep_ind["n"]))
print("[G3] §2.3 cross-check delete-162: E_fixed %.4f (doc 0.2039) maxdiff %.2e"
      % (scen["全部失效_162"]["E_fixed"], max(diff162.values())))
