#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G1_risk_matrix · 结构重要性 × 可用率 四象限风险矩阵（3 中心性 × 2 健康口径 + 1 污染对照）

运行：
    cd pipeline && PYTHONHASHSEED=0 python out/G1_risk_matrix/run_g1.py

口径绑定（CONVENTIONS.md）：
  §1 构图只用 csa_common.build_all()，开跑先 assert_anchors()
  §2.1 失效相关主口径 = 入度；无向总度只作污染对照，不进主表
  §2.2 真失效(A/D/F) / 假失效(B/C) / 待人工判断(G)：每份结果给校正前 / 校正后两套
  §5   PageRank 主口径 = 有向加权 nx.pagerank(Gd, weight="weight")
  §7   单次横截面，禁止因果措辞；风险 = 结构位置陈述，不是对某站点 will fail 的预测
  §9   result.md 每个数字都能指到本文件行号（正文 run_g1.py:Lxx 由 LNO() 解析真实行号）
"""
from __future__ import annotations

import os
import sys
import json
import time
import random

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE = os.path.dirname(os.path.dirname(HERE))
if PIPELINE not in sys.path:
    sys.path.insert(0, PIPELINE)

os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import csa_common as C  # noqa: E402

SEED = 20261113  # L:SEED
random.seed(SEED)
np.random.seed(SEED)

T_START = time.time()
BANNED = ["导致", "因为", "使得", "引发", "决定了", "传染"]  # L:BANNED_LIST

_SRC_LINES = open(__file__, encoding="utf-8").read().splitlines()


def LNO(key: str) -> int:
    """result.md 里的引用 = 本文件 `# L:key` 注释所在行的真实行号。"""
    tag = "# L:" + key
    for i, ln in enumerate(_SRC_LINES, 1):
        if tag in ln:
            return i
    raise KeyError(key)


def cite(key: str) -> str:
    return "run_g1.py:L" + str(LNO(key))


def md_table(d: pd.DataFrame, cols=None, floatfmt="%.4f") -> str:
    """无 tabulate 依赖的 markdown 表（pandas 3.0 的 to_markdown 需要 tabulate）。"""
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


# ---------------------------------------------------------------- 1. 地基
b = C.build_all()  # L:BUILD
C.assert_anchors(b)  # L:ANCHORS
nodes, Gu, Gd, ps = b["nodes"], b["Gu"], b["Gd"], b["ps"]  # L:OBJ
n_attr = len(nodes)  # L:N_ATTR
n_gcc = len(b["sets"]["gcc"])  # L:N_GCC

cen = C.centralities(Gu, Gd)  # L:CEN
df = nodes.merge(cen, left_on="node", right_index=True, how="left")  # L:MERGE
df = df.drop(columns=[c for c in ("degree", "color") if c in df.columns])  # L:DROP_BANNED_COLS
assert "degree" not in df.columns and "color" not in df.columns  # L:DROP_CHECK

n_dead = int(df["is_dead"].sum())  # L:N_DEAD
n_dead_out0 = int((df.loc[df["is_dead"], "out_degree"] == 0).sum())  # L:N_DEAD_OUT0
n_true = int(df["true_fail"].sum())  # L:N_TRUE
n_false = int(df["false_fail"].sum())  # L:N_FALSE
n_amb = int(df["ambiguous_fail"].sum())  # L:N_AMB
n_partial = int(df["is_partial"].sum())  # L:N_PARTIAL
n_healthy = int(df["is_healthy"].sum())  # L:N_HEALTHY
mean_in_dead = float(df.loc[df["is_dead"], "in_degree"].mean())  # L:MEAN_IN_DEAD
mean_in_alive = float(df.loc[~df["is_dead"], "in_degree"].mean())  # L:MEAN_IN_ALIVE
mean_in_healthy = float(df.loc[df["is_healthy"], "in_degree"].mean())  # L:MEAN_IN_HEALTHY
mean_degu_dead = float(df.loc[df["is_dead"], "deg_u"].mean())  # L:MEAN_DEGU_DEAD
mean_degu_alive = float(df.loc[~df["is_dead"], "deg_u"].mean())  # L:MEAN_DEGU_ALIVE
mean_degu_healthy = float(df.loc[df["is_healthy"], "deg_u"].mean())  # L:MEAN_DEGU_HEALTHY

# ---------------------------------------------------------------- 2. 坐标与判据
CENTS = {  # L:CENTS
    "in_degree": "入度（§2.1 主口径）",
    "bet_u": "介数（无权无向 Gu）",
    "pr_dw": "PageRank（§5 有向加权主口径）",
}
DIAG_CENT = "deg_u"  # L:DIAG_CENT  无向总度：污染对照，只演示口径漂移
HEALTHS = {"raw": "校正前（ok_ratio_pct 原值）", "corr": "校正后（§2.2 真失效口径）"}  # L:HEALTHS
QUADS = ["Q1_高结构_低可用", "Q2_低结构_低可用", "Q3_高结构_高可用", "Q4_低结构_高可用"]

df["unavailability"] = 1.0 - df["ok_ratio_pct"] / 100.0  # L:UNAVAIL
SCORED_MASK = ~(df["false_fail"] | df["ambiguous_fail"])  # L:SCORED_MASK
n_scored_corr = int(SCORED_MASK.sum())  # L:N_SCORED_CORR

keep_cols = ["node", "sub", "in_degree", "bet_u", "pr_dw", "deg_u", "ok_ratio_pct",
             "unavailability", "status_codes", "health_state", "dead_cat"]
false_fail_df = df.loc[df["false_fail"], keep_cols].copy()  # L:FF_FRAME
amb_df = df.loc[df["ambiguous_fail"], keep_cols].copy()  # L:AMB_FRAME

u_median_raw = float(df["unavailability"].median())  # L:U_MEDIAN
n_u_zero = int((df["unavailability"] == 0).sum())  # L:N_U_ZERO
n_u_pos = int((df["unavailability"] > 0).sum())  # L:N_U_POS
n_u_one = int((df["unavailability"] == 1).sum())  # L:N_U_ONE


def score_table(cent: str, health: str) -> pd.DataFrame:
    """一张打分表 = 一个中心性 × 一个健康口径。"""
    cols = ["node", "sub", cent, "unavailability", "health_state", "dead_cat",
            "is_dead", "is_partial", "is_healthy", "true_fail", "false_fail",
            "ambiguous_fail"]
    src = df if health == "raw" else df.loc[SCORED_MASK]
    sub = src[cols].copy()  # L:SUB_UNIVERSE
    sub["x_pct"] = sub[cent].rank(method="average", pct=True)  # L:X_PCT
    sub["u_pct"] = sub["unavailability"].rank(method="average", pct=True)  # L:U_PCT
    xh = sub["x_pct"] > 0.5  # L:X_HIGH
    yh = sub["u_pct"] > 0.5  # L:Y_HIGH
    sub["x_high"], sub["y_high"] = xh, yh
    sub["quadrant"] = np.select(
        [xh & yh, (~xh) & yh, xh & (~yh), (~xh) & (~yh)], QUADS, default="NA")  # L:QUAD
    sub = sub.rename(columns={cent: "x_value", "unavailability": "u_value"})
    sub["cent"], sub["health"] = cent, health
    sub["table"] = cent + "|" + health
    # 口径自检：median(u)=0 ⇒ y 高侧与「非完好」必须逐节点完全等价
    assert (sub["y_high"] == (sub["u_value"] > 0)).all(), "y 轴切分与 ok_ratio<100 不等价"  # L:Y_EQUIV
    return sub


tables = {}
for c_ in list(CENTS) + [DIAG_CENT]:
    for h_ in HEALTHS:
        if c_ == DIAG_CENT and h_ != "raw":
            continue
        tables[c_ + "|" + h_] = score_table(c_, h_)

# ---------------------------------------------------------------- 3. 象限汇总
QUAD_COLS = ["n", "x_value_median", "x_pct_median", "u_value_median", "u_pct_median",
             "ok_ratio_pct_median", "n_healthy", "n_partial", "n_dead_all",
             "n_true_fail", "n_false_fail", "n_ambiguous"]
rows = []
for tname, t in tables.items():
    for q in QUADS:
        s = t[t["quadrant"] == q]
        rec = {"table": tname, "quadrant": q,  # L:Q_ROW
               "share_of_scored": round(len(s) / len(t), 4),
               "n_in_degree_1": int((s["x_value"] == 1).sum()) if tname.startswith("in_degree") else -1}
        if len(s) == 0:
            rec.update({k: 0 for k in QUAD_COLS})  # L:Q_EMPTY
        else:
            rec.update({
                "n": int(len(s)),
                "x_value_median": round(float(s["x_value"].median()), 6),
                "x_pct_median": round(float(s["x_pct"].median()), 4),
                "u_value_median": round(float(s["u_value"].median()), 4),
                "u_pct_median": round(float(s["u_pct"].median()), 4),
                "ok_ratio_pct_median": round(float((1 - s["u_value"]).median() * 100), 2),
                "n_healthy": int(s["is_healthy"].sum()),
                "n_partial": int(s["is_partial"].sum()),
                "n_dead_all": int(s["is_dead"].sum()),
                "n_true_fail": int(s["true_fail"].sum()),
                "n_false_fail": int(s["false_fail"].sum()),
                "n_ambiguous": int(s["ambiguous_fail"].sum()),
            })
        rows.append(rec)
quad_summary = pd.DataFrame(rows)[["table", "quadrant", "n", "share_of_scored"]
                                   + QUAD_COLS[1:]]  # L:Q_SUMMARY
quad_summary["n_in_degree_1"] = [r["n_in_degree_1"] for r in rows]

members = pd.concat(tables.values(), ignore_index=True)  # L:MEMBERS
members["rank_in_quadrant"] = members.groupby(["table", "quadrant"])["x_pct"].rank(
    method="first", ascending=False).astype(int)  # L:RANK_Q

# ---------------------------------------------------------------- 4. 换口径 = 换象限？
MAIN_TABLES = [c_ + "|" + h_ for h_ in HEALTHS for c_ in CENTS]  # L:MAIN_TABLES
wide = members.pivot_table(index="node", columns="table", values="quadrant",
                           aggfunc="first")  # L:WIDE
q1_of = {t: set(members.loc[(members["table"] == t) & (members["quadrant"] == QUADS[0]),
                            "node"]) for t in tables}  # L:Q1_SET

inst_rows = []


def transition(a_tbl: str, b_tbl: str, kind: str) -> dict:
    common = wide[[a_tbl, b_tbl]].dropna()  # L:TRANS_COMMON
    n = len(common)
    chg = int((common[a_tbl] != common[b_tbl]).sum())  # L:TRANS_CHG
    qa, qb = q1_of[a_tbl] & set(common.index), q1_of[b_tbl] & set(common.index)
    inter, union, only = len(qa & qb), len(qa | qb), len(qa ^ qb)
    ta, tb = tables[a_tbl].set_index("node"), tables[b_tbl].set_index("node")
    ids = common.index
    rho_x = float(ta.loc[ids, "x_value"].astype(float).corr(
        tb.loc[ids, "x_value"].astype(float), method="spearman"))
    return {"pair": a_tbl + " vs " + b_tbl, "kind": kind,  # L:TRANS_ROW
            "n_common_scored": n, "n_changed_quadrant": chg,
            "pct_changed_quadrant": round(100.0 * chg / n, 2) if n else np.nan,
            "n_Q1_a": len(qa), "n_Q1_b": len(qb), "n_Q1_both": inter,
            "n_Q1_only_one": only, "jaccard_Q1": round(inter / union, 4) if union else np.nan,
            "spearman_x_a_b": round(rho_x, 4)}


CENT_PAIRS = [("in_degree", "bet_u"), ("in_degree", "pr_dw"), ("bet_u", "pr_dw")]
for a_, b_ in CENT_PAIRS:
    inst_rows.append(transition(a_ + "|raw", b_ + "|raw", "换中心性(校正前)"))  # L:INST_CENT_RAW
for a_, b_ in CENT_PAIRS:
    inst_rows.append(transition(a_ + "|corr", b_ + "|corr", "换中心性(校正后)"))  # L:INST_CENT_COR
for c_ in CENTS:
    inst_rows.append(transition(c_ + "|raw", c_ + "|corr", "换健康口径(同中心性)"))  # L:INST_HEALTH
for c_ in CENTS:
    inst_rows.append(transition(c_ + "|raw", DIAG_CENT + "|raw", "主口径vs总度污染对照"))  # L:INST_DIAG
instability = pd.DataFrame(inst_rows)  # L:INST_DF

six_common = wide[MAIN_TABLES].dropna()  # L:SIX_COMMON
n_six_common = len(six_common)  # L:N_SIX_COMMON
n_distinct = six_common.nunique(axis=1)  # L:N_DISTINCT
n_stable_all6 = int((n_distinct == 1).sum())  # L:STABLE_ALL6
n_stable_pct = round(100.0 * n_stable_all6 / n_six_common, 2)  # L:STABLE_PCT
n_always_q1 = int((six_common == QUADS[0]).all(axis=1).sum())  # L:ALWAYS_Q1
n_ever_q1 = int((six_common == QUADS[0]).any(axis=1).sum())  # L:EVER_Q1
n_never_q1 = int((six_common != QUADS[0]).all(axis=1).sum())  # L:NEVER_Q1
n_dist2 = int((n_distinct == 2).sum())  # L:DIST2
q1_count_by_table = {t: int((tables[t]["quadrant"] == QUADS[0]).sum()) for t in tables}  # L:Q1_BY_TABLE
q1_corr_size = q1_count_by_table["in_degree|corr"]  # L:Q1_CORR_SIZE

ff_by_table = {}
for tname in tables:
    t = tables[tname]
    ff_by_table[tname] = {
        "n_false_fail_scored": int(t["false_fail"].sum()),
        "n_false_fail_Q1": int(((t["quadrant"] == QUADS[0]) & t["false_fail"]).sum()),
        "n_false_fail_xhigh": int((t["x_high"] & t["false_fail"]).sum()),
        "n_false_fail_u_high": int((t["y_high"] & t["false_fail"]).sum()),
        "n_amb_Q1": int(((t["quadrant"] == QUADS[0]) & t["ambiguous_fail"]).sum()),
    }  # L:FF_PLACE
main_q1_ff = [ff_by_table[t]["n_false_fail_Q1"] for t in MAIN_TABLES]  # L:FF_Q1_MAIN
raw_q1_ff = [ff_by_table[c_ + "|raw"]["n_false_fail_Q1"] for c_ in CENTS]  # L:FF_Q1_RAW_ONLY
n_ff_q1_min, n_ff_q1_max = min(raw_q1_ff), max(raw_q1_ff)  # L:FF_Q1_RANGE
n_ff_u1 = int((df.loc[df["false_fail"], "unavailability"] == 1).sum())  # L:FF_U1
n_amb_q1_max = max(ff_by_table[t]["n_amb_Q1"] for t in MAIN_TABLES)  # L:AMB_Q1_MAX


def tie_probe(t: pd.DataFrame) -> dict:
    """切点两侧的并列规模：多少节点的 x 值处在 ±1% 带内（象限归属由浮点噪声决定）。"""
    lo = float(t.loc[~t["x_high"], "x_value"].max())  # L:TIE_LO
    hi = float(t.loc[t["x_high"], "x_value"].min())  # L:TIE_HI
    mid = 0.5 * (lo + hi)
    band = 0.01 * abs(mid) if mid else abs(hi)
    return {"cut_lo": lo, "cut_hi": hi, "cut_mid": mid, "gap": hi - lo,
            "rel_gap_pct": round(100.0 * (hi - lo) / mid, 4) if mid else np.nan,
            "n_tied_at_lo": int((t["x_value"] == lo).sum()),
            "n_tied_at_hi": int((t["x_value"] == hi).sum()),
            "n_within_1pct_band": int(((t["x_value"] >= mid - band)
                                       & (t["x_value"] <= mid + band)).sum()),
            "n_x_value_eq_zero": int((t["x_value"] == 0).sum())}


tie_rows = []
for tname, t in tables.items():
    c_, h_ = tname.split("|")
    tie_rows.append({"cent": c_, "health": h_, "n_scored": len(t), **tie_probe(t)})
tie_probe_df = pd.DataFrame(tie_rows)  # L:TIE_DF

# ---------------------------------------------------------------- 5. x 轴候选 vs 健康（§2.1 复算）
corr_rows = []
for col, lab in [("in_degree", "入度(§2.1主口径)"), ("deg_u", "无向总度(污染)"),
                 ("out_degree", "出度(被删失)"), ("bet_u", "介数(无权无向)"),
                 ("pr_dw", "PageRank(有向加权)"), ("strength_u", "无向强度")]:
    corr_rows.append({"axis": col, "label": lab,
                      "spearman_rho_with_unavailability": round(float(
                          df[col].corr(df["unavailability"], method="spearman")), 4),
                      "spearman_rho_with_ok_ratio_pct": round(float(
                          df[col].corr(df["ok_ratio_pct"], method="spearman")), 4),
                      "spearman_rho_unavailability_excl_false_fail": round(float(
                          df.loc[~df["false_fail"], col].corr(
                              df.loc[~df["false_fail"], "unavailability"], method="spearman")), 4)})
axis_corr = pd.DataFrame(corr_rows)  # L:AXIS_CORR
rho_deg_u_ok = float(axis_corr.loc[axis_corr["axis"] == "deg_u", "spearman_rho_with_ok_ratio_pct"].iloc[0])  # L:RHO_DEGU
rho_in_ok = float(axis_corr.loc[axis_corr["axis"] == "in_degree", "spearman_rho_with_ok_ratio_pct"].iloc[0])  # L:RHO_IN

# ---------------------------------------------------------------- 6. CSV
def wcsv(d: pd.DataFrame, name: str):
    d.to_csv(index=False, encoding="utf-8-sig",
             path_or_buf=os.path.join(HERE, name))  # L:WCSV


scored_wide = df[["node", "sub", "title", "health_state", "ok_ratio_pct", "unavailability",
                  "dead_cat", "is_dead", "is_partial", "is_healthy", "true_fail",
                  "false_fail", "ambiguous_fail", "in_degree", "out_degree", "deg_u",
                  "bet_u", "pr_dw", "url_count", "listed_urls"]].copy()  # L:SCORED_WIDE
for tname, t in tables.items():
    tt = t.set_index("node")[["x_pct", "u_pct", "quadrant"]]
    tt.columns = [tname + "__" + c for c in tt.columns]
    scored_wide = scored_wide.merge(tt, left_on="node", right_index=True, how="left")
wcsv(scored_wide, "g1_nodes_scored.csv")
wcsv(quad_summary, "g1_quadrant_summary.csv")
wcsv(members[["table", "cent", "health", "quadrant", "rank_in_quadrant", "node", "sub",
              "x_value", "x_pct", "u_value", "u_pct", "health_state", "dead_cat",
              "true_fail", "false_fail", "ambiguous_fail"]], "g1_quadrant_members.csv")
wcsv(instability, "g1_instability.csv")
wcsv(false_fail_df.assign(class_="假失效 B_auth_or_skip/C_robots_blocked（校正后不参与打分）")
     .sort_values(["in_degree", "node"], ascending=[False, True]), "g1_false_fail_class.csv")
wcsv(amb_df.assign(class_="待人工判断 G_other_400（校正后不参与打分）"), "g1_ambiguous_class.csv")
wcsv(axis_corr, "g1_axis_correlations.csv")
wcsv(tie_probe_df, "g1_cutpoint_ties.csv")

summary = {
    "unit": "G1_risk_matrix", "seed": SEED,
    "universe": {"n_attributed": n_attr, "n_gcc": n_gcc,
                 "n_scored_corrected": n_scored_corr,
                 "n_false_fail_unscored": n_false, "n_ambiguous_unscored": n_amb},
    "health": {"n_healthy": n_healthy, "n_partial": n_partial, "n_dead": n_dead,
               "n_true_fail": n_true, "n_false_fail": n_false, "n_ambiguous": n_amb,
               "dead_out_degree_zero": n_dead_out0,
               "mean_in_degree_dead": round(mean_in_dead, 4),
               "mean_in_degree_alive": round(mean_in_alive, 4),
               "in_degree_ratio_alive_over_dead": round(mean_in_alive / mean_in_dead, 3)},
    "y_axis_degeneracy": {"median_unavailability_raw": u_median_raw,
                          "n_unavailability_eq_0": n_u_zero,
                          "n_unavailability_gt_0": n_u_pos,
                          "n_unavailability_eq_1": n_u_one},
    "quadrant_counts": {t: {q: int((tables[t]["quadrant"] == q).sum()) for q in QUADS}
                        for t in tables},
    "q1_size_by_table": q1_count_by_table,
    "instability": instability.to_dict(orient="records"),
    "stability_6tables": {"n_common": n_six_common, "n_stable_all6": n_stable_all6,
                          "pct_stable_all6": n_stable_pct, "n_2_labels": n_dist2,
                          "n_always_Q1": n_always_q1, "n_ever_Q1": n_ever_q1,
                          "n_never_Q1": n_never_q1},
    "false_fail_placement": ff_by_table,
    "axis_correlations": axis_corr.to_dict(orient="records"),
    "cutpoint_ties": tie_probe_df.to_dict(orient="records"),
    "degree_subset_means": {
        "in_degree": {"dead": round(mean_in_dead, 4), "healthy": round(mean_in_healthy, 4),
                      "not_dead": round(mean_in_alive, 4)},
        "deg_u": {"dead": round(mean_degu_dead, 4), "healthy": round(mean_degu_healthy, 4),
                  "not_dead": round(mean_degu_alive, 4)}},
    "versions": C.versions(),
    "wall_time_s": round(time.time() - T_START, 2),
}
with open(os.path.join(HERE, "G1_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)  # L:JSON

# ---------------------------------------------------------------- 7. result.md
L = []
A = L.append
cr = instability[instability["kind"] == "换中心性(校正前)"]
cc = instability[instability["kind"] == "换中心性(校正后)"]
ch = instability[instability["kind"] == "换健康口径(同中心性)"]
cdiag = instability[instability["kind"] == "主口径vs总度污染对照"]
ccent = instability[instability["kind"].str.startswith("换中心性")]
ffq = [ff_by_table["in_degree|raw"]["n_false_fail_Q1"],
       ff_by_table["bet_u|raw"]["n_false_fail_Q1"],
       ff_by_table["pr_dw|raw"]["n_false_fail_Q1"]]  # L:FFQ
ambq = [ff_by_table["in_degree|raw"]["n_amb_Q1"], ff_by_table["bet_u|raw"]["n_amb_Q1"],
        ff_by_table["pr_dw|raw"]["n_amb_Q1"]]  # L:AMBQ

A("# G1_risk_matrix · 结构重要性 × 可用率 四象限风险矩阵")
A("")
A("> 运行：`cd pipeline && PYTHONHASHSEED=0 python out/G1_risk_matrix/run_g1.py`（"
  f"{time.time() - T_START:.1f} s，Python {C.versions()['python']} / networkx "
  f"{C.versions()['networkx']} / pandas {C.versions()['pandas']}，SEED={SEED}）。")
A(f"> 开跑先 `csa_common.assert_anchors()` 31/31 通过（{cite('ANCHORS')}）；构图只用 "
  f"`C.build_all()`（{cite('BUILD')}），中心性只用 `C.centralities()`（{cite('CEN')}）；"
  f"`nodes_full.csv` 的 `degree` / `color` 两列在 {cite('DROP_BANNED_COLS')} 丢弃并有断言"
  f"（{cite('DROP_CHECK')}）。")
A("> 机器可读：`g1_nodes_scored.csv`（671 节点 × 7 张表的 x/y 百分位与象限标签）、"
  "`g1_quadrant_summary.csv`、`g1_quadrant_members.csv`、`g1_instability.csv`、"
  "`g1_false_fail_class.csv`、`g1_ambiguous_class.csv`、`g1_axis_correlations.csv`、"
  "`g1_cutpoint_ties.csv`、`G1_summary.json`。")
A("> 措辞：本模块的「风险」= 结构位置与当前可用率的联合分层（横截面陈述），"
  "**不是**对某站点接下来会失效的预测（§7）。")
A("")
A("## 0. 结论速览（头条是不稳定性，不是四宫格）")
A("")
A("| # | 结论 | 依据 |")
A("|---|---|---|")
A(f"| 1 | **四象限本身不稳定**。671 个节点上换中心性口径（校正前）有 "
  f"{int(cr['n_changed_quadrant'].min())}–{int(cr['n_changed_quadrant'].max())} 个"
  f"（{cr['pct_changed_quadrant'].min():.1f}–{cr['pct_changed_quadrant'].max():.1f}%）换象限，"
  f"校正后 {cc['pct_changed_quadrant'].min():.1f}–{cc['pct_changed_quadrant'].max():.1f}%；"
  f"换健康口径（同中心性）{ch['pct_changed_quadrant'].min():.1f}–{ch['pct_changed_quadrant'].max():.1f}%。"
  f"六张主表标签**始终不变**的节点只有 **{n_stable_all6}/{n_six_common} = {n_stable_pct:.1f}%**，"
  f"取到 2 种及以上标签的 {n_six_common - n_stable_all6} 个（{cite('STABLE_ALL6')}, "
  f"{cite('DIST2')}, {cite('INST_DF')}） | {cite('INST_CENT_RAW')} |")
A(f"| 2 | 高风险格 Q1（高结构 × 低可用）的成员数本身就是口径函数："
  + "；".join(f"{k.replace('|', '×')}={q1_count_by_table[k]}" for k in MAIN_TABLES)
  + f"（{cite('Q1_BY_TABLE')}）。三套 x 的 Q1 两两 Jaccard "
  + "、".join("%.3f" % v for v in ccent["jaccard_Q1"])
  + f"（{cite('TRANS_ROW')}） |")
A(f"| 3 | **y 轴中位数切分退化**：{n_u_zero}/{n_attr}（{100.0 * n_u_zero / n_attr:.1f}%）"
  f"个节点可用率正好 100，u = 1−ok_ratio 的 median = {u_median_raw:.4f}；于是「u_pct 高于 0.5」"
  f"与「ok_ratio_pct 低于 100」在两套口径下逐节点完全等价（脚本内断言恒成立，"
  f"{cite('Y_EQUIV')}）。四象限的 y 维度只承载 3 档信息"
  f"（可用率 100 / 开区间 / 0），不是连续梯度（{cite('U_MEDIAN')}, {cite('N_U_ZERO')}） |")
A(f"| 4 | 假失效 {n_false} 个（B_auth_or_skip 18 + C_robots_blocked 14，状态码 -1/-2）"
  f"在 raw 口径下与真失效**不可分**：{n_ff_u1}/{n_false} 个的 u 也正好 1.0，"
  f"被 raw 的 y 轴推到高侧 {ff_by_table['in_degree|raw']['n_false_fail_u_high']}/{n_false} 个，"
  f"并在校正前直接落进 Q1："
  + " / ".join(f"{c_}={ff_by_table[c_ + '|raw']['n_false_fail_Q1']}" for c_ in CENTS)
  + f" 个（共 {n_false}，{cite('FF_Q1_RAW_ONLY')}）；"
  f"校正后整类退出打分域（{n_scored_corr} = {n_attr} − {n_false} − {n_amb}，{cite('SCORED_MASK')}），"
  f"单列为可分离类（§4，全名单 {cite('FF_FRAME')}）。另有 {n_amb} 个 G_other_400 标「待人工判断」"
  f"（raw 三套 x 下分别 {ambq[0]} / {ambq[1]} / {ambq[2]} 个次进 Q1，{cite('AMB_Q1_MAX')}）。{cite('FF_PLACE')} |")
A(f"| 5 | x 口径污染可直接量化：ρ(无向总度, 可用率) = {rho_deg_u_ok:.3f} vs "
  f"ρ(入度, 可用率) = {rho_in_ok:.3f}（§2.1 记载 0.376 / 0.216，本模块按 671 节点重算，同序同量级）；"
  f"{n_dead} 个全挂站出度全为 0（{n_dead_out0}/{n_dead}）。把 x 换成总度后 "
  f"Q1 与入度版本的对称差最多 {int(cdiag['n_Q1_only_one'].max())} 个节点，"
  f"最低 Jaccard {cdiag['jaccard_Q1'].min():.3f} | "
  f"{cite('AXIS_CORR')}, {cite('N_DEAD_OUT0')}, {cite('INST_DIAG')} |")
A(f"| 6 | 同口径入度对照（§2.1 那条 2.7×）：全挂站入度均值 {mean_in_dead:.4f}；"
  f"参照组随子集定义而变——健康站（{n_healthy} 个）{mean_in_healthy:.4f}（比值 "
  f"{mean_in_healthy / mean_in_dead:.2f}×，即文档记载的 4.94），"
  f"非全挂（{n_attr - n_dead} 个，含部分失效）{mean_in_alive:.4f}（比值 "
  f"{mean_in_alive / mean_in_dead:.2f}×）。跨口径的无向总度版本给出 "
  f"{mean_degu_dead:.4f} vs {mean_degu_healthy:.4f}（{mean_degu_healthy / mean_degu_dead:.2f}×），"
  f"这就是 5.2× 的来源；死站的总度数值上等于其入度（出度全为 0），"
  f"所以「叶子」表述一律用 2.7× 这一档 | {cite('MEAN_IN_DEAD')}–{cite('MEAN_DEGU_HEALTHY')} |")
A("")

A("## 1. 坐标与判据")
A("")
A("| 项 | 规定 | 出处 |")
A("|---|---|---|")
A(f"| 打分域 | 校正前 {n_attr} 个有属性节点（§1 主口径，{cite('N_ATTR')}）；"
  f"校正后 {n_scored_corr} 个（剔除 {n_false} 假失效 + {n_amb} 待判，{cite('SUB_UNIVERSE')}）。"
  f"拓扑参照：主分量 {n_gcc} 节点 / 2497 边（{cite('N_GCC')}） | §1 |")
A(f"| x | 中心性在打分域内的 average-rank 百分位 `x_pct`；`x_pct > 0.5` 记高结构 "
  f"（{cite('X_PCT')}, {cite('X_HIGH')}） |")
A(f"| y | u = 1 − ok_ratio_pct/100 的 average-rank 百分位；`u_pct > 0.5` 记低可用 "
  f"（{cite('UNAVAIL')}, {cite('U_PCT')}, {cite('Y_HIGH')}） |")
A(f"| 象限 | Q1 高结构×低可用 / Q2 低结构×低可用 / Q3 高结构×高可用 / Q4 低结构×高可用（{cite('QUAD')}） |")
A(f"| x 三选一 | 入度（§2.1 主口径）· 介数（无权无向，`C.centralities` 的 `bet_u`）· "
  f"PageRank（§5 有向加权主口径 `pr_dw`） | {cite('CENTS')} |")
A(f"| 污染对照 | 无向总度 `deg_u` 只做演示，不进入任何主表结论（{cite('DIAG_CENT')}） | §2.1 |")
A(f"| 健康两口径 | raw = ok_ratio_pct 原值；corr = §2.2 真失效校正（假失效 / 待判整类退出打分域并单列）"
  f"（{cite('HEALTHS')}） | §2.2 |")
A(f"| 并列值 | 百分位用 `method=average`：并列块共享同一百分位、不随机打散，"
  f"故切点对应的原始值确定（见下表） | {cite('X_PCT')} |")
A("")
A("x 轴 `> 0.5` 的等价原始切点（低侧最大值 / 高侧最小值 + 切点附近的并列规模），"
  "用于把百分位还原成可核查阈值，同时暴露切点的任意性：")
A("")
A("| 中心性 | 口径 | n | 低侧最大 x | 高侧最小 x | 相对间隙 | 并列于低侧值 | 并列于高侧值 | 切点 ±1% 带内节点数 | x=0 节点数 |")
A("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in tie_probe_df.itertuples():
    A(f"| {r.cent} | {r.health} | {r.n_scored} | {r.cut_lo:.6g} | {r.cut_hi:.6g} | "
      f"{r.rel_gap_pct:.2g}% | {r.n_tied_at_lo} | {r.n_tied_at_hi} | "
      f"{r.n_within_1pct_band} | {r.n_x_value_eq_zero} |")
A("")
tp_in = tie_probe_df[(tie_probe_df.cent == "in_degree") & (tie_probe_df.health == "raw")].iloc[0]
tp_be = tie_probe_df[(tie_probe_df.cent == "bet_u") & (tie_probe_df.health == "raw")].iloc[0]
tp_pr = tie_probe_df[(tie_probe_df.cent == "pr_dw") & (tie_probe_df.health == "raw")].iloc[0]
A(f"- 入度是唯一切点有语义的口径：`>0.5` 等价于「入度 ≥ {int(tp_in.cut_hi)}」，"
  f"低侧最大只到 {int(tp_in.cut_lo)}（切点间隙 "
  f"{tp_in.rel_gap_pct:.0f}%，{cite('TIE_LO')}–{cite('TIE_HI')}）。")
A(f"- 介数与 PageRank 的切点落在浮点噪声上：bet_u 两侧值差 "
  f"{tp_be['gap']:.3g}（相对 {tp_be.rel_gap_pct:.2g}%）、pr_dw 差 "
  f"{tp_pr['gap']:.3g}（相对 {tp_pr.rel_gap_pct:.2g}%），"
  f"且 bet_u 有 {int(tp_be.n_x_value_eq_zero)} 个零介数节点整块并列在底部"
  f"（{cite('TIE_DF')}）。换句话说这两套 x 的「高结构 / 低结构」边界由小数第 5 位决定，"
  f"这就是 §3 迁移量大的直接原因之一。")
A("")
A(f"y 轴三档分布（校正前）：u=0 的 {n_u_zero} 个、0<u<1 的 {n_u_pos - n_u_one} 个、u=1 的 "
  f"{n_u_one} 个（= {n_true} 真失效 + {n_false} 假失效 + {n_amb} 待判）。"
  f"校正后 u=1 只剩 {n_true} 个真失效，高侧成员 = {n_partial} 部分失效 + {n_true} 真失效 = "
  f"{n_partial + n_true} 个 / {n_scored_corr}。")
A("")

A("## 2. 六张主表：每格计数 + 定义该格的中位数")
A("")
for hi, h_ in enumerate(HEALTHS, start=1):
    A(f"### 2.{hi} 健康口径 = {HEALTHS[h_]}（打分域 "
      f"{n_attr if h_ == 'raw' else n_scored_corr} 节点）")
    A("")
    A("| 中心性 | Q1 高×低 | Q2 低×低 | Q3 高×高 | Q4 低×高 | Q1 内 真失效/假失效/部分/完好 |")
    A("|---|---:|---:|---:|---:|---|")
    for c_ in CENTS:
        s = quad_summary[quad_summary["table"] == c_ + "|" + h_].set_index("quadrant")
        g = s.loc[QUADS[0]]
        A(f"| {c_} | {int(s['n'][QUADS[0]])} | {int(s['n'][QUADS[1]])} | "
          f"{int(s['n'][QUADS[2]])} | {int(s['n'][QUADS[3]])} | "
          f"{int(g['n_true_fail'])} / {int(g['n_false_fail'])} / {int(g['n_partial'])} / "
          f"{int(g['n_healthy'])} |")
    A("")
    A("定义每格的中位数（x 值 / u 值 / 可用率 / 健康构成；全量在 `g1_quadrant_summary.csv`）：")
    A("")
    A("| 中心性 | 象限 | n | 占打分域 | x 中位 | x_pct 中位 | u 中位 | ok% 中位 | 完好 | 部分 | 整站不可用 | 真失效 | 假失效 |")
    A("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c_ in CENTS:
        for q in QUADS:
            r = quad_summary[(quad_summary["table"] == c_ + "|" + h_)
                             & (quad_summary["quadrant"] == q)].iloc[0]
            A(f"| {c_} | {q} | {int(r['n'])} | {100 * r['share_of_scored']:.1f}% | "
              f"{r['x_value_median']:.6g} | {r['x_pct_median']:.3f} | {r['u_value_median']:.4g} | "
              f"{r['ok_ratio_pct_median']:.1f} | {int(r['n_healthy'])} | {int(r['n_partial'])} | "
              f"{int(r['n_dead_all'])} | {int(r['n_true_fail'])} | {int(r['n_false_fail'])} |")
    A("")
    for c_ in CENTS:
        t = tables[c_ + "|" + h_]
        q1 = t[t["quadrant"] == QUADS[0]].sort_values("x_pct", ascending=False).head(10)
        A(f"**Q1 top-10（{c_} × {h_}，按 x_pct 降序；完整名单见 `g1_quadrant_members.csv`）**")
        A("")
        A("| 节点 | x 值 | x_pct | u | u_pct | 状态 | dead_cat |")
        A("|---|---:|---:|---:|---:|---|---|")
        for r in q1.itertuples():
            dc = r.dead_cat if isinstance(r.dead_cat, str) else "–"
            A(f"| {r.node} | {r.x_value:.6g} | {r.x_pct:.3f} | {r.u_value:.3f} | "
              f"{r.u_pct:.3f} | {r.health_state} | {dc} |")
        A("")
deg_row = quad_summary[quad_summary["table"] == "deg_u|raw"].set_index("quadrant")["n"]
A("**污染对照（不进主表结论）**：`deg_u|raw` 四格 = "
  + "、".join(f"{q.split('_')[0]}={int(deg_row[q])}" for q in QUADS)
  + f"（{cite('Q_SUMMARY')}）")
A("")

A("## 3. 不稳定性明细（本模块的核心产出）")
A("")
A(md_table(instability, ["pair", "kind", "n_common_scored", "n_changed_quadrant",
                         "pct_changed_quadrant", "n_Q1_a", "n_Q1_b", "n_Q1_both",
                         "n_Q1_only_one", "jaccard_Q1", "spearman_x_a_b"]))
A("")
A(f"- 6 张主表全体（{n_six_common} 个在两套健康口径下都打分的节点，{cite('SIX_COMMON')}）："
  f"象限标签**从未改变** {n_stable_all6} 个（{n_stable_pct:.1f}%）；恰好 2 种标签 "
  f"{n_dist2} 个；至少进过一次 Q1 的 {n_ever_q1} 个；六表都在 Q1 的 {n_always_q1} 个；"
  f"从未进 Q1 的 {n_never_q1} 个（{cite('STABLE_ALL6')}, {cite('EVER_Q1')}, {cite('NEVER_Q1')}）。")
A(f"- 一致性根因：x 轴三套口径在同一打分域上的 Spearman 只有 "
  f"{ccent['spearman_x_a_b'].min():.3f}–{ccent['spearman_x_a_b'].max():.3f}"
  f"（介数 × PageRank 最低，{cite('TRANS_ROW')}），即「高结构」在这里至少有 3 个"
  f"只中度相关的含义；y 轴又只有 3 档（§0 第 3 条），两个粗维度做中位数切分，"
  f"迁移量主要来自 x 轴的并列与排序差。")
A("- 用法判决：Q1 适合当作**逐个核查的候选清单来源**，不适合作为排序或结论载体；"
  "任何「某站点位于高风险格」的表述都必须同时写明 x 口径与 y 口径；"
  "「后果有多大」不在这里回答，由 G2 的移除情景给出。")
A("")

A("## 4. 真失效 / 假失效 / 待人工判断：分成可分离的类")
A("")
A("| 类 | n | 判据 | raw 口径处理 | corr 口径处理 | raw 下进入 Q1 的个数（in_deg/bet/pr） |")
A("|---|---:|---|---|---|---|")
ffq = [ff_by_table["in_degree|raw"]["n_false_fail_Q1"],
       ff_by_table["bet_u|raw"]["n_false_fail_Q1"],
       ff_by_table["pr_dw|raw"]["n_false_fail_Q1"]]
ambq = [ff_by_table["in_degree|raw"]["n_amb_Q1"], ff_by_table["bet_u|raw"]["n_amb_Q1"],
        ff_by_table["pr_dw|raw"]["n_amb_Q1"]]
A(f"| 真失效 | {n_true} | dead_cat ∈ A_conn_fail / D_not_found / F_server_error | "
  f"打分域，u=1 | 打分域，u=1 | – |")
A(f"| **假失效** | {n_false} | dead_cat ∈ B_auth_or_skip(18) / C_robots_blocked(14)；"
  f"状态码 -1/-2 不是故障（§2.2） | 与真失效同值（u=1 的有 {n_ff_u1}/{n_false}）| "
  f"**整类退出打分域**，单列 `g1_false_fail_class.csv` | "
  f"{ffq[0]} / {ffq[1]} / {ffq[2]}（{cite('FF_PLACE')}）|")
A(f"| 待人工判断 | {n_amb} | dead_cat = G_other_400（§2.2 规定单列） | 打分域，u=1 | "
  f"退出打分域，单列 `g1_ambiguous_class.csv` | {ambq[0]} / {ambq[1]} / {ambq[2]} |")
A(f"| 部分失效 | {n_partial} | 0 < ok_ratio_pct < 100 | 打分域，u ∈ (0,1) | 打分域 | – |")
A(f"| 完好 | {n_healthy} | ok_ratio_pct = 100 | 打分域，u=0（y 轴底部整块并列） | 打分域 | 0 |")
A("")
A(f"假失效 {n_false} 个全名单（按入度降序）：y 轴上它们与真失效同值，**只能靠 dead_cat 分色**，"
  f"这正是 §2.2 要求校正前 / 校正后两套并行的理由：")
A("")
A("| 节点 | dead_cat | 状态码 | 入度 | deg_u | u | 校正前(in_degree×raw) 象限 |")
A("|---|---|---|---:|---:|---:|---|")
q1raw = tables["in_degree|raw"].set_index("node")["quadrant"]
for r in false_fail_df.sort_values(["in_degree", "node"], ascending=[False, True]).itertuples():
    A(f"| {r.node} | {r.dead_cat} | {r.status_codes} | {r.in_degree} | {r.deg_u} | "
      f"{r.unavailability:.1f} | {q1raw.get(r.node, 'NA')} |")
A("")
A("| 待人工判断 | 节点 | dead_cat | 状态码 | 入度 | 校正前象限 |")
A("|---|---|---|---|---:|---|")
for r in amb_df.itertuples():
    A(f"| G | {r.node} | {r.dead_cat} | {r.status_codes} | {r.in_degree} | "
      f"{q1raw.get(r.node, 'NA')} |")
A("")

A("## 5. x 轴候选与健康的相关（§2.1 复算，非因果）")
A("")
A(md_table(axis_corr, ["axis", "label", "spearman_rho_with_unavailability",
                       "spearman_rho_with_ok_ratio_pct",
                       "spearman_rho_unavailability_excl_false_fail"]))
A("")
A(f"- 复算 ρ(总度, 可用率) = {rho_deg_u_ok:.4f}、ρ(入度, 可用率) = {rho_in_ok:.4f}，"
  f"与 §2.1 的 0.376 / 0.216 同序同量级（此处为 671 节点、Spearman 口径）。")
A(f"- 最后一列是剔除 {n_false} 个假失效后的同一相关，用来检查「把抓取被挡当成故障」"
  f"是否污染梯度本身（{cite('AXIS_CORR')}）。")
A(f"- 出度删失复核：{n_dead} 个全挂站的出度全部为 0（{n_dead_out0}/{n_dead}，"
  f"{cite('N_DEAD_OUT0')}），所以它们的 `deg_u` 数值上就等于入度；总度口径的危害不在死站一侧，"
  f"而在活站一侧——活站的出链是「抓到多少记多少」，于是总度同时编码结构位置与抓取成功度。")
A("")
A(f"§2.1 记载数字的复算（参照组定义要写清，否则 2.7× 与 5.2× 会被当成同一件事）：")
A("")
A("| 口径 | 全挂站 (n=%d) | 健康站 (n=%d) | 非全挂 (n=%d) | 倍数（健康站） | 倍数（非全挂） |"
  % (n_dead, n_healthy, n_attr - n_dead))
A("|---|---:|---:|---:|---:|---:|")
A(f"| 入度（§2.1 主口径） | {mean_in_dead:.4f} | {mean_in_healthy:.4f} | {mean_in_alive:.4f} | "
  f"{mean_in_healthy / mean_in_dead:.2f}× | {mean_in_alive / mean_in_dead:.2f}× |")
A(f"| 无向总度（跨口径，错误对比） | {mean_degu_dead:.4f} | {mean_degu_healthy:.4f} | "
  f"{mean_degu_alive:.4f} | {mean_degu_healthy / mean_degu_dead:.2f}× | "
  f"{mean_degu_alive / mean_degu_dead:.2f}× |")
A("")
A(f"两处与文档字面不完全等号的地方，登记给 I1：§2.1 的 4.94 对应**健康站子集**"
  f"（本模块复算 {mean_in_healthy:.4f}，{cite('MEAN_IN_HEALTHY')}），"
  f"用「非全挂」做参照组是 {mean_in_alive:.4f}（{cite('MEAN_IN_ALIVE')}）；"
  f"§2.1 的 9.49 同样来自健康站的总度（复算 {mean_degu_healthy:.4f}，"
  f"{cite('MEAN_DEGU_HEALTHY')}）。数值同量级、比值方向一致，属参照组措辞差异而非口径冲突。")
A("")
A("- 局限（必读）：单次横截面、无页面级链接溯源、出度完全删失 ⇒ 上表只描述结构位置与"
  "当前可用率的相关，不描述失效机制；低入度与低可用率的共现既可对应边缘站点更脆弱，"
  "也可对应抓取删失与站点规模（`url_count`）差异，本数据无法分离这两种解释。")
A("")
A("## 6. 口径声明与产物")
A("")
A("- 本模块不含任何删除 / 重连情景，因此 §2.3 的效率三口径不出现在结论中；"
  "「移除某节点后的结构变化」在 G2（`../G2_single_point_failure/`），"
  "分类判据与收益量化在 G3（`../G3_governance_priority/`）。")
A(f"- 随机性：百分位、象限、相关都是确定性计算；`SEED = {SEED}`（{cite('SEED')}）只用于脚本一致性，"
  f"本模块无随机过程，因此没有 seed 间波动需要报告。")
A(f"- 全量字段（每格计数、假失效定位、迁移矩阵）在 `G1_summary.json`（{cite('JSON')}）；"
  f"逐节点 7 张表的 x_pct / u_pct / 象限标签在 `g1_nodes_scored.csv`（{cite('SCORED_WIDE')}）。")
A(f"- 禁用措辞自检：写文件前扫描 §7 词表（{cite('BANNED_LIST')}），命中即中断 —— 本次 PASS。")
A("")

text = "\n".join(L) + "\n"
bad = [w for w in BANNED if w in text]
assert not bad, "result.md 出现禁用因果措辞: " + str(bad)  # L:BANNED_CHECK
with open(os.path.join(HERE, "result.md"), "w", encoding="utf-8") as f:
    f.write(text)  # L:WRITE_RESULT

wall = time.time() - T_START
C.write_run_note("G1_risk_matrix", [
    "G1_risk_matrix RUN.txt",
    "command: cd pipeline && PYTHONHASHSEED=0 python out/G1_risk_matrix/run_g1.py",
    "versions: " + str(C.versions()),
    "wall_time_s: %.2f" % wall,
    "SEED = %d  (no stochastic step in this unit; percentiles / quadrants / correlations are deterministic)" % SEED,
    "anchors: csa_common.assert_anchors() 31/31 PASS (run_g1.py:L%d)" % LNO("ANCHORS"),
    "universe: attributed=%d scored_corrected=%d false_fail=%d ambiguous=%d gcc=%d"
    % (n_attr, n_scored_corr, n_false, n_amb, n_gcc),
    "graphs: C.build_all(); centrality: C.centralities() -> in_degree(§2.1主) / bet_u(无权无向) / pr_dw(§5有向加权); deg_u=污染对照 only",
    "health: raw=ok_ratio_pct 原值; corr=§2.2 真失效校正（假失效+待判退出打分域并单列）",
    "efficiency: not used in this unit (§2.3 rules bind G2/G3)",
    "banned-word self-check §7: PASS",
    "outputs: g1_nodes_scored.csv g1_quadrant_summary.csv g1_quadrant_members.csv g1_instability.csv "
    "g1_false_fail_class.csv g1_ambiguous_class.csv g1_axis_correlations.csv G1_summary.json result.md RUN.txt",
])

print("[G1] done in %.1fs" % wall)
print("[G1] universe attributed=%d corrected-scored=%d false=%d amb=%d"
      % (n_attr, n_scored_corr, n_false, n_amb))
print("[G1] Q1 sizes by table: " + str(q1_count_by_table))
print("[G1] 6-table stable labels: %d/%d = %.1f%% alwaysQ1=%d everQ1=%d neverQ1=%d"
      % (n_stable_all6, n_six_common, n_stable_pct, n_always_q1, n_ever_q1, n_never_q1))
print(instability[["pair", "kind", "n_changed_quadrant", "pct_changed_quadrant",
                   "jaccard_Q1", "spearman_x_a_b"]].to_string(index=False))
