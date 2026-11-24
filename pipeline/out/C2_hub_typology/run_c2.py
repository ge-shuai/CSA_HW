# -*- coding: utf-8 -*-
"""C2_hub_typology — 出度 × 入度双轴角色分类（死站出度删失已隔离）。

口径来源（pipeline/CONVENTIONS.md）：
  §1 只用 csa_common.build_all() 构图；主分析节点集 671；开跑先 assert_anchors()。
     禁用 nodes_full.csv 的 degree / color 列 → 本脚本的度一律来自 C.add_degrees（图上重算），
     并在 L? 处断言与 Gd 直接读取一致。
  §2.1 本单元的核心纪律：**入度与出度分开成轴**，不折叠成总度；162 个整站不可用站点出度恒为 0
     是**删失**（crawler 抓不到页面就提不到出链），不是"不发链"。因此：
       主口径 = 活站子集（509 = 671 − 162），出度轴只在活站上可解释；
       死站单独成桶 "出度删失，无法归类"，**绝不**进入"导航型枢纽"；
       全样本(671) 版本只作对照，并逐条量化它被删失污染的方式（阈值本身被 162 个 0 拉低）。
  §7 无任何因果措辞：类别是"结构上对应"的描述，不是"因为它是导航站所以出度高"。
  §8 SEED=20261113；本脚本的随机过程只有"并列打散"与"随机分类基线"，均显式传 seed。

分类规则（可复现，写死）：
  轴 = in_degree（有向入度）与 out_degree（有向出度），都是二值度（不是强度，见敏感性）。
  阈值 = 该轴在**主口径子集**上的中位数**数值** t_in / t_out。
  high := 值 > t（严格大于）；值 == t 归 low。
  ⚠ 用"数值阈值"而不是"秩取前一半"：本数据在中位数上有大量并列
    （alive 子集 in_degree==2 有 105 个节点），秩切分必须在并列里硬挑，等于把随机噪声写进结论。
    数值阈值下，并列全部落 low，规则与节点身份一一对应，跨进程/跨排序稳定。
  四类：导航型枢纽(out high, in low) / 内容型权威(in high, out low) / 双向枢纽(both high) / 边缘(both low)

输出：hub_typology_alive509.csv（主口径名单）/ hub_typology_all671.csv（对照，标注污染）
      class_summary.csv / cut_sensitivity.csv / subset_sensitivity.csv / censoring_evidence.csv
      c2_summary.json
"""
from __future__ import annotations

import os
import sys
import time
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPELINE = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _PIPELINE)

import numpy as np
import pandas as pd
import networkx as nx

import csa_common as C

SEED = 20261113
MODULE = "C2_hub_typology"
CLASS_ORDER = ["导航型枢纽", "内容型权威", "双向枢纽", "边缘", "出度删失，无法归类"]
T0 = time.time()
TIMING: dict[str, float] = {}


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def stage(name: str):
    class _S:
        def __enter__(self):
            self.t = time.time()
            return self

        def __exit__(self, *a):
            TIMING[name] = round(time.time() - self.t, 2)

    return _S()


# ---------------------------------------------------------------- 1. 建图 + 锚点
with stage("build"):
    C.assert_anchors()
    log("assert_anchors() 31/31 PASS")
    b = C.build_all()
    Gu, Gd, m = b["Gu"], b["Gd"], b["nodes"]
    ATT = list(m["node"])
    # 度必须是图上重算的（CONVENTIONS §1 禁用 nodes_full.degree / color）
    assert (m["in_degree"].values == np.array([Gd.in_degree(n) for n in m["node"]])).all()
    assert (m["out_degree"].values == np.array([Gd.out_degree(n) for n in m["node"]])).all()
    assert "degree" not in m.columns, "nodes_full.csv 的 degree 列不得进入本模块"
    log(f"671 attributed / Gu {Gu.number_of_edges()} edges / Gd {Gd.number_of_edges()} arcs")

df = m.loc[:, ["node", "sub", "title", "in_degree", "out_degree", "in_strength", "out_strength",
               "degree_u", "strength_u", "health_state", "is_dead", "is_partial", "is_healthy",
               "true_fail", "false_fail", "ok_ratio_pct", "listed_urls", "url_count"]].copy()
df["is_nav_hub6"] = df["node"].isin(C.hub_nodes(Gd))          # 精确 FQDN，不用前缀匹配（§1）
ALIVE = df.loc[~df["is_dead"], "node"].tolist()               # 509 主口径
HEALTHY = df.loc[df["is_healthy"], "node"].tolist()           # 360 最干净子集
DEAD = df.loc[df["is_dead"], "node"].tolist()
assert (len(ALIVE), len(HEALTHY), len(DEAD)) == (509, 360, 162)


# ---------------------------------------------------------------- 2. 分类器
def classify(frame: pd.DataFrame, subset: list, quantile: float = 50.0,
             ax_in: str = "in_degree", ax_out: str = "out_degree"):
    """在 subset 上按轴分位数取阈值，对**全 frame** 打标；subset 外的节点标 NaN（不参与阈值）。

    返回 (labels, t_in, t_out)。labels 只保证对 subset 内节点有意义。
    """
    s = frame.set_index("node")
    vin = s.loc[subset, ax_in].astype(float)
    vout = s.loc[subset, ax_out].astype(float)
    t_in = float(np.percentile(vin.values, quantile))
    t_out = float(np.percentile(vout.values, quantile))
    hi_in = frame[ax_in] > t_in
    hi_out = frame[ax_out] > t_out
    lab = np.select(
        [hi_in & hi_out, hi_in & ~hi_out, ~hi_in & hi_out],
        ["双向枢纽", "内容型权威", "导航型枢纽"], default="边缘")
    out = pd.Series(lab, index=frame.index)
    out.loc[~frame["node"].isin(subset)] = np.nan
    return out, t_in, t_out


with stage("primary"):
    lab_p, t_in, t_out = classify(df, ALIVE, 50.0)
    df["class_alive"] = lab_p
    # 死站：出度删失 → 无论阈值怎么定都不进四类，改判为独立桶
    df.loc[df["is_dead"], "class_alive"] = "出度删失，无法归类"
    assert int((df.loc[df["is_dead"], "class_alive"] == "导航型枢纽").sum()) == 0
    prim = df["class_alive"].value_counts().reindex(CLASS_ORDER).fillna(0).astype(int)
    log(f"primary cut t_in={t_in} t_out={t_out} -> {prim.to_dict()}")

with stage("all671"):
    lab_a, ta_in, ta_out = classify(df, ATT, 50.0)
    df["class_all671"] = lab_a
    allc = df["class_all671"].value_counts().reindex(CLASS_ORDER[:4]).fillna(0).astype(int)
    # 污染量化 1：162 个删失站点被塞进哪几类
    dead_in_all = df.loc[df["is_dead"], "class_all671"].value_counts().to_dict()
    # 污染量化 2：阈值本身被 162 个 0 拉低
    thr_shift = {"t_in_alive": t_in, "t_out_alive": t_out, "t_in_all": ta_in, "t_out_all": ta_out}
    # 污染量化 3：活站的类别有多少被改掉
    both = df.loc[~df["is_dead"]].set_index("node")[["class_alive", "class_all671"]]
    same = int((both["class_alive"] == both["class_all671"]).sum())
    n_alive = len(both)
    flow = (both.groupby(["class_alive", "class_all671"]).size()
              .rename("n").reset_index())
    flow = flow[flow["class_alive"] != flow["class_all671"]].sort_values("n", ascending=False)
    log(f"all671 cut t_in={ta_in} t_out={ta_out}; alive 改类 {n_alive - same}/{n_alive}; "
        f"dead 去向 {dead_in_all}")

with stage("censoring"):
    # 删失证据表：每一类的出度是否可观测
    rows = []
    for hs in ["healthy", "partial", "dead"]:
        s = df.loc[df["health_state"] == hs]
        rows.append({"health_state": hs, "n": len(s),
                     "mean_in_degree": round(float(s["in_degree"].mean()), 3),
                     "mean_out_degree": round(float(s["out_degree"].mean()), 3),
                     "n_out_degree_zero": int((s["out_degree"] == 0).sum()),
                     "share_out_degree_zero": round(float((s["out_degree"] == 0).mean()), 4),
                     "n_listed_urls_at_cap20": int((s["listed_urls"] == 20).sum()),
                     "share_listed_urls_at_cap20": round(float((s["listed_urls"] == 20).mean()), 4)})
    cen = pd.DataFrame(rows)
    # 以死站为起点的弧数（必须 0）与以死站为终点的弧数（可观测）
    dead_set = set(DEAD)
    cen_extra = {
        "arcs_from_dead": int(sum(1 for u, v in Gd.edges() if u in dead_set)),
        "arcs_to_dead": int(sum(1 for u, v in Gd.edges() if v in dead_set)),
        "out_degree_zero_among_dead": int((df.loc[df["is_dead"], "out_degree"] == 0).sum()),
        "n_alive_out_zero": int((df.loc[~df["is_dead"], "out_degree"] == 0).sum()),
    }
    log(f"censoring: {cen_extra}")

with stage("sensitivity"):
    base = df.set_index("node")["class_alive"]          # 主口径标签，node 索引

    def agree_stats(lab_df_indexed: pd.Series, nodes: list) -> dict:
        """把 df 行索引的标签向量对齐到 node 索引，再与主口径在**同一批节点**上比。

        ⚠ 两套标签必须在同一节点集上比（CONVENTIONS 关于"不同节点集的分区不可直接比"的教训）。
        """
        cand = pd.Series(lab_df_indexed.values, index=df["node"].values).reindex(nodes)
        ref = base.reindex(nodes)
        assert cand.notna().all() and ref.notna().all()
        n = int((cand == ref).sum())
        return {"n_agree": n, "share_agree": round(n / len(nodes), 4)}

    sens = []
    for q in [50.0, 60.0, 75.0, 90.0]:
        lab, ti, to = classify(df, ALIVE, q)
        cnt = lab[df["node"].isin(ALIVE)].value_counts().reindex(CLASS_ORDER[:4]).fillna(0).astype(int)
        sens.append({"kind": "quantile_alive509", "rule": f"P{q:g}", "subset_n": len(ALIVE),
                     "t_in": ti, "t_out": to, **{k: int(cnt[k]) for k in CLASS_ORDER[:4]},
                     **agree_stats(lab, ALIVE)})
    for q in [50.0, 75.0]:
        lab, ti, to = classify(df, HEALTHY, q)
        cnt = lab[df["node"].isin(HEALTHY)].value_counts().reindex(CLASS_ORDER[:4]).fillna(0).astype(int)
        sens.append({"kind": "subset_healthy360", "rule": f"P{q:g}", "subset_n": len(HEALTHY),
                     "t_in": ti, "t_out": to, **{k: int(cnt[k]) for k in CLASS_ORDER[:4]},
                     **agree_stats(lab, HEALTHY)})
    lab, ti, to = classify(df, ALIVE, 50.0, ax_in="in_strength", ax_out="out_strength")
    cnt = lab[df["node"].isin(ALIVE)].value_counts().reindex(CLASS_ORDER[:4]).fillna(0).astype(int)
    sens.append({"kind": "strength_alive509", "rule": "P50 on in/out_strength", "subset_n": len(ALIVE),
                 "t_in": ti, "t_out": to, **{k: int(cnt[k]) for k in CLASS_ORDER[:4]},
                 **agree_stats(lab, ALIVE)})
    # CONVENTIONS §2.2 "校正前/校正后"两套：32 个假失效（B 认证/跳过、C robots）不是故障。
    # 它们的**出边同样不可观测**，所以不回填进分类集合，只回填进"阈值估计集合"(509+32=541)，
    # 用来回答：把假失效从"死"里摘出去，四类占比会动多少。
    ff = df.loc[df["false_fail"], "node"].tolist()
    lab, ti, to = classify(df, ALIVE + ff, 50.0)
    cnt = lab[df["node"].isin(ALIVE)].value_counts().reindex(CLASS_ORDER[:4]).fillna(0).astype(int)
    sens.append({"kind": "corrected_541_threshold", "rule": f"P50(509 活站+32 假失效={len(ALIVE) + len(ff)})",
                 "subset_n": len(ALIVE) + len(ff), "t_in": ti, "t_out": to,
                 **{k: int(cnt[k]) for k in CLASS_ORDER[:4]}, **agree_stats(lab, ALIVE)})
    lab_a2, ti, to = classify(df, ATT, 50.0)
    cnt = lab_a2.value_counts().reindex(CLASS_ORDER[:4]).fillna(0).astype(int)
    sens.append({"kind": "ALL671_censoring_contaminated", "rule": "P50(全样本 671)", "subset_n": len(ATT),
                 "t_in": ti, "t_out": to, **{k: int(cnt[k]) for k in CLASS_ORDER[:4]},
                 "n_agree": np.nan, "share_agree": np.nan})
    # 随机基线：把四类按同样边际分布随机分配，看"类别=度位置"的对照组
    rng = np.random.default_rng(SEED)
    pj = prim.reindex(CLASS_ORDER[:4]).values / prim.reindex(CLASS_ORDER[:4]).values.sum()
    jac_rand = []
    for r in range(200):
        rl = rng.choice(CLASS_ORDER[:4], size=len(ALIVE), p=pj)
        lab2 = pd.Series(rl, index=ALIVE)
        jac_rand.append(float((lab2.values == base[ALIVE].values).mean()))
    sens.append({"kind": "random_label_baseline", "rule": "按主口径边际分布随机打乱",
                 "subset_n": len(ALIVE), "t_in": np.nan, "t_out": np.nan,
                 **{k: np.nan for k in CLASS_ORDER[:4]},   # 同类列名，避免 CSV 出现空的重名列
                 "n_agree": int(np.mean(jac_rand) * len(ALIVE)),
                 "share_agree": round(float(np.mean(jac_rand)), 4)})
    sens = pd.DataFrame(sens)
    log("sensitivity rows: " + str(len(sens)))

with stage("summary_tables"):
    # 无向度 Top-20（671 有属性节点内，值降序 + 节点名字典序，确定性）——用来回答"高度是不是同一批"
    degu_top20 = set(df.sort_values(["degree_u", "node"], ascending=[False, True])["node"].head(20))
    prof = []
    for cl in CLASS_ORDER:
        s = df[df["class_alive"] == cl]
        if len(s) == 0:
            continue
        universe = len(df) if cl == "出度删失，无法归类" else len(ALIVE)
        prof.append({
            "class": cl, "n": len(s), "universe_n": universe,
            "share_of_universe": round(len(s) / universe, 4),
            "mean_in_degree": round(float(s["in_degree"].mean()), 3),
            "median_in_degree": float(s["in_degree"].median()),
            "mean_out_degree": round(float(s["out_degree"].mean()), 3),
            "median_out_degree": float(s["out_degree"].median()),
            "mean_degree_u": round(float(s["degree_u"].mean()), 3),
            "mean_in_strength": round(float(s["in_strength"].mean()), 1),
            "mean_out_strength": round(float(s["out_strength"].mean()), 1),
            "n_in_degree_u_top20": int(len(set(s["node"]) & degu_top20)),
            "share_of_degree_u_top20": round(len(set(s["node"]) & degu_top20) / 20, 4),
            "jaccard_with_degree_u_top20": round(
                len(set(s["node"]) & degu_top20) / (len(set(s["node"]) | degu_top20)), 4),
            "n_nav_hub6": int(s["is_nav_hub6"].sum()),
            "n_healthy": int(s["is_healthy"].sum()), "n_partial": int(s["is_partial"].sum()),
            "n_dead": int(s["is_dead"].sum()),
            "mean_ok_ratio_pct": round(float(s["ok_ratio_pct"].mean()), 2),
            "share_listed_urls_at_cap20": round(float((s["listed_urls"] == 20).mean()), 4),
        })
    prof = pd.DataFrame(prof)

    # 名单：每类按 (出度, 入度) 排序给全量成员（csv 给全名单，这里只留表）
    roster = df[df["class_alive"].notna()].copy()
    roster = roster.sort_values(["class_alive", "out_degree", "in_degree", "node"],
                                ascending=[True, False, False, True])

    # 6 个命名导航枢纽的结构类别（"名字叫导航" ≠ "结构上是导航型"）
    hubtab = df[df["is_nav_hub6"]].loc[:, ["node", "in_degree", "out_degree", "in_strength",
                                           "out_strength", "degree_u", "class_alive",
                                           "class_all671", "health_state"]].sort_values("out_degree",
                                                                                         ascending=False)
    log("nav6 structural classes:\n" + hubtab.to_string(index=False))

with stage("save"):
    keep = ["node", "sub", "title", "class_alive", "class_all671", "in_degree", "out_degree",
            "in_strength", "out_strength", "degree_u", "strength_u", "health_state", "is_dead",
            "true_fail", "false_fail", "ok_ratio_pct", "listed_urls", "is_nav_hub6"]
    df.loc[df["node"].isin(ALIVE + DEAD), keep].to_csv(
        os.path.join(_HERE, "hub_typology_alive509.csv"), index=False, encoding="utf-8-sig")
    df.loc[:, keep].to_csv(os.path.join(_HERE, "hub_typology_all671.csv"), index=False,
                           encoding="utf-8-sig")
    prof.to_csv(os.path.join(_HERE, "class_summary.csv"), index=False, encoding="utf-8-sig")
    sens.to_csv(os.path.join(_HERE, "cut_sensitivity.csv"), index=False, encoding="utf-8-sig")
    cen.to_csv(os.path.join(_HERE, "censoring_evidence.csv"), index=False, encoding="utf-8-sig")
    hubtab.to_csv(os.path.join(_HERE, "nav_hub6_structural_class.csv"), index=False, encoding="utf-8-sig")
    roster[keep].to_csv(os.path.join(_HERE, "class_rosters_alive509.csv"), index=False, encoding="utf-8-sig")
    flow.to_csv(os.path.join(_HERE, "contamination_flow_alive_classes.csv"), index=False, encoding="utf-8-sig")

    summary = {
        "module": MODULE, "versions": C.versions(), "seed": SEED,
        "primary_subset": "alive = 671 attributed - 162 整站不可用 = 509",
        "cut_rule": f"P50 of each axis on the primary subset, high = value strictly > threshold",
        "thresholds_primary": {"t_in": t_in, "t_out": t_out},
        "thresholds_all671": {"t_in": ta_in, "t_out": ta_out},
        "class_counts_primary": {k: int(prim[k]) for k in CLASS_ORDER},
        "class_counts_all671": {k: int(allc[k]) for k in CLASS_ORDER[:4]},
        "dead_nodes_redistributed_in_all671": dead_in_all,
        "alive_nodes_reclassified_by_fullsample_cut": {"changed": int(n_alive - same),
                                                       "of": int(n_alive),
                                                       "share": round(float((n_alive - same) / n_alive), 4)},
        "censoring_evidence": cen_extra,
        "degree_axis_note": "入度含 296 条活站指向死站的弧，可观测；出度对死站 100% 不可观测",
        "sensitivity_agreement_with_primary": sens[["kind", "rule", "share_agree"]].to_dict("records"),
        "random_label_baseline_share_agree": round(float(np.mean(jac_rand)), 4),
        "nav_hub6_class": hubtab.set_index("node")["class_alive"].to_dict(),
        "overlap_with_degree_u_top20": {r["class"]: [int(r["n_in_degree_u_top20"]), float(r["jaccard_with_degree_u_top20"])]
                                        for _, r in prof.iterrows()},
        "timing_s": TIMING, "wall_s": round(time.time() - T0, 2),
    }
    with open(os.path.join(_HERE, "c2_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    C.write_run_note(MODULE, [
        "C2_hub_typology RUN.txt",
        "command: set PYTHONHASHSEED=0 && python out/C2_hub_typology/run_c2.py "
        "(bash: PYTHONHASHSEED=0 python out/C2_hub_typology/run_c2.py; cwd=pipeline)",
        f"versions: {json.dumps(C.versions(), ensure_ascii=False)}",
        f"SEED={SEED}; rng: np.random.default_rng(SEED) 用于随机标签基线(200 次)",
        f"wall time: {summary['wall_s']}s ; per-stage: {json.dumps(TIMING)}",
        "anchors: csa_common.assert_anchors() 31/31 PASS",
        "graph: csa_common.build_all(); degrees from C.add_degrees (nodes_full 的 degree/color 列未使用)",
        f"primary口径: alive={len(ALIVE)} (671-162); cut = P50 per axis, high = value > t; "
        f"t_in={t_in}, t_out={t_out}",
        "dead sites -> class '出度删失，无法归类'（永不进入四类）",
        "outputs: hub_typology_alive509.csv hub_typology_all671.csv class_summary.csv "
        "cut_sensitivity.csv censoring_evidence.csv nav_hub6_structural_class.csv "
        "class_rosters_alive509.csv contamination_flow_alive_classes.csv c2_summary.json result.md",
    ])
    log("saved")
    print(json.dumps({k: summary[k] for k in
                      ["class_counts_primary", "class_counts_all671", "thresholds_primary",
                       "thresholds_all671", "dead_nodes_redistributed_in_all671",
                       "alive_nodes_reclassified_by_fullsample_cut", "censoring_evidence",
                       "nav_hub6_class", "overlap_with_degree_u_top20",
                       "sensitivity_agreement_with_primary", "random_label_baseline_share_agree"]},
                     ensure_ascii=False, indent=1))
