# -*- coding: utf-8 -*-
"""C3_centrality_robustness — Top-20 榜单在三种图口径下的稳定性 + 判决规则。

口径来源（pipeline/CONVENTIONS.md）：
  §1 构图只用 csa_common.build_all()；枢纽只用 C.hub_nodes()（精确 6 FQDN）/ C.strip_hub_edges；
     骨架用 C.strong_backbone(Gu, 10)。开跑先 assert_anchors()。
  §2.1 入度/出度/无向度分列，不折叠。
  §5 PageRank 变体分列（pr_dw 有向加权主口径 / pr_uu 无向无权对照）。
  §7 无因果措辞。  §8 SEED=20261113，所有随机过程显式传 seed。

三种图口径：
  (a) full   : Gu 全图（676 节点 / 2498 边），Gd 全图（2788 弧）
  (b) dehub  : C.strip_hub_edges(Gu) —— 删掉与 6 个导航枢纽相连的全部边（505 条 = 20.2%）
  (c) bb10   : C.strong_backbone(Gu, 10) —— 只留无向权重 >=10 的强边骨架（675 条边，307 个孤立点）
  有向指标用**同一批幸存无序点对**反推的弧集（(b) C.strip_hub_edges(Gd)；
  (c) 弧 (u,v) 保留 <=> 无序点对 {u,v} 的合并权重 >=10）。这条映射规则 CONVENTIONS 未定义，
  本单元自述并在 DISSENT.md 中标注为"扩展口径"，另给"按弧自身权重 >=10"的对照。

判据（写死在代码里，见 DECISION）：
  PASS := min(三个两两口径 Jaccard) >= 0.50  **且**  节点抽样稳定性(删 10% 节点) >= 0.50
  J=0.50 <=> 两榜至少共有 14/20 个成员；随机两张 20 榜（671 候选池）的期望 J=0.015（实测见
  random_baseline），0.50 ≈ 随机预期的 34 倍，这是"榜单可作为结论"的下限。
  并且取 **并列打散的最坏值**（100 次随机节点序），因为 k-core / H-index 这类离散指标
  在第 20 名上有几十个并列节点，任何单一打破顺序都是任意的。
  FAIL 的指标 → 其 Top-10 禁止作为结论上报（本脚本会打出被禁名单）。

输出：top20_jaccard_by_measure.csv / jaccard_matrix_long.csv / decision_table.csv
      bootstrap_stability.csv / rank_shift.csv / tie_boundary.csv / c3_summary.json
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
MODULE = "C3_centrality_robustness"
TOP_K = 20
N_TIE_PERM = 100          # 并列打散次数（同一随机节点序同时用于所有指标/口径，避免额外混淆）
N_BOOT = 50               # 删 10% 节点的重复次数（CONVENTIONS §6：随机策略 >=100 次是针对攻击曲线；
                          # 这里只做榜单稳定性，50 次已能把 mean 的 Monte-Carlo 误差压到 ~0.01）
BOOT_DROP = 0.10
J_PASS = 0.50
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


# ------------------------------------------------------------------ 指标族
def hits_pinned(G: nx.DiGraph):
    """nx.hits 在 networkx 3.6 里走 svds(v0=nstart)；nstart=None → ARPACK 随机起点，
    跨进程逐位不可复现。这里固定成均匀起点（同算法同参数），证据见 C1/DISSENT.md。"""
    return nx.hits(G, max_iter=200, tol=1e-8, nstart={n: 1.0 for n in G})


def hindex_in_in(Gd: nx.DiGraph) -> dict:
    """H-index（有向入度版）：最大的 h 使 u 至少有 h 个入邻域 x 满足 in_degree(x) >= h。"""
    indeg = dict(Gd.in_degree())
    out = {}
    for n in Gd.nodes():
        vals = sorted((indeg[x] for x in Gd.predecessors(n)), reverse=True)
        h = 0
        for i, v in enumerate(vals, start=1):
            if v >= i:
                h = i
            else:
                break
        out[n] = h
    return out


def measure_frame(Gu: nx.Graph, Gd: nx.DiGraph) -> pd.DataFrame:
    """12 个"重要性"指标在同一对图上的值（列名与 C1 完全一致）。

    无向指标跑在 Gu 上，有向指标跑在 Gd 上；两者由**同一批无序点对**定义（见模块 docstring）。
    """
    h, a = hits_pinned(Gd)
    d = {
        "deg_u": pd.Series(dict(Gu.degree())),
        "in_degree": pd.Series(dict(Gd.in_degree())),
        "out_degree": pd.Series(dict(Gd.out_degree())),
        "in_strength": pd.Series(dict(Gd.in_degree(weight="weight"))),
        "out_strength": pd.Series(dict(Gd.out_degree(weight="weight"))),
        "bet_u": pd.Series(nx.betweenness_centrality(Gu)),
        "pr_dw": pd.Series(nx.pagerank(Gd, weight="weight")),
        "pr_uu": pd.Series(nx.pagerank(nx.Graph(Gu))),
        "hits_hub": pd.Series(h),
        "hits_auth": pd.Series(a),
        "kcore": pd.Series(dict(nx.core_number(Gu))),
        "hidx_in_in": pd.Series(hindex_in_in(Gd)),
    }
    return pd.DataFrame(d)


MEAS = ["deg_u", "in_degree", "out_degree", "in_strength", "out_strength", "bet_u",
        "pr_dw", "pr_uu", "hits_hub", "hits_auth", "kcore", "hidx_in_in"]


def top_k(values: pd.Series, k: int, order_key: pd.Series | None = None) -> list:
    """Top-k 名单（有序）。并列时按 order_key 打破；order_key 缺省 = 节点名字典序 → 确定性。"""
    t = (values.index.to_series() if order_key is None
         else order_key.reindex(values.index).astype(str))
    o = pd.DataFrame({"v": values.to_numpy(dtype=float), "t": t.to_numpy()}, index=values.index)
    o = o.sort_values(["v", "t"], ascending=[False, True], kind="mergesort")
    return list(o.index[:k])


# ---------------------------------------------------------------- 1. 建图 + 锚点
with stage("build"):
    C.assert_anchors()
    log("assert_anchors() 31/31 PASS")
    b = C.build_all()
    Gu, Gd, m = b["Gu"], b["Gd"], b["nodes"]
    ATT = list(m["node"])                       # 671 候选池，三套口径完全一致
    HUBS = C.hub_nodes(Gu)
    assert len(HUBS) == 6

    pair_w = {(u, v): dd.get("weight", 1) for u, v, dd in Gu.edges(data=True)}
    pair_w.update({(v, u): w for (u, v), w in list(pair_w.items())})

    def arcs_backbone(Gd_: nx.DiGraph, min_w: int, by_arc: bool = False) -> nx.DiGraph:
        """有向骨架。(u,v) 保留 <=> 参照权重 >= min_w。
        参照权重：主口径 = 无序点对 {u,v} 在 Gu 上的**合并**权重（与 C.strong_backbone 同一批边）；
        by_arc=True 时退化成弧自身权重（诊断口径，写进 DISSENT.md）。"""
        H = nx.DiGraph()
        H.add_nodes_from(Gd_.nodes())
        for u, v, dd in Gd_.edges(data=True):
            w_ref = (float(dd.get("weight", 1)) if by_arc
                     else float(pair_w.get((u, v) if u <= v else (v, u), 0.0)))
            if w_ref >= min_w:
                H.add_edge(u, v, weight=int(dd.get("weight", 1)))
        return H

    GRAPH_VARIANTS = {
        "a_full": (Gu, Gd),
        "b_dehub": (C.strip_hub_edges(Gu), C.strip_hub_edges(Gd)),
        "c_bb10": (C.strong_backbone(Gu, 10), arcs_backbone(Gd, 10)),
        "c_bb10_byarc_DIAG": (C.strong_backbone(Gu, 10), arcs_backbone(Gd, 10, by_arc=True)),
    }
    edge_profile = []
    for kk, (ug, dg) in GRAPH_VARIANTS.items():
        iso = sum(1 for n in ATT if ug.degree(n) == 0)
        edge_profile.append({"variant": kk, "nodes": ug.number_of_nodes(), "uedges": ug.number_of_edges(),
                             "darcs": dg.number_of_edges(), "uweight_total": int(
                sum(dd.get("weight", 1) for _, _, dd in ug.edges(data=True))),
                             "n_attributed_isolated_in_u": iso,
                             "removed_uedges_vs_full": Gu.number_of_edges() - ug.number_of_edges(),
                             "share_removed_edges": round(1 - ug.number_of_edges() / Gu.number_of_edges(), 4),
                             "share_removed_weight": round(1 - sum(dd.get("weight", 1) for _, _, dd in ug.edges(data=True))
                                                           / sum(dd.get("weight", 1) for _, _, dd in Gu.edges(data=True)), 4)})
    ep = pd.DataFrame(edge_profile)
    log("graph variants:\n" + ep.to_string(index=False))

# ---------------------------------------------------------------- 2. 三套口径的指标值
with stage("measures"):
    FR = {k: measure_frame(ug, dg).loc[ATT] for k, (ug, dg) in GRAPH_VARIANTS.items()}
    for k, f in FR.items():
        assert len(f) == 671 and set(f.index) == set(ATT), k          # 候选池三套口径完全一致
        assert f.notna().all().all(), (k, f.columns[f.isna().any()].tolist())
    log("measure frames: " + str({k: v.shape for k, v in FR.items()}))

# ---------------------------------------------------------------- 3. 榜单 + 并列
with stage("rankings"):
    variants = ["a_full", "b_dehub", "c_bb10"]
    diag_variant = "c_bb10_byarc_DIAG"
    # 并列边界诊断：第 20 名的值上有多少节点并列
    tie_rows = []
    for v in variants + [diag_variant]:
        for c in MEAS:
            s = FR[v][c]
            vk = float(s.sort_values(ascending=False, kind="mergesort").iloc[TOP_K - 1])
            tie_rows.append({"variant": v, "measure": c, "value_at_20": vk,
                             "n_at_value": int((s == vk).sum()),
                             "n_strictly_above": int((s > vk).sum()),
                             "top20_fully_identified": bool(int((s > vk).sum()) + int((s == vk).sum()) <= TOP_K)})
    tie = pd.DataFrame(tie_rows)

    det_top = {(v, c): set(top_k(FR[v][c], TOP_K)) for v in variants + [diag_variant] for c in MEAS}

    # 并列打散：同一随机节点序作用于所有指标与口径（N_TIE_PERM 次）
    perm_top: dict[tuple[str, str], list[set]] = {(v, c): [] for v in variants for c in MEAS}
    for r in range(N_TIE_PERM):
        rng = np.random.default_rng(SEED + r)
        keys = pd.Series(rng.random(len(ATT)), index=ATT)
        for v in variants:
            for c in MEAS:
                perm_top[(v, c)].append(set(top_k(FR[v][c], TOP_K, order_key=keys)))
    log(f"tie permutations done: R={N_TIE_PERM}")

    PAIRS = [("a_full", "b_dehub"), ("a_full", "c_bb10"), ("b_dehub", "c_bb10")]

    def jac(x: set, y: set) -> float:
        i = len(x & y)
        return i / (2 * TOP_K - i)

    rows = []
    for c in MEAS:
        for (v1, v2) in PAIRS:
            js = [jac(perm_top[(v1, c)][r], perm_top[(v2, c)][r]) for r in range(N_TIE_PERM)]
            rows.append({"measure": c, "v1": v1, "v2": v2,
                         "jaccard_deterministic": round(jac(det_top[(v1, c)], det_top[(v2, c)]), 4),
                         "jaccard_perm_mean": round(float(np.mean(js)), 4),
                         "jaccard_perm_min": round(float(np.min(js)), 4),
                         "jaccard_perm_p05": round(float(np.percentile(js, 5)), 4),
                         "jaccard_perm_max": round(float(np.max(js)), 4),
                         "shared_deterministic": int(len(det_top[(v1, c)] & det_top[(v2, c)]))})
    J = pd.DataFrame(rows)
    cells = [f"{p[0]}|{p[1]}" for p in PAIRS]
    jmat = pd.DataFrame({c: {cell: float(J.loc[(J.measure == c) & (J.v1 == cell.split("|")[0])
                                               & (J.v2 == cell.split("|")[1]),
                                               "jaccard_deterministic"].iloc[0])
                             for cell in cells} for c in MEAS}).T
    jmat_worst = pd.DataFrame({c: {cell: float(J.loc[(J.measure == c) & (J.v1 == cell.split("|")[0])
                                                      & (J.v2 == cell.split("|")[1]),
                                                      "jaccard_perm_min"].iloc[0])
                                   for cell in cells} for c in MEAS}).T

# ---------------------------------------------------------------- 4. 随机基线
with stage("null"):
    rng = np.random.default_rng(SEED)
    nulls = []
    for _ in range(2000):
        x = set(rng.choice(len(ATT), TOP_K, replace=False))
        y = set(rng.choice(len(ATT), TOP_K, replace=False))
        i = len(x & y)
        nulls.append(i / (2 * TOP_K - i))
    null_mean, null_p99 = float(np.mean(nulls)), float(np.percentile(nulls, 99))
    log(f"random Top-20 pair Jaccard: mean={null_mean:.4f} p99={null_p99:.4f} max={max(nulls):.4f}")

# ---------------------------------------------------------------- 5. 节点抽样稳定性
with stage("bootstrap"):
    boot = []
    for r in range(N_BOOT):
        rng = np.random.default_rng(SEED + 7919 * (r + 1))
        n_keep = int(round(len(ATT) * (1 - BOOT_DROP)))
        keep = list(rng.choice(ATT, size=n_keep, replace=False))
        ks = set(keep)
        Gu_s = Gu.subgraph(keep).copy()
        Gd_s = Gd.subgraph(keep).copy()
        fs = measure_frame(Gu_s, Gd_s).loc[keep]
        keys = pd.Series(rng.random(len(keep)), index=keep)
        for c in MEAS:
            tb = set(top_k(fs[c], TOP_K, order_key=keys))
            tfull = det_top[("a_full", c)]
            inter_raw = len(tb & tfull)
            ref_cond = tfull & ks
            i2 = len(tb & ref_cond)
            boot.append({"rep": r, "measure": c, "n_kept": n_keep,
                         "jaccard_vs_full": inter_raw / (2 * TOP_K - inter_raw),
                         "jaccard_cond_on_survivors": i2 / (len(tb) + len(ref_cond) - i2),
                         "n_dropped_members": int(len(tfull - ks))})
    B = pd.DataFrame(boot)
    # 抽样的"天花板"：删掉的原始成员不可能再入选，故单次 Jaccard 上限 = (20-d)/(20+d)
    B["j_ceiling"] = (TOP_K - B["n_dropped_members"]) / (TOP_K + B["n_dropped_members"])
    bs = B.groupby("measure")[["jaccard_vs_full", "jaccard_cond_on_survivors"]].agg(
        ["mean", "std", "min", lambda s: float(np.percentile(s, 10))])
    bs.columns = ["j_mean", "j_std", "j_min", "j_p10", "cj_mean", "cj_std", "cj_min", "cj_p10"]
    bs = bs.reset_index()
    ceil = B.groupby("measure")["j_ceiling"].mean().rename("j_ceiling_mean").reset_index()
    dmean = B.groupby("measure")["n_dropped_members"].mean().rename("n_top20_dropped_mean").reset_index()
    bs = bs.merge(ceil, on="measure").merge(dmean, on="measure")
    bs["j_mean_relative_to_ceiling"] = (bs["j_mean"] / bs["j_ceiling_mean"]).round(4)
    log("bootstrap done: " + str(N_BOOT) + " reps; mean ceiling="
        + str(round(float(B["j_ceiling"].mean()), 4)))

    # 同样的抽样也跑一遍 (b)/(c) 口径的相对稳定性？成本可控但结论冗余 → 只做主口径，写明。

# ---------------------------------------------------------------- 6. 排名移动 + 判决
with stage("decision"):
    rank_tbl = []
    for c in MEAS:
        ranks = {}
        for v in variants + [diag_variant]:
            order = top_k(FR[v][c], len(ATT))
            ranks[v] = {n: i + 1 for i, n in enumerate(order)}
        for n in ATT:
            rank_tbl.append({"measure": c, "node": n,
                             **{f"rank_{v}": ranks[v][n] for v in ranks},
                             "in_top20_all3": all(ranks[v][n] <= TOP_K for v in variants)})
    RK = pd.DataFrame(rank_tbl)

    dec = []
    for c in MEAS:
        sub = J[J["measure"] == c]
        g = {(r.v1, r.v2): r.jaccard_perm_min for _, r in sub.iterrows()}
        j_all3 = float(min(g.values()))                                  # 三对里最坏的一对（主判据）
        j_vs_ref = float(min(g[("a_full", "b_dehub")], g[("a_full", "c_bb10")]))   # 只跟基准比
        j_det = float(sub["jaccard_deterministic"].min())
        bmean = float(bs.loc[bs["measure"] == c, "j_mean"].iloc[0])
        bmin = float(bs.loc[bs["measure"] == c, "j_min"].iloc[0])
        bp10 = float(bs.loc[bs["measure"] == c, "j_p10"].iloc[0])
        okj = j_all3 >= J_PASS
        okref = j_vs_ref >= J_PASS
        okb = bmean >= J_PASS
        frag = bp10 < 0.25 or bmin < 0.10            # 抽到一个关键点就整榜重排
        t10 = [set(top_k(FR[v][c], 10)) for v in variants]
        core10 = set.intersection(*t10)
        verdict = "PASS" if okj else "FAIL"
        combined = ("PASS" if (okj and okb and not frag) else
                    ("PASS·单点脆弱" if (okj and okb and frag) else
                     ("FAIL·口径不稳" if not okj else "FAIL·抽样不稳")))
        dec.append({
            "measure": c,
            "j_min_deterministic": round(j_det, 4),
            "j_min_worst_over_tie_perms": round(j_all3, 4),
            "j_min_vs_reference_only": round(j_vs_ref, 4),
            "pass_if_only_vs_reference": okref,
            "j_mean_over_perms": round(float(sub["jaccard_perm_mean"].mean()), 4),
            "boot_drop10_mean_j": round(bmean, 4),
            "boot_drop10_ceiling_mean": round(float(bs.loc[bs["measure"] == c, "j_ceiling_mean"].iloc[0]), 4),
            "boot_drop10_rel_to_ceiling": round(float(bs.loc[bs["measure"] == c,
                                                             "j_mean_relative_to_ceiling"].iloc[0]), 4),
            "boot_drop10_min_j": round(bmin, 4),
            "boot_drop10_p10_j": round(bp10, 4),
            "boot_cond_on_survivors_mean": round(float(bs.loc[bs["measure"] == c, "cj_mean"].iloc[0]), 4),
            "single_point_fragile": frag,
            "pass_graph_variants": okj, "pass_bootstrap": okb,
            "VERDICT": verdict, "COMBINED": combined,
            "top10_shared_all3_variants": len(core10),
            "null_random_j_mean": round(null_mean, 4),
            "j_min_over_null_ratio": round(j_all3 / null_mean, 1) if null_mean > 0 else np.nan,
            "tie_note": str(tie.loc[(tie.variant == "a_full") & (tie.measure == c),
                                    "n_at_value"].iloc[0]) + " nodes tied at rank20 (full)",
        })
    DEC = pd.DataFrame(dec).sort_values(["VERDICT", "j_min_worst_over_tie_perms"], ascending=[True, False])
    passed = DEC.loc[DEC.VERDICT == "PASS", "measure"].tolist()
    failed = DEC.loc[DEC.VERDICT == "FAIL", "measure"].tolist()
    strict_pass = DEC.loc[DEC.COMBINED == "PASS", "measure"].tolist()
    fragile_pass = DEC.loc[DEC.COMBINED == "PASS·单点脆弱", "measure"].tolist()
    would_flip = DEC.loc[(~DEC.pass_graph_variants) & (DEC.pass_if_only_vs_reference), "measure"].tolist()
    log("PASS(口径门): " + str(passed))
    log("FAIL(口径门): " + str(failed))
    log(f"PASS(口径门+抽样+无单点脆弱): {strict_pass} ; 单点脆弱: {fragile_pass} ; "
        f"若只与基准比会翻案: {would_flip}")

# ---------------------------------------------------------------- 7. 落盘
with stage("save"):
    ep.to_csv(os.path.join(_HERE, "graph_variant_profile.csv"), index=False, encoding="utf-8-sig")
    tie.to_csv(os.path.join(_HERE, "tie_boundary.csv"), index=False, encoding="utf-8-sig")
    J.to_csv(os.path.join(_HERE, "top20_jaccard_by_measure.csv"), index=False, encoding="utf-8-sig")
    jmat.round(4).to_csv(os.path.join(_HERE, "jaccard_matrix_deterministic.csv"), encoding="utf-8-sig")
    jmat_worst.round(4).to_csv(os.path.join(_HERE, "jaccard_matrix_worstcase.csv"), encoding="utf-8-sig")
    bs.round(4).to_csv(os.path.join(_HERE, "bootstrap_stability.csv"), index=False, encoding="utf-8-sig")
    B.round(6).to_csv(os.path.join(_HERE, "bootstrap_raw.csv"), index=False, encoding="utf-8-sig")
    DEC.to_csv(os.path.join(_HERE, "decision_table.csv"), index=False, encoding="utf-8-sig")
    RK[RK["in_top20_all3"]].to_csv(os.path.join(_HERE, "nodes_stable_top20.csv"), index=False,
                                   encoding="utf-8-sig")
    with open(os.path.join(_HERE, "top20_lists.csv"), "w", encoding="utf-8-sig") as f:
        f.write("variant,measure,rank,node,value\n")
        for v in variants + [diag_variant]:
            for c in MEAS:
                lst = top_k(FR[v][c], TOP_K)
                for i, n in enumerate(lst, 1):
                    f.write(f"{v},{c},{i},{n},{FR[v][c][n]:.10g}\n")

    summary = {
        "module": MODULE, "versions": C.versions(), "seed": SEED,
        "top_k": TOP_K, "n_tie_perm": N_TIE_PERM, "n_boot": N_BOOT, "boot_drop": BOOT_DROP,
        "decision_rule": f"PASS := 三对两两口径 Jaccard 的最小值（取 {N_TIE_PERM} 次并列打散的**最坏值**）"
                         f">= {J_PASS}；抽样稳定性（删 {BOOT_DROP:.0%} 节点，{N_BOOT} 次）作为第二道门与"
                         f"单点脆弱标记（p10<0.25 或 min<0.10）一并给出",
        "jaccard_pass_threshold": J_PASS,
        "threshold_justification": {"shared_members_at_0.5": 14,
                                    "random_top20_pair_jaccard_mean": round(null_mean, 5),
                                    "random_top20_pair_jaccard_p99": round(null_p99, 5),
                                    "ratio_of_threshold_to_random_mean": round(J_PASS / null_mean, 1)},
        "graph_variants": ep.to_dict("records"),
        "hubs_exact": sorted(HUBS),
        "passed_measures": passed, "failed_measures": failed,
        "strict_pass_measures": strict_pass, "fragile_pass_measures": fragile_pass,
        "flip_if_compared_only_to_reference": would_flip,
        "per_measure": DEC.to_dict("records"),
        "diag_arcs_by_own_weight": {
            "note": "c_bb10_byarc_DIAG = 弧按自身权重 >=10 过滤（vs 主口径按无序点对合并权重）",
            "j_vs_c_bb10_pairbased": {c: round(jac(det_top[(diag_variant, c)], det_top[("c_bb10", c)]), 4)
                                      for c in MEAS}},
        "timing_s": TIMING, "wall_s": round(time.time() - T0, 2),
    }
    with open(os.path.join(_HERE, "c3_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    C.write_run_note(MODULE, [
        "C3_centrality_robustness RUN.txt",
        "command: set PYTHONHASHSEED=0 && python out/C3_centrality_robustness/run_c3.py "
        "(bash: PYTHONHASHSEED=0 python out/C3_centrality_robustness/run_c3.py; cwd=pipeline)",
        f"versions: {json.dumps(C.versions(), ensure_ascii=False)}",
        f"SEED={SEED}; tie-permutation rng = np.random.default_rng(SEED + r), r=0..{N_TIE_PERM - 1}; "
        f"bootstrap rng = np.random.default_rng(SEED + 7919*(r+1)), r=0..{N_BOOT - 1}; "
        f"random-null rng = np.random.default_rng(SEED) (2000 draws)",
        f"wall time: {summary['wall_s']}s ; per-stage: {json.dumps(TIMING)}",
        "anchors: csa_common.assert_anchors() 31/31 PASS",
        "graphs: csa_common.build_all / strip_hub_edges / strong_backbone / hub_nodes (exact 6 FQDN)",
        f"candidate pool identical across variants = 671 attributed nodes; TOP_K={TOP_K}",
        f"HITS pinned via nstart=uniform (networkx 3.6 nx.hits -> svds(v0=None) is not process-reproducible)",
        f"decision rule: {summary['decision_rule']}",
        "outputs: graph_variant_profile.csv tie_boundary.csv top20_jaccard_by_measure.csv "
        "jaccard_matrix_deterministic.csv jaccard_matrix_worstcase.csv bootstrap_stability.csv "
        "bootstrap_raw.csv decision_table.csv nodes_stable_top20.csv top20_lists.csv c3_summary.json result.md",
    ])
    log("saved")
    print(json.dumps({k: summary[k] for k in
                      ["decision_rule", "threshold_justification", "passed_measures",
                       "failed_measures", "strict_pass_measures", "fragile_pass_measures",
                       "flip_if_compared_only_to_reference", "diag_arcs_by_own_weight"]},
                     ensure_ascii=False, indent=1))
