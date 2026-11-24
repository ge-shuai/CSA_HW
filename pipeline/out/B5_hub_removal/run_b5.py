# -*- coding: utf-8 -*-
"""B5_hub_removal — 去枢纽边的社区敏感性（≥41 seed，mean ± sd 与区间）。

口径来源（全部遵守 pipeline/CONVENTIONS.md）：
  §1 枢纽节点只用 csa_common.hub_nodes()（精确 6 个 FQDN）；前缀匹配作为"坑"单列诊断，
     不进主口径。建图只用 csa_common.build_all()。
  §3 社区数同时报 n_comms_total 与 n_comms_nonsingleton，主表用后者。
  §5 枢纽选择的 PageRank 口径 = 有向加权 nx.pagerank(Gd, weight="weight")。
  §8 所有随机过程显式传 seed；Louvain 用 csa_common.louvain() 逐 seed 跑分布。

分析：
  1) 主口径：删 6 枢纽关联边（C.strip_hub_edges，保留全部 676 节点）→ 41 个 seed 的
     社区总数 / 非单点社区数 / Q / 最大社区 / NMI / ARI，mean ± sd + 分位数；
     同时给出全图基线同 seed 分布。
  2) 孤立点两套：(a) keep_iso = 676 节点全保留（25 个零度点各成一个单点社区）；
     (b) drop_iso = 先剔除 25 个零度孤立点再跑。逐 seed 报差值。
  3) 枢纽选择敏感性：度 Top-k / PageRank Top-k（k∈{3,5,6,8,10,15}）、
     随机选 k 个"高度同量级"节点（度 >= min(deg(nav6))=63 的池，30 组 × 3 Louvain seed）、
     随机删 505 条边对照（30 组 × 3 seed）——分离"这 6 个站特殊"与"删 20% 边本身"。
  4) 流向表：seed=42 规范划分下 全图社区 → 去枢纽社区 的人数流动矩阵 + 节点级去向。
  5) 诊断：同学预实验"76 社区 Q=0.759"的三种可能来源逐一检验
     （前缀误删 / 无权图 / 删枢纽节点本身）。

输出全部写入 pipeline/out/B5_hub_removal/，不改动其它目录。
"""
from __future__ import annotations

import os
import sys
import time
import json
import random

# 允许从本模块目录直接运行：定位到 pipeline/ 根
_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPELINE = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _PIPELINE)

import numpy as np
import pandas as pd
import networkx as nx

import csa_common as C

MODULE = "B5_hub_removal"
KS = [3, 5, 6, 8, 10, 15]
SEEDS_MAIN = [42] + list(range(1, 41))        # 41 seeds，判据要求 >=30
SEEDS_SENS = list(range(1, 13))               # 确定性枢纽集 12 seeds
LOUV_SEEDS_RAND = [1, 2, 3]                   # 每个随机组 3 个 Louvain seed
N_DRAWS = 30                                  # 每组随机对照 >=30 组
T0 = time.time()
TIMING = {}


def log(msg: str) -> None:
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


def dstat(vals) -> dict:
    a = np.asarray(vals, dtype=float)
    return {
        "n": int(a.size),
        "mean": float(np.mean(a)),
        "sd": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
        "min": float(np.min(a)),
        "q25": float(np.quantile(a, 0.25)),
        "median": float(np.median(a)),
        "q75": float(np.quantile(a, 0.75)),
        "max": float(np.max(a)),
    }


def comm_lab(partition):
    """社区列表 → (node→整数标签, 按规模降序的社区列表)。标签 1..k。"""
    order = sorted(partition, key=lambda c: (-len(c), sorted(c)[0]))
    lab = {}
    for i, c in enumerate(order, 1):
        for n in c:
            lab[n] = i
    return lab, order


def main():
    # ---------------------------------------------------------------- 地基
    b = C.build_all()
    C.assert_anchors(b)
    Gu, Gd, Gb, nodes = b["Gu"], b["Gd"], b["Gb"], b["nodes"]
    hn = C.hub_nodes(Gu)                       # 精确 6 个 FQDN
    assert len(hn) == 6, f"hub_nodes() 解析到 {len(hn)} 个节点，预期 6"
    H = C.strip_hub_edges(Gu)                  # 只删边，保留全部节点
    n_edges_full = Gu.number_of_edges()
    n_edges_dh = H.number_of_edges()
    edges_removed = n_edges_full - n_edges_dh
    iso_nodes = sorted(n for n, d in H.degree() if d == 0)
    H_drop = nx.Graph(H)
    H_drop.remove_nodes_from(iso_nodes)
    lv42_check = C.louvain(H, seed=42)
    n_dh_singleton42 = lv42_check["n_comms_total"] - lv42_check["n_comms_nonsingleton"]

    hub_deg = {n: Gu.degree(n) for n in sorted(hn)}
    pool_deg_min = min(hub_deg.values())       # 63（en 枢纽）
    # "高度同量级"随机对照池：无向度 >= min(deg(nav6))/2 = 32 的节点（实测 29 个）。
    # 不用 deg>=63 的池：只有 8 个节点，k=10/15 时无法抽样。
    pool_floor = (pool_deg_min + 1) // 2       # 32
    pool = sorted(n for n in Gu.nodes() if Gu.degree(n) >= pool_floor)
    assert len(pool) >= max(KS), f"随机对照池过小: {len(pool)} < {max(KS)}"
    log(f"hub_edges removed={edges_removed} ({edges_removed / n_edges_full:.3%}) "
        f"isolated={len(iso_nodes)} pool(deg>={pool_floor})={len(pool)}")

    # 全图基线 Louvain：按 seed 缓存，供所有 NMI/ARI 对照复用
    cache_seeds = sorted(set(SEEDS_MAIN) | set(SEEDS_SENS) | set(LOUV_SEEDS_RAND))
    full = {}
    t = time.time()
    for s in cache_seeds:
        full[s] = C.louvain(Gu, seed=s)
    TIMING["full_louvain_cache"] = time.time() - t
    log(f"full-graph louvain cached for {len(full)} seeds")

    def eval_graph(Hx, seed, variant, edges_removed_x):
        lv = C.louvain(Hx, seed=seed)
        pm = C.partition_metrics(Gu, full[seed]["partition"], lv["partition"])
        return {
            "seed": seed,
            "variant": variant,
            "n_nodes_graph": Hx.number_of_nodes(),
            "n_edges_graph": Hx.number_of_edges(),
            "edges_removed": edges_removed_x,
            "isolated_nodes": sum(1 for _, d in Hx.degree() if d == 0),
            "n_comms_total": lv["n_comms_total"],
            "n_comms_nonsingleton": lv["n_comms_nonsingleton"],
            "n_singleton": lv["n_singleton"],
            "largest": lv["largest"],
            "Q": round(lv["Q"], 6),
            "NMI_vs_full": pm["NMI"],
            "ARI_vs_full": pm["ARI"],
            "n_compared": pm["n_compared"],
        }

    # ------------------------------------------------------------ 判据 1+2
    t = time.time()
    rows_main = []
    for s in SEEDS_MAIN:
        lf = full[s]
        rows_main.append({
            "seed": s, "variant": "full_graph",
            "n_nodes_graph": Gu.number_of_nodes(), "n_edges_graph": n_edges_full,
            "edges_removed": 0, "isolated_nodes": 0,
            "n_comms_total": lf["n_comms_total"], "n_comms_nonsingleton": lf["n_comms_nonsingleton"],
            "n_singleton": lf["n_singleton"], "largest": lf["largest"],
            "Q": round(lf["Q"], 6), "NMI_vs_full": 1.0, "ARI_vs_full": 1.0,
            "n_compared": Gu.number_of_nodes(),
        })
        rows_main.append(eval_graph(H, s, "dehub_keep_iso", edges_removed))
        rows_main.append(eval_graph(H_drop, s, "dehub_drop_iso", edges_removed))
    df_main = pd.DataFrame(rows_main)
    C.save_csv(df_main, MODULE, "dehub_summary.csv")
    TIMING["main_41seeds"] = time.time() - t
    log(f"main sweep done: {len(df_main)} rows")

    # ------------------------------------------------------- 判据 3 枢纽选法
    t = time.time()
    deg_sorted = sorted(Gu.nodes(), key=lambda n: (-Gu.degree(n), n))
    pr = nx.pagerank(Gd, weight="weight")      # CONVENTIONS §5 主口径：有向加权
    pr_sorted = sorted(pr.items(), key=lambda kv: (-kv[1], kv[0]))

    sens_rows = []

    def add_sens(method, k, draw_id, hubset, Hx, s):
        r = eval_graph(Hx, s, "keep_iso", n_edges_full - Hx.number_of_edges())
        r.update({
            "method": method, "k": k, "draw": draw_id,
            "louvain_seed": s,
            "n_hubs_resolved": len(hubset) if hubset is not None else 0,
            "hub_list": "|".join(sorted(hubset)) if hubset is not None and len(hubset) <= 16 else "",
            "edges_removed_pct": round((n_edges_full - Hx.number_of_edges()) / n_edges_full, 4),
        })
        r.pop("variant")
        sens_rows.append(r)

    # (0) 主口径 nav6（复用主循环结果，keep_iso）
    for s in SEEDS_MAIN:
        r = next(x for x in rows_main if x["seed"] == s and x["variant"] == "dehub_keep_iso")
        rr = dict(r)
        rr.update({
            "method": "nav6_exact", "k": 6, "draw": 0, "louvain_seed": s,
            "n_hubs_resolved": 6, "hub_list": "|".join(sorted(hn)),
            "edges_removed_pct": round(edges_removed / n_edges_full, 4),
        })
        rr.pop("variant")
        sens_rows.append(rr)

    # (1) 度 Top-k / (2) PageRank Top-k
    for k in KS:
        hset_d = set(deg_sorted[:k])
        Hd = C.strip_hub_edges(Gu, hubs=sorted(hset_d))
        hset_p = set(n for n, _ in pr_sorted[:k])
        Hp = C.strip_hub_edges(Gu, hubs=sorted(hset_p))
        for s in SEEDS_SENS:
            add_sens("deg_topk", k, 0, hset_d, Hd, s)
            add_sens("pagerank_topk", k, 0, hset_p, Hp, s)
    log("deterministic hub-choice done")

    # (3) 随机高度同量级对照：pool = deg >= min(deg(nav6))/2 = 32
    gu_edges = list(Gu.edges())
    for k in KS:
        for draw in range(N_DRAWS):
            rng = random.Random(C.SEED * 1000 + k * 100 + draw)
            hset_r = set(rng.sample(pool, k))
            Hr = C.strip_hub_edges(Gu, hubs=sorted(hset_r))
            for s in LOUV_SEEDS_RAND:
                add_sens("random_degpool", k, draw, hset_r, Hr, s)
        log(f"random_degpool k={k} done")

    # (4) 随机删等量边（505 条）对照——回答"删 20% 边本身"的效果
    for draw in range(N_DRAWS):
        rng = random.Random(C.SEED * 777 + draw)
        sel = rng.sample(gu_edges, edges_removed)
        Hx = Gu.copy()
        Hx.remove_edges_from(sel)
        for s in LOUV_SEEDS_RAND:
            add_sens("random_edges505", "NA", draw, None, Hx, s)
    log("random_edges505 done")
    TIMING["hub_choice"] = time.time() - t

    # ------------------------------------------- 判据 5 预实验数字来源诊断
    t = time.time()
    wrong_hn = sorted(n for n in Gu.nodes() if n.split(".")[0] in set(C.NAV_HUBS))  # 前缀匹配（错误示范）
    H_wrong = C.strip_hub_edges(Gu, hubs=wrong_hn)
    for s in SEEDS_MAIN:
        add_sens("prefix_broken_DIAGNOSTIC", 36, 0, set(wrong_hn), H_wrong, s)

    H_bin_dh = nx.Graph()
    H_bin_dh.add_nodes_from(H.nodes())
    H_bin_dh.add_edges_from(H.edges())        # 无权化
    for s in SEEDS_MAIN:
        add_sens("nav6_unweighted_DIAGNOSTIC", 6, 0, hn, H_bin_dh, s)

    H_nodedrop = Gu.copy()
    H_nodedrop.remove_nodes_from(hn)          # 删节点而不是只删边
    for s in SEEDS_MAIN:
        add_sens("nav6_nodedrop_DIAGNOSTIC", 6, 0, hn, H_nodedrop, s)
    log("diagnostics done")
    TIMING["diagnostics"] = time.time() - t

    df_sens = pd.DataFrame(sens_rows)
    front = ["method", "k", "draw", "louvain_seed", "n_hubs_resolved", "hub_list",
             "edges_removed", "edges_removed_pct", "isolated_nodes",
             "n_comms_total", "n_comms_nonsingleton", "largest", "Q",
             "NMI_vs_full", "ARI_vs_full", "seed", "n_nodes_graph",
             "n_edges_graph", "n_singleton", "n_compared"]
    df_sens = df_sens[front]
    C.save_csv(df_sens, MODULE, "hub_choice_sensitivity.csv")
    log(f"hub_choice_sensitivity rows={len(df_sens)}")

    # ------------------------------------------------------- 判据 4 流向表
    s0 = 42
    lab_full, comms_full = comm_lab(full[s0]["partition"])
    lv_dh = C.louvain(H, seed=s0)
    lab_dh, comms_dh = comm_lab(lv_dh["partition"])
    cells = {}
    for n in Gu.nodes():
        cells.setdefault((lab_full[n], lab_dh[n]), 0)
        cells[(lab_full[n], lab_dh[n])] += 1
    size_full = {i + 1: len(c) for i, c in enumerate(comms_full)}
    size_dh = {j: len(c) for j, c in enumerate(comms_dh, 1)}
    flow_rows = []
    for (i, j), cnt in sorted(cells.items(), key=lambda kv: -kv[1]):
        flow_rows.append({
            "full_comm": f"F{i}", "full_size": size_full[i],
            "dehub_comm": f"D{j}", "dehub_size": size_dh[j],
            "n_nodes": cnt,
            "share_of_full": round(cnt / size_full[i], 4),
            "share_of_dehub": round(cnt / size_dh[j], 4),
            "is_max_cell_of_full": int(cnt == max(v for (ii, jj), v in cells.items() if ii == i)),
            "example_nodes": "|".join(sorted(n for n in Gu.nodes() if lab_full[n] == i and lab_dh[n] == j)[:6]),
        })
    df_flow = pd.DataFrame(flow_rows)
    C.save_csv(df_flow, MODULE, "flow_table.csv")

    iso_set = set(iso_nodes)
    node_rows = []
    for n in sorted(Gu.nodes()):
        i, j = lab_full[n], lab_dh[n]
        best_j = max((jj for (ii, jj), v in cells.items() if ii == i), key=lambda jj: cells[(i, jj)])
        node_rows.append({
            "node": n,
            "full_comm": f"F{i}", "full_size": size_full[i],
            "dehub_comm": ("ISOLATED" if n in iso_set else f"D{j}"),
            "dehub_size": (0 if n in iso_set else size_dh[j]),
            "status": ("became_isolated" if n in iso_set else ("primary_flow" if j == best_j else "minority_flow")),
            "deg_u": Gu.degree(n),
        })
    df_nodes = pd.DataFrame(node_rows)
    C.save_csv(df_nodes, MODULE, "nodes_reassigned.csv")

    # --------------------------------------------------- 枢纽关联边画像
    prof_rows = []
    total_w = sum(d.get("weight", 1) for _, _, d in Gu.edges(data=True))
    removed_edges = [(u, v, Gu[u][v].get("weight", 1)) for u, v in Gu.edges()
                     if u in hn or v in hn]
    for h in sorted(hn):
        he = [(u, v, w) for (u, v, w) in removed_edges if h in (u, v)]
        cross = sum(1 for (u, v, w) in he if lab_full[u] != lab_full[v])
        prof_rows.append({
            "hub": h, "degree_u": Gu.degree(h), "strength_u": dict(Gu.degree(weight="weight"))[h],
            "n_incident_edges": len(he),
            "weight_sum": int(sum(w for _, _, w in he)),
            "cross_community_edges_seed42": cross,
            "cross_community_share": round(cross / len(he), 4) if he else np.nan,
        })
    all_cross = sum(1 for (u, v, w) in removed_edges if lab_full[u] != lab_full[v])
    base_cross = sum(1 for u, v in Gu.edges() if lab_full[u] != lab_full[v])
    df_prof = pd.DataFrame(prof_rows)
    df_prof.loc[len(df_prof)] = {
        "hub": "ALL_HUB_EDGES", "degree_u": sum(Gu.degree(h) for h in hn),
        "strength_u": sum(dict(Gu.degree(weight="weight"))[h] for h in hn),
        "n_incident_edges": len(removed_edges),
        "weight_sum": int(sum(w for _, _, w in removed_edges)),
        "cross_community_edges_seed42": all_cross,
        "cross_community_share": round(all_cross / len(removed_edges), 4),
    }
    df_prof.loc[len(df_prof)] = {
        "hub": "WHOLE_GRAPH", "degree_u": 0, "strength_u": 0,
        "n_incident_edges": n_edges_full, "weight_sum": int(total_w),
        "cross_community_edges_seed42": base_cross,
        "cross_community_share": round(base_cross / n_edges_full, 4),
    }
    C.save_csv(df_prof, MODULE, "hub_edges_profile.csv")

    # --------------------------------------------------------- 聚合统计
    agg = {}
    for var, g in df_main.groupby("variant"):
        agg[var] = {m: dstat(g[m].values) for m in
                    ["n_comms_total", "n_comms_nonsingleton", "largest", "Q", "NMI_vs_full", "ARI_vs_full"]}
    pair = df_main[df_main["variant"] != "full_graph"].pivot(index="seed", columns="variant")
    dif = {}
    for m in ["n_comms_total", "Q", "largest", "NMI_vs_full", "n_comms_nonsingleton"]:
        d = pair.loc[:, (m, "dehub_keep_iso")] - pair.loc[:, (m, "dehub_drop_iso")]
        dif[m + "_keep_minus_drop"] = dstat(d.values)
    agg["paired_keep_vs_drop"] = dif

    sens_agg = {}
    for (m, k), g in df_sens.groupby(["method", "k"], sort=False):
        sens_agg[f"{m}|k={k}"] = {
            "n_runs": int(len(g)),
            "edges_removed": dstat(g["edges_removed"].values),
            "n_comms_nonsingleton": dstat(g["n_comms_nonsingleton"].values),
            "Q": dstat(g["Q"].values),
            "largest": dstat(g["largest"].values),
            "NMI_vs_full": dstat(g["NMI_vs_full"].values),
            "ARI_vs_full": dstat(g["ARI_vs_full"].values),
        }

    flow_summary = {
        "seed": 42,
        "n_full_comms": len(comms_full), "n_dehub_comms": len(comms_dh),
        "n_cells": int((df_flow["n_nodes"] > 0).sum()),
        "full_to_dehub_primary_share_mean": float(np.mean([
            max(v for (ii, jj), v in cells.items() if ii == i) / size_full[i]
            for i in size_full])),
        "nodes_on_primary_flow": int(sum(
            max(v for (ii, jj), v in cells.items() if ii == i) for i in size_full)),
        "n_became_isolated": len(iso_nodes),
        "isolated_list": "|".join(iso_nodes),
        "isolated_full_comm_counts": {f"F{i}": int(sum(1 for n in iso_nodes if lab_full[n] == i))
                                       for i in sorted(set(lab_full[n] for n in iso_nodes))},
    }

    out = {
        "facts": {
            "hub_nodes": sorted(hn), "hub_degrees": hub_deg,
            "edges_full": n_edges_full, "edges_removed_nav6": edges_removed,
            "edges_removed_pct": round(edges_removed / n_edges_full, 4),
            "n_isolated_after_dehub": len(iso_nodes),
            "degpool_min_deg": pool_deg_min, "degpool_floor": pool_floor, "degpool_size": len(pool),
            "top12_degree_nodes": [f"{n}:{Gu.degree(n)}" for n in deg_sorted[:12]],
            "deg_topk_overlaps_nav6": {k: len(set(deg_sorted[:k]) & hn) for k in KS},
            "pr_topk_overlaps_nav6": {k: len(set(n for n, _ in pr_sorted[:k]) & hn) for k in KS},
            "deg_rank_of_nav6": {h: deg_sorted.index(h) + 1 for h in sorted(hn)},
            "pr_rank_of_nav6": {h: [n for n, _ in pr_sorted].index(h) + 1 for h in sorted(hn)},
            "removed_edges_weight_share": round(
                sum(w for _, _, w in removed_edges) / total_w, 4),
            "n_dh_singleton_keepiso_seed42": int(n_dh_singleton42),
        },
        "main_summary": agg,
        "sensitivity_summary": sens_agg,
        "flow_summary": flow_summary,
        "versions": C.versions(),
        "seeds_main": SEEDS_MAIN, "seeds_sens": SEEDS_SENS, "louvain_seeds_rand": LOUV_SEEDS_RAND,
        "n_random_draws": N_DRAWS,
    }
    p = os.path.join(C.module_dir(MODULE), "B5_summary.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    log("B5_summary.json written")

    # ------------------------------------------------------------- RUN.txt
    wall = time.time() - T0
    v = C.versions()
    lines = [
        "B5_hub_removal RUN.txt",
        "command: python run_b5.py  (working dir = pipeline/out/B5_hub_removal; 亦可从任意目录运行)",
        f"python={v['python']} networkx={v['networkx']} pandas={v['pandas']} numpy={v['numpy']} "
        f"PYTHONHASHSEED={v['PYTHONHASHSEED']} SEED={C.SEED}",
        f"total wall time: {wall:.1f}s; per-stage: " + json.dumps({k: round(x, 1) for k, x in TIMING.items()}),
        f"anchors: csa_common.assert_anchors() 31/31 PASS",
        f"seeds: main={len(SEEDS_MAIN)} (42 + 1..40), deterministic hub-choice={len(SEEDS_SENS)}, "
        f"random draws={N_DRAWS} x louvain seeds {LOUV_SEEDS_RAND}",
        "random control pool: nodes with degree_u >= ceil(min(deg(nav6))/2) "
        f"= {pool_floor}, pool size = {len(pool)} (min nav6 hub degree = {pool_deg_min}); "
        "rng seeds = C.SEED*1000 + k*100 + draw (nodes), "
        "C.SEED*777 + draw (edges)",
        f"louvain run count ~ {len(df_main) + len(df_sens):,}",
        "outputs: dehub_summary.csv hub_choice_sensitivity.csv flow_table.csv "
        "nodes_reassigned.csv hub_edges_profile.csv B5_summary.json result.md",
    ]
    with open(os.path.join(C.module_dir(MODULE), "RUN.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log("RUN.txt written")
    log(json.dumps(out["facts"], ensure_ascii=False, default=str)[:2000])


if __name__ == "__main__":
    main()
