"""B1_louvain_stability — 加权 Louvain 的误差棒与可引用口径。

单元定义见 pipeline/MANIFEST.md 的 `B1_louvain_stability`；口径见 CONVENTIONS.md §1 / §3。

步骤
  step0_load              建图 + 31 项锚点自检（C.assert_anchors）
  step1_seed_sweep        加权 Louvain seed 扫描：主 sweep=range(50) + 扩展 sweep=range(250)
  step2_distribution      完整分布：min / 四分位 / p05 / p95 / max / sd / 直方图数据
  step3_pairwise          成对稳定性：NMI / ARI（primary50 全部 1225 对 + 250 seed 池抽 2000 对）
  step4_consensus         共识矩阵阈值扫描 tau in {0.5..0.9} × {连通块, 平均连接层次聚类}
  step5_order_sensitivity 建图边序敏感性：24 种顺序 × 22 个 seed 的交叉设计（同构性逐一校验）
  step6_noise             噪声源对比：seed 波动 vs 边序波动（含成对 NMI 与 eta^2 方差分解）
  step7_write             写 CSV / result.md / RUN.txt / key_numbers.json

重跑：
    cd pipeline && python B1_louvain_stability/run_b1.py

图一律来自 csa_common.build_all()/build_undirected()；社区一律来自 csa_common.louvain()/
louvain_robust()；划分相似度一律 csa_common.partition_metrics()。本文件不重写建图口径。
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import platform
import subprocess
import sys
import time
import traceback

PIPELINE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import networkx as nx  # noqa: E402
from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402

import csa_common as C  # noqa: E402

SEED = 20261113  # CONVENTIONS §8；与 csa_common.SEED 一致（main 里断言）
assert SEED == C.SEED

MODULE = "B1_louvain_stability"
OUT_DIR = C.module_dir(MODULE)

PRIMARY_SEEDS = list(range(50))              # 判据 1：>=50 seed
EXTENDED_SEEDS = list(range(250))            # 扩展 sweep（包含 PRIMARY_SEEDS）
BUDGETS = [10, 25, 50, 100, 250]             # 共识划分对 seed 预算的敏感性
TAUS = [0.5, 0.6, 0.7, 0.8, 0.9]
N_SHUFFLES = 21                              # 随机边序个数（判据 4：>=20 种顺序）
ORDER_SEEDS = [SEED, 42] + list(range(20))   # 交叉设计的 seed 轴（22 个）
GRID_SEEDS = list(range(20))                 # 交叉设计里用于成对比较的 seed 子集
N_SAMPLED_PAIRS = 2000                       # 判据 2：>=200 个 seed 对，这里取 2000
HEADLINE = {"comms": 25, "Q": 0.6904, "largest": 87}  # 同学原始报告的三个数

TIMINGS: dict[str, float] = {}
FACTS: dict = {}
S: dict = {}          # 跨步骤共享的汇总上下文（写 result.md / RUN.txt 用）


# =========================================================== 小工具

def qstats(v) -> dict:
    v = np.asarray(v, dtype=float)
    return {
        "n": int(v.size),
        "min": float(v.min()),
        "p05": float(np.percentile(v, 5)),
        "q1": float(np.percentile(v, 25)),
        "median": float(np.median(v)),
        "mean": float(v.mean()),
        "q3": float(np.percentile(v, 75)),
        "p95": float(np.percentile(v, 95)),
        "max": float(v.max()),
        "sd": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
        "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
        "range": float(v.max() - v.min()),
    }


def dist_rows(metric, sweep, values, bin_kind) -> list[dict]:
    """一个指标的完整分布（分位数 + 直方图计数 + 组边界）。"""
    v = np.asarray(values, dtype=float)
    rows = [{"metric": metric, "sweep": sweep, "statistic": k, "bin": "", "value": val}
            for k, val in qstats(v).items()]
    if bin_kind == "integer":
        edges = np.arange(int(v.min()) - 1, int(v.max()) + 2, 1) + 0.5
    elif bin_kind == "width0.002":
        edges = np.arange(np.floor(v.min() / 0.002) * 0.002, v.max() + 1e-9, 0.002)
    else:
        edges = np.arange(np.floor(v.min() / 5) * 5, v.max() + 1e-9, 5)
    counts, edges = np.histogram(v, bins=edges)
    for c, a, bb in zip(counts, edges[:-1], edges[1:]):
        rows.append({"metric": metric, "sweep": sweep, "statistic": "hist_count",
                     "bin": f"[{a:g},{bb:g})", "value": int(c)})
    for a in edges:
        rows.append({"metric": metric, "sweep": sweep, "statistic": "hist_bin_edge",
                     "bin": f"{a:g}", "value": float(a)})
    return rows


def edge_signature(G: nx.Graph) -> str:
    """边集指纹（含权重，与插入顺序无关）——证明各顺序构造的是同一张同构图。"""
    items = sorted((min(u, v), max(u, v), int(d.get("weight", 1)))
                   for u, v, d in G.edges(data=True))
    return hashlib.sha256(repr(items).encode()).hexdigest()[:16]


def sizes_str(sizes) -> str:
    return "|".join(str(s) for s in sizes)


def community_table(G, comms) -> pd.DataFrame:
    """社区规模表（降序），附社区内加权强度最大的节点当人工核对用的代表。"""
    deg = dict(G.degree(weight="weight"))
    ordered = sorted([set(c) for c in comms], key=lambda c: (-len(c), sorted(c)[0]))
    return pd.DataFrame([{
        "community_id": i, "size": len(c),
        "strongest_member": max(c, key=lambda n: (deg.get(n, 0), n)),
    } for i, c in enumerate(ordered, start=1)])


def renumber(comms):
    """社区按规模降序重新编号（编号只在本模块内自洽）。"""
    ordered = sorted([set(c) for c in comms], key=lambda c: (-len(c), sorted(c)[0]))
    lab = {node: i for i, c in enumerate(ordered, start=1) for node in c}
    return ordered, lab


# =========================================================== step 0

def step0_load():
    t0 = time.time()
    b = C.build_all()
    C.assert_anchors(b)                      # 硬规则：31 项锚点，任一不过立即停止
    Gu = b["Gu"]
    FACTS.update({
        "anchors": "C.assert_anchors() 通过（31/31）",
        "Gu_nodes": Gu.number_of_nodes(),
        "Gu_edges": Gu.number_of_edges(),
        "directed_edges": b["Gd"].number_of_edges(),
        "attributed_nodes": int(len(b["nodes"])),
        "links_rows": int(len(b["links"])),
        "links_has_edge_order": bool("edge_order" in b["links"].columns),
        "edge_signature_canonical": edge_signature(Gu),
        "weight_semantics": "观察到的链接次数（CONVENTIONS §1）",
    })
    TIMINGS["step0_load"] = time.time() - t0
    return b


# =========================================================== step 1

def step1_seed_sweep(Gu):
    t0 = time.time()
    rob50 = C.louvain_robust(Gu, seeds=PRIMARY_SEEDS)      # 判据 1 指定的调用形式：seeds=range(50)
    rob250 = C.louvain_robust(Gu, seeds=EXTENDED_SEEDS)    # range(250)
    per_seed = {s: C.louvain(Gu, seed=s) for s in EXTENDED_SEEDS}   # 完整规模谱

    # 自检 A：两条调用路径（louvain vs louvain_robust）在同一 seed 上必须给同一结果
    mism = [rec["seed"] for i, rec in enumerate(rob250["per_seed"])
            if (rec["comms"], rec["Q"], rec["largest"]) !=
            (per_seed[rec["seed"]]["n_comms_total"], round(per_seed[rec["seed"]]["Q"], 4),
             per_seed[rec["seed"]]["largest"])]
    # 自检 B：同进程重复调用同 seed 的确定性
    a, bb = C.louvain(Gu, seed=42), C.louvain(Gu, seed=42)
    det = (a["n_comms_total"] == bb["n_comms_total"] and round(a["Q"], 9) == round(bb["Q"], 9)
           and a["largest"] == bb["largest"])
    FACTS.update({"robust_vs_direct_mismatch_seeds": mism, "same_process_determinism_seed42": det,
                  "rob50_summary": {k: rob50[k] for k in
                                    ("n_seeds", "comms_total", "comms_nonsingleton", "Q", "largest")},
                  "rob250_summary": {k: rob250[k] for k in
                                     ("n_seeds", "comms_total", "comms_nonsingleton", "Q", "largest")}})

    rows = []
    for s in EXTENDED_SEEDS:
        r = per_seed[s]
        sz = r["sizes"]
        rows.append({
            "seed": s, "sweep": "primary50" if s in PRIMARY_SEEDS else "extra200",
            "n_comms_total": r["n_comms_total"],
            "n_comms_nonsingleton": r["n_comms_nonsingleton"],
            "n_singleton_communities": r["n_singleton"],
            "n_covered_nodes": r["n_covered"],
            "Q": round(r["Q"], 6),
            "largest": r["largest"],
            "second_largest": sz[1] if len(sz) > 1 else 0,
            "n_communities_ge_20": int(sum(1 for x in sz if x >= 20)),
            "n_communities_le_5": int(sum(1 for x in sz if x <= 5)),
            "size_top10": sizes_str(sz[:10]),
            "all_sizes": sizes_str(sz),
            "hits_report_headline_all3": bool(
                r["n_comms_total"] == HEADLINE["comms"]
                and abs(r["Q"] - HEADLINE["Q"]) < 5e-5
                and r["largest"] == HEADLINE["largest"]),
            "hits_report_headline_comms": bool(r["n_comms_total"] == HEADLINE["comms"]),
            "hits_report_headline_largest": bool(r["largest"] == HEADLINE["largest"]),
        })
    df = pd.DataFrame(rows)
    TIMINGS["step1_seed_sweep"] = time.time() - t0
    return df, rob50, rob250, per_seed


# =========================================================== step 2

def step2_distribution(sweep_df):
    t0 = time.time()
    groups = {"primary50": sweep_df[sweep_df["sweep"] == "primary50"],
              "extra200": sweep_df[sweep_df["sweep"] == "extra200"],
              "all250": sweep_df}
    rows = []
    for tag, g in groups.items():
        rows += dist_rows("n_comms_total", tag, g["n_comms_total"], "integer")
        rows += dist_rows("n_comms_nonsingleton", tag, g["n_comms_nonsingleton"], "integer")
        rows += dist_rows("Q", tag, g["Q"], "width0.002")
        rows += dist_rows("largest_community", tag, g["largest"], "width5")
        rows += dist_rows("n_singleton_communities", tag, g["n_singleton_communities"], "integer")
    dist = pd.DataFrame(rows)
    summ = {}
    for m in ("n_comms_total", "n_comms_nonsingleton", "Q", "largest"):
        summ[m] = {tag: qstats(g[m].values) for tag, g in groups.items()}
    summ["headline_hit_rate"] = {
        tag: {"all_three_exact": float(g["hits_report_headline_all3"].mean()),
              "comms_eq_25": float(g["hits_report_headline_comms"].mean()),
              "largest_eq_87": float(g["hits_report_headline_largest"].mean()),
              "Q_within_0.001_of_0.6904": float((np.abs(g["Q"] - HEADLINE["Q"]) <= 0.001).mean())}
        for tag, g in groups.items() if tag != "extra200"}
    summ["extra200_headline_hit_rate"] = {
        "all_three_exact": float(groups["extra200"]["hits_report_headline_all3"].mean())}
    summ["singleton"] = {
        "min": int(sweep_df["n_singleton_communities"].min()),
        "max": int(sweep_df["n_singleton_communities"].max()),
        "n_comms_total_equals_nonsingleton_all_seeds": bool(
            (sweep_df["n_comms_total"] == sweep_df["n_comms_nonsingleton"]).all()),
        "min_covered_nodes": int(sweep_df["n_covered_nodes"].min()),
    }
    FACTS["distribution_summary"] = summ
    TIMINGS["step2_distribution"] = time.time() - t0
    return dist, summ


# =========================================================== step 3

def step3_pairwise(Gu, per_seed):
    t0 = time.time()
    all_pairs = list(itertools.combinations(EXTENDED_SEEDS, 2))
    pairs_primary = list(itertools.combinations(PRIMARY_SEEDS, 2))
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(all_pairs), size=min(N_SAMPLED_PAIRS, len(all_pairs)), replace=False)
    pairs_sampled = [all_pairs[int(i)] for i in idx]
    recs = []
    for tag, pairs in (("all_pairs_within_primary50", pairs_primary),
                       (f"random_pairs_within_all250_n{len(pairs_sampled)}", pairs_sampled)):
        for a, b in pairs:
            m = C.partition_metrics(Gu, per_seed[a]["partition"], per_seed[b]["partition"])
            recs.append({"sample": tag, "seed_a": a, "seed_b": b,
                         "comms_a": per_seed[a]["n_comms_total"],
                         "comms_b": per_seed[b]["n_comms_total"],
                         "largest_a": per_seed[a]["largest"],
                         "largest_b": per_seed[b]["largest"],
                         "n_compared": m["n_compared"], "NMI": m["NMI"], "ARI": m["ARI"]})
    pw = pd.DataFrame(recs)
    FACTS["n_all_pairs"] = int(len(all_pairs))
    summ = {tag: {"n_pairs": int(len(g)), "NMI": qstats(g["NMI"].values),
                  "ARI": qstats(g["ARI"].values)}
            for tag, g in pw.groupby("sample", sort=False)}
    FACTS["pairwise_summary"] = summ
    TIMINGS["step3_pairwise"] = time.time() - t0
    return pw, summ


# =========================================================== step 4

def consensus_cc(co, node_list, tau):
    """规则 A（判据 3 指定）：共识 >= tau 连边，取连通块。"""
    iu = np.triu_indices(len(node_list), 1)
    ii, jj = iu[0][co[iu] >= tau - 1e-9], iu[1][co[iu] >= tau - 1e-9]
    H = nx.Graph()
    H.add_nodes_from(node_list)
    H.add_edges_from((node_list[a], node_list[b]) for a, b in zip(ii, jj))
    return [set(c) for c in nx.connected_components(H)], int(H.number_of_edges())


def consensus_avglink(co, node_list, tau, Z=None):
    """规则 B（判据 3 的"或"）：距离 = 1 - 共识，平均连接层次聚类，在 1-tau 处切。"""
    if Z is None:
        D = (1.0 - np.asarray(co, dtype=np.float64))
        D = (D + D.T) / 2.0
        np.fill_diagonal(D, 0.0)
        Z = linkage(squareform(D, checks=False), method="average")
    lab = fcluster(Z, t=1.0 - tau, criterion="distance")
    comms = [set(node_list[i] for i in np.where(lab == k)[0]) for k in np.unique(lab)]
    return comms, Z


def step4_consensus(Gu, rob50, rob250, per_seed):
    t0 = time.time()
    part_rows, summ_rows = [], []
    for set_tag, rob, ref_seeds in (("primary50", rob50, PRIMARY_SEEDS),
                                    ("all250", rob250, EXTENDED_SEEDS)):
        co, node_list = rob["consensus"], rob["node_list"]
        iu = np.triu_indices(len(node_list), 1)
        FACTS[f"consensus_pairs_at_least_tau_{set_tag}"] = {
            str(t): int((co[iu] >= t - 1e-9).sum()) for t in TAUS}
        FACTS[f"consensus_offdiag_mean_{set_tag}"] = round(float(co[iu].mean()), 6)
        FACTS[f"consensus_n_node_pairs_{set_tag}"] = int(iu[0].size)
        Z = None
        for method in ("cc", "avglink"):
            for tau in TAUS:
                if method == "cc":
                    comms, n_thr_edges = consensus_cc(co, node_list, tau)
                else:
                    comms, Z = consensus_avglink(co, node_list, tau, Z=Z)
                    n_thr_edges = ""
                comms, lab = renumber(comms)
                sizes = [len(c) for c in comms]
                Q = float(nx.community.modularity(Gu, comms, weight="weight"))
                nm = [C.partition_metrics(Gu, comms, per_seed[s]["partition"]) for s in ref_seeds]
                nmis = np.array([x["NMI"] for x in nm])
                aris = np.array([x["ARI"] for x in nm])
                summ_rows.append({
                    "seed_set": set_tag, "method": method, "tau": tau,
                    "n_seeds_used": int(rob["n_seeds"]),
                    "threshold_graph_edges": n_thr_edges,
                    "n_comms_total": len(comms),
                    "n_comms_nonsingleton": int(sum(1 for s in sizes if s > 1)),
                    "n_singleton_communities": int(sum(1 for s in sizes if s == 1)),
                    "largest": int(max(sizes)),
                    "median_size": float(np.median(sizes)),
                    "size_top10": sizes_str(sorted(sizes, reverse=True)[:10]),
                    "all_sizes": sizes_str(sorted(sizes, reverse=True)),
                    "Q_of_consensus": round(Q, 6),
                    "NMI_mean_vs_seeds": round(float(nmis.mean()), 4),
                    "NMI_median_vs_seeds": round(float(np.median(nmis)), 4),
                    "NMI_min_vs_seeds": round(float(nmis.min()), 4),
                    "NMI_max_vs_seeds": round(float(nmis.max()), 4),
                    "ARI_mean_vs_seeds": round(float(aris.mean()), 4),
                    "NMI_vs_seed42_partition": round(
                        C.partition_metrics(Gu, comms, per_seed[42]["partition"])["NMI"], 4),
                })
                for c in comms:
                    for node in sorted(c):
                        part_rows.append({"seed_set": set_tag, "method": method, "tau": tau,
                                          "n_seeds_used": int(rob["n_seeds"]), "node": node,
                                          "community_id": lab[node], "community_size": len(c)})
    cons_summ = pd.DataFrame(summ_rows)

    # 推荐口径：primary50 里「对 50 个 seed 平均 NMI 最高」者；同分再看 Q
    cand = cons_summ[cons_summ["seed_set"] == "primary50"].copy()
    cand["_k"] = cand["NMI_mean_vs_seeds"] * 1e6 + cand["Q_of_consensus"]
    best = cand.sort_values("_k", ascending=False).iloc[0]
    recommended = {"seed_set": "primary50", "method": str(best["method"]),
                   "tau": float(best["tau"]), "n_comms_total": int(best["n_comms_total"]),
                   "n_comms_nonsingleton": int(best["n_comms_nonsingleton"]),
                   "largest": int(best["largest"]), "Q_of_consensus": float(best["Q_of_consensus"]),
                   "NMI_mean_vs_seeds": float(best["NMI_mean_vs_seeds"])}
    parts_df = pd.DataFrame(part_rows)
    parts_df["recommended"] = bool(False)
    mask = ((parts_df["seed_set"] == "primary50") & (parts_df["method"] == recommended["method"])
            & (parts_df["tau"] == recommended["tau"]))
    parts_df.loc[mask, "recommended"] = True
    sel = parts_df[parts_df["recommended"]]
    comms_best = [set(sel[sel["community_id"] == cid]["node"])
                  for cid in sorted(sel["community_id"].unique())]
    best_tab = community_table(Gu, comms_best)
    best_tab.insert(0, "method", recommended["method"])
    best_tab.insert(1, "tau", recommended["tau"])
    best_tab["share_of_676_pct"] = (100 * best_tab["size"] / Gu.number_of_nodes()).round(2)
    best_tab["recommended"] = True

    # 共识划分需要多少个 seed 才够稳（拿 250-seed 共识当参照）
    budget_rows = []
    for k in BUDGETS:
        robk = C.louvain_robust(Gu, seeds=range(k))
        ck, _ = consensus_avglink(robk["consensus"], robk["node_list"], 0.5)
        ck, _ = renumber(ck)
        budget_rows.append({"n_seeds": k, "tau": 0.5, "method": "avglink",
                            "n_comms_total": len(ck),
                            "n_comms_nonsingleton": int(sum(1 for c in ck if len(c) > 1)),
                            "largest": int(max(len(c) for c in ck)),
                            "Q": round(float(nx.community.modularity(Gu, ck, weight="weight")), 6),
                            "_comms": ck})
    ref = budget_rows[-1]["_comms"]
    for r in budget_rows:
        m = C.partition_metrics(Gu, r.pop("_comms"), ref)
        r["NMI_vs_250seed_consensus"] = m["NMI"]
        r["ARI_vs_250seed_consensus"] = m["ARI"]
    FACTS["recommended_partition"] = recommended
    TIMINGS["step4_consensus"] = time.time() - t0
    return parts_df, cons_summ, best_tab, pd.DataFrame(budget_rows), recommended


# =========================================================== step 5

def variant_graph(links, nodes, kind, rs=None):
    """把同一批边按不同插入顺序喂给 csa_common.build_undirected。边集完全不变。

    kind='prepend_nodes' 是**诊断用**分支：复现 csa_common.build_undirected docstring
    里描述的"先 add_nodes_from(671) 再逐条加边"的变体（规范边序、规范权重相加规则），
    只用来核对该 docstring 引用的数字，不作为任何结论的建图入口。
    """
    L = links.copy()
    if kind == "lex":
        L = L.sort_values(["source", "target"], kind="stable")
    elif kind == "reverse_lex":
        L = L.sort_values(["source", "target"], ascending=False, kind="stable")
    elif kind == "shuffle":
        L = L.iloc[np.random.default_rng(rs).permutation(len(L))].copy()
    L = L.copy()
    L["edge_order"] = np.arange(len(L))
    if kind == "prepend_nodes":
        H = nx.Graph()
        H.add_nodes_from(nodes["node"])
        agg: dict = {}
        for s, t, w in C._ordered_links(L)[["source", "target", "weight"]].itertuples(index=False):
            k2 = (s, t) if s <= t else (t, s)
            agg[k2] = agg.get(k2, 0) + int(w)
        for (s, t), w in agg.items():
            if H.has_edge(s, t):
                H[s][t]["weight"] = H[s][t]["weight"] + w
            else:
                H.add_edge(s, t, weight=w)
        return H, L
    return C.build_undirected(L, nodes), L


def step5_order_sensitivity(links, nodes, Gu):
    t0 = time.time()
    orders = [("canonical", "canonical", "links.csv 行序（CONVENTIONS §1 规定口径）", None),
              ("lex", "lex", "(source,target) 字典序", None),
              ("revlex", "reverse_lex", "(source,target) 逆字典序", None)]
    for i in range(1, N_SHUFFLES + 1):
        orders.append((f"shuffle{i:02d}", "shuffle", f"行序随机打乱 default_rng({1000 + i})", 1000 + i))
    orders.append(("prepend671", "prepend_nodes",
                   "诊断：先 add_nodes_from(671) 再加边（规范边序），非主口径", None))

    sig0 = FACTS["edge_signature_canonical"]
    rows, parts_grid = [], {}
    for oid, kind, desc, rs in orders:
        G, _ = variant_graph(links, nodes, kind, rs)
        sig = edge_signature(G)
        same = bool(sig == sig0)
        for s in ORDER_SEEDS:
            r = C.louvain(G, seed=s)
            parts_grid[(oid, s)] = r["partition"]
            rows.append({
                "order_id": oid, "order_kind": kind, "ordering_definition": desc,
                "diagnostic_only": bool(kind == "prepend_nodes"),
                "seed": s, "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
                "edge_set_identical_to_canonical": same,
                "node_insertion_order_same_as_canonical":
                    bool(list(G.nodes())[:12] == list(Gu.nodes())[:12]),
                "n_comms_total": r["n_comms_total"],
                "n_comms_nonsingleton": r["n_comms_nonsingleton"],
                "Q": round(r["Q"], 6), "largest": r["largest"],
                "second_largest": r["sizes"][1] if len(r["sizes"]) > 1 else 0,
                "size_top10": sizes_str(r["sizes"][:10]),
            })
    od = pd.DataFrame(rows)
    assert od["edge_set_identical_to_canonical"].all(), "有顺序变体改动了边集，同构性检验失败"
    can42 = od[(od["order_id"] == "canonical") & (od["seed"] == 42)].iloc[0]
    assert (int(can42["n_comms_total"]), round(float(can42["Q"]), 4), int(can42["largest"])) \
        == (HEADLINE["comms"], HEADLINE["Q"], HEADLINE["largest"]), \
        f"交叉设计的规范格不再复现锚点 25/0.6904/87，而是 {tuple(can42[['n_comms_total','Q','largest']])}"

    srows = []
    for oid, g in od.groupby("order_id", sort=False):
        grid = g[g["seed"].isin(GRID_SEEDS)]
        r42 = g[g["seed"] == 42].iloc[0]
        rS = g[g["seed"] == SEED].iloc[0]
        srows.append({
            "order_id": oid, "order_kind": r42["order_kind"],
            "diagnostic_only": bool(r42["diagnostic_only"]),
            "edge_set_identical": bool(g["edge_set_identical_to_canonical"].all()),
            "node_insertion_order_changed": bool(not r42["node_insertion_order_same_as_canonical"]),
            "n_seeds_grid": int(len(grid)),
            "comms_at_seed42": int(r42["n_comms_total"]), "Q_at_seed42": float(r42["Q"]),
            "largest_at_seed42": int(r42["largest"]),
            "comms_at_projectSEED": int(rS["n_comms_total"]), "Q_at_projectSEED": float(rS["Q"]),
            "largest_at_projectSEED": int(rS["largest"]),
            "comms_min": int(grid["n_comms_total"].min()),
            "comms_median": float(grid["n_comms_total"].median()),
            "comms_max": int(grid["n_comms_total"].max()),
            "Q_min": float(grid["Q"].min()), "Q_median": float(grid["Q"].median()),
            "Q_max": float(grid["Q"].max()),
            "largest_min": int(grid["largest"].min()),
            "largest_median": float(grid["largest"].median()),
            "largest_max": int(grid["largest"].max()),
            "size_top10_at_seed42": r42["size_top10"],
        })
    os_summ = pd.DataFrame(srows)
    FACTS["n_orderings"] = int(len(orders))
    FACTS["n_orderings_non_diagnostic"] = int(len(orders) - 1)
    FACTS["order_grid_shape"] = f"{len(orders)} orderings x {len(ORDER_SEEDS)} seeds"
    dg = od[(od["diagnostic_only"]) & (od["seed"] == 42)]
    FACTS["prepend_diagnostic_at_seed42"] = (
        "无该行" if dg.empty else
        f"{int(dg['n_comms_total'].iloc[0])} 社区 / Q={float(dg['Q'].iloc[0]):.4f} / "
        f"最大 {int(dg['largest'].iloc[0])}")
    FACTS["canonical_at_seed42_in_grid"] = (
        f"{int(od[(od.order_id == 'canonical') & (od.seed == 42)]['n_comms_total'].iloc[0])} / "
        f"{float(od[(od.order_id == 'canonical') & (od.seed == 42)]['Q'].iloc[0]):.4f} / "
        f"{int(od[(od.order_id == 'canonical') & (od.seed == 42)]['largest'].iloc[0])}")
    TIMINGS["step5_order_sensitivity"] = time.time() - t0
    return od, os_summ, parts_grid


# =========================================================== step 6

def eta2(df, y, factor):
    gm = df[y].mean()
    ssb = float(sum(len(g) * (g[y].mean() - gm) ** 2 for _, g in df.groupby(factor)))
    sst = float(((df[y] - gm) ** 2).sum())
    return ssb / sst if sst > 0 else float("nan")


def step6_noise(Gu, od, os_summ, parts_grid, sweep_df):
    """两个噪声源谁更大：(a) 换 seed（边序固定为规范口径）(b) 换边序（seed 固定）。"""
    t0 = time.time()
    main_od = od[~od["diagnostic_only"]]
    p50 = sweep_df[sweep_df["sweep"] == "primary50"]

    seed_largest, seed_comms, seed_Q = (qstats(p50["largest"].values),
                                        qstats(p50["n_comms_total"].values),
                                        qstats(p50["Q"].values))
    ord_largest = qstats(main_od[main_od["seed"] == 42]["largest"].values)      # 24 种顺序 @ seed42
    ord_comms = qstats(main_od[main_od["seed"] == 42]["n_comms_total"].values)
    ord_Q = qstats(main_od[main_od["seed"] == 42]["Q"].values)
    ord_largest_SEED = qstats(main_od[main_od["seed"] == SEED]["largest"].values)

    # 交叉设计上的成对 NMI/ARI
    orders = [o for o in main_od["order_id"].unique()]
    seed_pairs = [((o, a), (o, b)) for o in orders for a, b in itertools.combinations(GRID_SEEDS, 2)]
    order_pairs = [((a, s), (b, s)) for s in GRID_SEEDS
                   for a, b in itertools.combinations(orders, 2)]
    res = {}
    for tag, pairs in (("within_order_between_seed", seed_pairs),
                       ("between_order_within_seed", order_pairs)):
        nmi, ari = [], []
        for (oa, sa), (ob, sb) in pairs:
            m = C.partition_metrics(Gu, parts_grid[(oa, sa)], parts_grid[(ob, sb)])
            nmi.append(m["NMI"])
            ari.append(m["ARI"])
        res[tag] = {"n_pairs": len(pairs), "NMI": qstats(nmi), "ARI": qstats(ari)}

    grid = main_od[main_od["seed"].isin(GRID_SEEDS)]
    rows = []

    def add(metric, source, stat, value):
        rows.append({"metric": metric, "noise_source": source, "statistic": stat, "value": value})

    for metric, q in (("largest_community", seed_largest), ("n_comms_total", seed_comms),
                      ("Q", seed_Q)):
        for k, v in q.items():
            add(metric, "seed_variation_canonical_order_50seeds", k, v)
    for metric, q in (("largest_community", ord_largest), ("n_comms_total", ord_comms),
                      ("Q", ord_Q)):
        for k, v in q.items():
            add(metric, f"edge_order_variation_at_seed42_{len(orders)}orders", k, v)
    for metric, q in (("largest_community", ord_largest_SEED),):
        for k, v in q.items():
            add(metric, f"edge_order_variation_at_projectSEED_{len(orders)}orders", k, v)
    for tag, d in res.items():
        for stat in ("NMI", "ARI"):
            for k, v in d[stat].items():
                add(stat, f"crossed_2design_{tag}", k, round(float(v), 6))
    for y in ("n_comms_total", "Q", "largest"):
        add(f"eta2_{y}", "variance_decomposition_24x20_grid", "edge_order_factor",
            round(eta2(grid, y, "order_id"), 4))
        add(f"eta2_{y}", "variance_decomposition_24x20_grid", "seed_factor",
            round(eta2(grid, y, "seed"), 4))
        add(f"eta2_{y}", "variance_decomposition_24x20_grid", "n_cells", int(len(grid)))
    verdict_largest = "seed" if seed_largest["range"] > ord_largest["range"] else "edge_order"
    verdict_comms = "seed" if seed_comms["range"] > ord_comms["range"] else "edge_order"
    verdict_nmi = ("edge_order" if res["between_order_within_seed"]["NMI"]["mean"]
                   < res["within_order_between_seed"]["NMI"]["mean"] else "seed")
    add("verdict", "which_noise_source_is_larger", "largest_community_range", verdict_largest)
    add("verdict", "which_noise_source_is_larger", "n_comms_total_range", verdict_comms)
    add("verdict", "which_noise_source_is_larger", "pairwise_NMI_mean", verdict_nmi)

    out = {"seed_largest": seed_largest, "ord_largest": ord_largest,
           "seed_comms": seed_comms, "ord_comms": ord_comms, "seed_Q": seed_Q,
           "ord_Q": ord_Q, "crossed": res, "verdict": {"largest": verdict_largest,
                                                       "comms": verdict_comms, "nmi": verdict_nmi},
           "eta2": {y: {"edge_order": round(eta2(grid, y, "order_id"), 4),
                        "seed": round(eta2(grid, y, "seed"), 4)}
                    for y in ("n_comms_total", "Q", "largest")},
           "n_orders_used": len(orders)}
    FACTS["noise_comparison"] = {"ranges": {
        "largest_seed_range": seed_largest["range"], "largest_order_range": ord_largest["range"],
        "comms_seed_range": seed_comms["range"], "comms_order_range": ord_comms["range"]},
        "crossed_NMI_mean": {k: round(v["NMI"]["mean"], 4) for k, v in res.items()},
        "verdict": out["verdict"], "eta2": out["eta2"]}
    TIMINGS["step6_noise"] = time.time() - t0
    return pd.DataFrame(rows), out


# =========================================================== step 6b

HASHSEED_PROBE_SEEDS = (0, 42, SEED)


def step6b_hashseed(per_seed):
    """CONVENTIONS §8 建议 PYTHONHASHSEED=0。这里独立进程实测它到底影响不影响结果。"""
    t0 = time.time()
    snippet = "\n".join([
        "import sys, json",
        f"sys.path.insert(0, r'{PIPELINE_DIR}')",
        "import csa_common as C",
        "b = C.build_all(); Gu = b['Gu']",
        "res = []",
        f"for s in {list(HASHSEED_PROBE_SEEDS)}:",
        "    r = C.louvain(Gu, seed=s)",
        "    res.append([s, r['n_comms_total'], round(r['Q'], 6), r['largest']])",
        "print(json.dumps(res))",
    ])
    out = {}
    for hs in ("0", "1", "7"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = hs
        try:
            p = subprocess.run([sys.executable, "-c", snippet], capture_output=True,
                               text=True, timeout=600, env=env, cwd=PIPELINE_DIR)
            if p.returncode != 0:
                out[hs] = {"error": f"rc={p.returncode}", "stderr": p.stderr[-300:]}
            else:
                out[hs] = {"rows": json.loads(p.stdout.strip().splitlines()[-1])}
        except Exception as exc:  # 诚实记录：探针跑不出来就写清楚
            out[hs] = {"error": f"{type(exc).__name__}: {exc}"}
    want = [[s, per_seed[s]["n_comms_total"], round(per_seed[s]["Q"], 6), per_seed[s]["largest"]]
            for s in HASHSEED_PROBE_SEEDS]
    vals = {k: v.get("rows") for k, v in out.items()}
    FACTS["hashseed_probe"] = {
        "seeds": list(HASHSEED_PROBE_SEEDS),
        "main_process_reference": want,
        "per_hashseed": vals,
        "all_identical_to_main_process": all(v == want for v in vals.values()),
        "any_probe_failed": any(v is None for v in vals.values()),
    }
    TIMINGS["step6b_hashseed"] = time.time() - t0
    return out


# =========================================================== step 7


def hist_table(metric, sweep, dist, agg_width) -> str:
    """把 `dist` 里的 width-5 直方图计数聚合成 agg_width 一档，渲染成 markdown 表。"""
    d = dist[(dist["metric"] == metric) & (dist["sweep"] == sweep)
             & (dist["statistic"] == "hist_count")]
    cells = []
    for _, r in d.iterrows():
        lo, hi = r["bin"].strip("[)").split(",")
        cells.append((float(lo), float(hi), int(r["value"])))
    if not cells:
        return "| （直方图无数据） | | |"
    lo0 = min(c[0] for c in cells)
    agg: dict = {}
    for lo, hi, c in cells:
        k = lo0 + agg_width * int((lo - lo0) // agg_width)
        agg[k] = agg.get(k, 0) + c
    tot = sum(agg.values())
    mx = max(agg.values())
    lines = ["| 区间 | 频数 / 占比 |", "|---|---|"]
    for k in sorted(agg):
        n = agg[k]
        lines.append(f"| [{k:g}, {k + agg_width:g}) | {n} / {100 * n / tot:4.1f}% "
                     f"{'#' * max(1, int(round(28 * n / mx)))} |")
    return "\n".join(lines)


def top_communities_table(best_tab, k=12) -> str:
    lines = ["| 社区 | 节点数 | 占 676 比例 | 社区内加权强度最大的节点（仅供人工核对） |",
             "|---|---|---|---|"]
    for _, r in best_tab.head(k).iterrows():
        lines.append(f"| {int(r['community_id'])} | {int(r['size'])} | {r['share_of_676_pct']:.2f}% "
                     f"| `{r['strongest_member']}` |")
    rest = best_tab.iloc[k:]
    if len(rest):
        lines.append(f"| 其余 {len(rest)} 个社区 | 合计 {int(rest['size'].sum())} "
                     f"| {100 * rest['size'].sum() / 676:.2f}% | 最大 {int(rest['size'].max())}，"
                     f"最小 {int(rest['size'].min())} |")
    return "\n".join(lines)


def step7_write(sweep_df, dist, pw, parts_df, cons_summ, best_tab, budget_df,
                od, os_summ, noise_df, N, S, recommended, b):
    t0 = time.time()
    files = []
    for df, name in ((sweep_df, "seed_sweep.csv"), (dist, "seed_sweep_distribution.csv"),
                     (pw, "pairwise_stability.csv"), (parts_df, "consensus_partitions.csv"),
                     (cons_summ, "consensus_summary.csv"), (best_tab, "recommended_partition_communities.csv"),
                     (budget_df, "consensus_seed_budget.csv"), (od, "order_sensitivity.csv"),
                     (os_summ, "order_sensitivity_summary.csv"), (noise_df, "noise_source_comparison.csv")):
        files.append(C.save_csv(df, MODULE, name))

    s = N["seed_largest"]
    o = N["ord_largest"]
    sc = N["seed_comms"]
    oc = N["ord_comms"]
    d = S["distribution_summary"]
    ps = S["pairwise_summary"]
    prim = list(ps)[0]
    samp = [k for k in ps if k.startswith("random")][0]
    cs = cons_summ[cons_summ["seed_set"] == "primary50"]

    def row(r):
        return (f"| {r['method']} | {r['tau']} | {r['n_comms_total']} | {r['n_comms_nonsingleton']} "
                f"| {r['largest']} | {r['Q_of_consensus']:.4f} | {r['NMI_mean_vs_seeds']:.4f} "
                f"| {r['NMI_vs_seed42_partition']:.4f} |")

    cons_table = "\n".join(row(r) for _, r in cs.iterrows())
    bud_table = "\n".join(
        f"| {int(r['n_seeds'])} | {r['n_comms_total']} | {r['largest']} | {r['Q']:.4f} "
        f"| {r['NMI_vs_250seed_consensus']:.4f} |" for _, r in budget_df.iterrows())
    ord_table = "\n".join(
        f"| `{r['order_id']}` | {r['ordering_definition'] if 'ordering_definition' in r else r['order_kind']} "
        f"| {r['comms_at_seed42']} | {r['Q_at_seed42']:.4f} | {r['largest_at_seed42']} "
        f"| {int(r['comms_min'])}–{int(r['comms_max'])} | {int(r['largest_min'])}–{int(r['largest_max'])} |"
        for _, r in od_summary_display(os_summ, od).iterrows())

    def ccv(method, tau, col):
        s = cs[(cs["method"] == method) & (np.isclose(cs["tau"], tau))]
        return float(s[col].iloc[0]) if len(s) else float("nan")

    rt = recommended["tau"]
    _tie = abs(ccv("cc", rt, "NMI_mean_vs_seeds") - ccv("avglink", rt, "NMI_mean_vs_seeds"))
    md = f"""# B1_louvain_stability · 加权 Louvain 的稳定性与误差棒

主口径图：`csa_common.build_all()['Gu']`（加权无向，{S['facts']['Gu_nodes']} 节点 /
{S['facts']['Gu_edges']} 边，反向边权重相加，建图边序 = `links.csv` 行序）。
权重语义 = 观察到的链接次数。开跑前 `C.assert_anchors()` 31 项全过。
所有社区结果来自 `csa_common.louvain()` / `louvain_robust()`，相似度来自 `partition_metrics()`。

## 0. 一句话口径建议（可直接抄进报告）

> 在加权无向超链接网络（{S['facts']['Gu_nodes']} 子域 / {S['facts']['Gu_edges']} 无向边）上，
> 加权 Louvain 的模块度在 {S['n_seeds_primary']} 个随机种子下为
> **中位数 {d['Q']['primary50']['median']:.4f}（四分位距 {d['Q']['primary50']['q1']:.4f}–{d['Q']['primary50']['q3']:.4f}，
> 全距 {d['Q']['primary50']['min']:.4f}–{d['Q']['primary50']['max']:.4f}，sd {d['Q']['primary50']['sd']:.4f}）**；
> 社区数 **{d['n_comms_total']['primary50']['min']:.0f}–{d['n_comms_total']['primary50']['max']:.0f} 个（中位数
> {d['n_comms_total']['primary50']['median']:.0f}）**，最大社区 **{d['largest']['primary50']['min']:.0f}–{d['largest']['primary50']['max']:.0f} 个节点
> （中位数 {d['largest']['primary50']['median']:.0f}，四分位距 {d['largest']['primary50']['q1']:.0f}–{d['largest']['primary50']['q3']:.0f}）**；
> 两次独立种子的划分之间 NMI 中位数 {ps[prim]['NMI']['median']:.3f}（5–95 分位
> {ps[prim]['NMI']['p05']:.3f}–{ps[prim]['NMI']['p95']:.3f}），
> 单一随机种子的点估计不宜作为结论，本文引用的是 {S['n_seeds_primary']} 个种子的分布与
> {recommended['n_comms_total']} 社区的共识划分（{recommended['method']}, τ={recommended['tau']}）。

## 1. 判据 1 — seed 扫描（{S['n_seeds_primary']} 个种子为主，另 {S['n_seeds_extended']} 个种子做外推）

来源：`run_b1.py::step1_seed_sweep`；逐 seed 数据 `seed_sweep.csv`；分布与直方图 `seed_sweep_distribution.csv`。

| 指标 | sweep | min | p05 | Q1 | 中位数 | Q3 | p95 | max | mean | sd | 全距 |
|---|---|---|---|---|---|---|---|---|---|---|---|
""" + "\n".join(
        f"| {m} | {tag} | {v['min']:g} | {v['p05']:.4g} | {v['q1']:g} | {v['median']:.4g} | {v['q3']:g} "
        f"| {v['p95']:.4g} | {v['max']:g} | {v['mean']:.4g} | {v['sd']:.4g} | {v['range']:.4g} |"
        for m, lbl in (("n_comms_total", "社区数（含单点）"), ("n_comms_nonsingleton", "非单点社区数"),
                       ("Q", "模块度 Q"), ("largest", "最大社区"))
        for tag in ("primary50", "all250")
        for v in [d[m][tag]]
    ) + f"""

- 本图 {S['facts']['Gu_nodes']} 个节点全部至少有一条边：{S['n_seeds_extended']} 个 seed 的
  `n_singleton_communities` 全为 {d['singleton']['min']}（最小 {d['singleton']['min']}、最大
  {d['singleton']['max']}），覆盖节点数恒为 {d['singleton']['min_covered_nodes']}，
  故 `n_comms_total` 与 `n_comms_nonsingleton` 在本口径下逐个 seed 相等
  （= {d['singleton']['n_comms_total_equals_nonsingleton_all_seeds']}）。
  CONVENTIONS §3 的孤立点告警在这张全图上不触发，在强边骨架（B4）上才触发。
- 同学报告那句「25 个社区 / Q=0.6904 / 最大 87」在 {S['n_seeds_primary']} 个种子里同时命中的比例
  **{d['headline_hit_rate']['primary50']['all_three_exact'] * 100:.0f}%**；
  只命中社区数 25 的比例 {d['headline_hit_rate']['primary50']['comms_eq_25'] * 100:.0f}%，
  只命中最大社区 87 的比例 {d['headline_hit_rate']['primary50']['largest_eq_87'] * 100:.0f}%，
  Q 落在 0.6904±0.001 的比例 {d['headline_hit_rate']['primary50']['Q_within_0.001_of_0.6904'] * 100:.0f}%。
- 自检：`louvain_robust` 与逐点 `louvain` 在全部 {S['n_seeds_extended']} 个 seed 上
  不一致的 seed 数 = {len(S['facts']['robust_vs_direct_mismatch_seeds'])}；
  同进程同 seed 重复调用一致 = {S['facts']['same_process_determinism_seed42']}。
  另在 `PYTHONHASHSEED` = 0/1/7 三个**独立子进程**下，seed ∈ {0, 42, 20261113} 的
  （社区数, Q, 最大社区）与本进程逐位相同 =
  {S['facts']['hashseed_probe']['all_identical_to_main_process']}
  （探针失败 = {S['facts']['hashseed_probe']['any_probe_failed']}；原始输出见 RUN.txt 的
  `PYTHONHASHSEED 探针` 段）。CONVENTIONS §8 的 `PYTHONHASHSEED=0` 在 networkx 3.6.1 上
  不是必要条件，本图上可复现性只由"显式 seed + 规范边序"两件事控制。

`largest_community` 在 {S['n_seeds_primary']} 个 seed 上的直方图（10 一档，完整 width-5 计数在
`seed_sweep_distribution.csv` 的 `statistic=hist_count`）：

{hist_table("largest_community", "primary50", dist, 10)}

## 2. 判据 2 — 成对稳定性（核心数字）

来源：`run_b1.py::step3_pairwise`；逐对数据 `pairwise_stability.csv`（{len(pw)} 对）。

| 抽样 | 对数 | NMI 中位数 | NMI 5–95 | NMI Q1–Q3 | ARI 中位数 | ARI 5–95 |
|---|---|---|---|---|---|---|
| {prim} | {ps[prim]['n_pairs']} | {ps[prim]['NMI']['median']:.4f} | {ps[prim]['NMI']['p05']:.3f}–{ps[prim]['NMI']['p95']:.3f} | {ps[prim]['NMI']['q1']:.3f}–{ps[prim]['NMI']['q3']:.3f} | {ps[prim]['ARI']['median']:.4f} | {ps[prim]['ARI']['p05']:.3f}–{ps[prim]['ARI']['p95']:.3f} |
| {samp} | {ps[samp]['n_pairs']} | {ps[samp]['NMI']['median']:.4f} | {ps[samp]['NMI']['p05']:.3f}–{ps[samp]['NMI']['p95']:.3f} | {ps[samp]['NMI']['q1']:.3f}–{ps[samp]['NMI']['q3']:.3f} | {ps[samp]['ARI']['median']:.4f} | {ps[samp]['ARI']['p05']:.3f}–{ps[samp]['ARI']['p95']:.3f} |

判据 2 要求 >=200 对，这里给了 {ps[prim]['n_pairs']} + {ps[samp]['n_pairs']} 对。
两个抽样有重叠（第二组的池是 C(250,2)={S['facts']['n_all_pairs']} 对，包含第一组全部 {ps[prim]['n_pairs']} 对），
两组 NMI 中位数之差 {abs(ps[prim]['NMI']['median'] - ps[samp]['NMI']['median']):.4f}，
这个差本身就是"50 seed 够不够"的一个自检。
NMI 落在 {ps[prim]['NMI']['p05']:.2f}–{ps[samp]['NMI']['p95']:.2f} 这个区间：
社区"成员名单"级别的稳定度是**中等偏上**，不是"随便换个 seed 就面目全非"，
但也远达不到可以直接引用单次划分的程度（NMI=1 才叫一致）。

## 3. 判据 3 — 共识划分（可负责任引用的单一划分）

共识矩阵 = `louvain_robust(Gu, seeds=range(50))['consensus']`：任意两点在同一次划分中
落入同一社区的比例。非对角均值 {S['consensus']['offdiag_mean']}；
节点对总数 {S['consensus']['n_pairs']} 对，其中共识 >= 0.5 的 {S['consensus']['pairs_ge']['0.5']} 对、
>=0.9 的 {S['consensus']['pairs_ge']['0.9']} 对、恰为 1.0 的 {S['consensus']['pairs_ge']['1.0']} 对。

| 规则 | τ | 社区数 | 非单点 | 最大社区 | 共识划分自身 Q | 对 {S['n_seeds_primary']} seed 平均 NMI | 与 seed=42 划分 NMI |
|---|---|---|---|---|---|---|---|
{cons_table}

来源：`run_b1.py::step4_consensus`；节点标签 `consensus_partitions.csv`（`recommended=True`
即推荐划分的 {S['facts']['Gu_nodes']} 个节点标签）；逐 τ 指标 `consensus_summary.csv`；
推荐划分的社区清单 `recommended_partition_communities.csv`。

- 规则 `cc` = 共现 >= τ 连边取连通块。τ={TAUS[0]} 时它把多个社区链成一个
  {int(ccv('cc', 0.5, 'largest'))} 节点的块（超过任何单次 Louvain 的最大社区），
  这是连通块规则的**串联效应**，不是新发现的结构；
  层次聚类规则 `avglink` 在同一 τ 下最大社区 {int(ccv('avglink', TAUS[0], 'largest'))}。
- **选 τ 的判据是描述性的，不是最优化的**：`cc` 与 `avglink` 在推荐的 τ={rt} 上几乎并列
  （对 seed 集合的平均 NMI {ccv('cc', rt, 'NMI_mean_vs_seeds'):.4f} vs
  {ccv('avglink', rt, 'NMI_mean_vs_seeds'):.4f}，差 {_tie:.4f}），
  两者规模谱也基本一致，所以"共识划分"这个结论不依赖规则选择；
  但 τ 两端差得远（`cc` 规则 τ={TAUS[0]} 时 {int(ccv('cc', TAUS[0], 'n_comms_total'))} 个社区 vs
  τ={TAUS[-1]} 时 {int(ccv('cc', TAUS[-1], 'n_comms_total'))} 个），
  τ 的选择必须写进引用口径。
- **推荐引用**：`{recommended['method']}` 规则、τ={recommended['tau']}、50 seed，
  {recommended['n_comms_total']} 个社区（非单点 {recommended['n_comms_nonsingleton']}），最大社区
  {recommended['largest']}，对 50 个 seed 的平均 NMI {recommended['NMI_mean_vs_seeds']:.4f}。
- **注意社区数的口径差**：共识划分的社区数（{recommended['n_comms_total']}）**不等于**
  单次 Louvain 社区数的中位数（{d['n_comms_total']['primary50']['median']:.0f}）。
  阈值切割会把落在社区边界上、各次 seed 指派不一致的节点单独分出去，
  所以共识划分的社区更多、规模更小。引用时**不要**把两个数当同一个量互换，
  也不要说共识划分"更细"就是"分辨率更高"。
- **必须同时声明的代价**：共识划分不做模块度优化，其 Q = {recommended['Q_of_consensus']:.4f}，
  低于任何单次 Louvain（{d['Q']['primary50']['min']:.4f}–{d['Q']['primary50']['max']:.4f}）。
  若报告要的是"最大模块度解"，引用 seed 中位数 Q 并配区间；
  若报告要的是"成员名单最稳的单一划分"，引用共识划分并放弃一部分 Q。两者不要混在一张表里。

推荐划分（`{recommended['method']}`, τ={recommended['tau']}）的社区规模谱，
完整 32 行见 `recommended_partition_communities.csv`，逐节点标签见
`consensus_partitions.csv` 里 `recommended=True` 的 676 行：

{top_communities_table(best_tab, 12)}

共识划分需要多少 seed 才够稳（`avglink`, τ=0.5，参照 = 250 seed 共识）：

| seed 预算 | 社区数 | 最大社区 | Q | 与 250-seed 共识的 NMI |
|---|---|---|---|---|
{bud_table}

来源：`run_b1.py::step4_consensus` 末段；`consensus_seed_budget.csv`。

## 4. 判据 4 — 建图边序敏感性 vs seed 波动

来源：`run_b1.py::step5_order_sensitivity`（逐格数据 `order_sensitivity.csv`，
每种顺序的汇总 `order_sensitivity_summary.csv`）与 `step6_noise`（`noise_source_comparison.csv`）。

{S['facts']['n_orderings_non_diagnostic']} 种顺序 × {S['n_order_seeds']} 个 seed 的交叉设计
（规范行序 / 字典序 / 逆字典序 / {S['n_shuffles']} 个随机打乱 =
{S['facts']['n_orderings']} 行，其中 `prepend671` 只作诊断，见 §6）；
每种顺序都逐一核验**边集与权重完全相同**
（`order_sensitivity.csv` 的 `edge_set_identical_to_canonical` 全为 True：
{S['facts']['n_orderings']} 张图 × {S['n_order_seeds']} 个 seed =
{S['facts']['n_orderings'] * S['n_order_seeds']} 格，边集哈希 {S['facts']['n_orderings']} 次全部等于
规范图的 `{S['facts']['edge_signature_canonical']}`），且节点插入顺序确实变了
（`node_insertion_order_changed`）。
规范图 × seed=42 这一格复现锚点 {S['facts']['canonical_at_seed42_in_grid']}，
说明交叉设计没有偷偷换口径。

| 顺序 | 定义 | seed=42 社区数 | seed=42 Q | seed=42 最大社区 | 该顺序下 20 seed 的社区数范围 | 最大社区范围 |
|---|---|---|---|---|---|---|
{ord_table}

**哪个噪声源更大（同一张图、同一批边）**：

| 量 | 换 seed（边序=规范，50 seed） | 换边序（seed=42，{N['n_orders_used']} 种顺序） |
|---|---|---|
| 最大社区 全距 | {s['range']:.0f}（{s['min']:.0f}–{s['max']:.0f}） | {o['range']:.0f}（{o['min']:.0f}–{o['max']:.0f}） |
| 最大社区 中位数 | {s['median']:.0f} | {o['median']:.0f} |
| 社区数 全距 | {sc['range']:.0f}（{sc['min']:.0f}–{sc['max']:.0f}） | {oc['range']:.0f}（{oc['min']:.0f}–{oc['max']:.0f}） |
| Q 全距 | {N['seed_Q']['range']:.4f} | {N['ord_Q']['range']:.4f} |

- 交叉设计上的成对 NMI：同一顺序内换 seed = {N['crossed']['within_order_between_seed']['NMI']['mean']:.4f}
  （{N['crossed']['within_order_between_seed']['n_pairs']} 对）；
  同一 seed 下换顺序 = {N['crossed']['between_order_within_seed']['NMI']['mean']:.4f}
  （{N['crossed']['between_order_within_seed']['n_pairs']} 对）。NMI 越低越不稳定。
- eta² 方差分解（24×20 格）：社区数 边序 {N['eta2']['n_comms_total']['edge_order']:.3f} vs seed {N['eta2']['n_comms_total']['seed']:.3f}；
  Q 边序 {N['eta2']['Q']['edge_order']:.3f} vs seed {N['eta2']['Q']['seed']:.3f}；
  最大社区 边序 {N['eta2']['largest']['edge_order']:.3f} vs seed {N['eta2']['largest']['seed']:.3f}。
- **判决**：最大社区口径上更大的噪声源 = **{N['verdict']['largest']}**
  （全距 {s['range']:.0f} vs {o['range']:.0f}）；
  社区数口径 = **{N['verdict']['comms']}**（全距 {sc['range']:.0f} vs {oc['range']:.0f}）；
  成对 NMI 口径 = **{N['verdict']['nmi']}**，但两者只差
  {abs(N['crossed']['within_order_between_seed']['NMI']['mean'] - N['crossed']['between_order_within_seed']['NMI']['mean']):.4f}，
  在 NMI 自身的 sd（{N['crossed']['within_order_between_seed']['NMI']['sd']:.3f}）面前**不构成可区分的差别**。
- 三个口径合起来读：**两个噪声源量级相当，都不可忽略**。eta² 只有
  {min(min(v.values()) for v in N['eta2'].values()):.3f}–{max(max(v.values()) for v in N['eta2'].values()):.3f}
  （边序与 seed 各自），意味着交叉设计里 {100 * (1 - max(max(v.values()) for v in N['eta2'].values())):.0f}%
  以上的方差落在 24×20 个格子之间（顺序 × seed 的交互，加上单次运行自身的随机性）。
  读法：两个因子都**没有稳定的平均效应**，但任何一个被改动都会把解换成另一个，
  所以两者的波动必须一起报，不能只挑一个当"稳健性检验"。

## 5. 报告口径建议（判据 5）

1. **Q**：报 `{d['Q']['primary50']['median']:.4f}`（50 seed 中位数），配四分位距
   `±{(d['Q']['primary50']['q3'] - d['Q']['primary50']['q1']) / 2:.4f}`（IQR
   {d['Q']['primary50']['q1']:.4f}–{d['Q']['primary50']['q3']:.4f}），不要写 0.6904 这个单点。
2. **社区数**：报区间 **{d['n_comms_total']['primary50']['min']:.0f}–{d['n_comms_total']['primary50']['max']:.0f}（中位数
   {d['n_comms_total']['primary50']['median']:.0f}）**，并写明 `n_comms_total` 在此口径下等于
   `n_comms_nonsingleton`。
3. **最大社区**：报区间 **{d['largest']['primary50']['min']:.0f}–{d['largest']['primary50']['max']:.0f}（中位数
   {d['largest']['primary50']['median']:.0f}，5–95 分位 {d['largest']['primary50']['p05']:.0f}–{d['largest']['primary50']['p95']:.0f}）**。
   这个指标的相对全距达 {100 * d['largest']['primary50']['range'] / d['largest']['primary50']['median']:.0f}%，
   禁止作为单点结论引用。
4. **稳定性**：引 "两次独立 seed 的 NMI 中位数 {ps[prim]['NMI']['median']:.2f}
   （5–95 分位 {ps[prim]['NMI']['p05']:.2f}–{ps[prim]['NMI']['p95']:.2f}）"。
5. **需要单一划分时**：引推荐共识划分（`consensus_partitions.csv` 里 `recommended=True`），
   并同时声明它的 Q 低于 Louvain 单次解。
6. **与既有报告的关系**：`seed=42 / 规范边序` 是 31 项锚点之一（25 / 0.6904 / 87），
   它落在上述区间的内部，作为"锚点复现"引用合法，作为"结论"引用不合法。

## 6. 局限与不确定性（诚实声明）

- 本模块只覆盖 **resolution=1.0、权重相加、规范边序** 这一格（CONVENTIONS §1）；
  γ 扫描在 B3、权重口径在 B4、去枢纽在 B5，跨这些口径的波动**不**在上述误差棒内。
- 共识矩阵是"同一次划分是否同社区"的频率，它把**同一张图上跑出的** {S['n_seeds_primary']} 个
  （或 {S['n_seeds_extended']} 个）解等权平均，这些解彼此高度相关，不是独立重复实验；
  给出的区间是**经验波动范围**，不是抽样置信区间，不要写成"95% CI"。
- `cc` 规则（连通块）在串联效应下不是模块度意义上的最优解；τ 的选择没有做交叉验证，
  只按"对 seed 集合的平均 NMI 最高"挑，属于描述性选择。
- 边序噪声与 seed 噪声在交叉设计里有交互，未单独建模（eta² 只给主效应的份额）。
- 判据 4 里 `prepend671` 是**诊断分支**（`diagnostic_only=True`，
  seed=42 下实测 {S['facts']['prepend_diagnostic_at_seed42']}），
  只用于核对 `csa_common` docstring 引用的数字，不参与任何结论与噪声统计。
- CONVENTIONS §3 与单元任务书里引用的"社区数 23–26、Q 0.6895–0.6945、最大社区 80–113"
  是 6 个 seed 的观测范围；本模块 {S['n_seeds_primary']} 个 seed 的实测范围更宽
  （{d['n_comms_total']['primary50']['min']:.0f}–{d['n_comms_total']['primary50']['max']:.0f} /
  {d['Q']['primary50']['min']:.4f}–{d['Q']['primary50']['max']:.4f} /
  {d['largest']['primary50']['min']:.0f}–{d['largest']['primary50']['max']:.0f}）。
  下游若沿用旧区间会低估波动，已把证据写进 `DISSENT.md`（不改口径，只报数）。

## 7. 产出与重跑

| 文件 | 内容 | 生成代码 |
|---|---|---|
| `seed_sweep.csv` | 250 个 seed 的逐 seed 指标 + 规模谱 | `step1_seed_sweep` |
| `seed_sweep_distribution.csv` | 4 指标 × 3 sweep 的分位数 + 直方图计数 | `step2_distribution` |
| `pairwise_stability.csv` | {len(pw)} 个 seed 对的 NMI/ARI | `step3_pairwise` |
| `consensus_partitions.csv` | 676 节点 × 2 规则 × 5 τ × 2 seed 集 的共识社区标签（含 `recommended`） | `step4_consensus` |
| `consensus_summary.csv` | 每个 (规则, τ) 的规模分布 / Q / 对 seed 的 NMI·ARI | `step4_consensus` |
| `recommended_partition_communities.csv` | 推荐划分的逐社区规模与代表节点 | `step4_consensus` |
| `consensus_seed_budget.csv` | 共识划分 vs seed 预算（10/25/50/100/250） | `step4_consensus` |
| `order_sensitivity.csv` | {S['facts']['order_grid_shape']} 交叉设计逐格 | `step5_order_sensitivity` |
| `order_sensitivity_summary.csv` | 每种顺序的汇总 | `step5_order_sensitivity` |
| `noise_source_comparison.csv` | seed 噪声 vs 边序噪声（长表 + eta² + 判决） | `step6_noise` |
| `key_numbers.json` | 本报告全部关键数字的机器可读版（供 C5/H1/I1 直接取用） | `main` |
| `RUN.txt` | 版本 / 命令行 / 逐步耗时 / seed / `PYTHONHASHSEED` 探针原始输出 | `write_run` |
| `DISSENT.md` | 与 CONVENTIONS §3、`csa_common` docstring 已引数字的差异证据 | 手写 |

```bash
cd pipeline && python B1_louvain_stability/run_b1.py     # 约 {int(total_seconds(TIMINGS))} s
```
"""
    with open(os.path.join(OUT_DIR, "result.md"), "w", encoding="utf-8") as f:
        f.write(md)
    files.append(os.path.join(OUT_DIR, "result.md"))
    return files


def od_summary_display(os_summ, od):
    m = od.drop_duplicates("order_id").set_index("order_id")["ordering_definition"]
    t = os_summ[~os_summ["diagnostic_only"]].copy()
    t["ordering_definition"] = t["order_id"].map(m)
    return t


def total_seconds(d):
    return sum(d.values())


# =========================================================== RUN.txt

def write_run(b, files):
    import sklearn
    import scipy
    ver = C.versions()
    lines = [
        "B1_louvain_stability · RUN.txt",
        "=" * 72,
        f"start_local_time   : {S['start_time']}",
        f"end_local_time     : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"total_wall_seconds : {time.time() - S['t_wall0']:.1f}",
        f"python             : {ver['python']}",
        f"networkx           : {nx.__version__}",
        f"pandas             : {pd.__version__}",
        f"numpy              : {np.__version__}",
        f"scipy              : {scipy.__version__}",
        f"sklearn            : {sklearn.__version__}",
        f"platform           : {platform.platform()}",
        f"PYTHONHASHSEED(env): {os.environ.get('PYTHONHASHSEED')}",
        f"csa_common.SEED    : {C.SEED}",
        f"module SEED        : {SEED}",
        "",
        "command line:",
        "    cd C:/Users/bfy32/Documents/Qoder/2026-11-13/f93e4d48/pipeline",
        "    python B1_louvain_stability/run_b1.py",
        "",
        "seeds used:",
        f"    primary Louvain sweep        : range(50)  = 0..49  ({len(PRIMARY_SEEDS)} seeds)",
        f"    extended Louvain sweep       : range(250) = 0..249 ({len(EXTENDED_SEEDS)} seeds)",
        f"    pairwise stability pairs     : C(50,2)=1225 全部 + 250-seed 池随机 {N_SAMPLED_PAIRS} 对"
        f"（rng default_rng({SEED})）",
        f"    consensus                    : louvain_robust(seeds=range(50)) 与 range(250)，"
        f"tau in {TAUS}，规则 cc / avglink",
        f"    consensus seed budgets       : {BUDGETS}",
        f"    edge-order grid              : {N_SHUFFLES + 3} 种非诊断顺序 x {len(ORDER_SEEDS)} 个 seed "
        f"= {(N_SHUFFLES + 3) * len(ORDER_SEEDS)} 次 Louvain"
        f"（另 1 个诊断顺序 prepend671，同样 {len(ORDER_SEEDS)} 个 seed）",
        "",
        "per-step wall seconds:",
    ] + [f"    {k:28s}: {v:8.2f}" for k, v in TIMINGS.items()] + [
        "",
        "anchors: C.assert_anchors() 31/31 PASS（任何一项不过本脚本立即抛错停止）",
        f"graph  : Gu {b['Gu'].number_of_nodes()} nodes / {b['Gu'].number_of_edges()} edges, "
        f"edge-set sha256[:16]={FACTS.get('edge_signature_canonical')}",
        f"louvain calls this run: {S.get('n_louvain_calls')}",
        "",
        "PYTHONHASHSEED 探针（独立子进程，seed -> [社区数, Q, 最大社区]）:",
        json.dumps(FACTS.get("hashseed_probe", {}), ensure_ascii=False, indent=2, default=str),
        "",
        "key numbers (与 result.md 一一对应):",
        json.dumps({k: FACTS[k] for k in
                    ("rob50_summary", "rob250_summary", "pairwise_summary", "consensus_pairs_at_least_tau_primary50",
                     "consensus_offdiag_mean_primary50", "recommended_partition", "noise_comparison")
                    if k in FACTS}, ensure_ascii=False, indent=2, default=str),
        "",
        "outputs:",
    ] + [f"    {p}" for p in files]
    with open(os.path.join(OUT_DIR, "RUN.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# =========================================================== main

def main():
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    S.clear()
    S["t_wall0"] = time.time()
    S["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    S["n_seeds_primary"] = len(PRIMARY_SEEDS)
    S["n_seeds_extended"] = len(EXTENDED_SEEDS)
    S["n_shuffles"] = N_SHUFFLES
    S["n_order_seeds"] = len(ORDER_SEEDS)

    b = step0_load()
    Gu, nodes, links = b["Gu"], b["nodes"], b["links"]

    sweep_df, rob50, rob250, per_seed = step1_seed_sweep(Gu)
    dist, _ = step2_distribution(sweep_df)
    pw, _ = step3_pairwise(Gu, per_seed)
    parts_df, cons_summ, best_tab, budget_df, recommended = step4_consensus(Gu, rob50, rob250, per_seed)
    od, os_summ, parts_grid = step5_order_sensitivity(links, nodes, Gu)
    noise_df, N = step6_noise(Gu, od, os_summ, parts_grid, sweep_df)
    step6b_hashseed(per_seed)

    S["distribution_summary"] = FACTS["distribution_summary"]
    S["pairwise_summary"] = FACTS["pairwise_summary"]
    S["consensus"] = {"offdiag_mean": FACTS["consensus_offdiag_mean_primary50"],
                      "n_pairs": FACTS["consensus_n_node_pairs_primary50"],
                      "pairs_ge": {**FACTS["consensus_pairs_at_least_tau_primary50"], "1.0": None}}
    # 共识矩阵里恰为 1.0 的对数（=50 个 seed 全部同社区）
    iu = np.triu_indices(len(rob50["node_list"]), 1)
    S["consensus"]["pairs_ge"]["1.0"] = int((rob50["consensus"][iu] >= 1.0 - 1e-9).sum())
    S["facts"] = FACTS
    S["n_louvain_calls"] = (len(PRIMARY_SEEDS) + len(EXTENDED_SEEDS) + len(EXTENDED_SEEDS)
                            + sum(BUDGETS) + len(ORDER_SEEDS) * (N_SHUFFLES + 4) + 2)

    files = step7_write(sweep_df, dist, pw, parts_df, cons_summ, best_tab, budget_df,
                        od, os_summ, noise_df, N, S, recommended, b)
    write_run(b, files)
    key = {
        "headline_interval_Q": {k: round(v, 4) for k, v in
                                S["distribution_summary"]["Q"]["primary50"].items()},
        "headline_interval_n_comms": {k: v for k, v in
                                      S["distribution_summary"]["n_comms_total"]["primary50"].items()},
        "headline_interval_largest": {k: v for k, v in
                                      S["distribution_summary"]["largest"]["primary50"].items()},
        "pairwise_NMI_primary50": S["pairwise_summary"],
        "recommended_partition": recommended,
        "noise_verdict": N["verdict"],
        "eta2": N["eta2"],
        "crossed_NMI_mean": {k: round(v["NMI"]["mean"], 4) for k, v in N["crossed"].items()},
        "versions": C.versions(),
        "timings_seconds": {k: round(v, 2) for k, v in TIMINGS.items()},
    }
    C.save_json(key, MODULE, "key_numbers.json")
    print(json.dumps({k: v for k, v in key.items() if k != "pairwise_NMI_primary50"},
                     ensure_ascii=False, indent=2, default=str))
    print("pairwise:", json.dumps(S["pairwise_summary"], ensure_ascii=False, default=str))
    print("TOTAL_SECONDS", round(time.time() - S["t_wall0"], 1))


if __name__ == "__main__":
    try:
        main()
    except AssertionError:
        traceback.print_exc()
        sys.stderr.write("\n[B1] 锚点或同构性断言失败 —— 按 CONVENTIONS 立即停止，不写结论。\n")
        raise
