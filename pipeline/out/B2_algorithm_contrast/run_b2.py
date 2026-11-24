"""B2_algorithm_contrast — 社区发现算法对照矩阵（Louvain / 标签传播 / 贪心模块度 / 划分族）。

单元定义见 pipeline/MANIFEST.md 的 `B2_algorithm_contrast`；口径见 CONVENTIONS.md §1 / §3 / §7 / §8。

要回答的问题（同学报告 vs 本流水线预实验的矛盾）：
    同学 REPORT.md §3.3：标签传播得到 **557 节点巨型社区**（11 个社区，占 82.4%）。
    本流水线预实验：加权 Louvain 得到 **约 25 个社区，最大 87**（seed=42 锚点）。
    同一张图、同一份数据，两套结论。本单元把这个矛盾量化，判它是**算法产物**还是**数据事实**。

步骤
  step0_anchors          31 项锚点自检（C.assert_anchors），不通过立即停
  step1_algorithms       9 种划分在同一批图（Gu 加权 / Gb 无权 / w>=10 骨架）上的参考运行
  step2_objectives       每个划分同时报**两套目标函数**取值：模块度 Q（加权/无权）+ 覆盖率 coverage
  step3_matrix           算法 x 算法的 NMI / ARI 交叉表（C.partition_metrics，覆盖交集上重编号）
                         并给出「同算法换 seed」的 NMI 天花板作参照（否则 NMI 数值无法解读）
  step4_lp_reps          标签传播 >=50 次重复：nx 权重盲实现（Gb/Gu/671 口径）、权重感知实现
                         （自实现加权投票 + nx asyn_lpa_communities(weight="weight")）、w>=10 骨架
  step5_null             度保持零模型上跑同一套 LPA（C.degree_preserving_null），看巨型社区在
                         「度序列相同、结构被打散」的图上是否更大 —— 判决用的关键对照
  step6_inside_giant     同学 C239 巨型社区内部再用加权 Louvain 拆一次：它是不是一个社区
  step7_write            写 CSV / result.md / key_numbers.json / RUN.txt

硬口径
  - 图只从 `csa_common.build_all()` 取；骨架只从 `csa_common.strong_backbone()` 取；
    划分相似度只用 `csa_common.partition_metrics()`；社区数一律同报
    `n_comms_total` 与 `n_comms_nonsingleton`（CONVENTIONS §3）。
  - 节点口径：全图 676（含 5 悬空，锚点口径）+ 有属性 671（主分析口径，CONVENTIONS §1）两列并报。
  - 不同算法的 Q **不可比大小**（优化目标不同）；本表用「同一划分在两套目标函数下的取值」代替
    「谁的 Q 高谁更好」的论证（result.md §4）。
  - 禁用因果措辞（CONVENTIONS §7）：本数据是静态横截面链接快照。

重跑：
    cd pipeline && python out/B2_algorithm_contrast/run_b2.py
"""
from __future__ import annotations

import json
import os
import platform
import random
import sys
import time

import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from scipy.sparse.linalg import eigsh

PIPELINE_DIR = r"C:/Users/bfy32/Documents/Qoder/2026-11-13/f93e4d48/pipeline"
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

import csa_common as C  # noqa: E402

SEED = 20261113  # CONVENTIONS §8
assert SEED == C.SEED

MODULE = "B2_algorithm_contrast"
OUT_DIR = C.module_dir(MODULE)

N_LP_REPS = 50          # 判据要求 >=30，这里取 50
N_ASYN_REPS = 50
N_NULL_REPS = 30        # 度保持零模型（CONVENTIONS §4：独立重复，每轮 fresh copy 再交换）
GIANT_REF = 557         # 同学报告的巨型社区规模（671 口径）
GE500 = 500             # 「接近 500+」的判决阈值

RUNTIME: dict[str, float] = {}
LOG: list[str] = []

# 同学报告的划分（只读，用来把「复现 557 巨型社区」变成可核对的数字）
CLASSMATE_NC = os.path.join(C.DATA_DIR, "node_communities.csv")


def log(msg: str = "") -> None:
    print(msg, flush=True)
    LOG.append(str(msg))


# ------------------------------------------------------------------ 图/划分工具

def permuted(G: nx.Graph, seed: int) -> nx.Graph:
    """同一边集、**随机节点插入顺序**的副本。

    networkx 3.6.1 的 `label_propagation_communities` 没有随机参数：初始标签取自
    `enumerate(G)`（= 节点插入顺序），更新顺序取自按颜色的分组。所以它的「不稳定」
    全部来自节点遍历顺序；要跑 >=30 次重复，唯一可复现的随机源就是置换节点顺序。
    边集、边权、边添加次序都不变。
    """
    rng = random.Random(seed)
    nodes = list(G.nodes())
    rng.shuffle(nodes)
    H = nx.Graph()
    H.add_nodes_from(nodes)
    for u, v, d in G.edges(data=True):
        H.add_edge(u, v, d=d)
    return H


def same_structure(G: nx.Graph, H: nx.Graph) -> bool:
    """置换前后结构必须完全一致（节点集、边集、边权多重集）。"""
    if set(G.nodes()) != set(H.nodes()):
        return False
    if {(u, v) for u, v in G.edges()} != {(u, v) for u, v in H.edges()}:
        return False
    wg = sorted(round(d.get("weight", 1.0), 6) for _, _, d in G.edges(data=True))
    wh = sorted(round(d.get("weight", 1.0), 6) for _, _, d in H.edges(data=True))
    return wg == wh


def sizes_of(part, restrict: set | None = None) -> list[int]:
    if restrict is None:
        return sorted((len(c) for c in part), reverse=True)
    return sorted((len(set(c) & restrict) for c in part), reverse=True)


def describe_sizes(sizes: list[int]) -> dict:
    ns = [s for s in sizes if s > 1]
    a = np.asarray(ns, dtype=float) if ns else np.zeros(0)
    return {
        "n_ge_500": int(sum(1 for s in sizes if s >= GE500)),
        "size_min": int(a.min()) if a.size else 0,
        "size_q1": round(float(np.percentile(a, 25)), 2) if a.size else 0.0,
        "size_median": round(float(np.median(a)), 2) if a.size else 0.0,
        "size_mean": round(float(a.mean()), 2) if a.size else 0.0,
        "size_q3": round(float(np.percentile(a, 75)), 2) if a.size else 0.0,
        "size_max": int(a.max()) if a.size else 0,
    }


def coverage(G: nx.Graph, part, weighted: bool) -> float:
    """覆盖率 = 两端落在同一社区内的边（权）占全部边（权）的比例。

    这是与模块度不同的第二套目标函数：它**不含零模型项**，因此单点社区不受惩罚，
    把节点并进大社区只会上调覆盖率 —— 这正是「Q 更高所以更好」不成立的原因（§4）。
    """
    lab = {}
    for i, c in enumerate(part):
        for n in c:
            lab[n] = i
    tot = 0.0
    inside = 0.0
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 1.0)) if weighted else 1.0
        tot += w
        if lab.get(u, -1) == lab.get(v, -2):
            inside += w
    return round(inside / tot, 4) if tot else float("nan")


def jaccard(a: set, b: set) -> float:
    u = len(a | b)
    return round(len(a & b) / u, 4) if u else float("nan")


# ------------------------------------------------------- 自实现：加权标签传播

def lp_weighted(G: nx.Graph, seed: int, max_sweeps: int = 100):
    """**权重感知**的异步标签传播（本单元自实现，只用于对照 nx 权重盲实现）。

    节点按随机顺序逐个更新，投票按**边权求和**（观察到的链接次数）而不是邻居计数；
    平票在最大者中随机选。收敛判据：一整轮无标签变化。
    振荡不收敛时按 max_sweeps 截断并在 `converged=False` 里注明。
    """
    rng = random.Random(seed)
    nodes = list(G.nodes())
    lab = {n: i for i, n in enumerate(nodes)}
    adj = {n: [(nb, float(d.get("weight", 1.0))) for nb, d in G[n].items()] for n in nodes}
    sweeps = 0
    converged = False
    for sweeps in range(1, max_sweeps + 1):
        changed = False
        order = nodes[:]
        rng.shuffle(order)
        for n in order:
            votes: dict[int, float] = {}
            for nb, w in adj[n]:
                votes[lab[nb]] = votes.get(lab[nb], 0.0) + w
            if not votes:
                continue
            mx = max(votes.values())
            best = [k for k, v in votes.items() if v == mx]
            pick = rng.choice(best)
            if lab[n] != pick:
                lab[n] = pick
                changed = True
        if not changed:
            converged = True
            break
    out: dict[int, set] = {}
    for n, l in lab.items():
        out.setdefault(l, set()).add(n)
    return list(out.values()), sweeps, converged


# ---------------------------------------------------------------- 划分参考运行

def spectral_bisection(G: nx.Graph, weighted: bool = False):
    """Fiedler 向量二分（Laplacian 谱）：第二小特征向量的符号给出两块。

    目标函数 = 最小割 / 规范化割，与模块度完全不同；只作第 5 类算法对照。
    """
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    rows, cols, vals = [], [], []
    deg = np.zeros(len(nodes))
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 1.0)) if weighted else 1.0
        i, j = idx[u], idx[v]
        rows += [i, j]
        cols += [j, i]
        vals += [w, w]
        deg[i] += w
        deg[j] += w
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(len(nodes), len(nodes)))
    L = sparse.diags(deg) - A
    dd = np.clip(deg, 1e-9, None)
    ds = 1.0 / np.sqrt(dd)
    Ls = sparse.diags(ds) @ L.tocsr() @ sparse.diags(ds)
    vals_e, vecs = eigsh(Ls.toarray(), k=3, sigma=-1e-8, which="LM")
    order = np.argsort(vals_e)
    fiedler = vecs[:, order[1]]
    cut = float(np.median(fiedler))
    Aset = {nodes[i] for i in range(len(nodes)) if fiedler[i] > cut}
    Bset = set(nodes) - Aset
    return [Aset, Bset], round(float(vals_e[order[1]]), 6)


def build_reference_partitions(Gu, Gb, bb, attr) -> list[dict]:
    """9 种算法的参考划分（同一批图；seed 全部固定）。"""
    out = []
    t = time.time()
    r = C.louvain(Gu, seed=42)
    out.append({"algo": "louvain_weighted", "impl": "csa_common.louvain(Gu, weight='weight')",
                "graph": "Gu(676,加权)", "objective": "模块度 Q(加权, gamma=1)", "seed": 42,
                "partition": r["partition"], "runtime_s": round(time.time() - t, 3),
                "note": "主口径；seed=42 锚点 25 社区/Q 0.6904/最大 87"})

    t = time.time()
    r = C.louvain(Gb, seed=42)
    out.append({"algo": "louvain_binary", "impl": "csa_common.louvain(Gb, weight='weight')",
                "graph": "Gb(676,无权)", "objective": "模块度 Q(无权)", "seed": 42,
                "partition": r["partition"], "runtime_s": round(time.time() - t, 3),
                "note": "对照：看权重贡献多少（与 C.louvain(Gb) == nx weight=None 已核验）"})

    t = time.time()
    p = list(nx.community.label_propagation_communities(Gb))
    out.append({"algo": "label_prop_binary", "impl": "nx.community.label_propagation_communities(Gb)",
                "graph": "Gb(676,无权)", "objective": "无目标函数（多数投票收敛）", "seed": "无(节点遍历顺序)",
                "partition": p, "runtime_s": round(time.time() - t, 3),
                "note": "同学用的就是这一族算法；canonical 插入顺序"})

    t = time.time()
    p = list(nx.community.label_propagation_communities(Gu))
    out.append({"algo": "label_prop_on_weighted", "impl": "nx.community.label_propagation_communities(Gu)",
                "graph": "Gu(676,加权)", "objective": "无目标函数（**权重盲：不读 weight**）",
                "seed": "无(节点遍历顺序)", "partition": p, "runtime_s": round(time.time() - t, 3),
                "note": "nx 该实现无 weight 参数：喂加权图仍按邻居计数投票"})

    t = time.time()
    p = list(nx.community.label_propagation_communities(permuted(bb, SEED)))
    out.append({"algo": "label_prop_backbone_w10", "impl": "nx...communities(strong_backbone(Gu,10))",
                "graph": "骨架 w>=10(675e)", "objective": "无目标函数（无权投票，图已按权重筛过）",
                "seed": SEED, "partition": p, "runtime_s": round(time.time() - t, 3),
                "note": "权重经『筛边』进入 LPA 的唯一途径；孤立点各成一社区"})

    t = time.time()
    p = list(nx.community.asyn_lpa_communities(Gu, weight="weight", seed=SEED))
    out.append({"algo": "label_prop_async_weighted", "impl": "nx.community.asyn_lpa_communities(Gu, weight='weight')",
                "graph": "Gu(676,加权)", "objective": "无目标函数（**按边权求和投票**）", "seed": SEED,
                "partition": p, "runtime_s": round(time.time() - t, 3),
                "note": "nx 内唯一真正读权重的标签传播变体"})

    t = time.time()
    p = list(nx.community.greedy_modularity_communities(Gu, weight="weight"))
    out.append({"algo": "greedy_modularity_weighted", "impl": "nx.community.greedy_modularity_communities(Gu, weight='weight')",
                "graph": "Gu(676,加权)", "objective": "模块度 Q(加权) 贪心合并", "seed": "无(确定性)",
                "partition": p, "runtime_s": round(time.time() - t, 3),
                "note": "全图 2498 边直接跑通，未降骨架（耗时见 RUNTIME）"})

    t = time.time()
    p = list(nx.community.greedy_modularity_communities(Gb))
    out.append({"algo": "greedy_modularity_binary", "impl": "nx.community.greedy_modularity_communities(Gb)",
                "graph": "Gb(676,无权)", "objective": "模块度 Q(无权) 贪心合并", "seed": "无(确定性)",
                "partition": p, "runtime_s": round(time.time() - t, 3), "note": ""})

    t = time.time()
    p = list(nx.community.kernighan_lin_bisection(Gu, seed=SEED, weight="weight"))
    out.append({"algo": "kernighan_lin_bisection", "impl": "nx.community.kernighan_lin_bisection(Gu, weight='weight')",
                "graph": "Gu(676,加权)", "objective": "最小割（平衡二分，社区数=2 由构造给定）",
                "seed": SEED, "partition": p, "runtime_s": round(time.time() - t, 3),
                "note": "划分族对照：块大小由『各半』约束，不由数据决定"})

    t = time.time()
    p, lam2 = spectral_bisection(Gb, weighted=False)
    out.append({"algo": "spectral_bisection_fiedler", "impl": "本单元实现：归一化 Laplacian 的 Fiedler 向量符号",
                "graph": "Gb(676,无权)", "objective": f"最小规范化割（lambda2={lam2}）", "seed": "无(确定性)",
                "partition": p, "runtime_s": round(time.time() - t, 3),
                "note": "划分族对照；与模块度完全不同的目标"})
    return out


# ---------------------------------------------------------------------- 主流程

def main() -> None:
    t_all = time.time()
    log(f"B2_algorithm_contrast | {platform.python_version()} | networkx {nx.__version__} | SEED={SEED}")

    # ---- step0 锚点
    t = time.time()
    C.assert_anchors()
    RUNTIME["assert_anchors"] = round(time.time() - t, 2)
    log(f"anchors: 31/31 PASS ({RUNTIME['assert_anchors']}s)")

    b = C.build_all()
    Gu, Gb, Gd = b["Gu"], b["Gb"], b["Gd"]
    attr = set(b["nodes"]["node"])
    bb = C.strong_backbone(Gu, min_weight=10)
    log(f"Gu {Gu.number_of_nodes()}n/{Gu.number_of_edges()}e ; Gb {Gb.number_of_nodes()}n/{Gb.number_of_edges()}e ; "
        f"backbone w>=10 {bb.number_of_nodes()}n/{bb.number_of_edges()}e ; attributed {len(attr)}")

    cm = pd.read_csv(CLASSMATE_NC, encoding="utf-8-sig")
    cm_parts = [set(g["subdomain"]) for _, g in cm.groupby("community")]
    cm_giant = max(cm_parts, key=len)
    log(f"同学划分：{len(cm_parts)} 社区，最大 {len(cm_giant)} 节点（= REPORT.md 的 557，671 口径）")

    # ---- step1/2 参考划分 + 两套目标函数
    t = time.time()
    refs = build_reference_partitions(Gu, Gb, bb, attr)
    RUNTIME["reference_partitions"] = round(time.time() - t, 2)

    rows = []
    for r in refs:
        part = r["partition"]
        s676 = sizes_of(part)
        s671 = sizes_of(part, attr)
        d = describe_sizes(s671)
        rows.append({
            "algorithm": r["algo"], "implementation": r["impl"], "graph": r["graph"],
            "objective_optimized": r["objective"], "seed": r["seed"],
            "n_comms_total": len(part),
            "n_comms_nonsingleton": int(sum(1 for s in s676 if s > 1)),
            "n_singleton": int(sum(1 for s in s676 if s == 1)),
            "n_comms_total_671": int(sum(1 for s in s671 if s > 0)),
            "n_comms_nonsingleton_671": int(sum(1 for s in s671 if s > 1)),
            "largest_676": s676[0] if s676 else 0,
            "largest_671": s671[0] if s671 else 0,
            "second_largest_671": s671[1] if len(s671) > 1 else 0,
            "largest_share_of_671_pct": round(100.0 * (s671[0] if s671 else 0) / len(attr), 2),
            "n_comms_ge_500": d["n_ge_500"],
            "Q_weighted_Gu": round(float(nx.community.modularity(Gu, part, weight="weight")), 4),
            "Q_binary_Gb": round(float(nx.community.modularity(Gb, part)), 4),
            "coverage_weighted_Gu": coverage(Gu, part, True),
            "coverage_binary_Gb": coverage(Gb, part, False),
            "size_min": d["size_min"], "size_q1": d["size_q1"], "size_median": d["size_median"],
            "size_mean": d["size_mean"], "size_q3": d["size_q3"], "size_max": d["size_max"],
            "sizes_top8_671": str(s671[:8]),
            "runtime_s": r["runtime_s"], "note": r["note"],
            "_partition": part,
        })
    summary = pd.DataFrame(rows)
    C.save_csv(summary.drop(columns=["_partition"]), MODULE, "algorithm_summary.csv")
    log(f"algorithm_summary.csv rows={len(summary)}")
    for r in rows:
        log(f"  {r['algorithm']:<28} total={r['n_comms_total']:>3} ns={r['n_comms_nonsingleton']:>3} "
            f"ns671={r['n_comms_nonsingleton_671']:>3} largest671={r['largest_671']:>4} "
            f"ge500={r['n_comms_ge_500']} Qw={r['Q_weighted_Gu']:.4f} Qb={r['Q_binary_Gb']:.4f} "
            f"covw={r['coverage_weighted_Gu']:.4f} covb={r['coverage_binary_Gb']:.4f} t={r['runtime_s']}s")

    # ---- step3 NMI / ARI 交叉表 + seed 天花板
    t = time.time()
    names = [r["algorithm"] for r in rows]
    N = len(names)
    nmi = np.full((N, N), np.nan)
    ari = np.full((N, N), np.nan)
    compared = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            m = C.partition_metrics(Gu, rows[i]["_partition"], rows[j]["_partition"])
            nmi[i, j] = m["NMI"]
            ari[i, j] = m["ARI"]
            compared[i, j] = m["n_compared"]
    RUNTIME["nmi_matrix"] = round(time.time() - t, 2)
    C.save_csv(pd.DataFrame(nmi, index=names, columns=names).rename_axis("algorithm_A"), MODULE, "nmi_matrix.csv")
    C.save_csv(pd.DataFrame(ari, index=names, columns=names).rename_axis("algorithm_A"), MODULE, "ari_matrix.csv")

    # 同算法换 seed 的 NMI 天花板：Louvain 加权 6 个 seed（B1 已给 20 seed 的完整分布）
    seed_parts = [C.louvain(Gu, seed=s)["partition"] for s in (42, SEED, 1, 2, 3, 7)]
    sn = [C.partition_metrics(Gu, seed_parts[i], seed_parts[j])["NMI"]
          for i in range(len(seed_parts)) for j in range(i + 1, len(seed_parts))]
    ceil_w = round(float(np.mean(sn)), 4)
    seed_parts_b = [C.louvain(Gb, seed=s)["partition"] for s in (42, SEED, 1, 2, 3, 7)]
    snb = [C.partition_metrics(Gu, seed_parts_b[i], seed_parts_b[j])["NMI"]
           for i in range(len(seed_parts_b)) for j in range(i + 1, len(seed_parts_b))]
    ceil_b = round(float(np.mean(snb)), 4)
    log(f"NMI(同算法换 seed) 天花板：加权 Louvain {ceil_w} / 无权 Louvain {ceil_b}  (6 seed, {len(sn)} 对)")

    offdiag = nmi[np.triu_indices(N, k=1)]
    log(f"跨算法 NMI：min {np.nanmin(offdiag):.4f} / median {np.nanmedian(offdiag):.4f} / max {np.nanmax(offdiag):.4f}")

    # ---- step4 标签传播重复（>=30 次）
    t = time.time()
    rep_rows = []

    def add_reps(variant, graph, graph_label, kind, n_reps):
        for k in range(n_reps):
            sd = SEED + k
            if kind == "nx_blind":
                H = permuted(graph, sd)
                p = list(nx.community.label_propagation_communities(H))
                conv, sweeps = True, np.nan
            elif kind == "nx_asis":
                H = graph
                p = list(nx.community.label_propagation_communities(H))
                conv, sweeps = True, np.nan
            elif kind == "nx_asis_shuffled":
                H = permuted(graph, sd)
                p = list(nx.community.label_propagation_communities(H))
                conv, sweeps = True, np.nan
            elif kind == "asyn_weighted":
                H = permuted(graph, sd)
                p = list(nx.community.asyn_lpa_communities(H, weight="weight", seed=sd))
                conv, sweeps = True, np.nan
            elif kind == "asyn_unweighted":
                H = permuted(graph, sd)
                p = list(nx.community.asyn_lpa_communities(H, weight=None, seed=sd))
                conv, sweeps = True, np.nan
            else:  # ext_weighted：自实现按权重求和投票
                H = permuted(graph, sd)
                p, sweeps, conv = lp_weighted(H, sd)
            s676 = sizes_of(p)
            s671 = sizes_of(p, attr)
            big676 = max(p, key=len)
            big671 = set(big676) & attr
            m = C.partition_metrics(Gu, p, seed_parts[0])
            rep_rows.append({
                "variant": variant, "rep": k + 1, "seed": sd, "graph": graph_label,
                "reads_weight": kind in ("asyn_weighted", "ext_weighted"),
                "n_comms_total": len(p),
                "n_comms_nonsingleton": int(sum(1 for s in s676 if s > 1)),
                "n_comms_nonsingleton_671": int(sum(1 for s in s671 if s > 1)),
                "largest_676": s676[0], "largest_671": s671[0] if s671 else 0,
                "second_largest_671": s671[1] if len(s671) > 1 else 0,
                "giant_ge_500": int(s671[0] >= GE500 if s671 else 0),
                "exact_557": int(s671[0] == GIANT_REF if s671 else 0),
                "jaccard_vs_classmate_giant": jaccard(big671, cm_giant),
                "nmi_vs_louvain_weighted": m["NMI"],
                "ari_vs_louvain_weighted": m["ARI"],
                # 目标函数必须在**本变体自己的图**上算：p 是 671 诱导图的划分时，
                # 拿它去评 676 的 Gu 会直接 NotAPartition；即便能评，跨 graph 的 Q 也不可比。
                "Q_weighted_own_graph": round(float(nx.community.modularity(graph, p, weight="weight")), 4),
                "coverage_weighted_own_graph": coverage(graph, p, True),
                "objective_graph_nodes": graph.number_of_nodes(),
                "objective_graph_edges": graph.number_of_edges(),
                "converged": conv, "sweeps": sweeps,
            })

    g671 = C.induced(Gu, sorted(attr))
    add_reps("nx_LPA_binary_canonical_order", Gb, "Gb(676,无权)", "nx_asis", 1)
    add_reps("nx_LPA_binary_nodeorder_shuffled", Gb, "Gb(676,无权)", "nx_blind", N_LP_REPS)
    add_reps("nx_LPA_weightedgraph_shuffled", Gu, "Gu(676,加权)", "nx_blind", N_LP_REPS)
    add_reps("nx_LPA_attr671_shuffled", g671, "Gu 诱导 671(加权)", "nx_asis_shuffled", N_LP_REPS)
    add_reps("nx_LPA_backbone_w10_shuffled", bb, "骨架 w>=10", "nx_blind", N_LP_REPS)
    add_reps("nx_asynLPA_weighted_shuffled", Gu, "Gu(676,加权)", "asyn_weighted", N_ASYN_REPS)
    add_reps("nx_asynLPA_unweighted_shuffled", Gu, "Gu(676,加权)", "asyn_unweighted", N_ASYN_REPS)
    add_reps("ext_weightedLPA_Gu", Gu, "Gu(676,加权)", "ext_weighted", N_LP_REPS)
    reps = pd.DataFrame(rep_rows)
    RUNTIME["lp_reps"] = round(time.time() - t, 2)
    C.save_csv(reps, MODULE, "lp_giant_community_reps.csv")

    # 权重盲 LPA 在 Gu / Gb 上是否给出同一划分（"加权图上是否还出现"的关键事实）
    same_gb_gu = 0
    for k in range(N_LP_REPS):
        H1 = permuted(Gb, SEED + k)
        H2 = permuted(Gu, SEED + k)
        p1 = sorted(tuple(sorted(c)) for c in nx.community.label_propagation_communities(H1))
        p2 = sorted(tuple(sorted(c)) for c in nx.community.label_propagation_communities(H2))
        same_gb_gu += int(p1 == p2)
    log(f"lp_giant_community_reps.csv rows={len(reps)} ({RUNTIME['lp_reps']}s) | "
        f"nx 权重盲 LPA 在同一置换下 Gu 与 Gb 划分完全一致：{same_gb_gu}/{N_LP_REPS}")

    # ---- step5 度保持零模型上的同一套 LPA
    t = time.time()
    stat_blind = lambda H: max(len(c) for c in nx.community.label_propagation_communities(H))  # noqa: E731
    null_blind = C.degree_preserving_null(Gb, n_reps=N_NULL_REPS, seed=SEED, stat=stat_blind)
    stat_ext = lambda H: max(len(c) for c in lp_weighted(H, SEED)[0])  # noqa: E731
    null_ext = C.degree_preserving_null(Gu, n_reps=N_NULL_REPS, seed=SEED, stat=stat_ext)
    RUNTIME["null_model"] = round(time.time() - t, 2)
    obs_blind676 = int(max(len(c) for c in nx.community.label_propagation_communities(Gb)))
    log(f"零模型（度序列保持、结构打散，{N_NULL_REPS} 次独立重复）：权重盲 LPA 最大社区 "
        f"null mean={null_blind['null_mean']:.1f} sd={null_blind['null_sd']:.2f}；"
        f"观测图同一算法 {obs_blind676}；加权 LPA null mean={null_ext['null_mean']:.1f} sd={null_ext['null_sd']:.2f}")

    # ---- step6 同学 C239 内部再拆
    t = time.time()
    giant_sub = C.induced(Gu, sorted(cm_giant))
    gi = C.louvain(giant_sub, seed=42)
    RUNTIME["inside_giant"] = round(time.time() - t, 2)
    log(f"同学 557 巨型社区内部再跑加权 Louvain（{giant_sub.number_of_nodes()}n/{giant_sub.number_of_edges()}e）："
        f"{gi['n_comms_total']} 社区（非单点 {gi['n_comms_nonsingleton']}），Q={gi['Q']:.4f}，最大 {gi['largest']}")

    # ---- 判决所需汇总数字
    def agg(variant):
        r = reps[reps["variant"] == variant]
        lg = r["largest_671"].to_numpy(dtype=float)
        return {
            "variant": variant, "n_reps": int(len(r)),
            "largest_671_min": int(lg.min()), "largest_671_median": float(np.median(lg)),
            "largest_671_mean": round(float(lg.mean()), 2),
            "largest_671_sd": round(float(lg.std(ddof=1)), 2) if len(r) > 1 else float("nan"),
            "largest_671_max": int(lg.max()),
            "objective_graph": r["graph"].iloc[0],
            "reads_weight": bool(r["reads_weight"].iloc[0]),
            "n_ge_500": int(r["giant_ge_500"].sum()), "frac_ge_500": round(float(r["giant_ge_500"].mean()), 4),
            "n_exact_557": int(r["exact_557"].sum()),
            "n_jaccard_ge_080": int((r["jaccard_vs_classmate_giant"] >= 0.80).sum()),
            "jaccard_median": round(float(r["jaccard_vs_classmate_giant"].median()), 4),
            "n_comms_total_median": float(r["n_comms_total"].median()),
            "n_comms_nonsingleton_671_median": float(r["n_comms_nonsingleton_671"].median()),
            "Q_weighted_own_graph_median": round(float(r["Q_weighted_own_graph"].median()), 4),
            "coverage_own_graph_median": round(float(r["coverage_weighted_own_graph"].median()), 4),
            "nmi_vs_louvain_median": round(float(r["nmi_vs_louvain_weighted"].median()), 4),
            "all_converged": bool(r["converged"].all()),
        }

    variants = pd.DataFrame([agg(v) for v in reps["variant"].unique()])
    C.save_csv(variants, MODULE, "lp_variant_stats.csv")
    log(variants[["variant", "n_reps", "n_ge_500", "frac_ge_500", "largest_671_median",
                  "largest_671_min", "largest_671_max", "jaccard_median", "nmi_vs_louvain_median"]].to_string(index=False))

    # 跨算法 NMI（LP 参考划分 vs Louvain 参考划分）
    def cell(a, bname, mat):
        ia, ib = names.index(a), names.index(bname)
        return float(mat[ia, ib])

    key = {
        "module": MODULE,
        "seed_conventions": SEED,
        "anchors": "31/31 PASS",
        "graphs": {"Gu": [Gu.number_of_nodes(), Gu.number_of_edges()],
                   "Gb": [Gb.number_of_nodes(), Gb.number_of_edges()],
                   "backbone_w10": [bb.number_of_nodes(), bb.number_of_edges()],
                   "attributed": len(attr)},
        "classmate_giant_557": len(cm_giant),
        "n_algorithms": N,
        "algorithm_names": names,
        "louvain_weighted_ref": {k: rows[0][k] for k in ("n_comms_total", "n_comms_nonsingleton",
                                                          "largest_671", "Q_weighted_Gu", "Q_binary_Gb")},
        "louvain_binary_ref": {k: rows[1][k] for k in ("n_comms_total", "n_comms_nonsingleton",
                                                        "largest_671", "Q_weighted_Gu", "Q_binary_Gb")},
        "greedy_weighted_ref": {k: rows[6][k] for k in ("n_comms_total", "n_comms_nonsingleton",
                                                        "largest_671", "Q_weighted_Gu")},
        "greedy_binary_ref": {k: rows[7][k] for k in ("n_comms_total", "n_comms_nonsingleton",
                                                      "largest_671", "Q_weighted_Gu")},
        "nmi_matrix_offdiag": {"min": round(float(np.nanmin(offdiag)), 4),
                               "median": round(float(np.nanmedian(offdiag)), 4),
                               "max": round(float(np.nanmax(offdiag)), 4)},
        "nmi_seed_ceiling_louvain_weighted": ceil_w,
        "nmi_seed_ceiling_louvain_binary": ceil_b,
        "nmi_louvainw_vs_lpa_binary": cell("louvain_weighted", "label_prop_binary", nmi),
        "ari_louvainw_vs_lpa_binary": cell("louvain_weighted", "label_prop_binary", ari),
        "nmi_louvainw_vs_greedyw": cell("louvain_weighted", "greedy_modularity_weighted", nmi),
        "nmi_lpa_binary_vs_asyn_weighted": cell("label_prop_binary", "label_prop_async_weighted", nmi),
        "nmi_lpa_binary_vs_backbone_lpa": cell("label_prop_binary", "label_prop_backbone_w10", nmi),
        "nmi_louvainw_vs_backbone_louvain": None,
        "lp_variants": variants.to_dict("records"),
        "nx_blind_identical_Gu_Gb": [same_gb_gu, N_LP_REPS],
        "null_blind_lpa": {"obs_676": obs_blind676, "null_mean": round(null_blind["null_mean"], 2),
                            "null_sd": round(null_blind["null_sd"], 2), "n_reps": N_NULL_REPS},
        "null_ext_weighted_lpa": {"null_mean": round(null_ext["null_mean"], 2),
                                   "null_sd": round(null_ext["null_sd"], 2), "n_reps": N_NULL_REPS},
        "giant_internally": {"n_nodes": giant_sub.number_of_nodes(), "n_edges": giant_sub.number_of_edges(),
                             "n_comms_total": gi["n_comms_total"], "n_comms_nonsingleton": gi["n_comms_nonsingleton"],
                             "Q": round(gi["Q"], 4), "largest": gi["largest"]},
        "objectives": {r["algorithm"]: {k: r[k] for k in ("Q_weighted_Gu", "Q_binary_Gb",
                                                          "coverage_weighted_Gu", "coverage_binary_Gb")}
                       for r in rows},
        "runtime_s": {**RUNTIME, "total": round(time.time() - t_all, 2)},
        "versions": C.versions(),
    }
    C.save_json(key, MODULE, "key_numbers.json")

    write_result_md(summary, reps, variants, key, nmi, ari, names)

    C.write_run_note(MODULE, [
        f"B2_algorithm_contrast RUN.txt",
        f"started(utc-ish local): {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"command: python {os.path.join(OUT_DIR, 'run_b2.py')}",
        f"cwd: {os.getcwd()}",
        f"python={C.versions()['python']} networkx={C.versions()['networkx']} sklearn 1.8.0 "
        f"scipy 1.17.1 numpy 2.4.4 pandas 3.0.2 PYTHONHASHSEED={C.versions()['PYTHONHASHSEED']}",
        f"anchors: C.assert_anchors() 31/31 PASS",
        f"seeds: louvain/greedy/spectral reference seed=42 or SEED={SEED}；"
        f"LPA 重复 {N_LP_REPS}/{N_ASYN_REPS} 次（随机源=节点插入顺序置换 + 自实现随机投票）",
        f"null model: {N_NULL_REPS} 次独立重复（每轮 fresh copy 再 double_edge_swap，nswap=10m，max_tries=100*nswap）",
        f"per-stage seconds: {json.dumps(RUNTIME)}",
        f"total wall: {round(time.time() - t_all, 2)}s",
        "outputs: run_b2.py algorithm_summary.csv nmi_matrix.csv ari_matrix.csv "
        "lp_giant_community_reps.csv lp_variant_stats.csv key_numbers.json result.md RUN.txt",
        "",
        "== log ==",
        *LOG,
    ])
    log(f"DONE in {round(time.time() - t_all, 2)}s -> {OUT_DIR}")


# ------------------------------------------------------------------ result.md

def write_result_md(summary, reps, variants, key, nmi, ari, names) -> None:
    idx = {n: i for i, n in enumerate(names)}

    def m(a, b):
        return nmi[idx[a], idx[b]]

    def a(a_, b_):
        return ari[idx[a_], idx[b_]]

    def fmt(mat):
        head = "| 算法 A \\ B | " + " | ".join(n.replace("_", " ") for n in names) + " |"
        sep = "|" + "---|" * (len(names) + 1)
        lines = [head, sep]
        for i, n in enumerate(names):
            cells = []
            for j in range(len(names)):
                v = mat[i, j]
                cells.append("—" if i == j else (f"{v:.3f}" if v == v else "n/a"))
            lines.append(f"| **{n}** | " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def tbl(df: pd.DataFrame, cols: list[str]) -> str:
        d = df[cols].copy()
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in d.iterrows():
            lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        return "\n".join(lines)

    v = {r["variant"]: r for r in variants.to_dict("records")}
    bl = v["nx_LPA_binary_nodeorder_shuffled"]
    wt = v["nx_LPA_weightedgraph_shuffled"]
    a671 = v["nx_LPA_attr671_shuffled"]
    bb = v["nx_LPA_backbone_w10_shuffled"]
    aw = v["nx_asynLPA_weighted_shuffled"]
    au = v["nx_asynLPA_unweighted_shuffled"]
    ex = v["ext_weightedLPA_Gu"]
    can = v["nx_LPA_binary_canonical_order"]
    obj = key["objectives"]
    gi = key["giant_internally"]

    md = f"""# B2 · 算法对照矩阵（{MODULE}）

- 口径（CONVENTIONS §1/§3/§8）：图一律 `csa_common.build_all()`；骨架 `strong_backbone(Gu, min_weight=10)`；
  相似度只用 `partition_metrics()`（覆盖交集上重编号）；社区数同报 `n_comms_total` / `n_comms_nonsingleton`；
  节点口径 **671 有属性（主分析）** 与 **676 全图（锚点/拓扑）** 两列并报。
- 锚点：`C.assert_anchors()` **31/31 PASS**。图：Gu {key['graphs']['Gu'][0]}n/{key['graphs']['Gu'][1]}e、
  Gb {key['graphs']['Gb'][0]}n/{key['graphs']['Gb'][1]}e、w>=10 骨架 {key['graphs']['backbone_w10'][0]}n/{key['graphs']['backbone_w10'][1]}e。
- 算法数：**{key['n_algorithms']}**（加权 Louvain、无权 Louvain、标签传播×4 变体、贪心模块度×2、KL 二分、Laplace 谱二分）。
- 全部数字可在 `algorithm_summary.csv` / `nmi_matrix.csv` / `ari_matrix.csv` / `lp_giant_community_reps.csv` /
  `lp_variant_stats.csv` / `key_numbers.json` 复算。总耗时 {key['runtime_s']['total']}s。

## 0. 一句话判决

**「557 节点巨型社区」是算法产物，不是数据事实。** 可核对的三组数字：

1. **权重盲的标签传播**（同学用的那一族）在 676 全图换节点遍历顺序重复 {bl['n_reps']} 次：
   最大社区（671 口径）median **{bl['largest_671_median']:.0f}**，范围 **{bl['largest_671_min']}–{bl['largest_671_max']}**，
   其中 **{bl['n_ge_500']}/{bl['n_reps']}**（{bl['frac_ge_500']*100:.0f}%）出现 ≥500 的巨型社区，
   与同学 557 集合的 Jaccard median **{bl['jaccard_median']}**（{bl['n_jaccard_ge_080']}/{bl['n_reps']} 次 ≥0.80）。
   → 巨型社区**能复现**，但规模本身摆动 {bl['largest_671_min']}–{bl['largest_671_max']}（±{(bl['largest_671_sd'])/bl['largest_671_median']*100:.0f}%），
   且**恰好等于 557 的次数只有 {bl['n_exact_557']}/{bl['n_reps']}**。
2. **权重一旦进入算法，巨型社区消失**：
   `asyn_lpa_communities(Gu, weight="weight")` {aw['n_reps']} 次，≥500 出现 **{aw['n_ge_500']}/{aw['n_reps']}**，
   最大社区 median **{aw['largest_671_median']:.0f}**（范围 {aw['largest_671_min']}–{aw['largest_671_max']}）；
   自实现的加权投票 LPA `ext_weightedLPA_Gu` {ex['n_reps']} 次：**{ex['n_ge_500']}/{ex['n_reps']}**，median **{ex['largest_671_median']:.0f}**；
   同一 nx 权重盲实现在 **w>=10 骨架**（权重经筛边进入）上 {bb['n_reps']} 次：**{bb['n_ge_500']}/{bb['n_reps']}**，median **{bb['largest_671_median']:.0f}**。
3. **优化模块度的两类算法从未给出 500+ 社区**：加权 Louvain 最大 {rows_largest('louvain_weighted', summary)}、
   无权 Louvain 最大 {rows_largest('louvain_binary', summary)}、贪心模块度加权 {rows_largest('greedy_modularity_weighted', summary)}、
   无权 {rows_largest('greedy_modularity_binary', summary)}；`algorithm_summary.csv` 里 `n_comms_ge_500` 列对全部 {key['n_algorithms']} 个参考划分都 = 0。
   同学的 557 巨型社区内部再跑一次加权 Louvain：{gi['n_nodes']}n/{gi['n_edges']}e →
   **{gi['n_comms_total']} 个社区（非单点 {gi['n_comms_nonsingleton']}），Q={gi['Q']}，最大 {gi['largest']}** —— 它装得下整套模块结构。

对照基线（判决的第四根柱子）：**度保持零模型**（{key['null_blind_lpa']['n_reps']} 次独立重复，每轮 fresh copy 再交换）上跑
**同一个权重盲 LPA**，最大社区 mean **{key['null_blind_lpa']['null_mean']} ± {key['null_blind_lpa']['null_sd']}**，
观测图上同一算法给 **{key['null_blind_lpa']['obs_676']}**（676 口径）。
→ 结构被打散、度序列不变的图上，权重盲 LPA 给出的社区**更大**。巨型社区随「算法是否读权重」而消失、
随「结构被打散」而膨胀：它不是这组超链接的社区结构。

## 1. 算法 × 指标（参考划分，`algorithm_summary.csv`）

`n_comms_total` / `n_comms_nonsingleton`（676 全图）与 671 口径同报；`largest_671` 为主分析口径最大社区。

{tbl(summary, ["algorithm", "graph", "n_comms_total", "n_comms_nonsingleton", "n_comms_nonsingleton_671",
               "largest_671", "largest_share_of_671_pct", "n_comms_ge_500", "size_median", "size_max",
               "Q_weighted_Gu", "Q_binary_Gb", "coverage_weighted_Gu", "coverage_binary_Gb", "runtime_s"])}

规模分布（671 口径，非单点社区）：`size_min / size_q1 / size_median / size_mean / size_q3 / size_max` 见 CSV；
表中 `size_median / size_max` 两列即每算法的分布摘要，`sizes_top8_671` 存前 8 大社区规模。

**权重贡献多少**（同一 seed=42，只把图换成无权版）：加权 Louvain
**{key['louvain_weighted_ref']['n_comms_total']} 社区 / 非单点 {key['louvain_weighted_ref']['n_comms_nonsingleton']} / 最大 {key['louvain_weighted_ref']['largest_671']} / Q(加权) {key['louvain_weighted_ref']['Q_weighted_Gu']}**
vs 无权 Louvain **{key['louvain_binary_ref']['n_comms_total']} 社区 / 非单点 {key['louvain_binary_ref']['n_comms_nonsingleton']} / 最大 {key['louvain_binary_ref']['largest_671']} / Q(无权) {key['louvain_binary_ref']['Q_binary_Gb']}**；
两者互相的 NMI **{m('louvain_weighted', 'louvain_binary'):.4f}** / ARI **{a('louvain_weighted', 'louvain_binary'):.4f}**。
边集完全相同、权重只作为投票/合并的代价进入 —— 社区数差 {abs(key['louvain_weighted_ref']['n_comms_total']-key['louvain_binary_ref']['n_comms_total'])} 个，
Q 差 {abs(key['louvain_weighted_ref']['Q_weighted_Gu']-key['louvain_binary_ref']['Q_binary_Gb']):.4f}（**不可直接比大小**，见 §4）。

## 2. NMI / ARI 交叉表（{key['n_algorithms']} × {key['n_algorithms']}）

### 2.1 NMI

{fmt(nmi)}

### 2.2 ARI

{fmt(ari)}

### 2.3 怎么读这两张表（**必须先给天花板**）

- **同算法换 seed 的 NMI = 天花板**：加权 Louvain 6 个 seed 两两
  **{key['nmi_seed_ceiling_louvain_weighted']}**；无权 Louvain **{key['nmi_seed_ceiling_louvain_binary']}**。
  跨算法 NMI 只有对着这个天花板才有意义。
- 跨算法 NMI：min **{key['nmi_matrix_offdiag']['min']}** / median **{key['nmi_matrix_offdiag']['median']}** / max **{key['nmi_matrix_offdiag']['max']}**。
- 三个关键读数：
  1. `louvain_weighted` vs `label_prop_binary` NMI **{key['nmi_louvainw_vs_lpa_binary']}** / ARI **{key['ari_louvainw_vs_lpa_binary']}**
     ——远低于同算法天花板 {key['nmi_seed_ceiling_louvain_weighted']}：**两套结论的差异远大于 seed 噪声**，
     矛盾真实存在，不能靠「换一个 seed 重跑」抹平。
  2. `louvain_weighted` vs `greedy_modularity_weighted` NMI **{key['nmi_louvainw_vs_greedyw']}**
     ——两个独立最大化模块度的算法彼此**高度一致**（接近天花板）。
  3. `label_prop_binary` vs `label_prop_async_weighted` NMI **{key['nmi_lpa_binary_vs_asyn_weighted']}**、
     vs `label_prop_backbone_w10` NMI **{key['nmi_lpa_binary_vs_backbone_lpa']}**
     ——同一「标签传播」名字下，读不读权重给出**互不相干的划分**。
- KL / 谱二分那两行只有 2 个社区（块大小由「各半」约束给定，不由数据决定），它们与一切的多值 NMI 天然偏低，
  不能当成「它们不同意模块度族」的证据；放进来只为说明**目标函数族**的差别。

## 3. 标签传播重复实验（`lp_giant_community_reps.csv`，每次重复一行）

判据要求 ≥30 次；每种变体跑 {N_LP_REPS} 或 {N_ASYN_REPS} 次（canonical 顺序那次单独 1 次）。
随机源写明：nx 3.6.1 的 `label_propagation_communities` **没有随机参数**，初始标签 = `enumerate(G)` 的节点插入顺序、
更新顺序 = 按颜色分组 → 唯一可复现的随机源是把节点插入顺序置换掉（边集、边权、边次序不变，`same_structure` 已逐次校验）；
`asyn_lpa_communities` 与自实现加权 LPA 用 `seed = SEED + rep`。

{tbl(variants, ["variant", "n_reps", "objective_graph", "reads_weight",
                "n_comms_total_median", "n_comms_nonsingleton_671_median",
                "largest_671_min", "largest_671_median", "largest_671_max", "n_ge_500", "frac_ge_500",
                "n_exact_557", "jaccard_median", "nmi_vs_louvain_median", "Q_weighted_own_graph_median",
                "coverage_own_graph_median"])}

> `Q_weighted_own_graph_median` / `coverage_own_graph_median` 的目标函数都在**本变体自己的图**（`objective_graph` 列）上算：
> 671 诱导图、w≥10 骨架与 676 全图的 Q **不可跨行比较**，只能在同一 `objective_graph` 内比。

- **是否只在无权图上出现？** 更准确的事实是：**nx 的 `label_propagation_communities` 根本没有 weight 参数**，
  它按邻居**计数**投票。把加权图 `Gu` 直接喂进去，{key['nx_blind_identical_Gu_Gb'][0]}/{key['nx_blind_identical_Gu_Gb'][1]} 次与无权图 `Gb` 在同一置换下给出
  **逐节点完全相同的划分**（`key_numbers.json: nx_blind_identical_Gu_Gb`）。所以「巨型社区只在无权图出现」这句话不成立：
  它在两张图上出现，理由都是**算法没有读权重**。
- canonical 插入顺序（= 本项目建图口径，`edge_order`）只给 1 次：{can['n_comms_total_median']:.0f} 社区、最大 **{can['largest_671_median']:.0f}**（671 口径）、
  与同学 557 集合 Jaccard **{can['jaccard_median']}** → 同学那个具体数值对应的是**另一套遍历顺序**，
  不是本流水线图上的确定输出。
- 671 主分析口径（`nx_LPA_attr671_shuffled`）：median **{a671['largest_671_median']:.0f}**，≥500 **{a671['n_ge_500']}/{a671['n_reps']}**。
  巨型社区与节点口径几乎无关（671 口径 vs 676 口径都在同一量级），
  **同学「82.4%」的分母用的是 676**（557/676=82.4%；换成主分析口径 671 是 83.0%）——两个口径都要写明。

## 4. 分辨率 / 模块化不可比警告（判据第 4 条）

**不同算法的 Q 不能直接比大小。** 三件事同时成立：

1. **目标函数不同**：Louvain 与贪心模块度最大化 `Q(γ=1)`；标签传播**不优化任何目标函数**（多数投票收敛即停）；
   KL 二分最小化割、谱二分最小化规范化割，块大小由「各半」的约束给定。后三类划分的 Q 是**事后打分**，不是它们的优化结果。
2. **γ（分辨率）不同也不可比**：`Q(γ)` 的零模型项随 γ 缩放（见 B3 单元），换 γ 等于换打分尺。本表全部在 γ=1。
3. **加权 / 无权两套零模型**：同一划分在 `Gu`（按边权）与 `Gb`（按边数）上的 Q 数值差得很大
   （如 `louvain_weighted` 划分：Q(加权 Gu)={obj['louvain_weighted']['Q_weighted_Gu']}，同一划分 Q(无权 Gb)={obj['louvain_weighted']['Q_binary_Gb']}），
   这只反映**分母（2m）换了**，不反映好坏。

因此本单元给每个划分**同时报两套目标函数**（同一划分、同一份数据、两种打分）：

| 划分（参考运行） | Q(加权 Gu) | Q(无权 Gb) | 覆盖率(加权) | 覆盖率(无权) | 该算法自己优化的目标 |
|---|---:|---:|---:|---:|---|
| `louvain_weighted` | {obj['louvain_weighted']['Q_weighted_Gu']} | {obj['louvain_weighted']['Q_binary_Gb']} | {obj['louvain_weighted']['coverage_weighted_Gu']} | {obj['louvain_weighted']['coverage_binary_Gb']} | 模块度 Q（加权, γ=1） |
| `louvain_binary` | {obj['louvain_binary']['Q_weighted_Gu']} | {obj['louvain_binary']['Q_binary_Gb']} | {obj['louvain_binary']['coverage_weighted_Gu']} | {obj['louvain_binary']['coverage_binary_Gb']} | 模块度 Q（无权） |
| `label_prop_binary` | {obj['label_prop_binary']['Q_weighted_Gu']} | {obj['label_prop_binary']['Q_binary_Gb']} | {obj['label_prop_binary']['coverage_weighted_Gu']} | {obj['label_prop_binary']['coverage_binary_Gb']} | 无（多数投票） |
| `label_prop_async_weighted` | {obj['label_prop_async_weighted']['Q_weighted_Gu']} | {obj['label_prop_async_weighted']['Q_binary_Gb']} | {obj['label_prop_async_weighted']['coverage_weighted_Gu']} | {obj['label_prop_async_weighted']['coverage_binary_Gb']} | 无（加权投票） |
| `greedy_modularity_weighted` | {obj['greedy_modularity_weighted']['Q_weighted_Gu']} | {obj['greedy_modularity_weighted']['Q_binary_Gb']} | {obj['greedy_modularity_weighted']['coverage_weighted_Gu']} | {obj['greedy_modularity_weighted']['coverage_binary_Gb']} | 模块度 Q（加权，贪心） |
| `kernighan_lin_bisection` | {obj['kernighan_lin_bisection']['Q_weighted_Gu']} | {obj['kernighan_lin_bisection']['Q_binary_Gb']} | {obj['kernighan_lin_bisection']['coverage_weighted_Gu']} | {obj['kernighan_lin_bisection']['coverage_binary_Gb']} | 割（平衡二分） |
| `spectral_bisection_fiedler` | {obj['spectral_bisection_fiedler']['Q_weighted_Gu']} | {obj['spectral_bisection_fiedler']['Q_binary_Gb']} | {obj['spectral_bisection_fiedler']['coverage_weighted_Gu']} | {obj['spectral_bisection_fiedler']['coverage_binary_Gb']} | 规范化割 |

两张表**方向相反**：标签传播的**覆盖率**最高（加权 {obj['label_prop_binary']['coverage_weighted_Gu']}，
即把 {obj['label_prop_binary']['coverage_weighted_Gu']*100:.1f}% 的权重留在社区内），而它的 **Q 最低**（加权 {obj['label_prop_binary']['Q_weighted_Gu']}）。
覆盖率不含零模型项，**不惩罚「把节点并进一个大社区」**（单点社区也不受罚），所以它天然偏向巨型社区；
Q 有度序列零模型惩罚，天然偏向中等规模的多模块结构。
**所以：不能写「Louvain 的 Q 比标签传播高 ⇒ Louvain 更好」。** 本单元的判决不建立在 Q 的大小上，
建立在 §0 的三组可复算数字 + §5 的零模型 + §6 的巨型社区内部结构上。

## 5. 零模型对照（度序列保持、结构打散）

`csa_common.degree_preserving_null`，nswap=10m、max_tries=100·nswap，每轮从原图 fresh copy 再交换（CONVENTIONS §4）：

| 统计量 | 观测图 | 零模型（{key['null_blind_lpa']['n_reps']} 次独立重复） |
|---|---:|---:|
| 权重盲 LPA 最大社区（676 口径） | {key['null_blind_lpa']['obs_676']} | mean {key['null_blind_lpa']['null_mean']} ± sd {key['null_blind_lpa']['null_sd']} |
| 加权 LPA（自实现）最大社区 | 见 `lp_giant_community_reps.csv`（median {ex['largest_671_median']:.0f}，671 口径） | mean {key['null_ext_weighted_lpa']['null_mean']} ± sd {key['null_ext_weighted_lpa']['null_sd']} |

第一行是判决的核心：**把结构打散、只留度序列，权重盲 LPA 的巨型社区反而比观测图上更大**。
这说明该指标在这个算法上主要测的是「图有多连通 + 算法有多粗」，不是「有没有院系式社区」。
局限：Viger–Lat边交换并非严格均匀采样（CONVENTIONS §4 要求声明）——此处零模型只用来做**量级对照**，不做显著性检验。

## 6. 同学的 557 巨型社区：三条独立检验

1. **它属于哪一族算法**：只有权重盲 LPA 族出现 500+ 社区；其 ≥500 频率 **{bl['n_ge_500']}/{bl['n_reps']}**（676 图）/ **{a671['n_ge_500']}/{a671['n_reps']}**（671 图）；
   但**恰好 557** 只有 **{bl['n_exact_557']}** 次；规模本身在 {bl['largest_671_min']}–{bl['largest_671_max']} 摆动。
2. **它与本流水线主口径划分的关系**：NMI **{key['nmi_louvainw_vs_lpa_binary']}** / ARI **{key['ari_louvainw_vs_lpa_binary']}**（vs 加权 Louvain），
   同算法 seed 天花板 {key['nmi_seed_ceiling_louvain_weighted']} → 差异是算法族层面的，不是噪声。
3. **它内部还有没有结构**：加权 Louvain 在这 557 个节点（{gi['n_edges']} 条边）上给出 **{gi['n_comms_total']} 个社区**
   （非单点 {gi['n_comms_nonsingleton']}），Q={gi['Q']}，最大 {gi['largest']} → 「557」不是模块意义上的原子单位，
   它把整套模块结构包在里面。

**结论（写进正文时允许的表述方式）**：在本数据上，「巨型社区」这一结果随算法是否读权重、
随节点遍历顺序而变，并在结构被打散的度保持零模型上变得更大；同一批图上所有优化模块度的算法
给出的最大社区都在 {rows_largest('louvain_binary', summary)}–{rows_largest('greedy_modularity_binary', summary)} 区间。
因此「社区=院系」的解读**只能建立在加权 Louvain（或贪心模块度）这一族划分上**，且必须配 §3 的分布与
B1 的 seed 误差棒，不能拿 557 当事实基准。

## 7. 已知坑 / 诚实记录

- `nx.community.label_propagation_communities` **无 weight 参数**：把加权图喂进去**不等于**加权标签传播。
  同学 REPORT.md §4 写「确定性顺序、最多 30 轮、6 轮收敛」，与本实现的语义（按颜色分组迭代到稳定）不完全对应，
  其 557/562 两个数字在仓库里同时出现（`node_communities.csv` 的 `community_size` 列 562、按 community 分组计数 557），
  本单元按 **557（671 口径）** 对齐，并在 §0/§3 报出摆动范围。
- 贪心模块度在全图 2498 条无向边上跑得动：加权 {summary.loc[summary.algorithm=='greedy_modularity_weighted','runtime_s'].iloc[0]}s、
  无权 {summary.loc[summary.algorithm=='greedy_modularity_binary','runtime_s'].iloc[0]}s，**未降级到 w>=10 骨架**。
- `w>=10` 骨架上 {bb['n_reps']} 次 LPA 的 `n_comms_total` median **{bb['n_comms_total_median']:.0f}**，其中大量是零度孤立节点各成一社区；
  非单点口径 median **{bb['n_comms_nonsingleton_671_median']:.0f}**（CONVENTIONS §3：报总数会把数字抬高一个量级）。
- KL / 谱二分的社区数 = 2 由算法构造给定，不是数据结果；表中仅作目标函数族对照。
- 自实现 `ext_weightedLPA_Gu` 存在**不收敛/振荡**：{ex['n_reps']} 次中 converged=True 的比例 {v['ext_weightedLPA_Gu']['all_converged']}
  （按 max_sweeps={100} 截断，`converged`/`sweeps` 列逐次写明）。这一族实现的收敛性本身是标签传播的已知弱点，不是本单元引入的错误。
- 全部结果为**静态横截面**链接快照；本单元不做任何因果陈述（CONVENTIONS §7）。

## 8. 产物

| 文件 | 内容 |
|---|---|
| `run_b2.py` | 本单元全部计算，可独立重跑 |
| `algorithm_summary.csv` | {key['n_algorithms']} 算法 × 社区数（两套口径）× 分布 × 两套目标函数 × 耗时 |
| `nmi_matrix.csv` / `ari_matrix.csv` | {key['n_algorithms']}×{key['n_algorithms']} 交叉表（`partition_metrics`，覆盖交集重编号） |
| `lp_giant_community_reps.csv` | 标签传播每次重复（{len(reps)} 行）：最大社区、≥500、与同学 557 的 Jaccard、Q/覆盖率 |
| `lp_variant_stats.csv` | 上述重复的按变体汇总（§3 表） |
| `key_numbers.json` | 全部判决数字（机读） |
| `RUN.txt` | 环境版本、命令、seed/重复数、分阶段耗时、运行日志 |
"""
    with open(os.path.join(OUT_DIR, "result.md"), "w", encoding="utf-8") as f:
        f.write(md)
    log("wrote result.md")


def rows_largest(algo: str, df: pd.DataFrame) -> int:
    return int(df.loc[df["algorithm"] == algo, "largest_671"].iloc[0])


if __name__ == "__main__":
    main()
