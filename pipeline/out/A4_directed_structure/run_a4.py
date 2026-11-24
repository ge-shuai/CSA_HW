# -*- coding: utf-8 -*-
"""A4_directed_structure — 有向结构与「进得去出不来」

覆盖 MANIFEST A4 的 5 条判据：
  J1 SCC 规模谱 / 弱分量 / 缩点 DAG（层数、最长链、孤立 SCC 数）/ reciprocity
  J2 Bow-tie 分组（Broder et al. 2000, Computer Networks 33:309），本文件自写实现
     + 分量级缩点第二套代码路径交叉校验 + 3 个手工最小例
  J3 |in−out|/(in+out) 不对称度分布；入度=0 / 出度=0 节点清单与规模
  J4 出度删失（162 个整站不可用节点出度全为 0，edges_from_dead=0）的对照分析：
     只保留 360 个健康节点的有向结构，逐条判定哪些结论保留、哪些消失、哪些不可识别
  J5 listed_urls 的 20 条落库上限对出度的截断量级，以及对 sink 比例的高估幅度

口径全部取自 csa_common（构图 / 边序 / 健康分层 / 失效校正）；开跑先 C.assert_anchors()。
措辞遵循 CONVENTIONS §7：不写因果动词。「出度被删失 / 被截断」是测量边界陈述。

运行：python out/A4_directed_structure/run_a4.py    （工作目录 = pipeline/）
"""
from __future__ import annotations

import os
import sys
import time
import collections

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.dirname(os.path.dirname(HERE))
if PIPE not in sys.path:
    sys.path.insert(0, PIPE)

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.stats as ss  # noqa: E402

import csa_common as C  # noqa: E402

SEED = C.SEED  # 20261113。本单元全部为确定性计算，无随机过程，seed 仅按约定留痕。
MODULE = "A4_directed_structure"
Q = "\u201c"  # “
QE = "\u201d"  # ”

T0 = time.time()
LOG: list[str] = []


def log(msg: str) -> None:
    line = f"[{time.time() - T0:7.2f}s] {msg}"
    print(line, flush=True)
    LOG.append(line)


def pq(s: str) -> str:
    """把 |...| 形式的关键字包成中文引号，避免与 Python 字符串定界符冲突。"""
    return Q + s + QE


# =========================================================== bow-tie 实现（J2）

BOWTIE_GROUPS = ["CORE", "IN", "OUT", "TUBE", "IN-COMPONENTS", "OUT-COMPONENTS", "DISCONNECTED"]


def _fwd(G: nx.DiGraph, seeds) -> set:
    """多源前向可达集（含 seeds 自身）。手写 DFS 栈，不用 nx.descendants 以便逐步核对。"""
    seen = set(seeds)
    stack = list(seen)
    while stack:
        u = stack.pop()
        for v in G.succ[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


def _bwd(G: nx.DiGraph, seeds) -> set:
    """多源**反向**可达集：能到达 seeds 中任一点的节点集合（含 seeds 自身）。"""
    return _fwd(G.reverse(copy=True), seeds)


def bowtie(G: nx.DiGraph) -> dict:
    """Broder et al. (2000, Computer Networks 33:309) 的 7 组划分。

    记 down = CORE 可达点集，up = 可达 CORE 点集。操作性定义逐条对应本函数体
    （原文措辞 "pages on directed paths from IN to OUT"；转述见 Leskovec–
    Rajaraman–Ullman《Mining of Massive Datasets》3rd ed. Ch.5）：
      CORE            最大强连通分量（并列时取 min(节点名) 最小者，确定性）
      IN              能到达 CORE、但 CORE 到不了它              up − down
      OUT             CORE 能到它、但它到不了 CORE              down − up
      TUBE            未归组、且**在 IN→OUT 的有向通路上**：既可从 IN 到达、又能到 OUT
      IN-COMPONENTS   未归组、且**可从 IN 到达**（挂在 IN 下游的触须）
      OUT-COMPONENTS  未归组、且**能到达 OUT**（挂在 OUT 上游的触须，反向可达）
      DISCONNECTED    其余

    两个**可证明的恒空陷阱**（自查发现，已写成 selftest 断言）：
      (a) 常见的简化口径 `TUBE = up ∩ down − CORE` 对任何图都恒为空：
          若 v 可达 CORE 且被 CORE 可达，则 v 与核内所有点互达，v 必属于该 SCC。
          本函数把它降级为诊断量 `n_tube_vacuous_upcapdown`（应恒为 0）。
      (b) `OUT-COMPONENTS` 若按字面取「可从 OUT 到达」也恒为空：
          OUT = down − up，自 OUT 出发的点仍落在 down 内、已被 CORE/OUT 吸收。
          正确口径是「**能到达** OUT」（反向）。诊断量 `n_outcomp_naive_forward`。

    实现约定（可核对）：
      1) 前向可达 `_fwd`、反向可达 `_bwd`，都是手写 DFS 栈，不调用 nx.descendants；
      2) 抢占顺序 TUBE → IN-COMPONENTS → OUT-COMPONENTS；一个未归组点若既可自 IN
         到达又能到 OUT，按原文归 TUBE（这不是任意选择：它正是原文的 TUBE 定义）；
      3) 组内自检：七组两两不交、并集 = 全节点集，函数内 assert。
    """
    sccs = [set(c) for c in nx.strongly_connected_components(G)]
    mx = max(len(s) for s in sccs)
    core = set(sorted([frozenset(s) for s in sccs if len(s) == mx], key=lambda c: min(c))[0])

    down = _fwd(G, core)          # CORE 能到达的（含 CORE）
    up = _bwd(G, core)            # 能到达 CORE 的（含 CORE）

    IN = up - down
    OUT = down - up
    base = core | IN | OUT
    from_IN = _fwd(G, IN)         # 可从 IN 到达
    to_OUT = _bwd(G, OUT)         # 能到达 OUT

    TUBE = (from_IN & to_OUT) - base          # 在 IN→OUT 的有向通路上
    INc = from_IN - base - TUBE
    OUTc = to_OUT - base - TUBE - INc
    DIS = set(G.nodes()) - base - TUBE - INc - OUTc

    groups = {"CORE": core, "IN": IN, "OUT": OUT, "TUBE": TUBE,
              "IN-COMPONENTS": INc, "OUT-COMPONENTS": OUTc, "DISCONNECTED": DIS}

    assert sum(len(v) for v in groups.values()) == G.number_of_nodes(), "bow-tie 未覆盖全部节点"
    assert len(set().union(*groups.values())) == sum(len(v) for v in groups.values()), "分组有重叠"
    return {"groups": groups, "core": core, "n_core_tied": sum(1 for s in sccs if len(s) == mx),
            "n_tube_vacuous_upcapdown": len((up & down) - core),
            "n_outcomp_naive_forward": len(_fwd(G, OUT) - base - TUBE - INc),
            "n_in_out_tendril_overlap": len(TUBE),
            "scc_sizes_desc": sorted((len(s) for s in sccs), reverse=True)}


def bowtie_crosscheck(G: nx.DiGraph, bt: dict) -> dict:
    """第二套独立代码路径复算分组：自建缩点图 + `nx.transitive_closure`（闭包传递），
    在**分量级**判组，再展开到节点，与 bowtie() 的节点级可达集逐点比。

    两套路径共用定义、不共用可达性算法（DFS 栈 vs 传递闭包），也不共用节点标识
    （点级 FQDN vs 分量级整数 id）。
    """
    comps = [frozenset(c) for c in nx.strongly_connected_components(G)]
    cid = {v: i for i, c in enumerate(comps) for v in c}
    D = nx.DiGraph()
    D.add_nodes_from(range(len(comps)))
    for u, v in G.edges():
        if cid[u] != cid[v]:
            D.add_edge(cid[u], cid[v])

    tc = nx.transitive_closure(D)   # DAG 传递闭包（Warshall 式，与 DFS 栈完全不同的算法）
    tcr = nx.transitive_closure(D.reverse(copy=True))

    def fwd_c(S):
        o = set(S)
        for s in S:
            o |= set(tc.succ[s])
        return o

    def bwd_c(S):
        o = set(S)
        for s in S:
            o |= set(tcr.succ[s])
        return o

    mx = max(len(c) for c in comps)
    core_c = sorted([i for i, c in enumerate(comps) if len(c) == mx], key=lambda i: min(comps[i]))[0]
    down = fwd_c({core_c})          # CORE 可达的分量
    up = bwd_c({core_c})            # 可达 CORE 的分量
    in_c = up - down
    out_c = down - up
    base = {core_c} | in_c | out_c
    from_IN = fwd_c(in_c) if in_c else set()
    to_OUT = bwd_c(out_c) if out_c else set()
    tube_c = (from_IN & to_OUT) - base
    inc_c = from_IN - base - tube_c
    outc_c = to_OUT - base - tube_c - inc_c
    disc_c = set(D.nodes()) - base - tube_c - inc_c - outc_c

    lab = {}
    for i in D.nodes():
        g = ("CORE" if i == core_c else "IN" if i in in_c else "OUT" if i in out_c
             else "TUBE" if i in tube_c else "IN-COMPONENTS" if i in inc_c
             else "OUT-COMPONENTS" if i in outc_c else "DISCONNECTED")
        for v in comps[i]:
            lab[v] = g
    mism = [(v, k, lab.get(v)) for k in BOWTIE_GROUPS for v in bt["groups"][k] if lab.get(v) != k]
    return {"n_dag_components": D.number_of_nodes(), "n_mismatch": len(mism),
            "mismatch_examples": mism[:5],
            "core_identical": all(lab[v] == "CORE" for v in bt["core"])
            and sum(1 for v in lab.values() if v == "CORE") == len(bt["core"])}


def _selftest_bowtie() -> list[str]:
    """最小例的期望分组是人肉推出来的，不依赖被测实现。三例都同时过 cross-check。"""
    msgs = []
    # 例 1：a<->b 为核；x 只进(IN)；y、z 被核可达(OUT)；m 指向 y 但不被核可达(OUT-COMPONENTS)
    G1 = nx.DiGraph([("a", "b"), ("b", "a"), ("x", "a"), ("b", "y"), ("y", "z"), ("m", "y")])
    exp1 = {"CORE": {"a", "b"}, "IN": {"x"}, "OUT": {"y", "z"}, "TUBE": set(),
            "IN-COMPONENTS": set(), "OUT-COMPONENTS": {"m"}, "DISCONNECTED": set()}
    r1 = bowtie(G1)["groups"]
    assert r1 == exp1, f"selftest1 FAIL: {r1}"
    assert bowtie_crosscheck(G1, bowtie(G1))["n_mismatch"] == 0, "selftest1 crosscheck FAIL"
    msgs.append("selftest1 (6 点：CORE/IN/OUT/OUT-COMPONENTS) OK")
    # 例 2：人工构造覆盖全部 7 组非空。
    #   核 c1<->c2 | s->c1 (s 只进 -> IN) | c2->u->w (被核可达 -> OUT)
    #   s->t、t->u (t 在 IN→OUT 通路上、又不属于前三组 -> TUBE)
    #   s->r (r 可自 IN 到达、但到不了 OUT -> IN-COMPONENTS)
    #   n->u (n 能到 OUT、又不可自 IN 到达 -> OUT-COMPONENTS)
    #   p->q (独立弱分量 -> DISCONNECTED)
    G2 = nx.DiGraph([("c1", "c2"), ("c2", "c1"), ("s", "c1"), ("c2", "u"), ("u", "w"),
                     ("s", "t"), ("t", "u"), ("s", "r"), ("n", "u"), ("p", "q")])
    exp2 = {"CORE": {"c1", "c2"}, "IN": {"s"}, "OUT": {"u", "w"}, "TUBE": {"t"},
            "IN-COMPONENTS": {"r"}, "OUT-COMPONENTS": {"n"}, "DISCONNECTED": {"p", "q"}}
    r2 = bowtie(G2)["groups"]
    assert r2 == exp2, f"selftest2 FAIL: {r2}"
    assert bowtie_crosscheck(G2, bowtie(G2))["n_mismatch"] == 0, "selftest2 crosscheck FAIL"
    msgs.append("selftest2 (11 点：全 7 组均非空) OK")
    # 例 3：核外的孤立 2 点弱分量（本项目 bm->l.bm 的形态）必须落 DISCONNECTED
    G3 = nx.DiGraph([("c1", "c2"), ("c2", "c1"), ("p", "q")])
    r3 = bowtie(G3)["groups"]
    assert r3["DISCONNECTED"] == {"p", "q"}, f"selftest3 FAIL: {r3}"
    assert bowtie_crosscheck(G3, bowtie(G3))["n_mismatch"] == 0, "selftest3 crosscheck FAIL"
    msgs.append("selftest3 (核外孤立 2 点弱分量 -> DISCONNECTED) OK")
    # 例 4：钉死两条恒空陷阱 —— (a) 简化口径 TUBE=up∩down−CORE 必空
    #                              (b) OUT-COMPONENTS 取正向可达必空
    d2 = bowtie(G2)
    assert d2["n_tube_vacuous_upcapdown"] == 0, "selftest4a FAIL"
    assert d2["n_outcomp_naive_forward"] == 0, "selftest4b FAIL"
    # 随机图上再验一次 (a)：任何图都成立，若不成立说明实现或推理有误
    rng = np.random.default_rng(SEED)
    for rep in range(20):
        N = int(rng.integers(6, 30)); K = int(rng.integers(6, N * 3))
        GR = nx.DiGraph()
        for _ in range(K):
            u, v = int(rng.integers(N)), int(rng.integers(N))
            if u != v:
                GR.add_edge(u, v)
        if GR.number_of_edges():
            assert bowtie(GR)["n_tube_vacuous_upcapdown"] == 0, f"selftest4a FAIL rep{rep}"
            assert bowtie(GR)["n_outcomp_naive_forward"] == 0, f"selftest4b FAIL rep{rep}"
            assert bowtie_crosscheck(GR, bowtie(GR))["n_mismatch"] == 0, f"随机图 crosscheck FAIL rep{rep}"
    msgs.append("selftest4 (两条恒空陷阱 + 20 个随机有向图上双路径一致) OK")
    return msgs


# ============================================================ 度量装配（J1/J3）

def directed_metrics(G: nx.DiGraph) -> dict:
    """一张有向图的全部 A4 度量。度一律在 G 内部重算，不吃全图派生列。"""
    out: dict = {}
    n, m = G.number_of_nodes(), G.number_of_edges()
    ind = dict(G.in_degree())
    outd = dict(G.out_degree())
    out["n_nodes"] = n
    out["n_edges"] = m
    out["sum_edge_weight"] = int(G.size(weight="weight"))
    out["n_selfloops"] = sum(1 for u, v in G.edges() if u == v)
    out["density"] = round(nx.density(G), 6) if n > 1 else np.nan

    # --- J1 SCC / WCC / 缩点 DAG ------------------------------------------
    scc = [set(c) for c in nx.strongly_connected_components(G)]
    ssz = sorted((len(c) for c in scc), reverse=True)
    out["n_scc"] = len(scc)
    out["scc_largest"] = ssz[0]
    out["scc_largest_share"] = round(ssz[0] / n, 4)
    out["n_scc_nonsingleton"] = sum(1 for s in ssz if s > 1)
    out["n_scc_singleton"] = sum(1 for s in ssz if s == 1)
    out["n_nodes_in_nonsingleton_scc"] = sum(s for s in ssz if s > 1)
    out["scc_size_hist"] = ";".join(f"{k}pt:{v}个" for k, v in
                                    sorted(collections.Counter(ssz).items(), reverse=True))
    wcc = sorted((len(c) for c in nx.weakly_connected_components(G)), reverse=True)
    out["n_wcc"] = len(wcc)
    out["wcc_sizes_top5"] = ",".join(map(str, wcc[:5]))
    out["n_wcc_singleton"] = sum(1 for s in wcc if s == 1)
    out["wcc_largest"] = wcc[0]
    out["wcc_largest_share"] = round(wcc[0] / n, 4)

    D = nx.condensation(G)
    gens = [set(g) for g in nx.topological_generations(D)]
    lp = nx.dag_longest_path(D)
    mem_of = {i: frozenset(D.nodes[i]["members"]) for i in D.nodes()}
    assert len(ssz) == 1 or ssz[0] != ssz[1], "最大 SCC 存在并列，需人工定核"
    core_id = next(i for i in D.nodes() if mem_of[i] == frozenset(max(scc, key=len)))
    layer_of = {c: i for i, g in enumerate(gens) for c in g}
    out["n_dag_nodes"] = D.number_of_nodes()
    out["n_dag_edges"] = D.number_of_edges()
    out["dag_n_layers"] = len(gens)
    out["dag_layer_sizes"] = ",".join(str(len(g)) for g in gens)
    out["dag_layer_of_core"] = layer_of[core_id]
    out["dag_n_chain_in_core_layers"] = sum(1 for g in gens if core_id in g)
    out["dag_longest_chain_nodes"] = len(lp)
    out["dag_longest_chain_edges"] = max(len(lp) - 1, 0)
    out["dag_n_source_comps"] = sum(1 for _, d in D.in_degree() if d == 0)
    out["dag_n_sink_comps"] = sum(1 for _, d in D.out_degree() if d == 0)
    out["dag_n_isolated_comps"] = sum(1 for c in D.nodes()
                                      if D.in_degree(c) == 0 and D.out_degree(c) == 0)

    # --- 互惠性 -----------------------------------------------------------
    out["reciprocity"] = round(float(nx.reciprocity(G)), 4) if m else np.nan
    rp = sum(1 for u, v in G.edges() if G.has_edge(v, u))
    out["n_edges_in_reciprocal_pair"] = rp
    out["n_reciprocal_pairs"] = rp // 2
    out["n_one_sided_edges"] = m - rp
    out["one_sided_edge_share"] = round((m - rp) / m, 4) if m else np.nan

    # --- J3 不对称度（图内重算）-------------------------------------------
    ttot = np.array([ind[x] + outd[x] for x in G.nodes()], dtype=float)
    ai = np.array([abs(ind[x] - outd[x]) for x in G.nodes()], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        asym = np.where(ttot > 0, ai / np.where(ttot > 0, ttot, 1.0), np.nan)
    out["n_nodes_with_inout_gt0"] = int(np.nansum(ttot > 0))
    out["n_nodes_isolated"] = int(np.nansum(ttot == 0))
    out["asym_mean"] = round(float(np.nanmean(asym)), 4)
    out["asym_median"] = round(float(np.nanmedian(asym)), 4)
    out["asym_p25"] = round(float(np.nanpercentile(asym, 25)), 4)
    out["asym_p75"] = round(float(np.nanpercentile(asym, 75)), 4)
    out["asym_eq_0"] = int(np.nansum(asym == 0.0))
    out["asym_le_0p2"] = int(np.nansum(asym <= 0.2))
    out["asym_ge_0p8"] = int(np.nansum(asym >= 0.8))
    out["asym_eq_1"] = int(np.nansum(asym == 1.0))
    out["sum_in_degree"] = int(sum(ind.values()))
    out["sum_out_degree"] = int(sum(outd.values()))
    out["mean_in_degree"] = round(float(np.mean(list(ind.values()))), 4)
    out["mean_out_degree"] = round(float(np.mean(list(outd.values()))), 4)

    # --- sink / source ----------------------------------------------------
    s0 = [x for x in G.nodes() if outd[x] == 0]
    i0 = [x for x in G.nodes() if ind[x] == 0]
    out["n_sink_out0"] = len(s0)
    out["sink_share"] = round(len(s0) / n, 4)
    out["n_source_in0"] = len(i0)
    out["source_share"] = round(len(i0) / n, 4)
    out["n_both0"] = len(set(s0) & set(i0))
    out["sink_in_degree_sum"] = int(sum(ind[x] for x in s0))
    out["source_out_degree_sum"] = int(sum(outd[x] for x in i0))
    out["_in0_nodes"] = sorted(i0)

    # --- 可达对密度："进得去出不来"的直接量化 ------------------------------
    rc = sum(len(_fwd(G, [x])) - 1 for x in G.nodes())
    out["reachable_ordered_pairs"] = rc
    out["all_ordered_pairs"] = n * (n - 1)
    out["reach_density"] = round(rc / (n * (n - 1)), 4) if n > 1 else np.nan

    # --- J2 bow-tie -------------------------------------------------------
    bt = bowtie(G)
    for g in BOWTIE_GROUPS:
        out[f"bowtie_{g}"] = len(bt["groups"][g])
        out[f"bowtie_{g}_share"] = round(len(bt["groups"][g]) / n, 4)
    out["bowtie_TUBE_vacuous_shortcut"] = bt["n_tube_vacuous_upcapdown"]
    out["bowtie_OUT-COMPONENTS_naive_forward"] = bt["n_outcomp_naive_forward"]
    out["bowtie_tendril_overlap_assigned_to_TUBE"] = bt["n_in_out_tendril_overlap"]
    out["bowtie_core_tied_candidates"] = bt["n_core_tied"]
    out["_bt"] = bt
    return out


# ====================================================================== main

def main() -> None:
    log(f"versions {C.versions()}")
    b = C.build_all()
    C.assert_anchors(b)
    log("C.assert_anchors(b) -> 31/31 PASS")

    Gd: nx.DiGraph = b["Gd"]
    nodes: pd.DataFrame = b["nodes"].copy()

    # 建图与锚点之外的两条本地自检：属性表的度必须与图一致；死站出度必须全 0
    assert nodes["out_degree"].tolist() == [Gd.out_degree(x) for x in nodes["node"]]
    assert nodes["in_degree"].tolist() == [Gd.in_degree(x) for x in nodes["node"]]
    dead = set(nodes.loc[nodes["is_dead"], "node"])
    assert sum(1 for s, _ in Gd.edges() if s in dead) == C.ANCHORS["edges_from_dead"]
    log(f"Gd = {Gd.number_of_nodes()} 节点 / {Gd.number_of_edges()} 边；"
        f"dead={len(dead)} 且其中出度=0 的有 {int(nodes.loc[nodes.is_dead, 'out_degree'].eq(0).sum())} 个；"
        f"以死站为起点的边 = 0 已独立复算")

    for s in _selftest_bowtie():
        log(s)

    # 五套节点口径。CONVENTIONS §2.2 要求"校正前/校正后"两套 -> C 列即校正后。
    scopes = {
        "full_676": Gd.copy(),
        "attributed_671": C.induced(Gd, sorted(nodes["node"])),
        "ex_false_fail_639": C.induced(Gd, sorted(nodes.loc[~nodes["false_fail"], "node"])),
        "alive_no_dead_509": C.induced(Gd, sorted(nodes.loc[~nodes["is_dead"], "node"])),
        "healthy_only_360": C.induced(Gd, sorted(nodes.loc[nodes["is_healthy"], "node"])),
    }
    order = list(scopes)
    res = {}
    for k, G in scopes.items():
        res[k] = directed_metrics(G)
        r = res[k]
        log(f"{k:20s} n={r['n_nodes']:4d} m={r['n_edges']:5d} recip={r['reciprocity']} "
            f"core={r['scc_largest']:4d} sink={r['n_sink_out0']:4d}({r['sink_share']*100:5.1f}%) "
            f"src={r['n_source_in0']:2d} wcc={r['n_wcc']} layers={r['dag_n_layers']}")

    cc = {}
    for k, G in scopes.items():
        cc[k] = bowtie_crosscheck(G, res[k]["_bt"])
        log(f"bowtie cross-check {k:20s} 分量数={cc[k]['n_dag_components']:4d} "
            f"节点级vs闭包级不一致={cc[k]['n_mismatch']} core一致={cc[k]['core_identical']}")
        assert cc[k]["n_mismatch"] == 0, f"{k} bow-tie 两套实现不一致: {cc[k]['mismatch_examples']}"
        assert cc[k]["core_identical"]

    # ------------------------------------------------ bowtie_groups.csv
    rows: dict[str, dict] = {}
    for k in order:
        for gname, members in res[k]["_bt"]["groups"].items():
            for v in members:
                rows.setdefault(v, {})[f"group_{k}"] = gname
    bg = pd.DataFrame.from_dict(rows, orient="index")
    bg.index.name = "node"
    bg = bg.reset_index()
    at = nodes.set_index("node")
    for col in ["sub", "health_state", "is_dead", "is_partial", "is_healthy", "dead_cat",
                "true_fail", "false_fail", "url_count", "listed_urls", "ok",
                "in_strength", "out_strength"]:
        bg[col] = bg["node"].map(at[col])
    bg["in_graph_671"] = bg["node"].isin(set(nodes["node"]))
    bg["in_degree_full676"] = bg["node"].map(dict(Gd.in_degree()))
    bg["out_degree_full676"] = bg["node"].map(dict(Gd.out_degree()))
    Gh = scopes["healthy_only_360"]
    bg["in_degree_healthy360"] = bg["node"].map(dict(Gh.in_degree()))
    bg["out_degree_healthy360"] = bg["node"].map(dict(Gh.out_degree()))
    for tag in ["full676", "healthy360"]:
        i, o = bg[f"in_degree_{tag}"], bg[f"out_degree_{tag}"]
        bg[f"asym_{tag}"] = (i - o).abs() / (i + o).replace(0, np.nan)
    bg["unobserved_pages"] = (bg["url_count"] - bg["listed_urls"]).clip(lower=0)
    bg["page_truncated"] = bg["url_count"] > bg["listed_urls"]
    scc_size = {v: len(c) for c in nx.strongly_connected_components(Gd) for v in c}
    bg["scc_size_full676"] = bg["node"].map(scc_size)
    bg = bg.sort_values("node").reset_index(drop=True)
    p1 = C.save_csv(bg, MODULE, "bowtie_groups.csv")
    log(f"written {p1}  ({len(bg)} 节点 × 5 套分组标签 + 属性)")

    # ------------------------------------------------ 分组 × 健康列联表
    ct = pd.crosstab(bg["group_full_676"], bg["health_state"].fillna("dangling_ref"))
    ct = ct.reindex(index=BOWTIE_GROUPS, fill_value=0)
    for c in ["dead", "partial", "healthy", "dangling_ref"]:
        if c not in ct.columns:
            ct[c] = 0
    ct = ct[["dead", "partial", "healthy", "dangling_ref"]]
    ct["row_total"] = ct.sum(axis=1)
    ct.index.name = "group_full_676"
    ct = ct.reset_index()
    p1b = C.save_csv(ct, MODULE, "bowtie_by_health.csv")
    log(f"written {p1b}")

    # ------------------------------------------------ directed_metrics.csv
    names = [k for k in res["full_676"] if not k.startswith("_")]
    labels = {"full_676": "A_full676_anchor", "attributed_671": "B_attributed671_main",
              "ex_false_fail_639": "C_exFalseFail639_corrected",
              "alive_no_dead_509": "D_aliveNoDead509",
              "healthy_only_360": "E_healthyOnly360_censoringControl"}
    mt = pd.DataFrame({labels[k]: {nm: res[k][nm] for nm in names} for k in order})
    mt.insert(0, "metric", names)
    p2 = C.save_csv(mt, MODULE, "directed_metrics.csv")
    log(f"written {p2}  ({len(names)} 指标 × {len(order)} 口径)")

    # ------------------------------------------------ asymmetry_deciles.csv（pd.qcut）
    ab = []
    for k in order:
        G = scopes[k]
        i = np.array([d for _, d in G.in_degree()], dtype=float)
        o = np.array([d for _, d in G.out_degree()], dtype=float)
        keep = (i + o) > 0
        a = pd.Series(np.abs(i[keep] - o[keep]) / (i[keep] + o[keep]))
        bins = pd.qcut(a, 10, duplicates="drop")
        g = a.groupby(bins, observed=True).agg(n="size", mu="mean")
        for iv, r in g.iterrows():
            ab.append({"scope": labels[k], "asym_decile_bin": str(iv), "n_nodes": int(r["n"]),
                       "share_of_nodes": round(r["n"] / len(a), 4),
                       "mean_asym_in_bin": round(r["mu"], 4)})
    abs_df = pd.DataFrame(ab)
    p3 = C.save_csv(abs_df, MODULE, "asymmetry_deciles.csv")
    log(f"written {p3}")

    # ------------------------------------------------ zero_degree_nodes.csv（J3 清单）
    z = []
    for _, r in bg.iterrows():
        kinds = []
        if r["out_degree_full676"] == 0:
            kinds.append("out0_sink")
        if r["in_degree_full676"] == 0:
            kinds.append("in0_source")
        if not kinds:
            continue
        z.append({"node": r["node"], "kind": "+".join(kinds),
                  "in_degree_full676": r["in_degree_full676"],
                  "out_degree_full676": r["out_degree_full676"],
                  "out_degree_healthy360": r["out_degree_healthy360"],
                  "is_sink_in_healthy360": r["out_degree_healthy360"] == 0,
                  "health_state": r["health_state"] if pd.notna(r["health_state"]) else "dangling_ref",
                  "dead_cat": r["dead_cat"], "true_fail": r["true_fail"], "false_fail": r["false_fail"],
                  "group_full_676": r["group_full_676"],
                  "group_healthy_only_360": r.get("group_healthy_only_360"),
                  "url_count": r["url_count"], "listed_urls": r["listed_urls"], "ok": r["ok"],
                  "unobserved_pages": r["unobserved_pages"], "page_truncated": r["page_truncated"],
                  "out_degree_fully_censored_by_deadness": bool(r["is_dead"]) if pd.notna(r["is_dead"]) else False,
                  "sink_status_verifiable_from_pages": (bool(r["page_truncated"]) is False)
                  and (pd.notna(r["is_dead"]) and not r["is_dead"])})
    zd = pd.DataFrame(z).sort_values(["kind", "node"]).reset_index(drop=True)
    p4 = C.save_csv(zd, MODULE, "zero_degree_nodes.csv")
    log(f"written {p4}  ({len(zd)} 行: sink {(zd.kind == 'out0_sink').sum()} / "
        f"source {(zd.kind == 'in0_source').sum()})")

    # ------------------------------------------------ J5 / J4 证据表
    trunc = truncation_estimate(nodes, scopes, bg, res)
    p5 = C.save_csv(pd.DataFrame(trunc["rows"]), MODULE, "truncation_estimate.csv")
    log(f"written {p5}")
    cen = censoring_evidence(nodes, Gd, res, ct)
    p6 = C.save_csv(pd.DataFrame(cen["rows"]), MODULE, "censoring_evidence.csv")
    log(f"written {p6}")

    write_result(res, scopes, nodes, bg, ct, abs_df, zd, trunc, cen, cc, order, labels)
    log("written result.md")

    C.write_run_note(MODULE, [
        "A4_directed_structure run at pipeline/",
        "command: python out/A4_directed_structure/run_a4.py   (cwd = pipeline/)",
        f"versions: {C.versions()}",
        f"SEED = {SEED}（本单元全部确定性计算：SCC / 可达集 / 缩点 / qcut；无随机过程）",
        "C.assert_anchors(b) -> 31/31 PASS（含 reciprocity_gd=0.2081, edges_from_dead=0, "
        "dead_with_outdegree_zero=162, comp_sizes=[674,2]）",
        f"total elapsed: {time.time()-T0:.2f}s",
        "-" * 72,
        *LOG,
        "-" * 72,
        "outputs: run_a4.py bowtie_groups.csv directed_metrics.csv asymmetry_deciles.csv "
        "zero_degree_nodes.csv truncation_estimate.csv censoring_evidence.csv "
        "bowtie_by_health.csv result.md",
        "DISSENT.md: 未写 —— 本单元未发现与 CONVENTIONS 冲突的实测证据（见 result.md 末段）",
    ])


# ============================================================= J5 截断量级估计

def truncation_estimate(nodes: pd.DataFrame, scopes, bg: pd.DataFrame, res) -> dict:
    """用 url_count（抓取器报告的站点 URL 总数）与 listed_urls（实际落库明细，上限 20）
    之差估计出度截断量级，并给出对 sink 比例的高估幅度。

    三条通道，只有第一条能用现有文件量化：
      (1) 落库 20 条上限：url_count > listed_urls 的站点确有未落库页面 -> 可给
          "多少 sink 身份可被翻案"的上界，与"缺多少出链次数"的量级。
      (2) 抽取范围：一条边要求 href 指向另一 ustc.edu.cn 子域；站外链接、同子域内部
          链接、JS 渲染后的链接都不进 links.csv -> 无法从 urls.csv 4 列区分。无法量化。
      (3) url_count 自身可能也有每站预算 -> 无法量化（给出 99-105 堆积等证据）。
    """
    rows = []

    def add(k, v, note=""):
        rows.append({"item": k, "value": v, "note": note})

    h = nodes[nodes["is_healthy"]]
    trunc_mask = nodes["url_count"] > nodes["listed_urls"]
    add("cap_on_listed_urls", 20, "落库明细上限（DATA_DICTIONARY）")
    add("max_listed_urls_observed", int(nodes["listed_urls"].max()))
    add("n_sites_listed_at_cap", int((nodes["listed_urls"] == 20).sum()))
    add("n_sites_page_truncated", int(trunc_mask.sum()), "url_count>listed_urls，确有未落库页面")
    add("share_sites_page_truncated", round(float(trunc_mask.mean()), 4))
    add("n_sites_fully_enumerated", int((nodes["url_count"] == nodes["listed_urls"]).sum()),
        "url_count<=20，页面明细已穷尽")
    add("sum_url_count", int(nodes["url_count"].sum()), "抓取器报告的页面总数")
    add("sum_listed_urls", int(nodes["listed_urls"].sum()), "落库明细数（= n_urls 锚 8115）")
    add("page_coverage_share", round(float(nodes["listed_urls"].sum() / nodes["url_count"].sum()), 4),
        "页面级覆盖率 = Σlisted/Σurl_count")
    add("n_unobserved_pages_total", int((nodes["url_count"] - nodes["listed_urls"]).clip(lower=0).sum()))

    add("healthy_n_sites", int(len(h)))
    add("healthy_n_page_truncated", int((h["url_count"] > h["listed_urls"]).sum()))
    add("healthy_share_page_truncated", round(float((h["url_count"] > h["listed_urls"]).mean()), 4))
    add("healthy_sum_url_count", int(h["url_count"].sum()))
    add("healthy_sum_listed_urls", int(h["listed_urls"].sum()))
    add("healthy_page_coverage_share", round(float(h["listed_urls"].sum() / h["url_count"].sum()), 4))
    add("healthy_unobserved_pages", int((h["url_count"] - h["listed_urls"]).clip(lower=0).sum()))
    add("healthy_observed_out_hrefs", int(h["out_strength"].sum()), "已观测跨子域出链次数（=出强度之和）")
    add("healthy_observed_out_edges", int(h["out_degree"].sum()), "已观测不同出目标数之和")
    rate_pool = float(h["out_strength"].sum() / h["ok"].sum())
    per = (h["out_strength"] / h["ok"].replace(0, np.nan))
    add("href_per_ok_page_pooled_healthy", round(rate_pool, 4),
        "合并口径；模板重复会让它偏高（同一模板 href 在多页重复计入 weight）")
    add("href_per_ok_page_median_healthy", round(float(per.median()), 4))
    add("href_per_ok_page_p25_healthy", round(float(per.quantile(.25)), 4))
    add("href_per_ok_page_p75_healthy", round(float(per.quantile(.75)), 4))

    # (1a) 缺失出链「次数」量级
    add("missing_out_hrefs_pooled_upper", round(rate_pool * float(
        (h["url_count"] - h["listed_urls"]).clip(lower=0).sum()), 0),
        "上界：合并每页出链率 × 未落库页面数，完全忽略模板重复 -> 只作量级")
    add("missing_out_hrefs_median_rate", round(float(per.median()) * float(
        (h["url_count"] - h["listed_urls"]).clip(lower=0).sum()), 0), "保守：中位每页出链率外推")

    # (1b) 出度「不同目标数」的饱和标度：全枚举健康站点是天然对照组
    fe = h[h["url_count"] == h["listed_urls"]]
    add("fully_enumerated_healthy_n", int(len(fe)), "url_count==listed_urls<=20 的健康站点（未被截断对照组）")
    fs = fe[(fe["url_count"] >= 2) & (fe["out_degree"] >= 1)]
    lr = ss.linregress(np.log(fs["url_count"].astype(float)), np.log(fs["out_degree"].astype(float)))
    beta = float(lr.slope)
    add("outdeg_vs_pages_beta", round(beta, 4),
        f"全枚举健康站点内 log(out_degree)~log(url_count) 斜率；n={len(fs)}，"
        f"r={round(float(lr.rvalue), 3)}，p={lr.pvalue:.2e}")
    add("outdeg_vs_pages_beta_ci95",
        f"[{beta - 1.96 * lr.stderr:.3f},{beta + 1.96 * lr.stderr:.3f}]",
        "β<1 -> 出度随页面数亚线性饱和；观测范围 url_count∈[2,20]，用于外推有风险")
    tr_h = h[h["url_count"] > h["listed_urls"]]
    ratio = (tr_h["url_count"] / tr_h["listed_urls"]).astype(float)
    add("trunc_healthy_ratio_median", round(float(ratio.median()), 3))
    add("projection_factor_beta_model_at_ratio5", round(5 ** beta, 3),
        "url_count=100 只落库 20（比值 5）的站点，出度按 β 模型的放大倍数")
    add("projection_factor_linear_model_at_ratio5", 5.0, "β=1 线性模型给 5×（上界）")
    add("trunc_healthy_outdeg_observed", int(tr_h["out_degree"].sum()))
    add("trunc_healthy_outdeg_projected_beta",
        round(float((tr_h["out_degree"] * ratio ** beta).sum()), 1), "β 模型投影到全页面覆盖")
    add("trunc_healthy_outdeg_projected_linear",
        round(float((tr_h["out_degree"] * ratio).sum()), 1), "线性模型投影（上界）")
    add("net_directed_edges_now", int(res["full_676"]["n_edges"]))
    add("net_directed_edges_projected_beta", int(round(
        res["full_676"]["n_edges"] - tr_h["out_degree"].sum()
        + (tr_h["out_degree"] * ratio ** beta).sum())),
        "只放大被截断健康站点的出度、其余不动；不含死站（其出边完全不可观测）")
    add("net_directed_edges_projected_linear_upper", int(round(
        res["full_676"]["n_edges"] - tr_h["out_degree"].sum() + (tr_h["out_degree"] * ratio).sum())),
        "线性上界")

    # (1c) 对 sink 比例的高估幅度
    bh = bg[bg["in_graph_671"]]
    for tag in ["full676", "healthy360"]:
        s = bh[bh[f"out_degree_{tag}"] == 0]
        for st in ["dead", "partial", "healthy"]:
            g = s[s["health_state"] == st]
            add(f"sink_{tag}_out0_{st}", int(len(g)))
            add(f"sink_{tag}_out0_{st}_page_truncated", int(g["page_truncated"].sum()),
                "该组中确有未落库页面、sink 身份可被 20 条上限翻案的最大数目")
        add(f"sink_{tag}_total_attributed", int(len(s)))
    hs_f = bh[(bh["out_degree_full676"] == 0) & (bh["health_state"] == "healthy")]
    hs_h = bh[(bh["out_degree_healthy360"] == 0) & (bh["health_state"] == "healthy")]
    add("overestimate_healthy_sink_pp_max",
        round(100.0 * float(hs_h["page_truncated"].sum()) / len(h), 2),
        "以 360 子图 sink 率为基准：20 条上限至多抬高这么多（假设全部翻案）")
    add("healthy_sink_rate_360subgraph_now", res["healthy_only_360"]["sink_share"])
    add("healthy_sink_rate_360subgraph_if_all_flip",
        round((res["healthy_only_360"]["n_sink_out0"] - int(hs_h["page_truncated"].sum())) / len(h), 4),
        "下界：把全部有未落库页面的 360-sink 都算成非 sink")
    snk_f = bh[bh["out_degree_full676"] == 0]
    add("sink_full676_page_truncated_alive_only",
        int(snk_f[snk_f["page_truncated"] & (snk_f["health_state"] != "dead")].shape[0]),
        "出度=0、确有未落库页面、且不是整站不可用（截断这条通道对它另有主导解释）的点")
    add("sink_full676_page_truncated_all", int(snk_f["page_truncated"].sum()))
    add("overestimate_full676_sink_pp_max",
        round(100.0 * float(snk_f["page_truncated"].sum()) / 676, 2),
        "只算 20 条上限这条通道；一阶通道是 162 点完全删失，见 censoring_evidence.csv")
    add("flippable_360_sinks_n", int(len(hs_h)), "360 子图里 sink 且确有未落库页面的健康站点数")
    add("flippable_360_sinks_observed_out_hrefs", int(hs_h["out_strength"].sum()),
        "这些站 20 条已落库页面里的跨子域出链次数")
    add("flippable_360_sinks_missing_hrefs_95ub",
        round(float((3.0 / hs_h["ok"].replace(0, np.nan) * hs_h["unobserved_pages"]).sum()), 1),
        "rule of three：0 事件 / n 页的 95% 上界率 3/n，乘未落库页面数；是 15 站的合计")
    add("healthy_sinks_url_count_eq_1", int(((bh["out_degree_healthy360"] == 0)
                                             & (bh["health_state"] == "healthy")
                                             & (bh["url_count"] == 1)).sum()),
        "整站只报 1 页：在口径上算未截断，但真实页面存量不可核 -> 通道 (3)")

    # (3) url_count 自身可疑证据
    add("sites_url_count_in_99_105", int(((nodes["url_count"] >= 99) & (nodes["url_count"] <= 105)).sum()),
        "堆积提示抓取器有每站页面预算 -> 通道 (3) 无法量化")
    add("sites_url_count_eq_100", int((nodes["url_count"] == 100).sum()))
    add("sites_url_count_eq_1", int((nodes["url_count"] == 1).sum()))
    add("max_url_count", int(nodes["url_count"].max()))
    add("urls_csv_has_page_level_hrefs", 0, "urls.csv 仅 subdomain/status/url/title 4 列 -> 通道 (2) 无法量化")
    return {"rows": rows, "beta": beta}


# ========================================================= J4 删失证据与对照

def censoring_evidence(nodes: pd.DataFrame, Gd: nx.DiGraph, res, ct: pd.DataFrame) -> dict:
    dead = set(nodes.loc[nodes["is_dead"], "node"])
    rows = []

    def add(k, v, note=""):
        rows.append({"item": k, "value": v, "note": note})

    add("edges_total", Gd.number_of_edges())
    add("edges_from_dead", sum(1 for s, _ in Gd.edges() if s in dead), "锚点 0：出边完全不可观测")
    add("edges_dead_to_dead", sum(1 for s, t in Gd.edges() if s in dead and t in dead),
        "构造性不可识别，不是「测出不聚集」")
    add("weight_from_dead", int(sum(d.get("weight", 0) for s, _, d in Gd.edges(data=True) if s in dead)),
        "死站出边的权重合计也是 0")
    add("edges_to_dead_from_alive", sum(1 for s, t in Gd.edges() if t in dead and s not in dead),
        "入边可观测 -> 失效分析主口径=入度")
    add("dead_n", len(dead))
    add("dead_out0_n", int(nodes.loc[nodes["is_dead"], "out_degree"].eq(0).sum()))
    add("dead_out0_share", round(float(nodes.loc[nodes["is_dead"], "out_degree"].eq(0).mean()), 4))
    add("true_fail_n", int(nodes["true_fail"].sum()), "A_conn_fail+D_not_found+F_server_error")
    add("false_fail_n", int(nodes["false_fail"].sum()), "B_auth_or_skip+C_robots_blocked")
    add("ambiguous_n", int(nodes["ambiguous_fail"].sum()), "G_other_400，待人工判断")
    snk = nodes[nodes["out_degree"] == 0]
    add("out0_attributed_total", int(len(snk)))
    for st in ["dead", "partial", "healthy"]:
        add(f"out0_{st}", int((snk["health_state"] == st).sum()))
        add(f"out0_{st}_share_within_class",
            round(float((snk["health_state"] == st).sum() / max((nodes["health_state"] == st).sum(), 1)), 4))
    add("out0_dangling_ref", int(res["full_676"]["n_sink_out0"] - len(snk)),
        "5 个悬空引用点无 url_count，截断与否不可判")
    add("out0_dead_share_of_all_out0", round(float((snk["health_state"] == "dead").mean()), 4),
        "「出度=0」这个读数里有多大比例来自整站不可用")
    add("sink_share_full676", res["full_676"]["sink_share"])
    add("sink_share_healthy360", res["healthy_only_360"]["sink_share"])
    add("sink_share_drop_pp", round(100 * (res["full_676"]["sink_share"]
                                           - res["healthy_only_360"]["sink_share"]), 2))
    add("edges_kept_healthy360", res["healthy_only_360"]["n_edges"])
    add("edge_kept_share", round(res["healthy_only_360"]["n_edges"] / Gd.number_of_edges(), 4))
    add("weight_kept_healthy360", res["healthy_only_360"]["sum_edge_weight"])
    add("weight_kept_share", round(res["healthy_only_360"]["sum_edge_weight"]
                                   / Gd.size(weight="weight"), 4))
    add("node_kept_share", round(res["healthy_only_360"]["n_nodes"] / 676, 4))
    ct2 = ct.set_index("group_full_676")
    for g in BOWTIE_GROUPS:
        if int(ct2.loc[g, "dead"]) > 0:
            add(f"dead_in_bowtie_{g}", int(ct2.loc[g, "dead"]))
    add("partial_n", int((nodes["health_state"] == "partial").sum()))
    add("partial_out0", int((snk["health_state"] == "partial").sum()), "部分失效点也有不可观测页面")
    return {"rows": rows}


# ================================================================= result.md

def write_result(res, scopes, nodes, bg, ct, abs_df, zd, trunc, cen, cc, order, labels) -> None:
    R = {labels[k]: res[k] for k in order}
    f = R["A_full676_anchor"]
    a = R["B_attributed671_main"]
    cx = R["C_exFalseFail639_corrected"]
    dv = R["D_aliveNoDead509"]
    hm = R["E_healthyOnly360_censoringControl"]
    T = {r["item"]: r["value"] for r in trunc["rows"]}
    E = {r["item"]: r["value"] for r in cen["rows"]}
    ct2 = ct.set_index("group_full_676")

    def pc(x):
        return f"{100 * x:.2f}%"

    def rt(g, s):
        return f"{g / max(s, 1):.0f}:1" if s >= 1 else f"{g}/0"

    md: list[str] = []
    w = md.append

    w(f"# A4 · 有向结构与{pq('进得去出不来')}")
    w("")
    w(f"- 建图 / 边序 / 健康分层 / 失效校正全部来自 `csa_common.build_all()`"
      f"（读 `pipeline/data/clean_*.csv`）。开跑 `C.assert_anchors(b)` → **31/31 PASS**；"
      f"本单元独立复算 `reciprocity = {f['reciprocity']}`（锚点 0.2081）、"
      f"`edges_from_dead = {E['edges_from_dead']}`（锚点 0）、"
      f"`dead_out0 = {E['dead_out0_n']}/162`（锚点 162）。")
    w(f"- 主图：**{f['n_nodes']} 节点 / {f['n_edges']} 条有向边 / 权重合计 "
      f"{f['sum_edge_weight']}**，自环 {f['n_selfloops']}。"
      f"权重一律读作{pq('观察到的链接次数')}，不解释为流量或重要性（CONVENTIONS §1）。")
    w(f"- 全部为确定性计算（SCC / 可达集 / 缩点 / `pd.qcut` / 一元回归），无随机过程；"
      f"`SEED = {SEED}` 仅按约定留痕。所有数字可用 `run_a4.py` 一次重跑复现，"
      f"并可在 `directed_metrics.csv` / `censoring_evidence.csv` / `truncation_estimate.csv` "
      f"里按行名 grep 到。")
    w(f"- 落盘：`bowtie_groups.csv`（{len(bg)} 节点 × 5 套分组标签 + 属性）、"
      f"`directed_metrics.csv`（{len([k for k in f if not k.startswith('_')])} 指标 × 5 口径）、"
      f"`asymmetry_deciles.csv`、`zero_degree_nodes.csv`（{len(zd)} 行清单）、"
      f"`truncation_estimate.csv`、`censoring_evidence.csv`、`bowtie_by_health.csv`。")
    w("")
    w(f"> 措辞约束（CONVENTIONS §7）：本文件不出现因果动词。"
      f"{pq('出度被删失')}{pq('出度被截断')}是**测量边界**陈述，不是机制陈述。")
    w("")

    # ---------------------------------------------------------------- J1
    w("## J1 · SCC 规模谱 / 弱分量 / 缩点 DAG / 互惠性")
    w("")
    w("| 指标 | 全图 676（锚点口径） | 主分析 671 |")
    w("|---|---:|---:|")
    for k, lab in [("n_nodes", "节点数"), ("n_edges", "有向边数"),
                   ("reciprocity", "reciprocity = `nx.reciprocity(Gd)`"),
                   ("n_reciprocal_pairs", "互惠对数"),
                   ("n_edges_in_reciprocal_pair", "处在互惠对中的边数"),
                   ("n_one_sided_edges", "单边边数"),
                   ("one_sided_edge_share", "单边占比"),
                   ("n_scc", "SCC 个数"),
                   ("scc_largest", "最大 SCC 规模"),
                   ("scc_largest_share", "最大 SCC 占节点集比"),
                   ("n_scc_nonsingleton", "非单点 SCC 个数"),
                   ("n_scc_singleton", "单点 SCC 个数"),
                   ("n_nodes_in_nonsingleton_scc", "落在 >1 点 SCC 内的点数"),
                   ("n_wcc", "弱连通分量个数"),
                   ("wcc_sizes_top5", "弱分量规模 Top5"),
                   ("n_dag_nodes", "缩点 DAG 节点数"),
                   ("n_dag_edges", "缩点 DAG 弧数"),
                   ("dag_n_layers", "DAG 拓扑层数"),
                   ("dag_layer_sizes", "各层分量数"),
                   ("dag_layer_of_core", "CORE 所在层（0 起）"),
                   ("dag_longest_chain_nodes", "最长链（分量数）"),
                   ("dag_longest_chain_edges", "最长链（弧数）"),
                   ("dag_n_source_comps", "DAG 源分量（入度 0）"),
                   ("dag_n_sink_comps", "DAG 汇分量（出度 0）"),
                   ("dag_n_isolated_comps", "DAG 孤立分量数"),
                   ("reachable_ordered_pairs", "可达有序点对数"),
                   ("all_ordered_pairs", "全部有序点对数"),
                   ("reach_density", "可达对密度")]:
        w(f"| {lab} | {f[k]} | {a[k]} |")
    w("")
    w(f"- **SCC 规模谱（676）**：`{f['scc_size_hist']}` —— 1 个 301 点大核 + 1 个 4 点 + "
      f"4 个 2 点 + {f['n_scc_singleton']} 个单点。非单点 SCC 只有 "
      f"**{f['n_scc_nonsingleton']} 个**，覆盖 {f['n_nodes_in_nonsingleton_scc']} 点 = "
      f"{pc(f['n_nodes_in_nonsingleton_scc'] / f['n_nodes'])}；"
      f"其余 {pc(f['n_scc_singleton'] / f['n_nodes'])} 的点自成单点 SCC。")
    w(f"- **弱分量 {f['n_wcc']} 个**，规模 `{f['wcc_sizes_top5']}` —— 与 "
      f"`n_components=2 / comp_sizes=[674,2]` 两个锚点一致。")
    w(f"- **缩点 DAG**：{f['n_dag_nodes']} 个分量 / {f['n_dag_edges']} 条弧 / "
      f"拓扑层数 **{f['dag_n_layers']}**（各层 `{f['dag_layer_sizes']}`）/ "
      f"最长链 **{f['dag_longest_chain_nodes']} 个分量、{f['dag_longest_chain_edges']} 条弧** / "
      f"**孤立 SCC 数 = {f['dag_n_isolated_comps']}**。CORE 位于第 {f['dag_layer_of_core']} 层。")
    w(f"- DAG 的汇分量（出度 0 的 SCC）有 **{f['dag_n_sink_comps']}/{f['n_dag_nodes']} = "
      f"{pc(f['dag_n_sink_comps'] / f['n_dag_nodes'])}**，源分量只有 {f['dag_n_source_comps']} 个。"
      f"缩点后的形状是一个大核挂着大量只进不出的末端分量，不是多级瀑布。")
    w(f"- **互惠性 {f['reciprocity']}**（671 口径 {a['reciprocity']}）：全网 "
      f"{f['n_reciprocal_pairs']} 对互惠、{f['n_one_sided_edges']} 条"
      f"（{pc(f['one_sided_edge_share'])}）是单边。")
    w(f"- **可达对密度 {f['reach_density']}**：{f['reachable_ordered_pairs']} / "
      f"{f['all_ordered_pairs']} 个有序点对在有向意义下走通，"
      f"**{(1 - f['reach_density']):.1%} 的点对之间不连通** —— "
      f"而弱分量口径下 99.7% 的点在同一个分量里。这就是{pq('进得去出不来')}的定量表述。")
    w("")

    # ---------------------------------------------------------------- J2
    w("## J2 · Bow-tie 分组（Broder et al. 2000, Computer Networks 33:309）")
    w("")
    w(f"**实现自写**于 `run_a4.py::bowtie`（networkx 无 bow-tie 原语，"
      f"3.6.1 只有 `dominance_frontiers`/`quotient_graph`），7 组定义与抢占顺序写在 docstring 里。"
      f"三重可核对性：")
    w(f"1. **不变量 assert**：7 组两两不交、并集 = 全节点集（函数内直接断言）；")
    w(f"2. **第二套代码路径**：`run_a4.py::bowtie_crosscheck` 自建缩点图并用 "
      f"`nx.transitive_closure` 求闭包，在**分量级**判组后展开到节点，"
      f"与节点级 DFS 逐点比 —— 5 套口径下不一致数均为 "
      f"**{sum(cc[k]['n_mismatch'] for k in cc)}**，CORE 集合一致 = "
      f"{all(cc[k]['core_identical'] for k in cc)}；")
    w(f"3. **手工最小例**：`run_a4.py::_selftest_bowtie` 里 4 个构造图 —— 例 2 一个图就让全部 "
      f"7 组非空（CORE/IN/OUT/TUBE/IN-COMPONENTS/OUT-COMPONENTS/DISCONNECTED），"
      f"例 3 是「核外孤立 2 点弱分量」即本项目 `bm→l.bm` 的形态，"
      f"例 4 断言 OUT-COMPONENTS 取正向可达时必恒空。"
      f"期望值全部人工推出，运行时逐个 assert，并各自再过一遍 cross-check。")
    w("")
    w("| 组 | 全图 676 | 占比 | 仅健康 360 | 占比 | 676 中死站数 |")
    w("|---|---:|---:|---:|---:|---:|")
    for g in BOWTIE_GROUPS:
        w(f"| {g} | {f[f'bowtie_{g}']} | {pc(f[f'bowtie_{g}_share'])} | "
          f"{hm[f'bowtie_{g}']} | {pc(hm[f'bowtie_{g}_share'])} | {int(ct2.loc[g, 'dead'])} |")
    w(f"| **合计** | **{sum(f[f'bowtie_{g}'] for g in BOWTIE_GROUPS)}** | 100% | "
      f"**{sum(hm[f'bowtie_{g}'] for g in BOWTIE_GROUPS)}** | 100% | {E['dead_n']} |")
    w("")
    w(f"- CORE = 最大 SCC = **{f['bowtie_CORE']} 点（{pc(f['bowtie_CORE_share'])}）**。"
      f"OUT 侧 **{f['bowtie_OUT']} 点（{pc(f['bowtie_OUT_share'])}）** 对 "
      f"IN 侧 **{f['bowtie_IN']} 点（{pc(f['bowtie_IN_share'])}）**，"
      f"**OUT/IN = {f['bowtie_OUT']}/{f['bowtie_IN']} = {rt(f['bowtie_OUT'], f['bowtie_IN'])}**。"
      f"网络是「一个大核 + 一圈几乎只出不进的下游裙摆」，而不是对称的沙漏。")
    w(f"- **TUBE = {f['bowtie_TUBE']}**（口径：原文的「在 IN→OUT 的有向通路上」，"
      f"即未归组、既可自 IN 到达又能到 OUT）。"
      f"IN-COMPONENTS = {f['bowtie_IN-COMPONENTS']}、OUT-COMPONENTS = "
      f"{f['bowtie_OUT-COMPONENTS']}、DISCONNECTED = {f['bowtie_DISCONNECTED']}。")
    w(f"- **两个可证明的恒空陷阱（自查时发现，已写成 selftest4 断言并在 20 个随机有向图上"
      f"复验）**："
      f"(a) 网上常见的简化口径 `TUBE = up ∩ down − CORE` 对**任何图**都恒为 "
      f"**{f['bowtie_TUBE_vacuous_shortcut']}** —— 若 v 可达核又被核可达，"
      f"则 v 与核内所有点互达、v 必属该 SCC；所以本实现不用这条，改用原文的 IN→OUT 通路口径。"
      f"(b) `OUT-COMPONENTS` 若按字面取「可从 OUT 到达」也恒为 "
      f"**{f['bowtie_OUT-COMPONENTS_naive_forward']}** —— OUT = down − up，"
      f"自 OUT 出发的点仍在 down 内、已被 CORE/OUT 吃掉；正确口径是"
      f"「**能到达** OUT」（反向可达），本数据集给出 {f['bowtie_OUT-COMPONENTS']} 点"
      f"（360 口径 {hm['bowtie_OUT-COMPONENTS']} 点）。")
    w(f"- 抢占顺序 TUBE → IN-COMPONENTS → OUT-COMPONENTS；需要靠顺序裁决的重叠点"
      f"（既可自 IN 到达又能到 OUT）= **{f['bowtie_tendril_overlap_assigned_to_TUBE']}**，"
      f"它们按原文一律记 TUBE，不构成任意口径选择。")
    w(f"- CORE 并列候选数 = {f['bowtie_core_tied_candidates']}（最大 SCC 唯一），"
      f"不存在选核歧义。")
    w("")
    w("**分组 × 健康状态列联表（全图 676）**（完整表 = `bowtie_by_health.csv`）")
    w("")
    w("```")
    w(ct.to_string(index=False))
    w("```")
    w("")
    w(f"- **162 个整站不可用节点的落点**：" +
      "、".join(f"{g} {int(ct2.loc[g, 'dead'])}" for g in BOWTIE_GROUPS if int(ct2.loc[g, "dead"]) > 0) +
      f"（合计 {sum(int(ct2.loc[g, 'dead']) for g in BOWTIE_GROUPS)}，与 162 闭合）。")
    w(f"- 其中 **{E['dead_in_bowtie_OUT']}/{E['dead_n']} 落在 OUT**。这句话能读的部分是"
      f"{pq('有活站单向指向它们')}；**不能**读作{pq('死站在下游')}："
      f"它们本应有自己的出边，而那些出边被观测到 {E['edges_from_dead']} 条（J4）。")
    w("")
    w("**GCC 之外那个 2 节点分量落在哪一组（点名回答）**")
    w("")
    disc = sorted(bg.loc[bg["group_full_676"] == "DISCONNECTED", "node"])
    outc = sorted(bg.loc[bg["group_full_676"] == "OUT-COMPONENTS", "node"])
    inc = sorted(bg.loc[bg["group_full_676"] == "IN-COMPONENTS", "node"])
    Gf = scopes["full_676"]
    w(f"- 该分量 = `bm.ustc.edu.cn` + `l.bm.ustc.edu.cn`（无向图 `comp_sizes=[674,2]` 的那个 2）。"
      f"唯一一条边方向为 `bm → l.bm`：`bm` 入度 {Gf.in_degree('bm.ustc.edu.cn')}、"
      f"出度 {Gf.out_degree('bm.ustc.edu.cn')}；`l.bm` 入度 {Gf.in_degree('l.bm.ustc.edu.cn')}、"
      f"出度 {Gf.out_degree('l.bm.ustc.edu.cn')}。")
    w(f"- 判组：两端都不在 674 点弱分量内 → 既不能到达 CORE、也不被 CORE 到达，"
      f"也不可自 IN 到达、也不能到达 OUT → **两点都是 `DISCONNECTED`**"
      f"（676 与 671 两套口径同结论；传递闭包那条独立路径同结论）。")
    w(f"- 全图 DISCONNECTED 共 {f['bowtie_DISCONNECTED']} 点：`{'`, `'.join(disc)}`。")
    w(f"- 对照组：IN-COMPONENTS {f['bowtie_IN-COMPONENTS']} 点 "
      f"`{'`, `'.join(x.replace('.ustc.edu.cn', '') for x in inc)}`；"
      f"OUT-COMPONENTS {f['bowtie_OUT-COMPONENTS']} 点 "
      f"`{'`, `'.join(x.replace('.ustc.edu.cn', '') for x in outc)}` —— "
      f"OUT-COMPONENTS 这一组在改用反向口径后才非空，成员都是"
      f"「只向 OUT 侧单向出链、自己收不到核的链接」的站点。")
    w(f"- IN 组 {f['bowtie_IN']} 点：`{'`, `'.join(sorted(bg.loc[bg.group_full_676 == 'IN', 'node']))}` —— "
      f"**全部是入度 = 0 的纯源点**（有出边、指向核方向，但核回不到它们）。")
    w(f"- 5 个悬空引用（`985-211` `personnel` `recapi` `rmd.snst` `www2.lib`，无属性列）"
      f"全部落 OUT：各只有 1 条入边、0 条出边。")
    w("")

    # ---------------------------------------------------------------- J3
    w("## J3 · in/out 不对称度分布、零度节点清单")
    w("")
    w(f"不对称度 = `|in−out| / (in+out)`，度在**各自口径的图内**重算；"
      f"in+out = 0 的点记 NaN —— 676 口径下这样的点有 **{f['n_nodes_isolated']} 个**"
      f"（与 `n_isolated_in_gu = 0` 锚点一致），671 口径同样 {a['n_nodes_isolated']} 个，"
      f"故 671/676 的分布分母完整。360 口径有 {hm['n_nodes_isolated']} 个孤立点（J4-B4）。")
    w("")
    w("| 统计量 | 全图 676 | 主分析 671 | 仅健康 360 |")
    w("|---|---:|---:|---:|")
    for k, lab in [("asym_mean", "均值"), ("asym_median", "中位数"),
                   ("asym_p25", "P25"), ("asym_p75", "P75"),
                   ("asym_eq_0", "asym = 0（完全对称）"),
                   ("asym_le_0p2", "asym ≤ 0.2"),
                   ("asym_ge_0p8", "asym ≥ 0.8"),
                   ("asym_eq_1", "asym = 1（纯入或纯出）"),
                   ("mean_in_degree", "平均入度"), ("mean_out_degree", "平均出度")]:
        w(f"| {lab} | {f[k]} | {a[k]} | {hm[k]} |")
    w("")
    dd = abs_df[abs_df["scope"] == "A_full676_anchor"]
    w(f"**十分位分布（`pd.qcut(a, 10)`，全图 676）**（两套以上口径全表 = `asymmetry_deciles.csv`）")
    w("")
    w("| 分位箱 | 点数 | 占比 | 箱内均值 |")
    w("|---|---:|---:|---:|")
    for _, r in dd.iterrows():
        w(f"| {r['asym_decile_bin']} | {r['n_nodes']} | {pc(r['share_of_nodes'])} | "
          f"{r['mean_asym_in_bin']} |")
    w("")
    w(f"- 分布**两端堆积、中间稀疏**：一端 {f['asym_eq_1']} 个点 asym = 1"
      f"（{pc(f['asym_eq_1'] / f['n_nodes'])}）—— 其中 {f['n_sink_out0']} 个是出度 0、"
      f"{f['n_source_in0']} 个是入度 0，二者交集 {f['n_both0']} 个；"
      f"另一端只有 {f['asym_eq_0']} 个点完全对称。中位 asym = **{f['asym_median']}**。")
    w(f"- 注意 Σin 恒等于 Σout（同一批边的两种记法，均为 {f['sum_in_degree']}），"
      f"所以不对称**只在分布层面存在**：出度堆在少数点上，大量点出度为 0。")
    w("")
    w("**零度节点规模与清单**（逐点全列 = `zero_degree_nodes.csv`，"
      f"{len(zd)} 行，含健康状态 / 失效类别 / 两套 bow-tie 分组 / url_count / 是否被 20 条上限截断）")
    w("")
    w("| 类 | 全图 676 | 主分析 671 | 仅健康 360 | 备注 |")
    w("|---|---:|---:|---:|---|")
    w(f"| 出度 = 0（sink） | **{f['n_sink_out0']}**（{pc(f['sink_share'])}） | "
      f"{a['n_sink_out0']}（{pc(a['sink_share'])}） | "
      f"**{hm['n_sink_out0']}**（{pc(hm['sink_share'])}） | 676 比 671 多的 {f['n_sink_out0'] - a['n_sink_out0']} 个是悬空引用 |")
    w(f"| 入度 = 0（source） | {f['n_source_in0']}（{pc(f['source_share'])}） | "
      f"{a['n_source_in0']} | {hm['n_source_in0']}（{pc(hm['source_share'])}） | "
      f"676 口径全名单：`{'`, `'.join(x.replace('.ustc.edu.cn', '') for x in f['_in0_nodes'])}` |")
    w(f"| 两者都 = 0 | {f['n_both0']} | {a['n_both0']} | {hm['n_both0']} | "
      f"与 `n_isolated_in_gu=0` 锚点一致；360 子图另有 {hm['n_wcc_singleton']} 个孤立点 |")
    w(f"| sink 吸纳入边条数 | {f['sink_in_degree_sum']} | {a['sink_in_degree_sum']} | "
      f"{hm['sink_in_degree_sum']} | 这些入边就是「只进不出」的实际接口 |")
    w(f"| source 抛出出边条数 | {f['source_out_degree_sum']} | {a['source_out_degree_sum']} | "
      f"{hm['source_out_degree_sum']} | — |")
    w("")
    w(f"- 出度 = 0 的 671 个归属点按健康状态拆：**dead {E['out0_dead']} / "
      f"partial {E['out0_partial']} / healthy {E['out0_healthy']}**"
      f"（{E['out0_dead']}+{E['out0_partial']}+{E['out0_healthy']} = {E['out0_attributed_total']}，"
      f"闭合；加 {E['out0_dangling_ref']} 个悬空点 = {f['n_sink_out0']}）。")
    w(f"- 入度 = 0 的 {f['n_source_in0']} 个点**全部是活站**（健康 "
      f"{int((zd[zd.kind == 'in0_source']['health_state'] == 'healthy').sum())} + 部分失效 "
      f"{int((zd[zd.kind == 'in0_source']['health_state'] == 'partial').sum())}），"
      f"零个死站 —— 死站的入边可观测，与 CONVENTIONS §2.1 一致。"
      f"其中 5 个落 bow-tie 的 IN 组、2 个落 DISCONNECTED。")
    w("")

    # ---------------------------------------------------------------- J4
    w(f"## J4 · 出度删失：{E['dead_out0_n']} 个整站不可用节点的出度**恰好全为 0**")
    w("")
    w("爬虫只能从成功抓到的页面提取链接 → 死站的出链**完全不可观测**"
      "（CONVENTIONS §2.1，非「低估」）。本单元独立复算（`censoring_evidence.csv`）：")
    w("")
    w("| 事实 | 值 | 说明 |")
    w("|---|---:|---|")
    w(f"| 全网有向边数 | {E['edges_total']} | 锚点 |")
    w(f"| 以死站为**起点**的边数 | **{E['edges_from_dead']}** | 锚点 `edges_from_dead`；完全删失 |")
    w(f"| 起点是死站的边**权重**合计 | {E['weight_from_dead']} | 0 条、0 权重，不只是「偏少」 |")
    w(f"| 死站 → 死站 | {E['edges_dead_to_dead']} | 构造性不可识别，不能读作「死站不互链」 |")
    w(f"| 活站 → 死站 | {E['edges_to_dead_from_alive']} | 入边可观测，故失效分析主口径 = 入度 |")
    w(f"| 162 死站中出度 = 0 | **{E['dead_out0_n']}（{pc(E['dead_out0_share'])}）** | 锚点 |")
    w(f"| 真失效 / 假失效 / 待定 | {E['true_fail_n']} / {E['false_fail_n']} / {E['ambiguous_n']} | §2.2 |")
    w(f"| 部分失效点 / 其中出度 = 0 | {E['partial_n']} / {E['partial_out0']} | 部分不可观测的第二来源 |")
    w("")
    w("### ① 出度 = 0 的节点里有多少是 dead / partial / healthy")
    w("")
    w("| 健康状态 | 该类点数 | 出度 = 0 的点数 | 类内 sink 率 | 占全部归属 sink |")
    w("|---|---:|---:|---:|---:|")
    for st in ["dead", "partial", "healthy"]:
        tot = int((nodes["health_state"] == st).sum())
        k = int(E[f"out0_{st}"])
        w(f"| {st} | {tot} | **{k}** | {pc(E[f'out0_{st}_share_within_class'])} | "
          f"{pc(k / E['out0_attributed_total'])} |")
    w(f"| 悬空引用（无属性） | 5 | {E['out0_dangling_ref']} | 100% | "
      f"{pc(E['out0_dangling_ref'] / f['n_sink_out0'])} |")
    w(f"| **合计** | **676** | **{f['n_sink_out0']}** | {pc(f['sink_share'])} | 100% |")
    w("")
    w(f"→ **{pc(E['out0_dead_share_of_all_out0'])} 的「出度 = 0」读数来自整站不可用点**"
      f"（dead 类内 sink 率 = {pc(E['out0_dead_share_within_class'])}，即 100%）。"
      f"所以{pq('出度=0 的节点有 300 多个')}"
      f"**不能**解读为{pq('这些站点是导航死胡同')}：这是**采样可见性**的读数，"
      f"不是站点出链行为的读数。另有 {E['out0_partial']} 个部分失效点的出度也部分不可观测"
      f"（它们有失败 URL，页面级损失无从核对）。")
    w("")
    w("### ② 对照版本：只保留 360 个健康节点后的有向结构")
    w("")
    w("构造：`C.induced(Gd, 360 个 is_healthy 节点)` —— 诱导子图，只留两端都健康的边；"
      "不重建图、不改边序、不改权重口径。另附 D 列（剔 162 死站、保留 partial）与 "
      "C 列（CONVENTIONS §2.2 的**校正后**：剔 32 个假失效）作中间档。")
    w("")
    w("| 指标 | A 全图 676 | B 主分析 671 | C 剔 32 假失效 639（校正后） | D 剔 162 死站 509 | **E 仅健康 360** |")
    w("|---|---:|---:|---:|---:|---:|")
    for k, lab in [("n_nodes", "节点数"), ("n_edges", "有向边数"),
                   ("sum_edge_weight", "权重合计"),
                   ("reciprocity", "**reciprocity**"),
                   ("n_reciprocal_pairs", "互惠对数"),
                   ("scc_largest", "**最大 SCC**"),
                   ("scc_largest_share", "最大 SCC 占比"),
                   ("n_scc", "SCC 个数"), ("n_scc_nonsingleton", "非单点 SCC 个数"),
                   ("n_wcc", "**弱分量个数**"), ("wcc_largest", "最大弱分量"),
                   ("n_wcc_singleton", "孤立点弱分量数"),
                   ("dag_n_layers", "**DAG 层数**"),
                   ("dag_longest_chain_nodes", "**最长链（分量数）**"),
                   ("dag_n_isolated_comps", "DAG 孤立分量数"),
                   ("dag_n_sink_comps", "DAG 汇分量数"),
                   ("bowtie_CORE", "**bow-tie CORE**"), ("bowtie_IN", "bow-tie IN"),
                   ("bowtie_OUT", "bow-tie OUT"), ("bowtie_TUBE", "**bow-tie TUBE**"),
                   ("bowtie_IN-COMPONENTS", "bow-tie IN-COMP"),
                   ("bowtie_OUT-COMPONENTS", "bow-tie OUT-COMP"),
                   ("bowtie_DISCONNECTED", "bow-tie DISCONNECTED"),
                   ("n_sink_out0", "**出度=0 点数**"), ("sink_share", "**sink 比例**"),
                   ("n_source_in0", "入度=0 点数"), ("asym_median", "asym 中位数"),
                   ("asym_mean", "asym 均值"), ("reach_density", "可达对密度")]:
        w(f"| {lab} | {f[k]} | {a[k]} | {cx[k]} | {dv[k]} | **{hm[k]}** |")
    w("")
    w(f"- 边保留：{E['edges_kept_healthy360']}/{E['edges_total']} = {pc(E['edge_kept_share'])} 的有向边、"
      f"{pc(E['weight_kept_share'])} 的权重落在健康点之间；节点保留 {pc(E['node_kept_share'])}。"
      f"**这不叫「删失只影响少量边」**：E 是「把最可能缺出边的一整类点直接从样本里拿掉」的子样本，"
      f"不是对缺失出边的插补。")
    w("")
    w("### ③ 结论判定：删失对照后**依然成立 / 消失或改写 / 不可识别**")
    w("")
    w("**A. 依然成立（方向与量级都在对照后保留）**")
    w("")
    w(f"1. **强不对称的「大核 + 下游裙摆」形状**。CORE 占比 A {pc(f['bowtie_CORE_share'])} → "
      f"E {pc(hm['bowtie_CORE_share'])}；OUT/IN 比 A {f['bowtie_OUT']}:{f['bowtie_IN']} → "
      f"E {hm['bowtie_OUT']}:{hm['bowtie_IN']}（E 里 OUT 是 IN 的 "
      f"{hm['bowtie_OUT'] / max(hm['bowtie_IN'], 1):.1f} 倍）。裙摆远大于入口这条结构事实与删失无关。")
    w(f"2. **互惠性低**。A {f['reciprocity']} → E **{hm['reciprocity']}**"
      f"（+{round(hm['reciprocity'] - f['reciprocity'], 4)}）。五套口径全部落在 "
      f"{min(v['reciprocity'] for v in R.values()):.3f}–{max(v['reciprocity'] for v in R.values()):.3f}"
      f"，即便完全剔除删失点，也只有约 1/4 的边成对。"
      f"**「单边占绝对多数」可以下，且它属于「删失后更强」的方向**（原始值被 162 个"
      f"零出边点稀释）。")
    w(f"3. **缩点 DAG 很浅**：拓扑层数 A {f['dag_n_layers']} → E {hm['dag_n_layers']}，"
      f"最长链 A {f['dag_longest_chain_nodes']} → E {hm['dag_longest_chain_nodes']} 个分量"
      f"（弧数 {f['dag_longest_chain_edges']} → {hm['dag_longest_chain_edges']}）。**一字不差**。"
      f"「核外结构只有一两层裙摆、不是多级瀑布」不由删失产生。")
    w(f"4. **TUBE 组 = {f['bowtie_TUBE']} → {hm['bowtie_TUBE']}（两套口径同值，"
      f"IN→OUT 的中转通路几乎不存在）**；"
      f"DAG 孤立分量 {f['dag_n_isolated_comps']} → {hm['dag_n_isolated_comps']}；"
      f"DAG 汇分量占比 A {pc(f['dag_n_sink_comps'] / f['n_dag_nodes'])} → "
      f"E {pc(hm['dag_n_sink_comps'] / hm['n_dag_nodes'])}。"
      f"「核外几乎没有穿核/穿流的中转层」这条与删失无关。")
    w(f"5. **入度侧不受删失**：入度 = 0 的点 A {f['n_source_in0']} 个、E {hm['n_source_in0']} 个，"
      f"且两套都是**零个死站**。「网络有极少的纯源入口」成立，"
      f"与 CONVENTIONS §2.1「主口径 = 入度」互证。")
    w("")
    w("**B. 消失 / 必须改写（不得对全图下结论）**")
    w("")
    w(f"1. **「{f['n_sink_out0']} 个出度 = 0（{pc(f['sink_share'])}）的站点是导航死胡同」—— 不成立**。"
      f"对照口径 E 降到 **{hm['n_sink_out0']}/{hm['n_nodes']} = {pc(hm['sink_share'])}**，"
      f"绝对降 **{E['sink_share_drop_pp']:.2f} pp**；D 口径（只剔 162 死站）"
      f"= {pc(dv['sink_share'])}。也就是说 sink 比例的**主要部分是删失读数**，"
      f"剩下 {pc(hm['sink_share'])} 才是健康子样本里可辩护的规模。")
    w(f"2. **「{f['n_wcc']} 个弱分量 / [674, 2]」是全图口径的性质，不是网络性质**。"
      f"对照后 E 的弱分量数 = **{hm['n_wcc']}**、最大弱分量 {hm['wcc_largest']}、"
      f"孤立点 {hm['n_wcc_singleton']} 个。{E['edges_to_dead_from_alive']} 条指向死站的入边"
      f"把 162 个点拖进大分量 —— 这句话里「99.7% 连在一起」有一半靠只进不出的叶子。")
    w(f"3. **IN 组「入口只有 {f['bowtie_IN']} 个」必须降级**。死站无出边 → 在 IN/OUT/TUBE 的"
      f"定义下它们**只能**落 OUT / DISCONNECTED / IN-COMPONENTS（实测 OUT "
      f"{E['dead_in_bowtie_OUT']}、DISCONNECTED {int(ct2.loc['DISCONNECTED', 'dead'])}、"
      f"IN-COMPONENTS {int(ct2.loc['IN-COMPONENTS', 'dead'])}），永不可能进 IN。"
      f"E 口径 IN = {hm['bowtie_IN']}、DISCONNECTED = {hm['bowtie_DISCONNECTED']}，"
      f"结构已重排。**「全图 IN 只有 5 个」是删失的必然结果**，"
      f"不能写成「网站群几乎没有外部入口」。")
    w(f"4. **最大 SCC 的规模与占比不能脱离分母引用**：A {f['scc_largest']}"
      f"（{pc(f['scc_largest_share'])}）→ B {a['scc_largest']}（{pc(a['scc_largest_share'])}）→ "
      f"C {cx['scc_largest']} → D {dv['scc_largest']}（{pc(dv['scc_largest_share'])}）→ "
      f"E {hm['scc_largest']}（{pc(hm['scc_largest_share'])}）。"
      f"节点集从 676 缩到 360（{pc(E['node_kept_share'])}），核只从 {f['scc_largest']} 缩到 "
      f"{hm['scc_largest']}（-{100 * (1 - hm['scc_largest'] / f['scc_largest']):.0f}%）—— "
      f"核的「占比」在 44%–51% 之间随口径漂移，跨模块引用必须写分母。")
    w(f"5. **可达对密度 A {f['reach_density']} → E {hm['reach_density']}**"
      f"（+{round(hm['reach_density'] - f['reach_density'], 4)}）。这个数与 CONVENTIONS §2.3 的 "
      f"`E_full` 是**同一类分母收缩陷阱**：分母 = n(n−1) 随节点集缩小而缩，"
      f"**禁止**据此写「删掉死站后网络更可达」；合法比较只有同规模跨情景。")
    w(f"6. **sink 比例在 E 口径下仍高达 {pc(hm['sink_share'])}，但它不再等价于「死胡同」** —— "
      f"它是「在 360 个健康站点的互相可见范围内无人收到其出链」。"
      f"J5 的截断分析给出这一残余项中 ≤ {T['overestimate_healthy_sink_pp_max']} pp 可归到 20 条上限，"
      f"其余部分（{pc(hm['sink_share'])} 减去这一小块）主要由"
      f"「整站仅 1 页 / 站外与站内链接不入表」两项测量边界承担，无法进一步拆。")
    w("")
    w("**C. 需要降级为「不可识别」的命题（对照版本也回答不了）**")
    w("")
    w(f"- 「死站之间是否互链」：`edges_dead_to_dead = {E['edges_dead_to_dead']}`，"
      f"起点集合被清空后这条问题在数据里没有取值。")
    w(f"- 「死站在 bow-tie 里本应属于 IN 还是 OUT」：出边不可观测，"
      f"任何标签都只反映「别人单向指向它」这一半信息。{E['dead_in_bowtie_OUT']}/{E['dead_n']} 的 OUT 归属是构造结果。")
    w("- 「某站是否真的没有对外链接」：需要页面级 href 清单，`urls.csv` 只有 4 列"
      "（subdomain / status / url / title），无页面级出链信息。")
    w("- 「核外点为什么回不到核」：本单元只能刻画形状，无时间维与页面级来源，机制问题不可答。")
    w("")

    # ---------------------------------------------------------------- J5
    w("## J5 · 20 条落库上限对出度的截断量级 & sink 比例被高估多少")
    w("")
    w(f"- `listed_urls` 上限 **{T['cap_on_listed_urls']}**（{T['n_sites_listed_at_cap']} 个站点正好 20 条），"
      f"`url_count` 最大 {T['max_url_count']}。`Σurl_count = {T['sum_url_count']}` vs "
      f"`Σlisted_urls = {T['sum_listed_urls']}`（= `n_urls` 锚 8115）"
      f"→ **页面级覆盖率 {pc(T['page_coverage_share'])}**、"
      f"未落库页面 **{T['n_unobserved_pages_total']}** 个。")
    w(f"- **{T['n_sites_page_truncated']}/671 = {pc(T['share_sites_page_truncated'])} 个站点确有未落库页面**"
      f"（`url_count > listed_urls`）；{T['n_sites_fully_enumerated']} 个 "
      f"`url_count ≤ 20` 的站点明细已穷尽。健康站被截断比例更高："
      f"**{T['healthy_n_page_truncated']}/360 = {pc(T['healthy_share_page_truncated'])}**，"
      f"健康站页面覆盖率 {pc(T['healthy_page_coverage_share'])}。")
    w("")
    w("**(1) 落库 20 条上限 —— 可量化。**")
    w("")
    w(f"- 每页出链次数（健康站，合并口径）= {T['href_per_ok_page_pooled_healthy']} 次/页；"
      f"逐站分布**极偏**：中位 {T['href_per_ok_page_median_healthy']}、"
      f"P25 {T['href_per_ok_page_p25_healthy']}、P75 {T['href_per_ok_page_p75_healthy']}。"
      f"合并口径被少数大站主导，所以下面的「次数级」外推只能当量级。")
    w(f"- 缺失出链**次数**（不是边数）量级：合并速率线性外推 ≈ "
      f"**{T['missing_out_hrefs_pooled_upper']:.3g}**（上界，完全忽略模板重复 —— "
      f"同一模板 href 在多页重复出现，已被 `weight` 累加过）；"
      f"中位速率外推 ≈ {T['missing_out_hrefs_median_rate']:.3g}。")
    w(f"- 出度是「**不同**出目标数」，不可线性外推。用 `url_count == listed_urls ≤ 20` 的 "
      f"**{T['fully_enumerated_healthy_n']} 个未被截断健康站点**当天然对照组，"
      f"估亚线性饱和标度：`log(out_degree) ~ log(url_count)` 斜率 "
      f"**β = {T['outdeg_vs_pages_beta']}**，95% CI {T['outdeg_vs_pages_beta_ci95']}"
      f"（回归子集 n 见 `truncation_estimate.csv` 同行 note；β < 1 说明出度随页面数**饱和**）。")
    w(f"- 对 `url_count/listed_urls = {T['trunc_healthy_ratio_median']}`"
      f"（中位比值，多为 100/20 = 5）的站点：β 模型给 **×{T['projection_factor_beta_model_at_ratio5']}**，"
      f"β=1 线性模型给 ×{T['projection_factor_linear_model_at_ratio5']}。"
      f"落到全网边数：现 {T['net_directed_edges_now']} 条 → "
      f"**≈ {T['net_directed_edges_projected_beta']} 条**（β 模型）"
      f"至 **≈ {T['net_directed_edges_projected_linear_upper']} 条**（线性上界）；"
      f"被截断健康站点的出度之和 {T['trunc_healthy_outdeg_observed']} → "
      f"{T['trunc_healthy_outdeg_projected_beta']:.0f}（β） / "
      f"{T['trunc_healthy_outdeg_projected_linear']:.0f}（线性）。"
      f"**出度量级被压低的幅度约在 1×–4× 之间** —— 这两个投影是**模型外推、不是观测**，"
      f"且不含死站（其出边完全不可观测）。")
    w("")
    w("**对 sink 比例「高估多少」的直接回答**（关键：截断只能作用在**确有未落库页面**的点上）")
    w("")
    w("| 基准口径 | 现报 sink 比例 | 其中「有未落库页面、身份可翻案」的点数 | "
      "20 条上限至多抬高 | 全部翻案后的下界 |")
    w("|---|---:|---:|---:|---:|")
    fl_h = T['flippable_360_sinks_n']
    flip_f = T['sink_full676_page_truncated_alive_only']
    w(f"| E 仅健康 360 子图 | {pc(T['healthy_sink_rate_360subgraph_now'])} | **{fl_h}** | "
      f"**≤ +{T['overestimate_healthy_sink_pp_max']} pp** | "
      f"{pc(T['healthy_sink_rate_360subgraph_if_all_flip'])} |")
    w(f"| A 全图 676 | {pc(f['sink_share'])} | **{flip_f}**"
      f"（健康 {T['sink_full676_out0_healthy_page_truncated']} + 部分失效 "
      f"{T['sink_full676_out0_partial_page_truncated']}；死站另有主导通道，不重复计入） | "
      f"**≤ +{100 * flip_f / 676:.2f} pp** | "
      f"{pc((f['n_sink_out0'] - flip_f) / 676)} |")
    w("")
    w(f"- 支撑细节：可翻案的 {fl_h} 个健康 sink，其 **{T['cap_on_listed_urls']} 条已落库页面里的"
      f"跨子域出链次数恰好为 {T['flippable_360_sinks_observed_out_hrefs']}**（全为 0）。"
      f"要让它们翻案，需要「前 20 页一条不出、后面才出」的页间差异。"
      f"rule of three 给 95% 上界：{fl_h} 站合计缺失 ≤ "
      f"**{T['flippable_360_sinks_missing_hrefs_95ub']:.0f} 次出链**（合计，非每站）—— 量级很小。")
    w(f"- 所以 **20 条上限这条通道对 sink 比例是二阶的**（健康口径 ≤ "
      f"{T['overestimate_healthy_sink_pp_max']} pp、全图口径 ≤ "
      f"{T['overestimate_full676_sink_pp_max']} pp）。"
      f"**一阶通道是 162 个整站不可用点的完全删失**："
      f"{E['out0_dead']} 个 sink，占 A 口径 sink 的 {pc(E['out0_dead'] / f['n_sink_out0'])}"
      f"（= {pc(f['sink_share'])} 中的 {pc(E['out0_dead'] / 676)} 个百分点）。")
    w(f"- 另一面：截断对**出度大小**（非零/零之分以外）的影响是一阶的 —— "
      f"{pc(T['healthy_share_page_truncated'])} 的健康站点出度被压低 1×–4×，"
      f"所以任何用出度做的排序 / 分位 / 中心性（C 组、A1）都应视为**下界口径**。")
    w("")
    w(f"**(2) 链接抽取范围 —— 无法量化。** 一条有向边要求 href 指向**另一个 ustc.edu.cn 子域**；"
      f"站外链接、同子域内部链接、JS 渲染后才出现的链接都不进 `links.csv`。"
      f"现有文件无法把「出度 = 0」拆成「页面确实无跨站链接」与「链接存在但不在抽取口径内」"
      f"（`urls.csv` 只有 subdomain / status / url / title 四列，无页面级 href 清单）。"
      f"**本单元明确写为无法量化。**")
    w("")
    w(f"**(3) `url_count` 自身可能也有每站预算 —— 无法量化，但有强证据提示。** "
      f"{T['sites_url_count_in_99_105']} 个站点的 `url_count` 落在 99–105"
      f"（其中 {T['sites_url_count_eq_100']} 个正好 100），这种堆积不像自然分布；"
      f"另有 {T['sites_url_count_eq_1']} 个站点 `url_count == 1`，"
      f"其中健康 sink {T['healthy_sinks_url_count_eq_1']} 个 —— "
      f"`url_count == listed_urls == 1` 在口径上算「未截断」，"
      f"但它只说明抓取器报告了 1 个页面，**不等于该站只有 1 个页面**。"
      f"真实页面存量在本数据集里不可核对 → 上面 (1) 的 pp 数因此是**下界**。")
    w("")

    # ------------------------------------------------------------ 局限
    w("## 本单元自报的局限")
    w("")
    w(f"1. **出度删失是最大不确定源**：{E['dead_n']} 点（{pc(E['dead_n'] / 676)}）出边 0 条可观测，"
      f"另有 {E['partial_n']} 点（{pc(E['partial_n'] / 676)}）出边部分不可观测。"
      f"本单元**未做任何插补**：E 是子样本，不是缺失值填补。")
    w(f"2. **口径分母漂移**：五套节点集的最大 SCC 占比 "
      f"{pc(f['scc_largest_share'])} / {pc(a['scc_largest_share'])} / "
      f"{pc(cx['scc_largest_share'])} / {pc(dv['scc_largest_share'])} / "
      f"{pc(hm['scc_largest_share'])}，差最多 "
      f"{100 * (max(v['scc_largest_share'] for v in R.values()) - min(v['scc_largest_share'] for v in R.values())):.1f} pp。"
      f"引用「最大 SCC」必须同时报分母。")
    w(f"3. **DAG 层数只有一个口径**：`nx.topological_generations` 给的是「最早可能层」，"
      f"不是唯一分层定义（换成关键路径层级会得到不同层数）。"
      f"最长链用 `nx.dag_longest_path`（缩点图无环，精确算法）。CORE 唯一（并列候选 "
      f"{f['bowtie_core_tied_candidates']} 个），故无选核歧义。")
    w(f"4. **Bow-tie 的组定义有两个坑，本单元都按原文口径钉死并报出对照量**："
      f"(a) `TUBE` 用原文「IN→OUT 有向通路」，不用流传的 `up ∩ down − CORE`"
      f"（后者恒为 {f['bowtie_TUBE_vacuous_shortcut']}，在任何图上都给不出非零 TUBE）；"
      f"(b) `OUT-COMPONENTS` 用「能到达 OUT」的反向可达，不用「可从 OUT 到达」"
      f"（后者恒为 {f['bowtie_OUT-COMPONENTS_naive_forward']}）。"
      f"抢占顺序 TUBE→IN→OUT 需要裁决的重叠点 = "
      f"{f['bowtie_tendril_overlap_assigned_to_TUBE']}（= TUBE 本身，按原文归 TUBE，"
      f"非任意选择）。两条恒空性都在 selftest4 里对 20 个随机有向图做了断言。")
    w(f"5. **β 外推的适用范围**：β 在 `url_count ∈ [2, 20]` 区间内估计，"
      f"却被用于比值中位 {T['trunc_healthy_ratio_median']}（≈5）的外推；"
      f"对照组 n = {T['fully_enumerated_healthy_n']}（回归子集更小），模型风险不可忽略。"
      f"因此 β 投影值只当量级用，不进任何主结论。")
    w(f"6. **拓扑判定一律用无权结构**（「出度 = 0」= 无边），只有 J5 的量级估计用到权重；"
      f"权重语义 = 观察到的链接次数。")
    w(f"7. **5 个悬空引用点**（`985-211` 等）无 `url_count`，截断与否不可判；"
      f"它们全部是出度 0 的 OUT 点，在 A 口径里贡献 {E['out0_dangling_ref']}/{f['n_sink_out0']} 的 sink。")
    w("")
    w("## 异议（DISSENT）说明")
    w("")
    w("本单元**未发现与 `CONVENTIONS.md` 冲突**的实测证据：31/31 锚点通过；"
      "`reciprocity_gd`、`edges_from_dead`、`dead_with_outdegree_zero`、"
      "`n_components / comp_sizes`、`n_isolated_in_gu` 均独立复现；"
      "§2.1（主口径 = 入度）、§2.2（校正前/后两套）、§2.3（分母收缩禁令）、"
      "§7（措辞约束）在本文档全部按原文执行。故不写 `DISSENT.md`，本段即替代说明。")
    w("")
    w("---")
    w("生成脚本：`run_a4.py`（本目录，可独立重跑）。"
      "文中每个数字对应 `directed_metrics.csv`（`metric` 列）、"
      "`censoring_evidence.csv` / `truncation_estimate.csv`（`item` 列）的行名，"
      "或 `bowtie_groups.csv` / `zero_degree_nodes.csv` 的逐点行。")

    text = "\n".join(md) + "\n"
    banned = ["导致", "因为", "使得", "引发", "决定了", "传染"]
    hits = [(w2, text.count(w2)) for w2 in banned if w2 in text]
    if hits:
        raise AssertionError(f"result.md 含禁用因果措辞：{hits}")
    p = os.path.join(C.module_dir(MODULE), "result.md")
    with open(p, "w", encoding="utf-8") as fp:
        fp.write(text)


if __name__ == "__main__":
    main()
