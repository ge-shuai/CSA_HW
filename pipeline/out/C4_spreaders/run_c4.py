# -*- coding: utf-8 -*-
"""C4_spreaders — Kitsak et al. (2010) 主张结构复现：k-core 是否比度更能预测"影响力"

⚠ 本模块是 **情景模拟（scenario simulation），不是测量**。
   本数据集是单次横截面超链接图，无传播/点击/时序数据，"影响力"只能通过显式
   假设的传播模型来定义。下文所有"传播能力 / 影响力"字样均等于
   "在该 SIR 模型假设下的期望可达规模"，不是对真实信息扩散的观测。

任务书（§C4）
  §1 在 **有向图 Gd** 上实现显式离散时间 SIR：
     - 主变体 = **无权**：每条有向边以概率 β 独立尝试穿越，权重不进传播层。
       理由（CONVENTIONS §1）：权重语义 = "观察到的链接次数"，**不得**解释为流量，
       把 w 当传染率放大系数没有依据。
     - 对照变体 = **加权**：逐边概率 = β × (w / w_max_of_that_source)，
       仅用于量化"无权"这一选择本身的影响。
     - 固定恢复率 γ = 0.20（平均传染期 1/γ = 5 步）；β 网格 = 6 档跨临界区。
     - 每个 (节点, β, 变体) 跑 20 次独立试验，seed 由 SeedSequence 熵元组
       [SEED, node_i, beta_i, trial, variant] 显式给出 → 任一单次试验可独立复现。
     - 记录期望最终感染规模 R = 累计被感染节点数（含种子；R=1 表示未扩散）。
  §2 按 R 排名，与 k-core / 无向度 / 介数 / PageRank(有向加权) 四个代理比较，
     另加入度、出度作参照。指标 = Spearman ρ（+ percentile bootstrap CI）
     与 Top-k 重叠率 / Jaccard（k = 10, 20, 25, 50；平局按下标升序确定性打破）。
  §3 判决：每个 β 下哪个代理最贴合模拟；k-core 是否优于度（Kitsak 主张）。
     另给"控制度之后的 k-core 偏 Spearman"，直接检验 k-core 有无超出度的信息。
  §4 **删失警示（CONVENTIONS §2.1）**：162 个全挂站点出度 ≡ 0 → 作种子时 R 恒为 1，
     "代理 vs R"的相关会混入"代理 vs 死没死"。
     故 **主口径 = 活站 ∩ attributed（509 个）**；671 / 676 口径只作对照并列报告。

口径声明
  - 建图一律 C.build_all()；开跑先 C.assert_anchors()（31 项，不过即中止）。
  - 模拟在 676 个 in_graph 节点上进行（传播可穿过 5 个悬空节点），排名只在子集上做。
  - **禁用** nodes_full.csv 的 `degree` / `color` 列；全部度从图重算。
  - PageRank 主口径 = 有向加权（C.centralities 的 pr_dw）。
  - 因果措辞禁用（CONVENTIONS §7）。
  - SEED = 20261113；PYTHONHASHSEED=0。

运行：  cd pipeline && PYTHONHASHSEED=0 python out/C4_spreaders/run_c4.py
产出：  sir_by_node_beta.csv, sir_node_summary.csv, epidemic_size_by_beta.csv,
        proxy_vs_sir_spearman.csv, proxy_vs_sir_topk.csv, kitsak_verdict.csv,
        censoring_contrast_671_vs_509.csv, variant_weighted_vs_unweighted.csv,
        reach_deterministic.csv, top20_spreaders.csv,
        key_numbers.json, _number_provenance.csv, RUN.txt

数字溯源：每个 record(...) 调用行尾的标记注释即 result.md 里引用的行号来源，
         映射表见产出 _number_provenance.csv 与 key_numbers.json 的 provenance 字段。
"""
from __future__ import annotations

import sys, os, json, time, re
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats as sps

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.dirname(os.path.dirname(_HERE))
if _PIPE not in sys.path:
    sys.path.insert(0, _PIPE)

import csa_common as C

SEED = 20261113
assert SEED == C.SEED, "SEED 与 csa_common.SEED 漂移"
MODULE = "C4_spreaders"
OUT = C.module_dir(MODULE)

# ------------------------------------------------ 模拟参数（跑前设定，不看结果回调）
GAMMA = 0.20                                  # 固定恢复率：每步恢复概率
BETAS = [0.01, 0.03, 0.06, 0.12, 0.25, 0.50]  # 6 档 β，近似几何间隔，跨临界区
N_TRIALS = 20                                 # 每 (节点,β,变体) 独立试验数（≥20，任务要求）
TOPK = [10, 20, 25, 50]                       # Top-k 重叠的 k
N_BOOT = 1000                                 # Spearman bootstrap 重采样次数
MAX_STEPS = 3000                              # 单条 SIR 轨迹最大步数（防病态循环）
VARIANTS = ["unweighted", "weighted"]         # 无权=主口径，加权=对照
PROXIES = ["kcore", "deg_u", "bet_u", "pr_dw", "in_degree", "out_degree"]
PROXY_CN = {"kcore": "k-core(核数)", "deg_u": "无向度", "bet_u": "介数(无向无权)",
            "pr_dw": "PageRank(有向加权,主口径)", "in_degree": "入度", "out_degree": "出度"}
SUBSETS = ["alive_attr509", "attr671", "graph676"]
SUBSET_CN = {"alive_attr509": "活站∩attributed(主口径)",
             "attr671": "attributed 671(对照)",
             "graph676": "全图 676(对照)"}

_T0 = time.perf_counter()
_LOG = []


def log(msg: str):
    line = "[%7.2fs] %s" % (time.perf_counter() - _T0, msg)
    print(line, flush=True)
    _LOG.append(line)


# ==================================================== 数字溯源（result.md 引用的行号）
_PROV = []
_TAG_RE = re.compile(r"#\s*@num\s+([A-Za-z0-9_]+)")
_SRC = open(os.path.join(_HERE, "run_c4.py"), encoding="utf-8").read().splitlines()
_TAG_AT = {}
for _i, _l in enumerate(_SRC, start=1):
    _m = _TAG_RE.search(_l)
    if _m:
        _TAG_AT.setdefault(_m.group(1), _i)


def record(tag, value):
    """登记关键数字；cite 指向源码中带 `# @num <tag>` 标记的那一行。"""
    ln = _TAG_AT.get(tag)
    if ln is None:
        raise KeyError("missing @num marker for tag: %s" % tag)
    v = value
    if isinstance(v, (np.integer,)):
        v = int(v)
    elif isinstance(v, (np.floating,)):
        v = float(v)
    elif isinstance(v, np.ndarray):
        v = v.tolist()
    _PROV.append({"tag": tag, "cite": "run_c4.py:L%d" % ln, "value": v})
    return value


# ================================================================ 图 → 前驱表
def build_adj(G, nodes):
    """CSR 式出边表：tgt[i] = 后继下标数组；wn[i] = 该源出边的归一化权重(w/w_max_source)。"""
    idx = {nd: i for i, nd in enumerate(nodes)}
    tgt, wn = [], []
    for nd in nodes:
        succ = list(G.successors(nd))
        tgt.append(np.array([idx[v] for v in succ], dtype=np.int64))
        if succ:
            w = np.array([G[nd][v]["weight"] for v in succ], dtype=float)
            wn.append(w / w.max() if w.max() > 0 else np.zeros_like(w))
        else:
            wn.append(np.empty(0, dtype=float))
    return idx, tgt, wn


# ==================================================================== SIR 单次实现
def sir_size(seed_i, beta, gamma, rng, tgt, wn, n, weighted):
    """离散时间 SIR。每步：当前感染者先以 γ 判恢复；存活者对每条出边以 β 尝试感染。

    weighted=False → 逐边传染概率恒为 beta（无权主变体）
    weighted=True  → 逐边传染概率 = beta * (w / w_max_of_source)（对照变体）
    返回累计被感染节点数（含种子）。
    """
    state = np.zeros(n, dtype=np.int8)                     # 0=S 1=I 2=R
    state[seed_i] = 1
    total = 1
    cur = np.array([seed_i], dtype=np.int64)
    steps = 0
    while cur.size and steps < MAX_STEPS:
        steps += 1
        rec_mask = rng.random(cur.size) < gamma
        rec = cur[rec_mask]
        surv = cur[~rec_mask]
        if rec.size:
            state[rec] = 2
        newly = np.empty(0, dtype=np.int64)
        if surv.size:
            lens = np.fromiter((tgt[i].size for i in surv), dtype=np.int64, count=surv.size)
            tl = int(lens.sum())
            if tl:
                t = np.concatenate([tgt[i] for i in surv])
                if weighted:
                    p = beta * np.concatenate([wn[i] for i in surv])
                else:
                    p = beta
                hit = rng.random(tl) < p
                cand = t[hit]
                cand = cand[state[cand] == 0]
                if cand.size:
                    newly = np.unique(cand)
                    state[newly] = 1
                    total += int(newly.size)
        parts = ([surv] if surv.size else []) + ([newly] if newly.size else [])
        cur = np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)
    return int(total)


def ent(node_i, beta_i, trial, variant_i):
    """显式 seed：SeedSequence 熵元组，与遍历顺序无关，任一试验可单独复现。"""
    return [SEED, int(node_i), int(beta_i), int(trial), int(variant_i)]


# ==================================================================== 统计小工具
def topk(score, k):
    """按 score 降序取 Top-k 下标集合；平局用下标升序确定性打破。"""
    order = np.lexsort((np.arange(len(score)), -np.asarray(score, dtype=float)))
    return set(order[:k].tolist())


def spearman_boot(x, y, n_boot, seed):
    """Spearman ρ + percentile bootstrap CI。

    做法：先对全样本做一次秩变换（Spearman = 秩上的 Pearson），再对"节点配对"有放回
    重采样、在每个重样本内计算秩的 Pearson 相关。这是总体 Spearman 相关系数的标准
    nonparametric bootstrap；CI 取 2.5/97.5 百分位。seed = [seed, 7777, m] 显式给出。
    """
    r = sps.spearmanr(x, y)
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = len(x)
    rx = sps.rankdata(x); ry = sps.rankdata(y)
    rng = np.random.default_rng([seed, 7777, m])
    vals = np.empty(n_boot)
    for k in range(n_boot):
        j = rng.integers(0, m, m)
        a = rx[j]; b = ry[j]
        sa = a.std(); sb = b.std()
        vals[k] = np.nan if (sa == 0 or sb == 0) else \
            float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))
    v = vals[np.isfinite(vals)]
    return (float(r.statistic), float(r.pvalue),
            float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), int(v.size))


def partial_spearman(x, ctrl, y):
    """控制 ctrl 之后 x 与 y 的偏 Spearman（秩残差法，线性残差于秩 ctrl）。"""
    rx = sps.rankdata(np.asarray(x, float))
    ry = sps.rankdata(np.asarray(y, float))
    rz = sps.rankdata(np.asarray(ctrl, float))
    B = np.column_stack([np.ones_like(rz), rz])

    def resid(a):
        return a - B @ np.linalg.lstsq(B, a, rcond=None)[0]
    return float(sps.spearmanr(resid(rx), resid(ry)).statistic)


# ==================================================================== 自检 / 校验
def self_checks(sir_size, ent, tgt, wn, n, idx, rec_nodes, Rm, det, mask):
    """SIR 实现的 5 条内建校验（含 1 条解析真值对照）。

    解析真值：若种子 i 的所有后继出度都为 0（深度 1 的星形），则
    R = 1 + Binomial(outdeg_i, β)（无权变体逐边独立），E[R] = 1 + outdeg_i·β。
    本图确实存在这样的节点（活站里出度>0 且全部后继为汇点）。
    """
    chk = {}
    is_sink = np.array([tgt[i].size == 0 for i in range(n)])
    stars = [i for i in range(n)
             if tgt[i].size >= 3 and bool(np.all(is_sink[tgt[i]])) and mask["alive_attr509"][i]]
    chk["n_depth1_star_nodes_found"] = len(stars)
    star = stars[:3]
    if not star:
        raise AssertionError("SIR 校验失败：图中找不到可用于解析对照的深度1星形节点")
    # A. β=0 → R ≡ 1（与 γ 无关）
    badA = sum(1 for i in star + [0, 1, 2]
               for tr in range(5)
               if sir_size(i, 0.0, GAMMA, np.random.default_rng(ent(i, 0, tr, 0)),
                           tgt, wn, n, False) != 1)
    chk["A_beta0_all_R_equal_1_violations"] = badA
    # B. γ=1 → R ≡ 1（每步先判恢复，恢复者不再传播 → 本实现的分步语义）
    badB = sum(1 for i in range(0, n, 11)
               if sir_size(i, 0.9, 1.0, np.random.default_rng([SEED, 400 + i, 0]),
                           tgt, wn, n, False) != 1)
    chk["B_gamma1_all_R_equal_1_violations"] = badB
    chk["B_nodes_tested"] = len(range(0, n, 11))
    # C. 星形种子解析真值。本实现每步"先判恢复、存活者再传播"，故种子在恢复前
    #    会反复尝试：尝试次数 K ~ 位移几何 P(K=k)=γ(1-γ)^k（k=0,1,...），均值 (1-γ)/γ。
    #    每条出边在 K 次机会内被穿越的概率 = 1-(1-β)^K，对 K 取期望：
    #      p_edge = E[1-(1-β)^K] = 1 - γ / (1-(1-γ)(1-β))
    #    → E[R] = 1 + d·p_edge。（注意：**不是** 1+γ·1+(1-γ)(1+dβ)——那是"只尝试一次"的错式）
    b_c, n_c = 0.30, 4000
    p_edge = 1.0 - GAMMA / (1.0 - (1.0 - GAMMA) * (1.0 - b_c))
    c_res = []
    for i in star:
        d = int(tgt[i].size)
        vals = [sir_size(i, b_c, GAMMA, np.random.default_rng([SEED, 5000 + i, tr]),
                         tgt, wn, n, False) for tr in range(n_c)]
        mu = float(np.mean(vals))
        se = float(np.std(vals, ddof=1) / np.sqrt(n_c))
        theo = 1.0 + d * p_edge
        c_res.append({"node": rec_nodes[i], "outdeg": d, "beta": b_c, "gamma": GAMMA,
                      "analytic_p_edge": round(p_edge, 6),
                      "mc_mean_R": round(mu, 4), "analytic_E_R": round(theo, 4),
                      "mc_se": round(se, 4),
                      "abs_z": round(abs(mu - theo) / se, 3) if se > 0 else None})
    chk["C_analytic_check"] = c_res
    chk["C_max_abs_z"] = max(r["abs_z"] for r in c_res)
    chk["C_all_within_3se"] = bool(all(r["abs_z"] < 3.0 for r in c_res))
    if not chk["C_all_within_3se"]:
        raise AssertionError("SIR 解析真值校验未通过： " + json.dumps(c_res))
    # D. 试验确实有随机性（活站、β=0.25 的 20 次试验 sd>0 的比例）
    arr = Rm[(0, BETAS.index(0.25))]
    chk["D_frac_alive_seeds_with_sd_gt0_b025"] = round(
        float((arr.std(1, ddof=1) > 0)[mask["alive_attr509"]].mean()), 4)
    # E. 出度为 0 的种子 → R ≡ 1（删失后果）
    zero_out = [i for i in range(n) if is_sink[i] and mask["alive_attr509"][i]]
    chk["E_n_alive_seeds_with_outdeg0"] = len(zero_out)
    chk["E_R_is_1_for_all_zero_outdeg_seeds"] = bool(
        all(Rm[(0, bi)][i].max() == 1 for bi in range(len(BETAS)) for i in zero_out))
    # F. seed 熵互异
    chk["F_seed_entropies_unique"] = (
        len({tuple(ent(i, 1, t, 0)) for i in range(4) for t in range(N_TRIALS)})
        == 4 * N_TRIALS)
    return chk



# ========================================================================= 主流程
def main():
    wall0 = time.time()
    log("step0 build_all + assert_anchors")
    b = C.build_all()
    C.assert_anchors(b)                                            # @num ANCHORS
    record("ANCHORS", "31/31 PASS")
    nodes_df, Gd, Gu = b["nodes"], b["Gd"], b["Gu"]
    rec_nodes = sorted(Gd.nodes())                                 # 排序→与集合迭代顺序无关
    n = len(rec_nodes)                                   # @num N_SIM_NODES
    record("N_SIM_NODES", n)
    record("N_EDGES_DIRECTED", Gd.number_of_edges())    # @num N_EDGES_DIRECTED
    idx, tgt, wn = build_adj(Gd, rec_nodes)
    record("GAMMA", GAMMA)                                   # @num GAMMA
    record("BETAS", BETAS)                                   # @num BETAS
    record("N_TRIALS", N_TRIALS)                             # @num N_TRIALS
    record("N_BOOT", N_BOOT)                                 # @num N_BOOT
    record("TOPK", TOPK)                                     # @num TOPK

    # ------------------------------------------------------------- 节点子集口径
    attr = set(nodes_df["node"])
    dead = b["fail"]["dead_all"]
    mask = {
        "alive_attr509": np.array([x in attr and x not in dead for x in rec_nodes]),
        "attr671": np.array([x in attr for x in rec_nodes]),
        "graph676": np.ones(n, dtype=bool),
    }
    record("SUBSET_N_ALIVE509", int(mask["alive_attr509"].sum()))   # @num SUBSET_N_ALIVE509
    record("SUBSET_N_ATTR671", int(mask["attr671"].sum()))          # @num SUBSET_N_ATTR671
    record("SUBSET_N_GRAPH676", int(mask["graph676"].sum()))        # @num SUBSET_N_GRAPH676
    outd = dict(Gd.out_degree())
    record("DEAD_OUTDEG_NONZERO",                                  # @num DEAD_OUTDEG_NONZERO
           int(sum(1 for d in dead if outd.get(d, 0) != 0)))
    record("N_DEAD_SEEDS_FORCED_R1",                                # @num N_DEAD_SEEDS_FORCED_R1
           int(sum(1 for x in rec_nodes
                   if x in attr and x in dead and outd.get(x, 0) == 0)))

    # ------------------------------------------- 确定性可达闭包（β→1 / γ→0 上界参照）
    log("step1 deterministic reachable closure (beta->1 upper bound)")
    det = np.zeros(n, dtype=np.int64)
    for i in range(n):
        seen = {i}
        fr = [i]
        while fr:
            nf = []
            for u in fr:
                for v in tgt[u]:
                    if v not in seen:
                        seen.add(v); nf.append(v)
            fr = nf
        det[i] = len(seen)
    record("DET_REACH_MAX", int(det.max()))                  # @num DET_REACH_MAX
    record("DET_REACH_MEAN_ALIVE509",                        # @num DET_REACH_MEAN_ALIVE509
           float(det[mask["alive_attr509"]].mean()))
    record("DET_REACH_MEDIAN_ALIVE509",                      # @num DET_REACH_MEDIAN_ALIVE509
           float(np.median(det[mask["alive_attr509"]])))
    record("DET_REACH_N_GE_100_ALIVE509",                    # @num DET_REACH_N_GE_100_ALIVE509
           int((det[mask["alive_attr509"]] >= 100).sum()))
    pd.DataFrame({
        "node": rec_nodes,
        "det_reach_size": det,
        "det_reach_pct_of_676": np.round(100.0 * det / n, 3),
        "out_degree": [outd[x] for x in rec_nodes],
        "in_alive_attr509": mask["alive_attr509"],
    }).to_csv(os.path.join(OUT, "reach_deterministic.csv"), index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------- SIR 扫描
    log("step2 SIR sweep: %d nodes x %d betas x %d trials x %d variants = %d trials"
        % (n, len(BETAS), N_TRIALS, len(VARIANTS), n * len(BETAS) * N_TRIALS * len(VARIANTS)))
    t_sir = time.perf_counter()
    Rm = {}                                          # (variant_i, beta_i) -> (n, N_TRIALS) int32
    for vi, var in enumerate(VARIANTS):
        for bi, beta in enumerate(BETAS):
            arr = np.zeros((n, N_TRIALS), dtype=np.int32)
            for i in range(n):
                for tr in range(N_TRIALS):
                    rng = np.random.default_rng(ent(i, bi, tr, vi))
                    arr[i, tr] = sir_size(i, beta, GAMMA, rng, tgt, wn, n, var == "weighted")
            Rm[(vi, bi)] = arr
            log("   %s beta=%.2f done (elapsed %.1fs)" % (var, beta, time.perf_counter() - t_sir))
    sir_wall = time.perf_counter() - t_sir
    record("SIR_WALL_SECONDS", round(sir_wall, 1))          # @num SIR_WALL_SECONDS
    record("SIR_TRIALS_RUN", int(sum(a.size for a in Rm.values())))   # @num SIR_TRIALS_RUN

    log("step2b SIR internal validation")
    chk = self_checks(Rm=Rm, det=det, mask=mask, sir_size=sir_size, ent=ent,
                      tgt=tgt, wn=wn, n=n, idx=idx, rec_nodes=rec_nodes)
    record("VALIDATION", chk)                                # @num VALIDATION
    record("N_ALIVE_SINKS", chk["E_n_alive_seeds_with_outdeg0"])   # @num N_ALIVE_SINKS
    record("C_MAX_ABS_Z", chk["C_max_abs_z"])                       # @num C_MAX_ABS_Z
    record("N_STAR_ANALYTIC", chk["n_depth1_star_nodes_found"])      # @num N_STAR_ANALYTIC
    for _k, _v in chk.items():
        log("   check %-42s = %s" % (_k, _v))

    # ------------------------------------------- 代理指标（全部从图重算，禁用 degree 列）
    log("step3 centralities recomputed from graph")
    cen = C.centralities(Gu, Gd).reindex(rec_nodes)
    cen["in_degree"] = [Gd.in_degree(x) for x in rec_nodes]
    cen["out_degree"] = [outd[x] for x in rec_nodes]
    record("MAX_DEG_U_RECOMPUTED", int(cen["deg_u"].max()))  # @num MAX_DEG_U_RECOMPUTED
    record("KCORE_MAX", int(cen["kcore"].max()))             # @num KCORE_MAX
    kc_counts = cen["kcore"].astype(int).value_counts().sort_index()
    record("KCORE_SHELL_COUNTS", kc_counts.to_dict())        # @num KCORE_SHELL_COUNTS
    record("KCORE_N_AT_KMAX", int(kc_counts.loc[int(cen["kcore"].max())]))      # @num KCORE_N_AT_KMAX
    record("KCORE_N_AT_KMAX_MINUS_1", int(kc_counts.loc[int(cen["kcore"].max()) - 1]))  # @num KCORE_N_AT_KMAX_MINUS_1
    record("MAX_OUT_DEGREE", int(cen["out_degree"].max()))   # @num MAX_OUT_DEGREE

    # --------------------------------------------------- 长表 + 每 β 宏观流行规模
    log("step4 long table + epidemic size")
    rows = []
    ep = []
    for vi, var in enumerate(VARIANTS):
        for bi, beta in enumerate(BETAS):
            arr = Rm[(vi, bi)]
            mn = arr.mean(1); sd = arr.std(1, ddof=1)
            for i in range(n):
                rows.append((rec_nodes[i], var, beta, int(arr[i].min()), int(arr[i].max()),
                             round(float(mn[i]), 4), round(float(sd[i]), 4),
                             int(np.argmax(arr[i]) + 1), int(det[i]),
                             bool(mask["alive_attr509"][i]), bool(mask["attr671"][i])))
            sub = mn[mask["alive_attr509"]]
            ep.append({"variant": var, "beta": beta, "gamma": GAMMA,
                       "R_mean_alive509": round(float(sub.mean()), 4),
                       "R_median_alive509": float(np.median(sub)),
                       "R_max_alive509": int(sub.max()),
                       "frac_seeds_R_gt_1": round(float((sub > 1).mean()), 4),
                       "frac_seeds_R_gt_10": round(float((sub > 10).mean()), 4),
                       "frac_seeds_R_eq_1": round(float((sub == 1).mean()), 4),
                       "n_distinct_R_alive509": int(np.unique(np.round(sub, 4)).size)})
    sir_long = pd.DataFrame(rows, columns=[
        "node", "variant", "beta", "R_min", "R_max", "R_mean", "R_sd",
        "trial_index_of_max_R", "det_reach_size", "in_alive_attr509", "in_attr671"])
    sir_long.to_csv(os.path.join(OUT, "sir_by_node_beta.csv"), index=False, encoding="utf-8-sig")
    record("SIR_LONG_ROWS", len(sir_long))                   # @num SIR_LONG_ROWS
    edf = pd.DataFrame(ep)
    edf.to_csv(os.path.join(OUT, "epidemic_size_by_beta.csv"), index=False, encoding="utf-8-sig")

    # ---------------------------- 试验数是否够：折半信度（奇/偶试验各自求 R 再比排名）
    log("step4b split-half reliability of the R ranking")
    sh = []
    for vi, var in enumerate(VARIANTS):
        for bi, beta in enumerate(BETAS):
            arr = Rm[(vi, bi)]
            m = mask["alive_attr509"]
            a = arr[m, 0::2].mean(1); a2 = arr[m, 1::2].mean(1)
            sh.append({"variant": var, "beta": beta, "subset": "alive_attr509",
                       "n_nodes": int(a.size), "n_trials_per_half": N_TRIALS // 2,
                       "spearman_halves": round(float(sps.spearmanr(a, a2).statistic), 4),
                       "kendall_tau_halves": round(float(sps.kendalltau(a, a2).statistic), 4),
                       "top20_overlap_halves": round(len(topk(a, 20) & topk(a2, 20)) / 20, 3),
                       "distinct_R_per_half": int(np.unique(np.round(a, 3)).size)})
    shdf = pd.DataFrame(sh)
    shdf.to_csv(os.path.join(OUT, "split_half_reliability.csv"), index=False, encoding="utf-8-sig")
    record("SPLIT_HALF_RHO_MIN", float(shdf["spearman_halves"].min()))       # @num SPLIT_HALF_RHO_MIN
    record("SPLIT_HALF_RHO_MAX", float(shdf["spearman_halves"].max()))       # @num SPLIT_HALF_RHO_MAX
    record("SPLIT_HALF_RHO_B06",                                            # @num SPLIT_HALF_RHO_B06
           float(shdf.loc[(shdf.variant == "unweighted") & (shdf.beta == 0.06),
                          "spearman_halves"].iloc[0]))
    record("SPLIT_HALF_TOP20_MIN", float(shdf["top20_overlap_halves"].min()))  # @num SPLIT_HALF_TOP20_MIN

    # ---- Spearman–Brown 推算 20 次试验的整体信度 + 试验数预算曲线
    log("step4c trial budget / Spearman-Brown")
    bud = []
    for vi, var in enumerate(VARIANTS):
        for bi, beta in enumerate(BETAS):
            arr = Rm[(vi, bi)][mask["alive_attr509"]]
            for nh in (2, 5, 10):
                rs = []
                ov = []
                for rep in range(8):
                    perm = np.random.default_rng([SEED, vi, bi, nh, rep]).permutation(N_TRIALS)
                    a = arr[:, perm[:nh]].mean(1); b = arr[:, perm[nh:2 * nh]].mean(1)
                    rs.append(float(sps.spearmanr(a, b).statistic))
                    ov.append(len(topk(a, 20) & topk(b, 20)) / 20)
                rh = float(np.mean(rs))
                bud.append({"variant": var, "beta": beta, "trials_per_half": nh,
                            "trials_total": 2 * nh, "splits_used": len(rs),
                            "halves_rho_mean": round(rh, 4), "halves_rho_sd": round(float(np.std(rs, ddof=1)), 4),
                            "spearman_brown_full_rho": round(2 * rh / (1 + rh), 4),
                            "halves_top20_overlap_mean": round(float(np.mean(ov)), 3)})
    buddf = pd.DataFrame(bud)
    buddf.to_csv(os.path.join(OUT, "trial_budget.csv"), index=False, encoding="utf-8-sig")
    record("SB_RHO_AT_20_B06",                                        # @num SB_RHO_AT_20_B06
           float(buddf.loc[(buddf.variant == "unweighted") & (buddf.beta == 0.06) &
                           (buddf.trials_total == 20), "spearman_brown_full_rho"].iloc[0]))
    record("SB_RHO_AT_20_MIN", float(buddf["spearman_brown_full_rho"].min()))  # @num SB_RHO_AT_20_MIN

    # ---------------------------------------------- 代理 vs 模拟：Spearman + Top-k
    log("step5 proxy vs SIR (spearman + topk)")
    sp_rows, tk_rows = [], []
    for vi, var in enumerate(VARIANTS):
        for bi, beta in enumerate(BETAS):
            R = Rm[(vi, bi)].mean(1)
            n_distinct_all = int(np.unique(np.round(R, 4)).size)
            for ss in SUBSETS:
                mm = mask[ss]
                Rs = R[mm]
                if np.unique(np.round(Rs, 4)).size < 3:
                    continue
                for p in PROXIES:
                    Ps = cen[p].to_numpy()[mm]
                    rho, pval, lo, hi, nb = spearman_boot(Ps, Rs, N_BOOT, SEED)
                    sp_rows.append({
                        "variant": var, "beta": beta, "subset": ss, "subset_cn": SUBSET_CN[ss],
                        "proxy": p, "proxy_cn": PROXY_CN[p], "n_nodes": int(mm.sum()),
                        "spearman_rho": round(rho, 4), "p_value": float(pval),
                        "boot_ci_lo": round(lo, 4), "boot_ci_hi": round(hi, 4),
                        "n_boot_valid": nb, "n_distinct_R": n_distinct_all})
                    for k in TOPK:
                        if k >= len(Rs):
                            continue
                        sa, sb = topk(Rs, k), topk(Ps, k)
                        inter = len(sa & sb)
                        tk_rows.append({
                            "variant": var, "beta": beta, "subset": ss, "proxy": p, "k": k,
                            "n_top_inter": inter, "overlap_ratio": round(inter / k, 4),
                            "jaccard": round(inter / (2 * k - inter), 4),
                            "random_expect_overlap_ratio": round(k / len(Rs), 4)})
    sp_df = pd.DataFrame(sp_rows)
    sp_df.to_csv(os.path.join(OUT, "proxy_vs_sir_spearman.csv"), index=False, encoding="utf-8-sig")
    tk_df = pd.DataFrame(tk_rows)
    tk_df.to_csv(os.path.join(OUT, "proxy_vs_sir_topk.csv"), index=False, encoding="utf-8-sig")
    record("N_SPEARMAN_CELLS", len(sp_df))                   # @num N_SPEARMAN_CELLS

    # ----------------------------------------------------- Kitsak 判决（主变体主口径）
    log("step6 kitsak verdict")
    vd = []
    for bi, beta in enumerate(BETAS):
        s = sp_df[(sp_df["variant"] == "unweighted") & (sp_df["beta"] == beta) &
                  (sp_df["subset"] == "alive_attr509")]
        if not len(s):
            continue
        s = s.sort_values("spearman_rho", ascending=False)
        g = lambda p: float(s.loc[s["proxy"] == p, "spearman_rho"].iloc[0])
        R = Rm[(0, bi)].mean(1)[mask["alive_attr509"]]
        col = {p: cen[p].to_numpy()[mask["alive_attr509"]] for p in PROXIES}
        ov = {p: round(len(topk(R, 20) & topk(col[p], 20)) / 20, 3) for p in PROXIES}
        vd.append({
            "beta": beta, "gamma": GAMMA, "subset": "alive_attr509",
            "n_distinct_R": int(np.unique(np.round(R, 4)).size),
            "best_proxy": s.iloc[0]["proxy"], "best_rho": round(float(s.iloc[0]["spearman_rho"]), 4),
            "rho_kcore": round(g("kcore"), 4), "rho_deg_u": round(g("deg_u"), 4),
            "rho_bet_u": round(g("bet_u"), 4), "rho_pr_dw": round(g("pr_dw"), 4),
            "rho_in_degree": round(g("in_degree"), 4), "rho_out_degree": round(g("out_degree"), 4),
            "kcore_beats_deg_spearman": bool(g("kcore") > g("deg_u")),
            "gap_kcore_minus_deg": round(g("kcore") - g("deg_u"), 4),
            "top20_kcore": ov["kcore"], "top20_deg_u": ov["deg_u"], "top20_bet_u": ov["bet_u"],
            "top20_pr_dw": ov["pr_dw"],
            "top20_kcore_beats_deg": bool(ov["kcore"] > ov["deg_u"]),
            "partial_kcore_ctrl_deg": round(partial_spearman(col["kcore"], col["deg_u"], R), 4),
            "partial_deg_ctrl_kcore": round(partial_spearman(col["deg_u"], col["kcore"], R), 4),
            "partial_kcore_ctrl_outdeg": round(partial_spearman(col["kcore"], col["out_degree"], R), 4),
            "partial_deg_ctrl_outdeg": round(partial_spearman(col["deg_u"], col["out_degree"], R), 4),
            "partial_bet_ctrl_outdeg": round(partial_spearman(col["bet_u"], col["out_degree"], R), 4),
            "partial_pr_ctrl_outdeg": round(partial_spearman(col["pr_dw"], col["out_degree"], R), 4)})
    vdf = pd.DataFrame(vd)
    vdf.to_csv(os.path.join(OUT, "kitsak_verdict.csv"), index=False, encoding="utf-8-sig")
    record("VERDICT_ROWS", len(vdf))                         # @num VERDICT_ROWS
    record("N_BETA_KCORE_BEATS_DEG",                         # @num N_BETA_KCORE_BEATS_DEG
           int(vdf["kcore_beats_deg_spearman"].sum()))
    record("BETAS_KCORE_BEATS_DEG",
           [float(x) for x in vdf.loc[vdf["kcore_beats_deg_spearman"], "beta"]])  # @num BETAS_KCORE_BEATS_DEG
    record("N_BETA_TOP20_KCORE_BEATS_DEG",                   # @num N_BETA_TOP20_KCORE_BEATS_DEG
           int(vdf["top20_kcore_beats_deg"].sum()))
    record("BEST_PROXY_BY_BETA", {float(r["beta"]): r["best_proxy"] for _, r in vdf.iterrows()})  # @num BEST_PROXY_BY_BETA
    record("PARTIAL_KCORE_RANGE",                             # @num PARTIAL_KCORE_RANGE
           [float(vdf["partial_kcore_ctrl_deg"].min()), float(vdf["partial_kcore_ctrl_deg"].max())])

    # -------------------------------------------------------------- 节点汇总宽表
    log("step7 node summary wide table")
    w = {"node": rec_nodes, "det_reach_size": det}
    for vi, var in enumerate(VARIANTS):
        for bi, beta in enumerate(BETAS):
            w["R_%s_b%05.2f" % (var[:2], beta)] = np.round(Rm[(vi, bi)].mean(1), 3)
    for p in PROXIES + ["pr_uu", "hits_hub", "hits_auth"]:
        w[p] = cen[p].to_numpy()
    summ = pd.DataFrame(w)
    key = nodes_df.set_index("node")
    summ["ok_ratio_pct"] = summ["node"].map(key["ok_ratio_pct"])
    summ["health_state"] = summ["node"].map(key["health_state"])
    summ["true_fail"] = summ["node"].map(key["true_fail"])
    summ["is_dead"] = summ["node"].map(key["is_dead"])
    MID_BETA = 0.12
    mid_bi = BETAS.index(MID_BETA)
    Rmid = Rm[(0, mid_bi)].mean(1)
    summ["R_mean_primary_b%.2f" % MID_BETA] = np.round(Rmid, 3)
    summ["rank_by_R_b%.2f" % MID_BETA] = sps.rankdata(-Rmid, method="min").astype(int)
    for p in ["kcore", "deg_u", "bet_u", "pr_dw"]:
        summ["rank_by_" + p] = sps.rankdata(-cen[p].to_numpy(), method="min").astype(int)
    summ["subset_primary"] = mask["alive_attr509"]
    summ.to_csv(os.path.join(OUT, "sir_node_summary.csv"), index=False, encoding="utf-8-sig")
    record("MID_BETA_USED", MID_BETA)                        # @num MID_BETA_USED

    # --------------------------------------------------- 主口径 Top-20 传播者名单
    log("step8 top20 spreaders")
    mm = mask["alive_attr509"]
    names = np.array(rec_nodes, dtype=object)
    Rsub = np.where(mm, Rmid, -1.0)
    order = np.lexsort((np.arange(n), -Rsub))
    top20 = [names[i] for i in order[:20]]
    rec_top = []
    for nd in top20:
        i = int(idx[nd])
        row = {"node": nd, "R_at_each_beta(unweighted)":
               [round(float(Rm[(0, bi)].mean(1)[i]), 2) for bi in range(len(BETAS))]}
        row.update({"kcore": int(cen["kcore"].iloc[i]), "deg_u": int(cen["deg_u"].iloc[i]),
                    "in_degree": int(cen["in_degree"].iloc[i]),
                    "out_degree": int(cen["out_degree"].iloc[i]),
                    "bet_u": round(float(cen["bet_u"].iloc[i]), 6),
                    "pr_dw": round(float(cen["pr_dw"].iloc[i]), 6),
                    "det_reach_size": int(det[i]),
                    "ok_ratio_pct": float(key["ok_ratio_pct"].get(nd, float("nan")))})
        rec_top.append(row)
    pd.DataFrame(rec_top).to_csv(os.path.join(OUT, "top20_spreaders.csv"),
                                 index=False, encoding="utf-8-sig")
    record("TOP20_FIRST", str(top20[0]))                      # @num TOP20_FIRST
    record("TOP20_NODES", [str(x) for x in top20])            # @num TOP20_NODES
    # 主口径 Top-20 与各代理 Top-20 的实际成员（用于人工核对）
    top_by_proxy = {p: [str(names[j]) for j in topk(np.where(mm, cen[p].to_numpy(), -1e9), 20)]
                    for p in ("kcore", "deg_u", "bet_u", "pr_dw")}
    record("TOP20_BY_PROXY", top_by_proxy)                    # @num TOP20_BY_PROXY

    # -------------------------------------- 删失对照：671 口径 vs 509 口径（结论会翻）
    log("step9 censoring contrast 671 vs 509")
    contrast = []
    for bi, beta in enumerate(BETAS):
        R = Rm[(0, bi)].mean(1)
        row = {"beta": beta}
        for ss in ("alive_attr509", "attr671"):
            m2 = mask[ss]
            for p in ("kcore", "deg_u", "bet_u", "pr_dw"):
                row["rho__%s__%s" % (p, ss)] = round(float(
                    sps.spearmanr(cen[p].to_numpy()[m2], R[m2]).statistic), 4)
        contrast.append(row)
    cdf = pd.DataFrame(contrast)
    cdf.to_csv(os.path.join(OUT, "censoring_contrast_671_vs_509.csv"),
               index=False, encoding="utf-8-sig")
    record("CENSOR_RHO_KCORE_B06_509",                        # @num CENSOR_RHO_KCORE_B06_509
           float(cdf.loc[cdf.beta == 0.06, "rho__kcore__alive_attr509"].iloc[0]))
    record("CENSOR_RHO_KCORE_B06_671",                        # @num CENSOR_RHO_KCORE_B06_671
           float(cdf.loc[cdf.beta == 0.06, "rho__kcore__attr671"].iloc[0]))
    record("CENSOR_RHO_DEGU_B06_509",                         # @num CENSOR_RHO_DEGU_B06_509
           float(cdf.loc[cdf.beta == 0.06, "rho__deg_u__alive_attr509"].iloc[0]))
    record("CENSOR_RHO_DEGU_B06_671",                         # @num CENSOR_RHO_DEGU_B06_671
           float(cdf.loc[cdf.beta == 0.06, "rho__deg_u__attr671"].iloc[0]))

    # ------------------------------------------------------ 加权 vs 无权变体对照
    log("step10 weighted vs unweighted variant")
    vw = []
    for bi, beta in enumerate(BETAS):
        Ru = Rm[(0, bi)].mean(1)[mm]
        Rw = Rm[(1, bi)].mean(1)[mm]
        vw.append({"beta": beta,
                   "R_mean_unweighted": round(float(Ru.mean()), 3),
                   "R_mean_weighted": round(float(Rw.mean()), 3),
                   "ratio_weighted_over_unweighted": round(float(Rw.mean() / Ru.mean()), 4),
                   "spearman_variants": round(float(sps.spearmanr(Ru, Rw).statistic), 4)})
    vwdf = pd.DataFrame(vw)
    vwdf.to_csv(os.path.join(OUT, "variant_weighted_vs_unweighted.csv"),
                index=False, encoding="utf-8-sig")
    record("VARIANT_CONCORDANCE_MIN", float(vwdf["spearman_variants"].min()))  # @num VARIANT_CONCORDANCE_MIN
    record("VARIANT_CONCORDANCE_MAX", float(vwdf["spearman_variants"].max()))  # @num VARIANT_CONCORDANCE_MAX
    record("RATIO_W_OVER_U_RANGE",                                  # @num RATIO_W_OVER_U_RANGE
           [float(vwdf["ratio_weighted_over_unweighted"].min()),
            float(vwdf["ratio_weighted_over_unweighted"].max())])

    # ------------------------------------------------------------------- 落盘
    prov = pd.DataFrame(_PROV)
    prov["value"] = prov["value"].astype(str)
    prov.to_csv(os.path.join(OUT, "_number_provenance.csv"), index=False, encoding="utf-8-sig")
    total = time.time() - wall0
    key_numbers = {
        "module": MODULE, "kind": "scenario_simulation_NOT_measurement",
        "seed": SEED, "gamma": GAMMA, "betas": BETAS, "n_trials": N_TRIALS,
        "n_boot": N_BOOT, "topk": TOPK, "variants": VARIANTS,
        "primary_subset": "alive_attr509", "mid_beta_for_roster": MID_BETA,
        "wall_seconds": round(total, 1), "sir_wall_seconds": round(sir_wall, 1),
        "provenance": {r["tag"]: {"cite": r["cite"], "value": r["value"]} for r in _PROV},
        "verdict_table": vdf.to_dict(orient="records"),
        "validation": chk,
        "epidemic_size": ep, "variant_contrast": vw,
        "top20_spreaders": rec_top, "censoring_contrast": cdf.to_dict(orient="records"),
        "versions": C.versions(),
    }
    C.save_json(key_numbers, MODULE, "key_numbers.json")

    run_lines = [
        "C4_spreaders RUN record",
        "start_local_time: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(wall0)),
        "end_local_time  : " + time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_wall_seconds: %.1f" % total,
        "sir_wall_seconds: %.1f" % sir_wall,
        "command line: cd C:/Users/bfy32/Documents/Qoder/2026-11-13/f93e4d48/pipeline && "
        "PYTHONHASHSEED=0 python out/C4_spreaders/run_c4.py",
        "versions: " + json.dumps(C.versions(), ensure_ascii=False),
        "SEED=%d ; csa_common.SEED=%d" % (SEED, C.SEED),
        "anchors: C.assert_anchors() -> 31/31 PASS (script aborts on any mismatch)",
        "graph: C.build_all(); Gd 676 nodes / 2788 directed edges; no re-implementation",
        "SIR model: discrete-time on DIRECTED Gd; gamma=%.2f; betas=%s; %d independent trials "
        "per (node, beta, variant)" % (GAMMA, BETAS, N_TRIALS),
        "SIR variant primary = UNWEIGHTED (each directed edge crossed w.p. beta). "
        "Weighted variant (p=beta*w/w_max_source) run only as a sensitivity check.",
        "seeds: explicit SeedSequence entropy [SEED, node_i, beta_i, trial, variant]; "
        "bootstrap entropy [SEED, 7777, n_nodes]",
        "trials run: %d ; SIR trajectories: %d" % (sum(a.size for a in Rm.values()), len(Rm) * n),
        "primary subset: alive & attributed (%d nodes); 671 / 676 reported as contrast"
        % int(mask["alive_attr509"].sum()),
        "note: R counts the seed node; R==1 means the process died out immediately.",
        "note: %d dead sites have out-degree 0 and are therefore FORCED to R==1 at every beta "
        "(censoring, CONVENTIONS 2.1) - hence the alive-only primary subset."
        % int(sum(1 for d in dead if outd.get(d, 0) == 0)),
        "note: this is a SCENARIO SIMULATION. No temporal / traffic / propagation data exists "
        "in this dataset; conclusions are model-dependent by construction.",
        "note: edge weights are 'observed link counts' and are NOT interpreted as traffic.",
    ] + _LOG
    C.write_run_note(MODULE, run_lines)
    log("done %.1fs -> %s" % (total, OUT))


if __name__ == "__main__":
    main()
