# -*- coding: utf-8 -*-
"""C5_functional_roles — Guimerà & Amaral (2005) z–P 功能角色 + 社区划分口径误差棒

任务书（§C5）
  §1 在 **canonical 口径**（加权 Gu 上做 Louvain，seed=42 → 25 社区 / Q=0.6904 /
     最大社区 87，即 CONVENTIONS §1 的三个 Louvain 锚点）计算
       z_i = (k_i^in − <k^in>_{s_i}) / sd_{s_i}(k^in)     模块内度 z-score
       P_i = 1 − Σ_s (k_i^s / k_i)^2                       参与系数
     并按 GA05 七类角色分类（阈值 z*=2.5、P*=0.62、P**=0.80；网络枢纽需模块产出 Y>平均）。
  §2 再取 **11 个其它 seed**（合计 12 个划分）重算，对每个节点报告
     "在该角色类中出现的 seed 占比" —— 这就是任务书要求的误差棒。
  §3 判决：某角色类若有 **>30% 的节点会翻**，该类标注"**口径依赖，不可作为名单结论**"。
  §4 每类给出代表节点 + 其稳定比例。

口径声明（全部显式，可复核）
  - 社区划分：`C.louvain(Gu, seed=s)`（加权 Louvain；权重语义 = "观察到的链接次数"）。
    主口径 seed=42；其它 11 个 seed = [20261113, 1, 2, …, 10]。
  - **z/P 用的图**：主口径 = **无权无向 Gb**（GA05 原文用二值邻接，k 为"链路条数"）；
    另跑一版 **加权 Gu**（k 换成强度）作对照，量化"用不用权重"对角色名单的影响。
  - 节点集：主表 671 attributed（CONVENTIONS §1）；5 个悬空节点参与社区构建但不进主表。
  - 社区数同时报 `n_comms_total` 与 `n_comms_nonsingleton`（CONVENTIONS §3）。
  - 禁用 nodes_full 的 `degree` / `color`；度全部从图重算。
  - 因果措辞禁用（CONVENTIONS §7）："角色"只描述结构位置，不是功能/重要性/健康判断。
  - SEED = 20261113；Louvain seed 全部显式列出。

继承的不确定性（必须写进结论，不能报单值）
  本图的加权 Louvain 对 **seed 和节点插入顺序都敏感**：20 seed 下社区数 23–26、
  Q 0.6895–0.6945、**最大社区 80–113**；仅改插入顺序（边集不变）最大社区 87→84。
  z 的分母是"本模块内部度的均值/标准差"，P 依赖"模块编号集合"，
  两者**直接继承**上述不确定性 → 本模块的所有角色数字都带 seed 误差棒。

运行：  cd pipeline && PYTHONHASHSEED=0 python out/C5_functional_roles/run_c5.py
产出：  zp_roles_canonical.csv, zp_distribution_canonical.csv, zp_by_seed.csv,
        role_counts_by_seed.csv, role_stability_by_node.csv, role_class_stability.csv,
        representatives.csv, weighted_vs_binary_roles.csv, seed_partition_summary.csv,
        module_size_stability.csv, module_match_jaccard.csv, role_health_summary.csv,
        key_numbers.json, _number_provenance.csv, RUN.txt
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
assert SEED == C.SEED
MODULE = "C5_functional_roles"
OUT = C.module_dir(MODULE)

# ------------------------------------------------------- 阈值（GA05 惯例，跑前设定）
Z_STAR = 2.5                       # 模块内度阈值 z*
P_STAR = 0.62                      # connector 下界 P*
P_DOUBLE_STAR = 0.80               # kinless 下界 P**
FLIP_TOL = 0.30                    # 任务书硬阈值：>30% 节点翻转 => 名单不可用
CANON_SEED = 42                    # canonical 社区划分（CONVENTIONS §1 锚点）
EXTRA_SEEDS = [SEED] + list(range(1, 11))   # 11 个"其它 seed"（任务要求 ≥10）
ALL_SEEDS = [CANON_SEED] + EXTRA_SEEDS      # 合计 12 个划分

ROLE_ORDER = ["peripheral", "non-hub_connector", "non-hub_kinless",
              "provincial_hub", "connector_hub", "kinless_hub", "network_hub"]
ROLE_CN = {
    "peripheral": "外围节点 (peripheral)",
    "non-hub_connector": "非枢纽连接者 (non-hub connector)",
    "non-hub_kinless": "非枢纽无亲属 (non-hub kinless)",
    "provincial_hub": "行省枢纽 (provincial hub)",
    "connector_hub": "连接枢纽 (connector hub)",
    "kinless_hub": "无亲属枢纽 (kinless hub)",
    "network_hub": "网络枢纽 (network hub)",
}
HUB_ROLES = ["provincial_hub", "connector_hub", "kinless_hub", "network_hub"]

_T0 = time.perf_counter()
_LOG = []


def log(msg):
    line = "[%7.2fs] %s" % (time.perf_counter() - _T0, msg)
    print(line, flush=True)
    _LOG.append(line)


# ==================================================== 数字溯源（result.md 引用行号）
_PROV = []
_TAG_RE = re.compile(r"#\s*@num\s+([A-Za-z0-9_]+)")
_SRC = open(os.path.join(_HERE, "run_c5.py"), encoding="utf-8").read().splitlines()
_TAG_AT = {}
for _i, _l in enumerate(_SRC, start=1):
    _m = _TAG_RE.search(_l)
    if _m:
        _TAG_AT.setdefault(_m.group(1), _i)


def record(tag, value):
    ln = _TAG_AT.get(tag)
    if ln is None:
        raise KeyError("missing @num marker for tag: %s" % tag)
    if isinstance(value, np.integer):
        value = int(value)
    elif isinstance(value, np.floating):
        value = float(value)
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    _PROV.append({"tag": tag, "cite": "run_c5.py:L%d" % ln, "value": value})
    return value


# ==================================================== z / P / 角色
def zpart(G, part, weighted=False):
    """Guimerà–Amaral 的 z 与 P + 模块产出 Y。

    返回与 G 中"被划分覆盖的节点"对齐的：
      z, P, P_norm, k_in, k_total, mod_of, yld, by_mod
    weighted=False → k = 链路条数（二值无向图）
    weighted=True  → k = 强度（同社区多条反向边权重已相加，见 csa_common.build_undirected）
    """
    mod_of = {}
    for ci, c in enumerate(part):
        for nd in c:
            mod_of[nd] = ci
    M = len(part)
    k_of, k_in_of, dist = {}, {}, {}
    for nd in G.nodes():
        if nd not in mod_of:
            continue
        cnt = {}
        kk = 0.0
        for x, d in G[nd].items():
            if x not in mod_of:
                continue
            w = float(d.get("weight", 1)) if weighted else 1.0
            cnt[mod_of[x]] = cnt.get(mod_of[x], 0.0) + w
            kk += w
        k_of[nd] = kk
        k_in_of[nd] = cnt.get(mod_of[nd], 0.0)
        dist[nd] = cnt
    by_mod = {}
    for nd, m in mod_of.items():
        if nd in k_in_of:
            by_mod.setdefault(m, []).append(k_in_of[nd])
    z = {}
    for nd, m in mod_of.items():
        if nd not in k_in_of:
            continue
        v = by_mod[m]
        mu = float(np.mean(v)); sd = float(np.std(v))       # 总体 sd（模块全体为总体）
        z[nd] = (k_in_of[nd] - mu) / sd if sd > 0 else 0.0
    P, Pn, yld = {}, {}, {}
    for nd, cnt in dist.items():
        kk = k_of[nd]
        if kk <= 0:
            P[nd] = 0.0; Pn[nd] = 0.0; yld[nd] = 0.0
            continue
        p = 1.0 - sum((v / kk) ** 2 for v in cnt.values())
        P[nd] = p
        Pn[nd] = p / (1.0 - 1.0 / M) if M > 1 else 0.0
        n_s = len(by_mod.get(mod_of[nd], [1]))
        yld[nd] = (k_in_of[nd] / kk) / n_s if n_s else 0.0
    return z, P, Pn, k_in_of, k_of, mod_of, yld, by_mod


def classify(z, P, yld, ybar):
    """GA05 七类判定。先按 z* 分枢纽/非枢纽，再按 P*/P** 分档，最后按 Y 升级为网络枢纽。"""
    r = {}
    for nd in z:
        zi, pi = z[nd], P[nd]
        if zi <= Z_STAR:
            cls = ("peripheral" if pi <= P_STAR else
                   "non-hub_connector" if pi <= P_DOUBLE_STAR else "non-hub_kinless")
        else:
            cls = ("provincial_hub" if pi <= P_STAR else
                   "connector_hub" if pi <= P_DOUBLE_STAR else "kinless_hub")
            if pi > P_STAR and nd in yld and yld[nd] > ybar:
                cls = "network_hub"
        r[nd] = cls
    return r


# ==================================================== 主流程
def main():
    wall0 = time.time()
    log("step0 build_all + assert_anchors")
    b = C.build_all()
    C.assert_anchors(b)                                          # @num ANCHORS
    record("ANCHORS", "31/31 PASS")
    nodes_df, Gd, Gu, Gb = b["nodes"], b["Gd"], b["Gu"], b["Gb"]
    attr = list(nodes_df["node"])
    attr_set = set(attr)
    nkey = nodes_df.set_index("node")
    record("N_ATTRIBUTED", len(attr))                            # @num N_ATTRIBUTED
    record("N_GU", Gu.number_of_nodes())                         # @num N_GU
    record("N_GU_EDGES", Gu.number_of_edges())                   # @num N_GU_EDGES
    record("N_GB_EDGES", Gb.number_of_edges())                   # @num N_GB_EDGES
    record("Z_STAR", Z_STAR)                                     # @num Z_STAR
    record("P_STAR", P_STAR)                                     # @num P_STAR
    record("P_DOUBLE_STAR", P_DOUBLE_STAR)                       # @num P_DOUBLE_STAR
    record("FLIP_TOL", FLIP_TOL)                                 # @num FLIP_TOL
    record("CANON_SEED", CANON_SEED)                             # @num CANON_SEED
    record("EXTRA_SEEDS", EXTRA_SEEDS)                           # @num EXTRA_SEEDS
    record("N_SEEDS_TOTAL", len(ALL_SEEDS))                      # @num N_SEEDS_TOTAL

    # ------------------------------------------------ 12 个社区划分
    log("step1 louvain x %d seeds on weighted Gu" % len(ALL_SEEDS))
    t_l = time.perf_counter()
    parts, sumrows = {}, []
    for s in ALL_SEEDS:
        r = C.louvain(Gu, seed=s)
        parts[s] = r["partition"]
        sumrows.append({"seed": s, "is_canonical": s == CANON_SEED,
                        "n_comms_total": r["n_comms_total"],
                        "n_comms_nonsingleton": r["n_comms_nonsingleton"],
                        "n_singleton": r["n_singleton"], "Q": round(r["Q"], 5),
                        "largest": r["largest"], "n_covered": r["n_covered"]})
    sdf = pd.DataFrame(sumrows)
    sdf.to_csv(os.path.join(OUT, "seed_partition_summary.csv"), index=False, encoding="utf-8-sig")
    record("LOUVAIN_WALL_SECONDS", round(time.perf_counter() - t_l, 2))          # @num LOUVAIN_WALL_SECONDS
    cr = sdf.loc[sdf.seed == CANON_SEED].iloc[0]
    record("CANON_COMMS", int(cr.n_comms_total))                                 # @num CANON_COMMS
    record("CANON_COMMS_NONSINGLETON", int(cr.n_comms_nonsingleton))             # @num CANON_COMMS_NONSINGLETON
    record("CANON_Q", float(cr.Q))                                               # @num CANON_Q
    record("CANON_LARGEST", int(cr.largest))                                     # @num CANON_LARGEST
    record("COMMS_RANGE", [int(sdf.n_comms_total.min()), int(sdf.n_comms_total.max())])   # @num COMMS_RANGE
    record("Q_RANGE", [float(sdf.Q.min()), float(sdf.Q.max())])                            # @num Q_RANGE
    record("LARGEST_RANGE", [int(sdf.largest.min()), int(sdf.largest.max())])               # @num LARGEST_RANGE
    record("COMMS_SD", round(float(sdf.n_comms_total.std(ddof=1)), 3))                       # @num COMMS_SD
    record("LARGEST_SD", round(float(sdf.largest.std(ddof=1)), 3))                           # @num LARGEST_SD

    # ------------------------------ canonical 主口径（二值 Gb + seed=42）
    log("step2 z/P on canonical partition (binary Gb)")
    cp = parts[CANON_SEED]
    zc, Pc, Pnc, kinc, ktot, modc, ydc, bym = zpart(Gb, cp, weighted=False)
    ybar = float(np.mean(list(ydc.values())))
    record("MEAN_MODULE_YIELD", round(ybar, 8))                                  # @num MEAN_MODULE_YIELD
    rolec = classify(zc, Pc, ydc, ybar)
    canon = pd.DataFrame({"node": attr})
    canon["module_id"] = canon["node"].map(modc)
    canon["module_size"] = canon["module_id"].map({m: len(v) for m, v in bym.items()})
    canon["k_binary"] = canon["node"].map(ktot)
    canon["k_in_module"] = canon["node"].map(kinc)
    canon["z"] = np.round(canon["node"].map(zc).astype(float), 4)
    canon["P"] = np.round(canon["node"].map(Pc).astype(float), 4)
    canon["P_normalized"] = np.round(canon["node"].map(Pnc).astype(float), 4)
    canon["module_yield"] = np.round(canon["node"].map(ydc).astype(float), 8)
    canon["role"] = canon["node"].map(rolec)
    canon["role_cn"] = canon["role"].map(ROLE_CN)
    for c in ["in_degree", "out_degree", "degree_u", "ok_ratio_pct", "health_state", "true_fail"]:
        canon[c] = canon["node"].map(nkey[c])
    canon.to_csv(os.path.join(OUT, "zp_roles_canonical.csv"), index=False, encoding="utf-8-sig")
    record("CANON_N_ZP_NODES", len(canon))                                        # @num CANON_N_ZP_NODES
    cnt = canon["role"].value_counts().reindex(ROLE_ORDER).fillna(0).astype(int)
    record("CANON_ROLE_COUNTS", cnt.to_dict())                                    # @num CANON_ROLE_COUNTS
    record("N_Z_OVER_STAR", int((canon["z"] > Z_STAR).sum()))                     # @num N_Z_OVER_STAR
    record("N_P_OVER_STAR", int((canon["P"] > P_STAR).sum()))                     # @num N_P_OVER_STAR
    record("N_CONNECTOR_NONHUB", int(((canon["P"] > P_STAR) & (canon["z"] <= Z_STAR)).sum()))  # @num N_CONNECTOR_NONHUB
    record("Z_PERCENTILES", [float(x) for x in np.percentile(canon["z"], [50, 90, 95, 99, 100])])   # @num Z_PERCENTILES
    record("P_PERCENTILES", [float(x) for x in np.percentile(canon["P"], [50, 90, 95, 99, 100])])   # @num P_PERCENTILES
    record("P_MAX", float(canon["P"].max()))                                      # @num P_MAX
    record("Z_MAX", float(canon["z"].max()))                                      # @num Z_MAX
    record("N_K_IN_SINGLE_MODULE_NODES", int((canon["P"] == 0).sum()))            # @num N_K_IN_SINGLE_MODULE_NODES
    pd.DataFrame([{"stat": "z", "thresh": Z_STAR, "n_above": int((canon["z"] > Z_STAR).sum()),
                   **{f"p{q}": float(np.percentile(canon["z"], q)) for q in (50, 90, 95, 99, 100)}},
                  {"stat": "P", "thresh": P_STAR, "n_above": int((canon["P"] > P_STAR).sum()),
                   **{f"p{q}": float(np.percentile(canon["P"], q)) for q in (50, 90, 95, 99, 100)}}]
                 ).to_csv(os.path.join(OUT, "zp_distribution_canonical.csv"),
                          index=False, encoding="utf-8-sig")

    # ---------------------------- 12 seed 的 z/P/角色（误差棒核心）
    log("step3 z/P for all %d seeds (binary Gb)" % len(ALL_SEEDS))
    per_seed_role, mod_by_seed, zp_rows, rc_rows = {}, {}, [], []
    for s in ALL_SEEDS:
        z, P, Pn, kin, kt, mod, yd, bymm = zpart(Gb, parts[s], weighted=False)
        yb = float(np.mean(list(yd.values())))
        rl = classify(z, P, yd, yb)
        per_seed_role[s] = rl
        mod_by_seed[s] = mod
        v = pd.Series([rl[x] for x in attr if x in rl]).value_counts()
        row = {"seed": s, "is_canonical": s == CANON_SEED, "mean_Y": round(yb, 8)}
        for r_ in ROLE_ORDER:
            row[r_] = int(v.get(r_, 0))
        row["n_nodes_scored"] = int(sum(row[r_] for r_ in ROLE_ORDER))
        rc_rows.append(row)
        for nd in attr:
            if nd not in z:
                continue
            zp_rows.append({"seed": s, "node": nd, "module_id": mod.get(nd),
                            "module_size": len(bymm.get(mod.get(nd), [])),
                            "z": round(z[nd], 4), "P": round(P[nd], 4),
                            "module_yield": round(yd.get(nd, 0.0), 8),
                            "role": rl[nd], "role_cn": ROLE_CN[rl[nd]]})
    zp_all = pd.DataFrame(zp_rows)
    zp_all.to_csv(os.path.join(OUT, "zp_by_seed.csv"), index=False, encoding="utf-8-sig")
    record("ZP_BY_SEED_ROWS", len(zp_all))                                        # @num ZP_BY_SEED_ROWS
    record("ZP_NODES_PER_SEED", int(zp_all.groupby("seed").size().max()))          # @num ZP_NODES_PER_SEED
    rcd = pd.DataFrame(rc_rows)
    rcd.to_csv(os.path.join(OUT, "role_counts_by_seed.csv"), index=False, encoding="utf-8-sig")
    record("ROLE_COUNTS_BY_SEED", {r_: [int(x) for x in rcd[r_]] for r_ in ROLE_ORDER})  # @num ROLE_COUNTS_BY_SEED
    record("ROLE_RANGE_BY_SEED", {r_: [int(rcd[r_].min()), int(rcd[r_].max())] for r_ in ROLE_ORDER})  # @num ROLE_RANGE_BY_SEED
    record("ROLE_CV_BY_SEED", {r_: round(float(rcd[r_].std(ddof=1) / max(rcd[r_].mean(), 1e-9)), 3)
                               for r_ in ROLE_ORDER})                             # @num ROLE_CV_BY_SEED

    # ----------------------- 每节点跨 seed 角色占比 = 任务书要求的误差棒
    log("step4 per-node role stability across %d seeds" % len(ALL_SEEDS))
    stab_rows = []
    for nd in attr:
        labs = [per_seed_role[s][nd] for s in ALL_SEEDS if nd in per_seed_role[s]]
        if not labs:
            continue
        vc = pd.Series(labs).value_counts()
        k = len(labs)
        rec = {"node": nd, "n_seeds_scored": k,
               "canonical_role": rolec[nd], "canonical_role_cn": ROLE_CN[rolec[nd]]}
        for r_ in ROLE_ORDER:
            rec["frac_" + r_] = round(float(vc.get(r_, 0)) / k, 4)
        rec["modal_role"] = vc.index[0]
        rec["modal_role_cn"] = ROLE_CN[vc.index[0]]
        rec["modal_frac"] = round(float(vc.iloc[0]) / k, 4)
        rec["n_distinct_roles_across_seeds"] = int(vc.size)
        rec["canonical_kept_frac"] = round(float(vc.get(rolec[nd], 0)) / k, 4)
        rec["stable_ge_70pct"] = bool(rec["canonical_kept_frac"] >= 1 - FLIP_TOL)
        rec["agrees_with_modal"] = bool(rolec[nd] == vc.index[0])
        rec["z_canonical"] = round(zc[nd], 4)
        rec["P_canonical"] = round(Pc[nd], 4)
        rec["k_binary"] = ktot[nd]
        rec["module_id_canonical"] = modc[nd]
        rec["module_size_canonical"] = len(bym.get(modc[nd], []))
        rec["health_state"] = nkey["health_state"].get(nd)
        rec["ok_ratio_pct"] = nkey["ok_ratio_pct"].get(nd)
        rec["in_degree"] = nkey["in_degree"].get(nd)
        stab_rows.append(rec)
    st = pd.DataFrame(stab_rows)
    st.to_csv(os.path.join(OUT, "role_stability_by_node.csv"), index=False, encoding="utf-8-sig")
    record("N_NODES_WITH_ROLE", len(st))                                          # @num N_NODES_WITH_ROLE
    record("FRAC_STABLE_GE70", round(float(st["stable_ge_70pct"].mean()), 4))       # @num FRAC_STABLE_GE70
    record("FRAC_MODAL_AGREES_CANON", round(float(st["agrees_with_modal"].mean()), 4))  # @num FRAC_MODAL_AGREES_CANON
    record("KEPT_FRAC_MEAN", round(float(st["canonical_kept_frac"].mean()), 4))      # @num KEPT_FRAC_MEAN
    record("KEPT_FRAC_MEDIAN", float(st["canonical_kept_frac"].median()))            # @num KEPT_FRAC_MEDIAN
    record("N_DISTINCT_ROLES_HIST", st["n_distinct_roles_across_seeds"].value_counts().sort_index().to_dict())  # @num N_DISTINCT_ROLES_HIST

    # ------------------- 每个角色类的"名单可用性"判决（任务书 §3 硬要求）
    log("step5 per-class roster verdict (>%.0f%% flipping => 口径依赖)" % (FLIP_TOL * 100))
    cls_rows = []
    for r_ in ROLE_ORDER:
        sub = st[st["canonical_role"] == r_]
        n = len(sub)
        if n == 0:
            cls_rows.append({"role": r_, "role_cn": ROLE_CN[r_], "n_canonical": 0,
                             "n_flipping": 0, "flip_pct": None, "kept_frac_mean": None,
                             "kept_frac_median": None, "kept_ge_half": None,
                             "verdict": "空类（canonical 划分下无成员）", "usable_as_roster": False})
            continue
        kf = sub["canonical_kept_frac"]
        flipping = int((kf < 1 - FLIP_TOL).sum())
        pct = flipping / n
        cls_rows.append({
            "role": r_, "role_cn": ROLE_CN[r_], "n_canonical": n,
            "n_flipping": flipping, "flip_pct": round(pct, 4),
            "kept_frac_mean": round(float(kf.mean()), 4),
            "kept_frac_median": float(kf.median()),
            "kept_ge_half": round(float((kf >= 0.5).mean()), 4),
            "verdict": ("口径依赖，不可作为名单结论" if pct > FLIP_TOL
                        else "名单可用（翻转节点 <=%d%%）" % int(FLIP_TOL * 100)),
            "usable_as_roster": bool(pct <= FLIP_TOL)})
    cdf = pd.DataFrame(cls_rows)
    cdf.to_csv(os.path.join(OUT, "role_class_stability.csv"), index=False, encoding="utf-8-sig")
    record("CLASS_VERDICT_TABLE", cdf.to_dict(orient="records"))                   # @num CLASS_VERDICT_TABLE
    record("N_CLASSES_UNUSABLE", int((~cdf["usable_as_roster"].astype(bool)).sum()))  # @num N_CLASSES_UNUSABLE
    record("UNUSABLE_CLASS_NAMES", [r for r, u in zip(cdf.role, cdf.usable_as_roster) if not u])  # @num UNUSABLE_CLASS_NAMES
    record("OVERALL_FLIP_GT30_PCT", round(float((st["canonical_kept_frac"] < 1 - FLIP_TOL).mean()), 4))  # @num OVERALL_FLIP_GT30_PCT
    record("OVERALL_FLIP_ANY_PCT", round(float((st["canonical_kept_frac"] < 1.0).mean()), 4))            # @num OVERALL_FLIP_ANY_PCT
    record("HUB_FLIP_GT30_PCT", round(float(
        (st[st.canonical_role.isin(HUB_ROLES)]["canonical_kept_frac"] < 1 - FLIP_TOL).mean()), 4))       # @num HUB_FLIP_GT30_PCT
    record("N_HUBS_CANON_TOTAL", int(st.canonical_role.isin(HUB_ROLES).sum()))                             # @num N_HUBS_CANON_TOTAL

    # ------------------------------------------- 每类代表节点 + 稳定比例
    log("step6 representatives")
    reps = []
    for r_ in ROLE_ORDER:
        sub = st[st["canonical_role"] == r_].copy()
        if not len(sub):
            continue
        sortcol = "z_canonical" if r_ in HUB_ROLES else "P_canonical"
        sub = sub.sort_values(["canonical_kept_frac", sortcol], ascending=False)
        for _, x in sub.head(5).iterrows():
            reps.append({"role": r_, "role_cn": ROLE_CN[r_], "node": x["node"],
                         "canonical_kept_frac": x["canonical_kept_frac"],
                         "modal_role": x["modal_role"], "modal_frac": x["modal_frac"],
                         "n_distinct_roles": x["n_distinct_roles_across_seeds"],
                         "z_canonical": x["z_canonical"], "P_canonical": x["P_canonical"],
                         "k_binary": x["k_binary"], "k_in_module": kinc[x["node"]],
                         "module_id_canonical": x["module_id_canonical"],
                         "module_size_canonical": x["module_size_canonical"],
                         "in_degree": x["in_degree"], "health_state": x["health_state"],
                         "ok_ratio_pct": x["ok_ratio_pct"]})
    rdf = pd.DataFrame(reps)
    rdf.to_csv(os.path.join(OUT, "representatives.csv"), index=False, encoding="utf-8-sig")
    record("N_REP_ROWS", len(rdf))                                                  # @num N_REP_ROWS
    record("REP_MOST_STABLE", {r_: (rdf[rdf.role == r_]["canonical_kept_frac"].max()
                                    if len(rdf[rdf.role == r_]) else None) for r_ in ROLE_ORDER})  # @num REP_MOST_STABLE

    # --------------------- 二值 vs 加权 z/P 对照（同一 canonical 划分）
    log("step7 binary vs weighted layer on canonical partition")
    zw, Pw, Pnw, kinw, ktw, modw, ydw, bymw = zpart(Gu, cp, weighted=True)
    ybw = float(np.mean(list(ydw.values())))
    rw = classify(zw, Pw, ydw, ybw)
    cmp_rows = [{"node": nd, "role_binary": rolec[nd], "role_weighted": rw[nd],
                 "z_binary": round(zc[nd], 4), "z_weighted": round(zw[nd], 4),
                 "P_binary": round(Pc[nd], 4), "P_weighted": round(Pw[nd], 4),
                 "k_binary": ktot[nd], "strength": ktw[nd],
                 "same_role": bool(rolec[nd] == rw[nd])} for nd in attr if nd in rw]
    cmpdf = pd.DataFrame(cmp_rows)
    cmpdf.to_csv(os.path.join(OUT, "weighted_vs_binary_roles.csv"), index=False, encoding="utf-8-sig")
    record("BINARY_VS_WEIGHTED_AGREE", round(float(cmpdf["same_role"].mean()), 4))   # @num BINARY_VS_WEIGHTED_AGREE
    rec_h = cmpdf[cmpdf.role_binary.isin(HUB_ROLES)]
    record("BINARY_VS_WEIGHTED_AGREE_HUBS", round(float(rec_h["same_role"].mean()), 4) if len(rec_h) else None)  # @num BINARY_VS_WEIGHTED_AGREE_HUBS
    record("N_HUBS_BINARY_CANON", int(len(rec_h)))                                   # @num N_HUBS_BINARY_CANON
    record("SPEARMAN_Z_BINARY_VS_WEIGHTED", round(float(sps.spearmanr(
        cmpdf.z_binary, cmpdf.z_weighted).statistic), 4))                            # @num SPEARMAN_Z_BINARY_VS_WEIGHTED
    record("SPEARMAN_P_BINARY_VS_WEIGHTED", round(float(sps.spearmanr(
        cmpdf.P_binary, cmpdf.P_weighted).statistic), 4))                             # @num SPEARMAN_P_BINARY_VS_WEIGHTED

    # ------------------- 模块成员跨 seed 的最好匹配 Jaccard（角色不确定性的来源）
    log("step8 module membership stability (best-match Jaccard)")
    ms = []
    for s in ALL_SEEDS:
        sizes = sorted((len(c) for c in parts[s]), reverse=True)
        ms.append({"seed": s, "is_canonical": s == CANON_SEED, "n_comms": len(parts[s]),
                   "largest": sizes[0], "median_size": float(np.median(sizes)),
                   "n_singleton": int(sum(1 for x in sizes if x == 1))})
    pd.DataFrame(ms).to_csv(os.path.join(OUT, "module_size_stability.csv"),
                            index=False, encoding="utf-8-sig")
    attr_arr = set(attr)
    jac = []
    # bym 是 {module_id: [module 内度...]}，只够算规模；成员集合要从划分本身取
    bym_sets = {m: set(c) for m, c in enumerate(cp)}
    for m, members in bym_sets.items():
        mm = set(members) & attr_arr
        if not mm:
            continue
        best = []
        for s in ALL_SEEDS:
            smap = {}
            for nd, cid in mod_by_seed[s].items():
                smap.setdefault(cid, set()).add(nd)
            bj = 0.0
            for c2 in smap.values():
                c2a = c2 & attr_arr
                d = len(mm | c2a)
                bj = max(bj, (len(mm & c2a) / d) if d else 0.0)
            best.append(bj)
        jac.append({"canonical_module_id": m, "size_all": len(set(members)),
                    "size_attr": len(mm),
                    "best_jaccard_mean": round(float(np.mean(best)), 4),
                    "best_jaccard_min": round(float(np.min(best)), 4),
                    "best_jaccard_canon_self": round(float(best[ALL_SEEDS.index(CANON_SEED)]), 4)})
    jdf = pd.DataFrame(jac)
    jdf.to_csv(os.path.join(OUT, "module_match_jaccard.csv"), index=False, encoding="utf-8-sig")
    record("N_CANON_MODULES_ATTR", len(jdf))                                         # @num N_CANON_MODULES_ATTR
    record("MODULE_JACCARD_MEAN", round(float(jdf["best_jaccard_mean"].mean()), 4))   # @num MODULE_JACCARD_MEAN
    record("MODULE_JACCARD_MIN", float(jdf["best_jaccard_min"].min()))                # @num MODULE_JACCARD_MIN
    record("MODULE_JACCARD_P10", round(float(np.percentile(jdf["best_jaccard_mean"], 10)), 4))  # @num MODULE_JACCARD_P10
    record("MODULE_JACCARD_MAX", float(jdf["best_jaccard_mean"].max()))               # @num MODULE_JACCARD_MAX
    record("N_MODULES_JACCARD_LT_05", int((jdf["best_jaccard_mean"] < 0.5).sum()))    # @num N_MODULES_JACCARD_LT_05

    # ------------------------------------------------- 角色 × 健康（描述性，不作推断）
    log("step9 role x health (descriptive)")
    hb = canon.groupby("role").agg(
        n=("node", "size"), mean_ok_ratio_pct=("ok_ratio_pct", "mean"),
        pct_dead=("health_state", lambda x: 100.0 * (x == "dead").mean()),
        pct_healthy=("health_state", lambda x: 100.0 * (x == "healthy").mean()),
        n_true_fail=("true_fail", "sum"), mean_in_degree=("in_degree", "mean"),
        mean_k_binary=("k_binary", "mean"), mean_z=("z", "mean"), mean_P=("P", "mean"),
        mean_kept_frac=("node", lambda x: float(st.loc[st.node.isin(x), "canonical_kept_frac"].mean()))
    ).reindex([r_ for r_ in ROLE_ORDER if r_ in set(canon["role"])]).round(4)
    hb.reset_index().to_csv(os.path.join(OUT, "role_health_summary.csv"),
                            index=False, encoding="utf-8-sig")
    record("ROLE_HEALTH_TABLE", hb.reset_index().to_dict(orient="records"))           # @num ROLE_HEALTH_TABLE

    # ------------------------------------------------------------- 落盘
    prov = pd.DataFrame(_PROV)
    prov["value"] = prov["value"].astype(str)
    prov.to_csv(os.path.join(OUT, "_number_provenance.csv"), index=False, encoding="utf-8-sig")
    total = time.time() - wall0
    C.save_json({"module": MODULE, "seed": SEED, "canon_seed": CANON_SEED,
                 "all_seeds": ALL_SEEDS, "z_star": Z_STAR, "p_star": P_STAR,
                 "p_double_star": P_DOUBLE_STAR, "flip_tol": FLIP_TOL,
                 "zP_graph": "binary undirected Gb (primary); weighted Gu (control)",
                 "wall_seconds": round(total, 1),
                 "provenance": {r["tag"]: {"cite": r["cite"], "value": r["value"]} for r in _PROV},
                 "partition_summary": sdf.to_dict(orient="records"),
                 "role_counts_by_seed": rc_rows, "class_stability": cls_rows,
                 "representatives": reps, "module_jaccard": jdf.to_dict(orient="records"),
                 "binary_vs_weighted": {"agree": round(float(cmpdf.same_role.mean()), 4),
                                        "agree_hubs": round(float(rec_h.same_role.mean()), 4) if len(rec_h) else None},
                 "role_health": hb.reset_index().to_dict(orient="records"),
                 "versions": C.versions()}, MODULE, "key_numbers.json")
    C.write_run_note(MODULE, [
        "C5_functional_roles RUN record",
        "start: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(wall0)),
        "end  : " + time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_wall_seconds: %.1f" % total,
        "command: cd /home/baifeiyang2026/csa_run_20261123/pipeline && "
        "PYTHONHASHSEED=0 ~/miniconda3/envs/csa/bin/python out/C5_functional_roles/run_c5.py",
        "versions: " + json.dumps(C.versions(), ensure_ascii=False),
        "SEED=%d ; csa_common.SEED=%d ; PYTHONHASHSEED=%s"
        % (SEED, C.SEED, os.environ.get("PYTHONHASHSEED")),
        "anchors: C.assert_anchors() 31/31 PASS (script aborts on any mismatch)",
        "partition: weighted-Gu Louvain via C.louvain; canonical seed=%d -> "
        "%d comms / Q=%.4f / largest %d" % (CANON_SEED, cr.n_comms_total, cr.Q, cr.largest),
        "stability seeds: %d partitions = canonical %d + %s" % (len(ALL_SEEDS), CANON_SEED, EXTRA_SEEDS),
        "z/P layer: primary = BINARY undirected Gb (GA05 convention, k = link count); "
        "control = weighted Gu (k = strength)",
        "thresholds: z*=%.1f P*=%.2f P**=%.2f; network_hub requires P>P* AND module_yield>mean(Y)"
        % (Z_STAR, P_STAR, P_DOUBLE_STAR),
        "roster rule: a class is labelled '口径依赖，不可作为名单结论' when >%d%% of its canonical "
        "members are re-assigned in >=%.0f%% of the %d seeds (kept_frac < %.2f)"
        % (int(FLIP_TOL * 100), (1 - FLIP_TOL) * 100, len(ALL_SEEDS), 1 - FLIP_TOL),
        "node set: %d attributed; the 5 dangling nodes help build communities but are not in main tables" % len(attr),
        "louvain wall: %.2fs" % (time.perf_counter() - t_l),
        "note: 'functional role' here = a position in the z-P plane. It carries no claim about "
        "site function, importance, or health (CONVENTIONS §7).",
    ] + _LOG)
    log("done %.1fs -> %s" % (total, OUT))


if __name__ == "__main__":
    main()
