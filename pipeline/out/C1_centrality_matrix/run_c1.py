# -*- coding: utf-8 -*-
"""C1_centrality_matrix — "重要性"有几个彼此一致的含义。

口径全部来自 pipeline/CONVENTIONS.md：
  §1 构图只用 csa_common.build_all()；主分析节点集 = 671 个有属性节点；开跑先 assert_anchors()。
  §2.1 入度 / 出度 / 无向总度必须分列，不得混用 → 本表 12 个"重要性"指标里三套度各自成列。
  §5 PageRank 变体分列：pr_dw = 有向加权（主口径），pr_uu = 无向无权（对照）。
  §7 只做排序/相关描述，不做任何因果表述。
  §8 SEED=20261113，所有随机过程显式传 seed。

H-index 定义（本单元自述，Hirsch 2005 的引用网络移植版，见 hindex_in_in）：
  有向图 Gd 中，指向 u 的每个入邻域都被视为一次"引用"，被引者的"影响力"用**它自己的入度**
  衡量（入度 = 收到多少条链接）。则
      h_in_in(u) = max{ h : u 至少有 h 个入邻域 x，且 in_degree(x) >= h }
  这是 Hirsch 原始定义（作者有 h 篇论文各被引 >= h 次）把"论文"换成"入邻域"、
  "该论文被引次数"换成"该入邻域的入度"的直接改写。
  备选变体（只做敏感性，不进相关族）：
      h_in_deg(u) = max{ h : u 至少有 h 个入邻域 x，且 degree_u(x) >= h }  # 用无向度当邻域影响力
  两个变体都是"只看入边"的口径，避开死站出度删失（CONVENTIONS §2.1）对 h_out_* 的污染。

输出：centrality_matrix.csv / spearman_matrix.csv / kendall_matrix.csv / pairwise_stats.csv
      / top20_membership.csv / top20_jaccard.csv / meaning_clusters.csv / c1_summary.json
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
from scipy import stats as sps
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

import csa_common as C

SEED = 20261113                      # 全局种子（本单元只有"并列打散"用到随机数）
MODULE = "C1_centrality_matrix"
TOP_K = 20                           # Top-K 榜单长度
N_TIE_PERM = 200                     # 并列打散重复次数（度量 Top-K 集合本身的可辨识性）
THETAS = [0.95, 0.90, 0.85, 0.80]    # "同一含义"的判定阈值（Spearman rho）
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
    Gu, Gd, m, sets = b["Gu"], b["Gd"], b["nodes"], b["sets"]
    ATT = list(m["node"])                       # 671 主分析节点集（有序、可复现）
    assert len(ATT) == 671 and Gu.number_of_nodes() == 676
    log(f"Gu {Gu.number_of_nodes()}/{Gu.number_of_edges()}  Gd {Gd.number_of_nodes()}/{Gd.number_of_edges()}"
        f"  attributed={len(ATT)}")


# ---------------------------------------------------------------- 2. H-index
def hindex_in_in(Gd: nx.DiGraph) -> dict:
    """主口径 H-index：h 个入邻域，每个入邻域的**入度** >= h。"""
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


def hindex_in_deg(Gd: nx.DiGraph, Gu: nx.Graph) -> dict:
    """备选口径：入邻域的**无向度** >= h。只作敏感性。"""
    degu = dict(Gu.degree())
    out = {}
    for n in Gd.nodes():
        vals = sorted((degu[x] for x in Gd.predecessors(n)), reverse=True)
        h = 0
        for i, v in enumerate(vals, start=1):
            if v >= i:
                h = i
            else:
                break
        out[n] = h
    return out


# ---------------------------------------------------------------- 3. 中心性全表
def hits_pinned(G: nx.DiGraph):
    """可复现的 HITS。

    坑（本单元实测，详见 DISSENT.md）：networkx 3.6 的 nx.hits 用
    `scipy.sparse.linalg.svds(A, k=1, v0=nstart)`，`nstart=None` 时 v0 由 ARPACK
    从 numpy 全局随机流抽取 → **同一个 csa_common.centralities() 在不同进程里
    返回逐位不同的 hits_hub/hits_auth**（实测 3 次运行 3 个不同 sha1）。
    数值差只有 ~2e-16，但 HITS 尾部有 354/676 个节点落在 |v|<1e-10 的噪声区，
    这点噪声足以改变它们的相对秩，从而把 Kendall tau 挪动 ~0.002。
    本函数只把 v0 固定成全 1（= 网络 X 的均匀初始向量），算法/参数与 csa_common 完全一致，
    因此是"可复现化"，不是换口径；主表仍调用 C.centralities 取其余 5 个指标。
    """
    h, a = nx.hits(G, max_iter=200, tol=1e-8, nstart={n: 1.0 for n in G})
    return h, a


def collapse_noise(v: pd.Series, rel: float = 1e-12) -> pd.Series:
    """把 HITS 的 |值| <= rel*max|值| 视为数值噪声，一律并成 0（真并列）。"""
    mx = float(v.abs().max())
    return v.where(v.abs() > rel * mx, 0.0)


with stage("centralities"):
    cen = C.centralities(Gu, Gd)               # deg_u bet_u pr_dw pr_uu hits_hub hits_auth kcore
    log(f"C.centralities -> {cen.shape[0]} rows x {cen.shape[1]} cols")
    h_pin, a_pin = hits_pinned(Gd)
    # 与规范实现的偏差（必须是噪声级，否则说明口径变了）
    dev_h = float((cen["hits_hub"] - pd.Series(h_pin)).abs().max())
    dev_a = float((cen["hits_auth"] - pd.Series(a_pin)).abs().max())
    assert dev_h < 1e-9 and dev_a < 1e-9, (dev_h, dev_a)
    log(f"HITS pinned (nstart=uniform): max |dev vs csa_common| hub={dev_h:.2e} auth={dev_a:.2e}")
    dd = dict(m.set_index("node")["in_degree"])
    de = dict(m.set_index("node")["out_degree"])
    hii = hindex_in_in(Gd)
    hid = hindex_in_deg(Gd, Gu)
    # 加权介数敏感性：networkx 把 weight 当"距离"，链接次数越多应越"近" → 用 1/w 作距离
    Gw = Gu.copy()
    for u, v, d in Gw.edges(data=True):
        d["dist"] = 1.0 / float(d["weight"])
    bet_w = nx.betweenness_centrality(Gw, weight="dist")

    df = cen.copy()
    df["hits_hub"] = pd.Series(h_pin)           # 同口径，固定 ARPACK 起点（见 hits_pinned）
    df["hits_auth"] = pd.Series(a_pin)
    df["in_degree"] = pd.Series(dd)
    df["out_degree"] = pd.Series(de)
    df["in_strength"] = m.set_index("node")["in_strength"]
    df["out_strength"] = m.set_index("node")["out_strength"]
    df["strength_u"] = m.set_index("node")["strength_u"]
    df["hidx_in_in"] = pd.Series(hii)
    df["hidx_in_deg"] = pd.Series(hid)
    df["bet_uw"] = pd.Series(bet_w)
    df = df.loc[ATT]
    df["title"] = m.set_index("node")["title"]
    df["health_state"] = m.set_index("node")["health_state"]
    df["is_dead"] = m.set_index("node")["is_dead"]
    df["ok_ratio_pct"] = m.set_index("node")["ok_ratio_pct"]
    df = df.reset_index()
    log(f"matrix {df.shape}")

# 12 个"重要性"指标族（进相关矩阵与 Top-K）；其余列只作描述/敏感性
MEAS = ["deg_u", "in_degree", "out_degree", "in_strength", "out_strength", "bet_u",
        "pr_dw", "pr_uu", "hits_hub", "hits_auth", "kcore", "hidx_in_in"]
DESC = {
    "deg_u": "无向度（二值，csa_common.centralities）",
    "in_degree": "有向入度（谁指向我；死站唯一可用口径）",
    "out_degree": "有向出度（我指向谁；死站恒为 0=删失）",
    "in_strength": "入强度=收到的链接次数之和",
    "out_strength": "出强度=发出的链接次数之和",
    "bet_u": "无向介数（无权，normalized）",
    "pr_dw": "PageRank 有向加权（主口径，§5）",
    "pr_uu": "PageRank 无向无权（对照口径，§5）",
    "hits_hub": "HITS 枢纽权（有向无权）",
    "hits_auth": "HITS 权威权（有向无权）",
    "kcore": "k-core 层级（无向）",
    "hidx_in_in": "H-index（入邻域×入度，本单元主口径）",
}

with stage("corr"):
    sub = df[MEAS]
    rho_s = sub.corr(method="spearman")
    rho_k = sub.corr(method="kendall")
    # 自校验：pandas spearman/kendall 与 scipy 口径一致（防止 pandas 3.0 行为漂移）
    chk = sub[["deg_u", "hidx_in_in"]]
    s_chk = sps.spearmanr(chk["deg_u"], chk["hidx_in_in"]).statistic
    k_chk = sps.kendalltau(chk["deg_u"], chk["hidx_in_in"]).statistic   # tau-b
    assert abs(s_chk - rho_s.loc["deg_u", "hidx_in_in"]) < 1e-9, (s_chk, rho_s.loc["deg_u", "hidx_in_in"])
    assert abs(k_chk - rho_k.loc["deg_u", "hidx_in_in"]) < 1e-9, (k_chk, rho_k.loc["deg_u", "hidx_in_in"])
    log(f"spearman/kendall vs scipy OK (tau-b={k_chk:.4f})")
    iu = np.triu_indices(len(MEAS), 1)
    sv = rho_s.values[iu]
    kv = rho_k.values[iu]

    # --- HITS 噪声地板敏感性：把 |v|<=1e-12*max 的尾部并成 0（真并列）再算一遍
    df_c = df.copy()
    n_noise = {}
    for c in ("hits_hub", "hits_auth"):
        cc = collapse_noise(df[c])
        n_noise[c] = {"n_below_floor": int((cc == 0).sum()),
                      "n_distinct_exact": int(df[c].nunique()),
                      "n_distinct_after_collapse": int(cc.nunique()),
                      "noise_floor_abs": float(1e-12 * df[c].abs().max())}
        df_c[c] = cc
    rho_s_c = df_c[MEAS].corr(method="spearman")
    rho_k_c = df_c[MEAS].corr(method="kendall")
    ds = (rho_s_c.values - rho_s.values)[iu]
    dk = (rho_k_c.values - rho_k.values)[iu]
    flip090 = int(((rho_s.values[iu] >= 0.90) != (rho_s_c.values[iu] >= 0.90)).sum())
    log(f"HITS noise floor: |Δrho_s| max={np.abs(ds).max():.4f} median={np.median(np.abs(ds)):.4f} "
        f"pairs crossing 0.90={flip090}")


# ---------------------------------------------------------------- 4. Top-K 与并列
def top_sets(values: pd.Series, k: int) -> set:
    """Top-k 集合：值降序，稳定排序 → 并列保持调用方传入的索引顺序。

    调用方一律传 `df.set_index("node")` 的列，而候选池 ATT（=nodes_full 的 FQDN 列）
    在 stage("top20") 开头断言为**字典升序**，所以并列 = 按 FQDN 字典序升序打破，
    跨进程/跨运行确定（C3 用同一约定）。
    """
    order = values.sort_values(ascending=False, kind="mergesort")
    return set(order.index[:k])


with stage("top20"):
    assert ATT == sorted(ATT), "并列打破依赖候选池按 FQDN 升序排列"
    dv = df.set_index("node")                       # 一律以 FQDN 为索引取榜单，避免整数行号泄漏
    assert dv.index.is_unique and dv.index.name == "node"
    det = {c: top_sets(dv[c], TOP_K) for c in MEAS}
    # 确定性字典序下的并列可辨识度：第 k 名的值上并列了多少节点
    bound = {}
    for c in MEAS:
        v = dv[c].sort_values(ascending=False, kind="mergesort")
        vk = float(v.iloc[TOP_K - 1])
        bound[c] = {
            "value_at_20": vk,
            "n_tied_at_value": int((dv[c] == vk).sum()),
            "n_strictly_above": int((dv[c] > vk).sum()),
            "identifiable_core": int((dv[c] > vk).sum()),   # 无论怎么打散都必进 Top-20 的节点数
        }
    # 并列打散：同一份值、R 次随机打破并列，看 Top-20 集合本身的抖动
    R = N_TIE_PERM
    perm_sets = {c: [] for c in MEAS}
    for r in range(R):
        rng = np.random.default_rng(SEED + r)
        jitter = pd.Series(rng.random(len(dv)), index=dv.index)
        for c in MEAS:
            j = dv[[c]].copy()
            j["j"] = jitter
            o = j.sort_values([c, "j"], ascending=[False, False], kind="mergesort").index
            perm_sets[c].append(set(o[:TOP_K]))
    tie_stab = {}
    for c in MEAS:
        inter = set.intersection(*perm_sets[c])
        union = set.union(*perm_sets[c])
        pairwise = []
        for a in range(0, R, 2):
            x, y = perm_sets[c][a], perm_sets[c][min(a + 1, R - 1)]
            pairwise.append(len(x & y) / len(x | y))
        tie_stab[c] = {
            "n_always_in_top20": len(inter),
            "n_ever_in_top20": len(union),
            "jaccard_within_permutations": round(float(np.mean(pairwise)), 4),
        }
    log(f"tie stability done, R={R}")

    # 两两 Top-20 Jaccard（确定性字典序口径）
    jac = pd.DataFrame(np.eye(len(MEAS)), index=MEAS, columns=MEAS)
    for i, a in enumerate(MEAS):
        for jj, bb in enumerate(MEAS):
            if jj <= i:
                continue
            inter = len(det[a] & det[bb])
            jac.loc[a, bb] = jac.loc[bb, a] = inter / (2 * TOP_K - inter)

    # 噪声地板是否动过榜单（预期不动：Top-20 都在 HITS 的可用量级上）
    dvc = df_c.set_index("node")
    det_c = {c: top_sets(dvc[c], TOP_K) for c in MEAS}
    top_changed = {c: len(det[c] ^ det_c[c]) // 2 for c in MEAS}
    log("Top-20 changed by noise collapse: " + str({k: v for k, v in top_changed.items() if v}))

with stage("cluster"):
    # "几个彼此一致的含义"：在 1-rho 距离上做层次聚类，多个阈值
    D = 1.0 - rho_s.values
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0
    Z = linkage(squareform(D, checks=False), method="average")
    Zc = linkage(squareform(np.maximum(1e-9, 1 - np.minimum(1, rho_s.values)), checks=False),
                 method="complete")
    cl_rows = []
    for th in THETAS:
        lab = fcluster(Z, t=1 - th, criterion="distance")
        lab_c = fcluster(Zc, t=1 - th, criterion="distance")
        cl_rows.append({
            "theta_rho": th,
            "n_clusters_average": int(len(set(lab))),
            "n_clusters_complete": int(len(set(lab_c))),
            "blocks_average": " | ".join(
                "{" + ",".join(sorted([MEAS[i] for i in range(len(MEAS)) if lab[i] == g])) + "}"
                for g in sorted(set(lab))),
        })
    cl = pd.DataFrame(cl_rows)
    lab90 = fcluster(Z, t=1 - 0.90, criterion="distance")
    lab90c = fcluster(Zc, t=1 - 0.90, criterion="distance")
    # 严格块：两两 rho>=0.90 的最大团（networkx.find_cliques，只保留规模 >=2 的）
    Gcl = nx.from_numpy_array((rho_s.values >= 0.90) & ~np.eye(len(MEAS), dtype=bool))
    Gcl.remove_nodes_from([i for i in Gcl.nodes() if Gcl.degree(i) == 0])   # 孤立柱 = 独立含义
    cliques_090 = sorted([sorted(MEAS[i] for i in c) for c in nx.find_cliques(Gcl) if len(c) >= 2],
                         key=lambda x: (-len(x), x))
    log(f"clusters@0.90 average={cl_rows[1]['n_clusters_average']} complete={cl_rows[1]['n_clusters_complete']}"
        f" cliques090={len(cliques_090)}")

# 每指标与"其余 11 个指标"的一致性汇总
with stage("summary"):
    prof = []
    for c in MEAS:
        others = [x for x in MEAS if x != c]
        prof.append({
            "measure": c,
            "spearman_min_vs_others": round(float(rho_s.loc[c, others].min()), 4),
            "spearman_median_vs_others": round(float(rho_s.loc[c, others].median()), 4),
            "spearman_max_vs_others": round(float(rho_s.loc[c, others].max()), 4),
            "kendall_min_vs_others": round(float(rho_k.loc[c, others].min()), 4),
            "kendall_median_vs_others": round(float(rho_k.loc[c, others].median()), 4),
            "kendall_max_vs_others": round(float(rho_k.loc[c, others].max()), 4),
            "top20_jaccard_mean": round(float(jac.loc[c, others].mean()), 4),
            "top20_jaccard_min": round(float(jac.loc[c, others].min()), 4),
            "n_others_rho_ge_0.9": int((rho_s.loc[c, others] >= 0.90).sum()),
            "n_others_rho_ge_0.95": int((rho_s.loc[c, others] >= 0.95).sum()),
            "top20_jaccard_ge_0.6": int((jac.loc[c, others] >= 0.60).sum()),
            "cluster_id_avg090": int(lab90[MEAS.index(c)]),
            "cluster_id_comp090": int(lab90c[MEAS.index(c)]),
            **{f"tie_{k2}": v2 for k2, v2 in tie_stab[c].items()},
            **{f"bound_{k2}": v2 for k2, v2 in bound[c].items()},
        })
    prof = pd.DataFrame(prof)

    # 一致性对：rho_s>=0.9 且 Top20 Jaccard>=0.6 才算"同一含义"的可操作判据
    pair_rows = []
    for i, a in enumerate(MEAS):
        for jj in range(i + 1, len(MEAS)):
            bb = MEAS[jj]
            inter = len(det[a] & det[bb])
            pair_rows.append({
                "m1": a, "m2": bb,
                "spearman": round(float(rho_s.loc[a, bb]), 4),
                "kendall_tau_b": round(float(rho_k.loc[a, bb]), 4),
                "top20_shared": inter, "top20_jaccard": round(inter / (2 * TOP_K - inter), 4),
                "same_meaning_rho090": bool(rho_s.loc[a, bb] >= 0.90),
                "same_meaning_both": bool(rho_s.loc[a, bb] >= 0.90 and inter / (40 - inter) >= 0.60),
            })
    pairs = pd.DataFrame(pair_rows).sort_values("spearman", ascending=False)
    n_same = int(pairs["same_meaning_rho090"].sum())
    n_same_both = int(pairs["same_meaning_both"].sum())

    # 共识名单：节点进入几个指标的 Top-20
    cnt = pd.Series(0, index=df["node"].values, dtype=int)
    for c in MEAS:
        for n in det[c]:
            cnt[n] += 1
    df["_n_top20_lists"] = df["node"].map(cnt)
    members = df[["node", "_n_top20_lists"]].sort_values(
        ["_n_top20_lists", "node"], ascending=[False, True])
    for c in MEAS:
        members[f"in_top20_{c}"] = members["node"].isin(det[c])

    hs = members[members["_n_top20_lists"] == len(MEAS)]["node"].tolist()
    ge10 = int((members["_n_top20_lists"] >= 10).sum())
    ge6 = int((members["_n_top20_lists"] >= 6).sum())
    hist = members["_n_top20_lists"].value_counts().sort_index(ascending=False)

# ---------------------------------------------------------------- 5. 落盘
with stage("save"):
    df.to_csv(os.path.join(_HERE, "centrality_matrix.csv"), index=False, encoding="utf-8-sig")
    rho_s.round(4).to_csv(os.path.join(_HERE, "spearman_matrix.csv"), encoding="utf-8-sig")
    rho_k.round(4).to_csv(os.path.join(_HERE, "kendall_matrix.csv"), encoding="utf-8-sig")
    pairs.to_csv(os.path.join(_HERE, "pairwise_stats.csv"), index=False, encoding="utf-8-sig")
    jac.round(4).to_csv(os.path.join(_HERE, "top20_jaccard.csv"), encoding="utf-8-sig")
    members.rename(columns={"_n_top20_lists": "n_top20_lists"}).to_csv(
        os.path.join(_HERE, "top20_membership.csv"), index=False, encoding="utf-8-sig")
    prof.to_csv(os.path.join(_HERE, "measure_profile.csv"), index=False, encoding="utf-8-sig")
    cl.to_csv(os.path.join(_HERE, "meaning_clusters.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame([{"measure": k, "desc": v} for k, v in DESC.items()]).to_csv(
        os.path.join(_HERE, "measure_definitions.csv"), index=False, encoding="utf-8-sig")
    noise = pd.DataFrame([
        {"pair": f"{MEAS[i[0]]}|{MEAS[i[1]]}", "m1": MEAS[i[0]], "m2": MEAS[i[1]],
         "spearman": round(float(rho_s.values[i]), 4), "spearman_hits_collapsed": round(float(rho_s_c.values[i]), 4),
         "delta_spearman": round(float(ds[j]), 4),
         "kendall": round(float(rho_k.values[i]), 4), "kendall_hits_collapsed": round(float(rho_k_c.values[i]), 4),
         "delta_kendall": round(float(dk[j]), 4),
         "crosses_theta090": bool((rho_s.values[i] >= 0.90) != (rho_s_c.values[i] >= 0.90))}
        for j, i in enumerate(zip(iu[0], iu[1]))])
    noise.to_csv(os.path.join(_HERE, "hits_noise_sensitivity.csv"), index=False, encoding="utf-8-sig")

    summary = {
        "module": MODULE,
        "versions": C.versions(),
        "seed": SEED,
        "hits_reproducibility": {
            "note": "csa_common.centralities 的 hits_* 用 svds(v0=None) → ARPACK 随机起点，跨进程逐位不可复现；"
                    "本单元用 nstart=uniform 固定（同算法同参数），见 hits_pinned / DISSENT.md",
            # 这个偏差本身就是"不可复现"的度量，跨进程只能保证 <5e-16（浮点噪声量级），
            # 所以这里写**上界字符串**而不是具体数值——写死数值会造出一个假的不可复现源。
            "max_abs_dev_vs_csa_common": "lt_5e-16" if max(dev_h, dev_a) < 5e-16 else "GE_5e-16_FAIL",
            "noise_floor": n_noise,
            "abs_delta_spearman_max": round(float(np.abs(ds).max()), 4),
            "abs_delta_spearman_median": round(float(np.median(np.abs(ds))), 4),
            "abs_delta_kendall_max": round(float(np.abs(dk).max()), 4),
            "pairs_crossing_theta090": flip090,
            "top20_members_changed": top_changed,
        },
        "n_nodes_attributed": len(df),
        "n_nodes_in_graph": int(Gu.number_of_nodes()),
        "n_measures": len(MEAS),
        "spearman_pairs": {"n": int(len(sv)), "min": round(float(sv.min()), 4),
                           "p25": round(float(np.percentile(sv, 25)), 4),
                           "median": round(float(np.median(sv)), 4),
                           "p75": round(float(np.percentile(sv, 75)), 4),
                           "max": round(float(sv.max()), 4),
                           "argmin": list(pairs.loc[pairs["spearman"].idxmin(), ["m1", "m2"]]),
                           "argmax": list(pairs.loc[pairs["spearman"].idxmax(), ["m1", "m2"]])},
        "kendall_pairs": {"n": int(len(kv)), "min": round(float(kv.min()), 4),
                          "median": round(float(np.median(kv)), 4),
                          "p75": round(float(np.percentile(kv, 75)), 4),
                          "max": round(float(kv.max()), 4),
                          "argmin": list(pairs.loc[pairs["kendall_tau_b"].idxmin(), ["m1", "m2"]]),
                          "argmax": list(pairs.loc[pairs["kendall_tau_b"].idxmax(), ["m1", "m2"]])},
        "top20_jaccard_pairs": {"min": round(float(jac.values[iu].min()), 4),
                                "median": round(float(np.median(jac.values[iu])), 4),
                                "max": round(float(jac.values[iu].max()), 4)},
        "n_pairs_same_meaning_rho090": n_same,
        "n_pairs_same_meaning_both": n_same_both,
        "n_pairs_total": int(len(pairs)),
        "meanings": {f"theta_{th}": {"avg": int(r["n_clusters_average"]),
                                     "complete": int(r["n_clusters_complete"])}
                     for th, r in zip(THETAS, cl_rows)},
        "cliques_rho_ge_090": cliques_090,
        "consensus_all12": hs,
        "n_in_ge10_lists": ge10,
        "n_in_ge6_lists": ge6,
        "lists_count_hist": {int(k): int(v) for k, v in hist.items()},
        "hindex": {"max": int(df["hidx_in_in"].max()), "mean": round(float(df["hidx_in_in"].mean()), 4),
                   "zero": int((df["hidx_in_in"] == 0).sum()),
                   "spearman_vs_in_degree": round(float(rho_s.loc["hidx_in_in", "in_degree"]), 4),
                   "spearman_vs_in_deg_alt_variant": round(float(df["hidx_in_deg"].corr(df["hidx_in_in"],
                                                                                          method="spearman")), 4)},
        "timing_s": TIMING,
        "wall_s": round(time.time() - T0, 2),
    }
    with open(os.path.join(_HERE, "c1_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    run_lines = [
        "C1_centrality_matrix RUN.txt",
        "command: set PYTHONHASHSEED=0 && python out/C1_centrality_matrix/run_c1.py  "
        "(bash form: PYTHONHASHSEED=0 python out/C1_centrality_matrix/run_c1.py; cwd=pipeline)",
        f"versions: {json.dumps(C.versions(), ensure_ascii=False)}",
        f"scipy={sps.__name__} {__import__('scipy').__version__}  (spearman/kendall self-check vs scipy: PASS)",
        f"SEED={SEED} (numpy tie-jitter rng = np.random.default_rng(SEED + r), r=0..{N_TIE_PERM - 1})",
        f"wall time: {summary['wall_s']}s ; per-stage: {json.dumps(TIMING)}",
        "anchors: csa_common.assert_anchors() 31/31 PASS",
        f"node set: 671 attributed (CONVENTIONS 1); graph Gu 676/2498, Gd 676/2788",
        f"measures ({len(MEAS)}): {', '.join(MEAS)}",
        "TOP_K=20 deterministic tie-break = node FQDN ascending; N_TIE_PERM=200 tie-permutation replicates",
        "outputs: centrality_matrix.csv spearman_matrix.csv kendall_matrix.csv pairwise_stats.csv "
        "top20_jaccard.csv top20_membership.csv measure_profile.csv meaning_clusters.csv "
        "measure_definitions.csv c1_summary.json result.md",
    ]
    C.write_run_note(MODULE, run_lines)
    log("saved all csv/json")
    print(json.dumps({k: summary[k] for k in
                      ["spearman_pairs", "kendall_pairs", "top20_jaccard_pairs",
                       "n_pairs_same_meaning_rho090", "n_pairs_same_meaning_both",
                       "meanings", "cliques_rho_ge_090", "consensus_all12",
                       "n_in_ge10_lists", "hindex"]}, ensure_ascii=False, indent=1))
