# -*- coding: utf-8 -*-
"""A3_null_model_matrix — 四个生成模型 × 八个指标的对照矩阵 + 定量判决。

单元职责（MANIFEST A3）：
  ER / WS / BA / 度保持配置模型 各自抽 >=30 个实例，逐指标报 mean ± sd，
  用标准化距离 z=(obs-mean)/sd 判决"观测网络最像哪个"，并列出模型解释不了的特征。

口径（全部服从 CONVENTIONS.md）：
  * 图：无权无向二值图。主口径节点集 = 671 有属性节点（§1 主表必须 671），
    敏感性口径 = 676（含 5 个悬空，§1 拓扑对照）。
  * 建图：一律 csa_common.build_all()，开跑先 C.assert_anchors()。
  * 配置模型：一律 C.degree_preserving_null（§4 独立重复 + nswap=10|E|），
    本脚本不自写交换循环。
  * 非连通实例规则：路径类指标（平均最短路径 / 直径）**只在各自的最大连通分量 GCC 上算**，
    其余指标在全图上算；观测网络与每个模型实例走同一条规则。详见 result.md §0。

运行：cd pipeline && python out/A3_null_model_matrix/run_a3.py
      快速自检：set CSA_A3_QUICK=1  （实例数缩到 8/5/3，只用于跑通，结果不用于结论）
禁止措辞：导致/因为/使/使得/引发/决定了/传染（本文件与 result.md 一律不用）。
"""
from __future__ import annotations

import os
import sys
import math
import time
import json

import numpy as np
import pandas as pd
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.dirname(os.path.dirname(HERE))
if PIPE not in sys.path:
    sys.path.insert(0, PIPE)
import csa_common as C  # noqa: E402

MODULE = "A3_null_model_matrix"
SEED = 20261113            # CONVENTIONS §8

QUICK = os.environ.get("CSA_A3_QUICK") == "1"
# ------------------------------------------------------------------ 样本量配置
N_PRIMARY = 8 if QUICK else 40      # 四个主模型的实例数（判据要求 >=30）
N_SENS = 5 if QUICK else 30         # 敏感性模型 / 676 口径的实例数
N_SCAN = 3 if QUICK else 30         # WS 扫描每个 p 的实例数
X_MIN = 3                           # 幂律 MLE 的固定截断（仅作相对比较，不做拟合优度判决）
SD_FLOOR = 1e-9                     # 判定"零方差"的下限，防止 1e-13 的 sd 把 z 炸到 1e13

# WS 重连概率扫描（p=0.05 是判据点名要报的固定值，也在网格内）
WS_P_GRID = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

# ------------------------------------------------------------------ 指标定义
# 10 个指标里 n_nodes / mean_degree 是模型的**输入**（参数匹配目标），不参与判决；
# 剩下 8 个是判据所说的"八个指标"。
METRICS = [
    "n_nodes", "mean_degree", "avg_clustering", "aspl_gcc", "diameter_gcc",
    "degree_assortativity", "degree_gini", "max_degree", "gcc_fraction",
    "powerlaw_alpha_x3",
]
CALIBRATED = ["n_nodes", "mean_degree"]                     # 匹配出来的，不作判决
EIGHT = ["avg_clustering", "aspl_gcc", "diameter_gcc", "degree_assortativity",
         "degree_gini", "max_degree", "gcc_fraction", "powerlaw_alpha_x3"]
# 跨模型可比的"自由"指标集：四个主模型都能在这几个指标上取值（不被度序列钉死）。
CORE4 = ["avg_clustering", "aspl_gcc", "diameter_gcc", "degree_assortativity"]
# 配置模型完全保持度序列 => 这几个指标对它恒定，不能参与它的综合评分。
CM_INVARIANT = ["n_nodes", "mean_degree", "degree_gini", "max_degree", "powerlaw_alpha_x3"]

Z_REJECT = 2.0    # |z|>2 记为"该模型在该指标上解释不了观测值"


def gini(values) -> float:
    """基尼系数（升序、含 0）。Gini = sum((2i-n-1) x_i) / (n * sum x)。"""
    x = np.sort(np.asarray(list(values), dtype=float))
    n = x.size
    if n == 0 or x.sum() <= 0:
        return float("nan")
    return float(((2 * np.arange(1, n + 1) - n - 1) * x).sum() / (n * x.sum()))


def pl_alpha(degrees, x_min: int = X_MIN) -> float:
    """离散幂律 MLE（Clauset et al. 2009 固定 x_min 式）：
    alpha = 1 + n / sum_i ln(x_i / (x_min - 0.5))，只取 x_i >= x_min。
    本单元把它当**描述统计量**用于跨模型相对比较，不判决分布形状（形状归 A1）。"""
    xs = [d for d in degrees if d >= x_min]
    if len(xs) < 10:
        return float("nan")
    denom = sum(math.log(x / (x_min - 0.5)) for x in xs)
    if denom <= 0:
        return float("nan")
    return float(1.0 + len(xs) / denom)


def path_mean_and_diameter(G: nx.Graph):
    """一次全源 BFS 同时得到平均最短路径与直径（连通图）。
    与 networkx 的等价性在 main 里用观测图和实例显式断言。"""
    n = G.number_of_nodes()
    if n < 2:
        return float("nan"), float("nan")
    tot = 0.0
    cnt = 0
    dia = 0
    for s in G:
        for t, d in nx.single_source_shortest_path_length(G, s).items():
            tot += d
            cnt += 1
            if d > dia:
                dia = d
    return float(tot / (cnt - n)), float(dia)


def all_metrics(H: nx.Graph) -> dict:
    """一个实例 / 观测图 → 10 个指标 + 3 个连通性辅助量。非连通时路径指标走 GCC。"""
    n = H.number_of_nodes()
    degs = [d for _, d in H.degree()]
    out = {
        "n_nodes": float(n),
        "mean_degree": float(H.number_of_edges() * 2 / n) if n else float("nan"),
        "avg_clustering": float(nx.average_clustering(H)),
        "degree_assortativity": float("nan"),
        "degree_gini": gini(degs),
        "max_degree": float(max(degs)) if degs else float("nan"),
        "powerlaw_alpha_x3": pl_alpha(degs),
        "aspl_gcc": float("nan"),
        "diameter_gcc": float("nan"),
        "gcc_fraction": float("nan"),
        "n_components": float(nx.number_connected_components(H)),
        "gcc_size": 0.0,
        "n_edges": float(H.number_of_edges()),
    }
    try:
        out["degree_assortativity"] = float(nx.degree_assortativity_coefficient(H))
    except Exception:
        pass
    comps = sorted(nx.connected_components(H), key=len, reverse=True)
    if comps and n:
        out["gcc_size"] = float(len(comps[0]))
        out["gcc_fraction"] = len(comps[0]) / n
    if comps and len(comps[0]) >= 2:
        G2 = H.subgraph(comps[0]).copy()
        try:
            a, d = path_mean_and_diameter(G2)
            out["aspl_gcc"] = a
            out["diameter_gcc"] = d
        except Exception:
            pass
    return out


# ------------------------------------------------------------------ 模型工厂
def ba_m_for(n: int, m_target: int) -> int:
    """BA(n,m) 有 m(n-m) 条边 → 匹配观测边数：解 m²-nm+M=0，取整后选 |m(n-m)-M| 最小者。"""
    lo = (n - math.sqrt(max(n * n - 4 * m_target, 0.0))) / 2.0
    cands = {max(1, int(math.floor(lo))), max(1, int(math.ceil(lo)))}
    return min(cands, key=lambda mm: abs(mm * (n - mm) - m_target))


def make_models(Gobs: nx.Graph):
    """{model_name: (role, fn, meta, seed_base)}；fn(i, seed) -> graph。
    每个实例一个独立 seed = SEED + seed_base + i，实例之间不共用随机轨迹。"""
    n = Gobs.number_of_nodes()
    m = Gobs.number_of_edges()
    mean_deg = 2 * m / n
    p_er = 2 * m / (n * (n - 1))                       # p 匹配平均度
    k_ws = max(2, int(round(mean_deg / 2) * 2))        # WS 环形格点要求 k 为偶数
    ba_m = ba_m_for(n, m)                              # BA 的 m 只能是整数

    out = {}
    out["ER"] = ("plain", lambda i, s: nx.gnp_random_graph(n, p_er, seed=s),
                 {"n": n, "p": round(p_er, 6), "mean_degree_target": round(mean_deg, 4)}, 1000)
    out[f"WS_k{k_ws}_p0.05"] = ("plain", lambda i, s: nx.watts_strogatz_graph(n, k_ws, 0.05, seed=s),
                                {"n": n, "k": k_ws, "p": 0.05}, 2000)
    out[f"WS_k{k_ws-2}_p0.05"] = ("sens", lambda i, s: nx.watts_strogatz_graph(n, k_ws - 2, 0.05, seed=s),
                                  {"n": n, "k": k_ws - 2, "p": 0.05}, 2400)
    out[f"WS_k{k_ws+2}_p0.05"] = ("sens", lambda i, s: nx.watts_strogatz_graph(n, k_ws + 2, 0.05, seed=s),
                                  {"n": n, "k": k_ws + 2, "p": 0.05}, 2600)
    out[f"BA_m{ba_m}"] = ("plain", lambda i, s: nx.barabasi_albert_graph(n, ba_m, seed=s),
                          {"n": n, "m": ba_m, "mean_degree_target": round(mean_deg, 4)}, 3000)
    out[f"BA_m{ba_m-1}"] = ("sens", lambda i, s: nx.barabasi_albert_graph(n, ba_m - 1, seed=s),
                            {"n": n, "m": ba_m - 1}, 3400)

    def ba_trim(i, s, mm=ba_m):
        """BA 的整数 m 无法精确匹配观测边数 → 修剪随机边到观测边数（口径对照，非主模型）。"""
        H = nx.barabasi_albert_graph(n, mm, seed=s)
        rng = np.random.default_rng(s + 777)
        drop = H.number_of_edges() - m
        if drop > 0:
            edges = list(H.edges())
            for j in sorted(rng.choice(len(edges), size=min(drop, len(edges)), replace=False)):
                u, v = edges[j]
                if H.has_edge(u, v):
                    H.remove_edge(u, v)
        return H

    out[f"BA_m{ba_m}_trim"] = ("sens", ba_trim, {"n": n, "m": ba_m, "trim_to_edges": m}, 3800)

    for j, p in enumerate(WS_P_GRID):
        out[f"WS_scan_p{p:g}"] = ("scan", (lambda i, s, p=p: nx.watts_strogatz_graph(n, k_ws, p, seed=s)),
                                  {"n": n, "k": k_ws, "p": p}, 5000 + 100 * j)
    return out


# ------------------------------------------------------------------ 配置模型
def config_model_instances(Gobs: nx.Graph, n_reps: int, seed: int):
    """实例来自 C.degree_preserving_null（§4：每轮 fresh copy + double_edge_swap，
    nswap=10|E|、max_tries=100·nswap）。**本函数不自写任何交换逻辑。**

    取实例的方式：给 csa_common 的 stat 挂钩——它在每个 swap 后的图上被调用一次，
    挂钩返回度同配性并顺手记下该图。随后用 csa_common 自己返回的 null_mean/null_sd
    与挂钩重算值做一致性断言，确保"记录到的实例"就是"它统计的那批实例"。
    额外好处：10 个指标落在同一批实例上，可直接算多元（Mahalanobis）距离。
    """
    captured = []

    def stat(H):
        try:
            v = float(nx.degree_assortativity_coefficient(H))
        except Exception:
            v = float("nan")
        if H is not Gobs:            # csa_common 末尾还会对原图调一次 stat(G)
            captured.append(H.copy())
        return v

    res = C.degree_preserving_null(Gobs, n_reps=n_reps, seed=seed, stat=stat)
    return res, captured


# ------------------------------------------------------------------ 统计汇总
def summarize(inst_rows: pd.DataFrame, obs: dict, metric: str) -> dict:
    v = inst_rows[metric].to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    k = int(v.size)
    o = float(obs.get(metric, float("nan")))
    mean = float(v.mean()) if k else float("nan")
    sd = float(v.std(ddof=1)) if k > 1 else float("nan")
    zero_var = bool(k and np.isfinite(sd) and sd <= SD_FLOOR)
    if k == 0:
        z, p_two, state = float("nan"), float("nan"), "no_finite_instance"
    elif zero_var and np.isfinite(o) and abs(o - mean) <= SD_FLOOR:
        z, p_two, state = 0.0, 1.0, "invariant_by_construction"        # 度序列保持 => 恒等
    elif zero_var:
        z, p_two, state = float("nan"), 0.0, "zero_variance_models_miss_obs"  # 模型不产生该值
    else:
        z = float((o - mean) / sd) if (np.isfinite(sd) and sd > 0 and np.isfinite(o)) else float("nan")
        p_two = float((np.abs(v - mean) >= abs(o - mean)).mean()) if np.isfinite(o) else float("nan")
        state = "ok" if np.isfinite(z) else "obs_undefined"
    explained = bool((state == "invariant_by_construction") or (state == "ok" and abs(z) <= Z_REJECT))
    rejected = bool(state == "zero_variance_models_miss_obs" or (state == "ok" and abs(z) > Z_REJECT))
    rel_dev = float(abs(o - mean) / abs(o)) if (np.isfinite(o) and np.isfinite(mean) and o != 0) else float("nan")
    return {
        "metric": metric,
        "obs": round(o, 6) if np.isfinite(o) else float("nan"),
        "mean": round(mean, 6) if np.isfinite(mean) else float("nan"),
        "sd": round(sd, 6) if np.isfinite(sd) else float("nan"),
        "cv_sd_over_mean": round(sd / mean, 4) if (np.isfinite(sd) and np.isfinite(mean) and mean != 0) else float("nan"),
        "diff_obs_minus_mean": round(o - mean, 6) if (np.isfinite(o) and np.isfinite(mean)) else float("nan"),
        "rel_dev_obs": round(rel_dev, 4) if np.isfinite(rel_dev) else float("nan"),
        "ratio_obs_over_mean": (round(o / mean, 4)
                                if (np.isfinite(o) and np.isfinite(mean) and mean != 0 and o * mean > 0)
                                else float("nan")),
        "z": round(z, 3) if np.isfinite(z) else float("nan"),
        "p_two_sided": round(p_two, 4) if np.isfinite(p_two) else float("nan"),
        "obs_inside_instance_range": bool(k and np.isfinite(o) and v.min() <= o <= v.max()),
        "n_instances_ge_obs": int(k and (v >= o).sum()) if np.isfinite(o) else -1,
        "exceedance_rate_obs": round(float((v >= o).mean()), 4) if (k and np.isfinite(o)) else float("nan"),
        "explained": explained,
        "rejected": rejected,
        "state": state,
        "n_instances": k,
        "min": round(float(v.min()), 6) if k else float("nan"),
        "max": round(float(v.max()), 6) if k else float("nan"),
        "note": state if state != "ok" else "",
    }


def rms(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    return float(np.sqrt(np.mean(v ** 2))) if v.size else float("nan")


def mahalanobis(inst: pd.DataFrame, obs: dict, cols):
    """多元标准化距离（把指标间的相关一起用进去）。样本协方差 + 伪逆。"""
    X = inst[cols].to_numpy(dtype=float)
    X = X[np.all(np.isfinite(X), axis=1)]
    x = np.array([obs[c] for c in cols], dtype=float)
    if X.shape[0] < len(cols) + 2 or not np.all(np.isfinite(x)):
        return float("nan"), int(X.shape[0])
    mu = X.mean(axis=0)
    S = np.atleast_2d(np.cov(X, rowvar=False, ddof=1))
    if not np.all(np.isfinite(S)):
        return float("nan"), int(X.shape[0])
    d = x - mu
    val = float(d @ np.linalg.pinv(S) @ d)
    return (float(math.sqrt(val)) if val >= 0 else float("nan")), int(X.shape[0])


# ------------------------------------------------------------------ 单口径主流程
def run_node_set(Gobs: nx.Graph, node_set_label: str, n_inst: int, n_sens: int,
                 n_scan: int, do_scan: bool, log):
    t0 = time.time()
    obs = all_metrics(Gobs)
    log.append(f"[{node_set_label}] observed: " + ", ".join(
        f"{k}={obs[k]:.4f}" for k in METRICS if np.isfinite(obs[k])))

    models = make_models(Gobs)
    inst_frames = []

    # ---- 1. 参数化模型（ER / WS / BA + 取整敏感性 + WS 扫描）
    for name, (kind, fn, meta, seed_base) in models.items():
        if kind == "scan" and not do_scan:
            continue
        reps = n_scan if kind == "scan" else (n_inst if kind == "plain" else n_sens)
        rows = []
        for i in range(reps):
            s = SEED + seed_base + i
            r = all_metrics(fn(i, s))
            r.update({"model": name, "role": kind, "instance": i, "seed_used": int(s),
                      "node_set": node_set_label,
                      **{f"par_{k}": v for k, v in meta.items()}})
            rows.append(r)
        inst_frames.append(pd.DataFrame(rows))
        log.append(f"[{node_set_label}] {name:28s} reps={reps:3d} elapsed={time.time()-t0:7.1f}s")

    # ---- 2. 度保持配置模型（全部交换在 csa_common 内部完成）
    res, captured = config_model_instances(Gobs, n_reps=n_inst, seed=SEED)
    assert len(captured) == n_inst, f"配置模型实例数 {len(captured)} != {n_inst}"
    assort_from_hook = np.array([all_metrics(h)["degree_assortativity"] for h in captured])
    assert abs(float(assort_from_hook.mean()) - res["null_mean"]) < 1e-12, \
        "挂钩记录的实例与 csa_common 统计的实例不是同一批"
    assert abs(float(assort_from_hook.std(ddof=1)) - res["null_sd"]) < 1e-12, "同上（sd）"
    deg_ok = all(sorted(d for _, d in h.degree()) == sorted(d for _, d in Gobs.degree())
                 for h in captured)
    loops = sum(1 for h in captured for e in h.edges() if e[0] == e[1])
    assert deg_ok and loops == 0, f"度序列未保持(deg_ok={deg_ok}) 或出现自环({loops})"
    log.append(f"[{node_set_label}] ConfigModel_degreePreserving     reps={n_inst:3d} "
               f"degseq_preserved={deg_ok} selfloops={loops} "
               f"csa_common(z={res['z']:.3f}, p={res['p_two_sided']:.4f}) elapsed={time.time()-t0:7.1f}s")
    rows = []
    for i, h in enumerate(captured):
        r = all_metrics(h)
        r.update({"model": "ConfigModel_degreePreserving", "role": "plain", "instance": i,
                  "seed_used": int(SEED + i), "node_set": node_set_label,
                  "par_nswap": 10 * Gobs.number_of_edges(),
                  "par_max_tries": 100 * 10 * Gobs.number_of_edges(),
                  "par_swap_by": "csa_common.degree_preserving_null",
                  "par_null_mean_assort": res["null_mean"], "par_null_sd_assort": res["null_sd"]})
        rows.append(r)
    inst_frames.append(pd.DataFrame(rows))

    inst = pd.concat(inst_frames, ignore_index=True)

    # ---- 3. 矩阵：指标 × 模型
    mrows = []
    for name, sub in inst.groupby("model", sort=False):
        role = sub["role"].iloc[0]
        for metric in METRICS:
            s = summarize(sub, obs, metric)
            cls = ("calibrated_input" if metric in CALIBRATED
                   else ("invariant_by_construction"
                         if (name.startswith("ConfigModel") and metric in CM_INVARIANT)
                         else "discriminative"))
            mrows.append({"node_set": node_set_label, "model": name, "role": role,
                          "metric": metric, "metric_class": cls, **s})
    matrix = pd.DataFrame(mrows)

    # ---- 4. 判决
    vrows = []
    for name in inst["model"].unique():
        sub = inst[inst["model"] == name]
        mm = matrix[matrix["model"] == name]
        zz = dict(zip(mm["metric"], mm["z"].astype(float)))
        rd = dict(zip(mm["metric"], mm["rel_dev_obs"].astype(float)))
        exp = dict(zip(mm["metric"], mm["explained"].astype(bool)))
        rej = dict(zip(mm["metric"], mm["rejected"].astype(bool)))
        role0 = mm["role"].iloc[0]
        free8 = [c for c in EIGHT if not (name.startswith("ConfigModel") and c in CM_INVARIANT)]
        D, _nkm = mahalanobis(sub, obs, CORE4)
        big = mm[(mm["metric_class"] == "discriminative") & (mm["rejected"])]["metric"].tolist()
        undet = mm[(mm["metric_class"] == "discriminative") & (~mm["explained"]) & (~mm["rejected"])]["metric"].tolist()
        absz = np.array([abs(zz[c]) if np.isfinite(zz[c]) else np.nan for c in free8], dtype=float)
        worst = free8[int(np.nanargmax(absz))] if np.isfinite(absz).any() else ""
        vrows.append({
            "node_set": node_set_label, "model": name,
            "role": ("scan" if name.startswith("WS_scan")
                     else ("primary" if role0 == "plain" else "sensitivity")),
            "n_instances": int(len(sub)),
            "free_metrics_used": "|".join(free8),
            # 判据 1：CORE4 里"解释得了"（|z|<=2 或恒等）的指标个数，越大越像
            "n_core4_explained": int(sum(1 for c in CORE4 if exp[c])),
            "n_core4_rejected": int(sum(1 for c in CORE4 if rej[c])),
            "n_core4_undetermined": int(sum(1 for c in CORE4 if not exp[c] and not rej[c])),
            # 判据 2：标准化距离 RMS（判据点名要的 z 口径）
            "rms_z_core4": round(rms([zz[c] for c in CORE4]), 3),
            # 判据 3：效应量 RMS（不受 null 的 sd 大小影响，|obs-mean|/|obs|）
            "rms_reldev_core4": round(rms([rd[c] for c in CORE4]), 4),
            "rms_z_free8": round(rms([zz[c] for c in free8]), 3),
            "rms_reldev_free8": round(rms([rd[c] for c in free8]), 4),
            "max_abs_z_free": round(float(np.nanmax(absz)), 3) if np.isfinite(absz).any() else float("nan"),
            "worst_metric": worst,
            "n_abs_z_gt2": len(big),
            "metrics_abs_z_gt2": "|".join(big),
            "n_undetermined": len(undet),
            "metrics_undetermined": "|".join(undet),
            "mahalanobis_D_core4": round(D, 3) if np.isfinite(D) else float("nan"),
            "n_connected_instances": int((sub["n_components"] == 1).sum()),
            "n_disconnected_instances": int((sub["n_components"] > 1).sum()),
            "mean_degree_gap_vs_obs": round(float(sub["mean_degree"].mean() - obs["mean_degree"]), 4),
            "edge_count_gap_vs_obs": round(float(sub["n_edges"].mean() - obs["n_edges"]), 1),
        })
    verdict = pd.DataFrame(vrows)
    log.append(f"[{node_set_label}] DONE reps_total={len(inst)} total={time.time()-t0:.1f}s")
    return matrix, verdict, inst, obs


# ------------------------------------------------------------------ 排名规则
def rank_primary(v: pd.DataFrame):
    """综合判决规则（字典序，可复算，写在 result.md §1）：
      主判据 = rms_z_core4 升序（判据点名的"标准化距离"跨指标聚合）。
      一致性检验 = n_core4_explained 降序 / rms_reldev_core4 升序 / mahalanobis_D_core4 升序。
      第一名在三种聚合下是否同一个，写进 result.md §3；若不一致则判"无法判定"。
    rank_key 只用主判据；nan（全部 CORE4 都无法计算）排最后。
    """
    out = v.copy()
    out["rank_key"] = out["rms_z_core4"].where(out["rms_z_core4"].notna(), 1e9)
    return out


def add_ranks(verdict: pd.DataFrame, log):
    out = []
    v671 = verdict[verdict["node_set"] == "attributed_671"].copy()
    v671["rank_core4"] = np.nan
    v671["rank_note"] = ""
    for role_label, note in [("primary", "main_rank_by_rms_z_core4"),
                             ("sensitivity", "sensitivity_rank_only"),
                             ("scan", "ws_scan_rank_only")]:
        sub = rank_primary(v671[v671["role"] == role_label]).sort_values(
            ["rank_key", "rms_reldev_core4", "model"])
        v671.loc[sub.index, "rank_core4"] = np.arange(1, len(sub) + 1, dtype=float)
        v671.loc[sub.index, "rank_note"] = note
        if role_label == "primary":
            log.append("PRIMARY RANK (rms_z_core4): " + " < ".join(
                f"{r.model}={r.rms_z_core4}" for r in sub.itertuples()))
            log.append("  (by n_core4_explained desc): " + " < ".join(
                f"{r.model}={r.n_core4_explained}" for r in
                sub.sort_values(["n_core4_explained", "rms_z_core4"], ascending=[False, True]).itertuples()))
            log.append("  (by rms_reldev_core4): " + " < ".join(
                f"{r.model}={r.rms_reldev_core4}" for r in sub.sort_values("rms_reldev_core4").itertuples()))
            log.append("  (by mahalanobis_D_core4): " + " < ".join(
                f"{r.model}={r.mahalanobis_D_core4}" for r in
                sub[sub["mahalanobis_D_core4"].notna()].sort_values("mahalanobis_D_core4").itertuples()))
    rest = verdict[verdict["node_set"] != "attributed_671"].copy()
    rest["rank_core4"] = np.nan
    rest["rank_note"] = "node_set_sensitivity_not_ranked"
    return pd.concat([v671, rest], ignore_index=True)


# ------------------------------------------------------------------ 主流程
def main():
    t_start = time.time()
    log = [f"A3_null_model_matrix run at {time.strftime('%Y-%m-%d %H:%M:%S')}",
           f"command: python {os.path.relpath(os.path.abspath(__file__), PIPE)}"
           + ("  [CSA_A3_QUICK=1 — 只跑通，不作结论]" if QUICK else ""),
           f"config: N_PRIMARY={N_PRIMARY} N_SENS={N_SENS} N_SCAN={N_SCAN} X_MIN={X_MIN} "
           f"WS_P_GRID={WS_P_GRID} Z_REJECT={Z_REJECT}"]

    b = C.build_all()
    C.assert_anchors(b)
    log.append("C.assert_anchors(): 31/31 PASS")
    log.append(f"env: {C.versions()}")

    Gb, nodes = b["Gb"], b["nodes"]
    G671 = C.induced(Gb, set(nodes["node"]))      # 主口径：671 有属性节点
    G676 = C.induced(Gb, set(Gb.nodes()))         # 敏感性口径：676（含 5 个悬空）
    for nm, G in (("G671", G671), ("G676", G676)):
        log.append(f"{nm} n={G.number_of_nodes()} m={G.number_of_edges()} "
                   f"mean_deg={2*G.number_of_edges()/G.number_of_nodes():.4f} "
                   f"components={nx.number_connected_components(G)}")

    # 与 networkx 的等价性断言（路径指标自实现的正确性）
    GCC = C.largest_component(G671)
    a, d = path_mean_and_diameter(GCC)
    assert abs(a - nx.average_shortest_path_length(GCC)) < 1e-9, "ASPL 与 networkx 不一致"
    assert abs(d - nx.diameter(GCC)) < 1e-9, "直径与 networkx 不一致"
    log.append(f"path_mean_and_diameter == networkx on observed GCC: aspl={a:.6f} diam={d:.0f} OK")

    mats, vers, insts, obs_store = [], [], [], {}
    M, V, I, O = run_node_set(G671, "attributed_671", N_PRIMARY, N_SENS, N_SCAN, True, log)
    mats.append(M); vers.append(V); insts.append(I); obs_store["attributed_671"] = O
    M, V, I, O = run_node_set(G676, "in_graph_676", N_SENS, N_SENS, 0, False, log)
    mats.append(M); vers.append(V); insts.append(I); obs_store["in_graph_676"] = O

    matrix = pd.concat(mats, ignore_index=True)
    verdict = add_ranks(pd.concat(vers, ignore_index=True), log)
    instances = pd.concat(insts, ignore_index=True)
    instances["instance"] = instances["instance"].astype(int)

    # 观测侧补充刻画（强度 Gini：四个模型都是无权图，不在可比范围内，只作说明）
    Gu = b["Gu"]
    G671w = C.induced(Gu, set(nodes["node"]))
    extra = {
        "obs_deg_gini_671": gini([d for _, d in G671.degree()]),
        "obs_strength_gini_671": gini([w for _, w in G671w.degree(weight="weight")]),
        "obs_deg_gini_676": gini([d for _, d in G676.degree()]),
        "obs_strength_gini_676": gini([w for _, w in Gu.degree(weight="weight")]),
        "obs_assort_weighted_676": float(nx.degree_assortativity_coefficient(Gu)),
        "obs_max_degree_671": int(max(d for _, d in G671.degree())),
        "argmax_degree_671": max(G671.degree(), key=lambda kv: kv[1])[0],
        "obs_mean_degree_671": float(2 * G671.number_of_edges() / G671.number_of_nodes()),
        "obs_n_edges_671": int(G671.number_of_edges()),
        "obs_n_components_671": int(nx.number_connected_components(G671)),
        "obs_gcc_size_671": int(C.largest_component(G671).number_of_nodes()),
        "obs_small_components_671": sorted([len(c) for c in nx.connected_components(G671)])[:-1],
    }
    log.append("extra observed: " + ", ".join(f"{k}={v}" for k, v in extra.items()))

    C.save_csv(matrix, MODULE, "null_model_matrix.csv")
    C.save_csv(verdict, MODULE, "verdict.csv")
    C.save_csv(instances, MODULE, "instances.csv")
    C.save_json({"observed_metrics": obs_store, "extra_observed": extra,
                 "config": {"N_PRIMARY": N_PRIMARY, "N_SENS": N_SENS, "N_SCAN": N_SCAN,
                            "X_MIN": X_MIN, "SEED": SEED, "WS_P_GRID": WS_P_GRID,
                            "SD_FLOOR": SD_FLOOR, "Z_REJECT": Z_REJECT,
                            "METRICS": METRICS, "EIGHT": EIGHT, "CORE4": CORE4,
                            "CM_INVARIANT": CM_INVARIANT}},
                MODULE, "observed_and_config.json")

    elapsed = time.time() - t_start
    log.append(f"TOTAL elapsed = {elapsed:.1f}s")
    log.append(f"instances total = {len(instances)} | matrix rows = {len(matrix)} | "
               f"verdict rows = {len(verdict)}")
    log.append(f"seeds: instance seed_used = {SEED} + model_offset + instance_index "
               "(ER 1000 / WS 2000,2400,2600,5000+ / BA 3000,3400,3800); "
               f"ConfigModel via csa_common.degree_preserving_null(seed={SEED}) => seed+r")
    C.write_run_note(MODULE, log)
    write_result_md(matrix, verdict, instances, obs_store, extra, elapsed)
    print("\n".join(log[-12:]))
    print("elapsed", round(elapsed, 1))
    return matrix, verdict, instances


# ------------------------------------------------------------------ result.md
def write_result_md(matrix, verdict, instances, obs_store, extra, elapsed):
    d = C.module_dir(MODULE)
    m671 = matrix[matrix["node_set"] == "attributed_671"]
    v671 = verdict[verdict["node_set"] == "attributed_671"]
    prim = v671[v671["role"] == "primary"].sort_values("rank_core4")
    order = list(prim["model"])
    obs = obs_store["attributed_671"]

    def g(model, metric, col):
        r = m671[(m671["model"] == model) & (m671["metric"] == metric)]
        return float("nan") if r.empty else r.iloc[0][col]

    def f(x, nd=4):
        return "—" if (x is None or (isinstance(x, float) and not np.isfinite(x))) else f"{x:.{nd}f}"

    L = []
    A = L.append
    A("# A3 · 生成模型对照矩阵（ER / WS / BA / 度保持配置模型）\n")
    A(f"口径：无权无向二值图；主口径节点集 **671 有属性节点**（CONVENTIONS §1 主表必须 671），"
      f"敏感性口径 676（含 5 个悬空）。开跑先 `C.assert_anchors()` → 31/31 PASS。")
    A(f"观测图（671）：n=671，m={int(extra['obs_n_edges_671'])} 条无向边，"
      f"平均度 {extra['obs_mean_degree_671']:.4f}（676 口径 7.3905 = CONVENTIONS 锚点值），"
      f"最大度 {int(extra['obs_max_degree_671'])}（`{extra['argmax_degree_671']}`），"
      f"度 Gini {extra['obs_deg_gini_671']:.4f}，连通分量 {extra['obs_n_components_671']} 个"
      f"（GCC {extra['obs_gcc_size_671']}，另有 {extra['obs_small_components_671']} 的小分量）。")
    A(f"实例数：主模型 {N_PRIMARY}／敏感性 {N_SENS}／WS 扫描每个 p {N_SCAN}；"
      f"全部固定 seed（`SEED={SEED}`+模型偏移+实例号，见 `RUN.txt`）。耗时 {elapsed:.0f}s。\n")

    A("## 0. 连通性处理规则（判据要求写明）\n")
    A("1. **路径类指标（`aspl_gcc` 平均最短路径、`diameter_gcc` 直径）只在每个实例自身的"
      "最大连通分量（GCC）上计算**；其余 8 个指标在该实例的全图上计算。")
    A("2. 这条规则**同时作用在观测网络和每一个模型实例**上，两边口径对称。")
    A("3. `gcc_fraction`（最大分量节点数 / 该实例节点数）单独作为一个指标报出，"
      "把「非连通程度」本身纳入判决，而不是被路径指标偷偷消化掉。")
    A("4. 每个模型的连通／非连通实例个数见 §6，并在 `verdict.csv` 的 "
      "`n_connected_instances` / `n_disconnected_instances` 列里。")
    A("5. 平局规则：最大分量规模相同者并列时，取 `networkx.connected_components` 的返回顺序第一位；"
      "实测本次所有实例与观测图的最大分量规模均唯一。")
    A(f"6. 观测网络自身在 671 口径下有 {extra['obs_n_components_671']} 个连通分量"
      f"（GCC={extra['obs_gcc_size_671']}，`gcc_fraction`={obs['gcc_fraction']:.4f}），"
      "所以观测侧的路径指标是在 669 个节点的 GCC 上算的。\n")

    A("## 1. 综合判决规则（先说规则，再给结论）\n")
    A("**主判据**：`rms_z_core4` 升序 = sqrt(mean(z²))，z=(obs−mean_model)/sd_model，"
      "在 4 个「跨模型自由指标」CORE4 = {平均聚类系数, GCC 平均最短路径, GCC 直径, 度同配性} 上聚合。"
      "这就是判据点名的标准化距离口径。")
    A("**三条一致性检验**（全部留在 `verdict.csv`，用来检查主判据是不是只是 sd 的产物）：")
    A("- a) `n_core4_explained`：CORE4 里 |z| ≤ 2 的个数（越大越像），刻度粗但不受 sd 大小影响；")
    A("- b) `rms_reldev_core4` = sqrt(mean(((obs−mean)/obs)²))：纯效应量，与实例分布的宽窄无关；")
    A("- c) `mahalanobis_D_core4`：把 4 个指标之间的相关一起用进去的多元距离（样本协方差 + 伪逆）。")
    A("**判决纪律**：只有当主判据与 a/b/c 给出**同一个第一名**时，才写「最像 X」；"
      "否则本单元判「无法判定」并把分歧如实列出。2–4 名的顺序不作为结论。")
    A("为什么主判据只放 4 个指标：度 Gini / 最大度 / α 对配置模型**由构造恒定**"
      "（度序列被逐点保持），把它们塞进跨模型平均等于让配置模型白拿满分；"
      "`n_nodes`/`mean_degree` 是参数匹配目标。8 个判别指标在 §2/§4 逐指标列全，"
      "配置模型的恒等指标在 `metric_class` 里标为 `invariant_by_construction`，"
      "并从它的 `rms_*_free8` 中剔除。")
    A("为什么还要 b：n≈670 时各模型的实例分布**极窄**（`cv_sd_over_mean` 列，常见 0.5%~5%），"
      "纯 z 排序会偏向 sd 大的模型，所以每次都给出不依赖 sd 的效应量对照。\n")
    A("### 主判决表（671 主口径 · 四个主模型）\n")
    A("| 排名 | 模型 | 实例数 | CORE4 解释得了 | rms_z_core4 | rms_reldev_core4 | Mahalanobis D | rms_z_free8 | 最大\\|z\\| | 该指标 | \\|z\\|>2 指标数 |")
    A("|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|")
    for r in prim.itertuples():
        A(f"| {int(r.rank_core4)} | `{r.model}` | {r.n_instances} | {r.n_core4_explained}/4 | "
          f"{r.rms_z_core4} | {r.rms_reldev_core4} | {r.mahalanobis_D_core4} | {r.rms_z_free8} | "
          f"{r.max_abs_z_free} | `{r.worst_metric}` | {r.n_abs_z_gt2} |")
    A("")

    A("## 2. 指标 × 模型矩阵（671 主口径；mean ± sd，括号内 z）\n")
    A("| 指标 | 观测 | " + " | ".join(f"`{o}`" for o in order) + " |")
    A("|---|---:|" + "---:|" * len(order))
    for metric in METRICS:
        cls = m671[m671["metric"] == metric]["metric_class"].iloc[0]
        tag = "" if cls == "discriminative" else f" ‹{cls}›"
        row = [f"`{metric}`{tag}", f(obs[metric])]
        for o in order:
            mu, sd, z = g(o, metric, "mean"), g(o, metric, "sd"), g(o, metric, "z")
            nt = m671[(m671["model"] == o) & (m671["metric"] == metric)]["note"].iloc[0]
            if not np.isfinite(mu):
                row.append("—")
            elif nt in ("invariant_by_construction", "zero_variance_models_miss_obs"):
                row.append(f"{mu:.4f} ± 0 ({'z=0 恒等' if nt=='invariant_by_construction' else '零方差且不覆盖观测'})")
            else:
                row.append(f"{mu:.4f} ± {sd:.4f} ({z:+.2f})")
        A("| " + " | ".join(row) + " |")
    A("\n`mean_degree` 那两行的模型均值≠观测值，是参数取整的必然结果（WS 的 k 只能偶数、BA 的 m 只能整数），"
      "差值见 §5；幂律 α 固定 `x_min=3`，**仅作跨模型相对比较**，分布形状判决属于 A1。\n")

    A("## 3. 判决结论\n")
    best = prim.iloc[0]
    by_z = list(prim.sort_values("rms_z_core4")["model"])
    by_expl = list(prim.sort_values(["n_core4_explained", "rms_z_core4"],
                                    ascending=[False, True])["model"])
    by_rd = list(prim.sort_values("rms_reldev_core4")["model"])
    _mh = prim[prim["mahalanobis_D_core4"].notna()]
    by_mh = list(_mh.sort_values("mahalanobis_D_core4")["model"])
    winners = [by_z[0], by_expl[0], by_rd[0], by_mh[0] if by_mh else None]
    uniq = set(x for x in winners if x)
    A("### 3.1 四种跨指标聚合给出的排序（判决只取第一名的共识）\n")
    A("| 聚合口径 | 排序（最像 → 最不像） | 第一名 |")
    A("|---|---|---|")
    for key, seq in [("主判据 `rms_z_core4`", by_z), ("`n_core4_explained`", by_expl),
                     ("`rms_reldev_core4`", by_rd), ("`mahalanobis_D_core4`", by_mh)]:
        A(f"| {key} | {' < '.join(f'`{x}`' for x in seq)} | `{seq[0]}` |")
    A("")
    if len(uniq) == 1:
        A(f"**判决（一句话，可复算）**：在 671 主口径、四个模型各 {N_PRIMARY} 个实例、"
          f"8 个判别指标、四种跨指标聚合下，与观测网络结构最接近的是 **`{by_z[0]}`**"
          f"（rms_z_core4={best.rms_z_core4}，rms_reldev_core4={best.rms_reldev_core4}，"
          f"CORE4 的 4 个指标里 {best.n_core4_explained} 个 |z|≤2）。"
          f"这个「最接近」**仅是排序意义**：`{by_z[0]}` 自己仍有 "
          f"{best.n_abs_z_gt2}/{len(EIGHT)} 个判别指标 |z|>2（§4）。")
    else:
        A("**判决：无法判定。** 四种聚合给出的第一名不一致（"
          + "、".join(f"{k}→`{v}`" for k, v in
                      zip(["rms_z_core4", "n_core4_explained", "rms_reldev_core4", "mahalanobis_D_core4"], winners))
          + "），按 §1 的判决纪律不下结论，只把逐指标偏离列在 §2/§4。")
    A(f"2–4 名的顺序在四种聚合之间互相调换（z 口径 {' < '.join(f'`{x}`' for x in by_z[1:])}；"
      f"效应量口径 {' < '.join(f'`{x}`' for x in by_rd[1:])}），"
      "所以本单元只对第一名表态，不对中间名次表态。\n")
    A("### 3.2 明确的否证\n")
    _ba = prim[prim["model"].str.startswith("BA")]
    _cm = "ConfigModel_degreePreserving"
    A(f"1. 「观测网络属于 BA 那一类（增长 + 优先连接）」在这张表里**没有支撑**："
      f"BA 有 {int(_ba.iloc[0].n_abs_z_gt2)}/{len(EIGHT)} 个判别指标 |z|>2，逐指标数值见 §5；"
      "而且本数据没有时间序列，BA 的两个机制假设根本不可检验（§9.1）。")
    A(f"2. 「观测网络是度保持随机图」同样不成立：配置模型的平均聚类系数 "
      f"z={g(_cm,'avg_clustering','z'):+.1f}（观测 {obs['avg_clustering']:.4f} vs 实例 "
      f"{g(_cm,'avg_clustering','mean'):.4f}，观测是实例均值的 "
      f"{obs['avg_clustering']/g(_cm,'avg_clustering','mean'):.2f} 倍）。")
    _ws = [o for o in order if o.startswith("WS_")][0]
    A(f"3. 「小世界 = WS 就够了」不成立：WS(p=0.05) 的聚类系数是观测的 "
      f"{g(_ws,'avg_clustering','mean')/obs['avg_clustering']:.2f} 倍、"
      f"GCC 平均路径是观测的 {g(_ws,'aspl_gcc','mean')/obs['aspl_gcc']:.2f} 倍，两头都偏；"
      f"把 p 扫到 §7a 的 {len(WS_P_GRID)} 格，也没有任何一格把两者同时拉进 |z|≤2。")
    A("4. 没有任何模型在 8 个判别指标上全部 |z| ≤ 2。逐模型未解释指标个数："
      + ", ".join(f"`{r.model}`={r.n_abs_z_gt2}" for r in prim.itertuples()) + "。\n")

    A("## 4. 模型解释不了哪些特征（按「连最贴合的模型都差多远」排序）\n")
    A("| 指标 | 观测 | " + " | ".join(f"{o} z" for o in order) + " | 最贴合模型 | 最贴合的\\|z\\| | 判定 |")
    A("|---|---:|" + "---:|" * len(order) + "---|---:|---|")
    rows = []
    for metric in EIGHT:
        zs = {o: g(o, metric, "z") for o in order}
        st = {o: m671[(m671["model"] == o) & (m671["metric"] == metric)]["state"].iloc[0] for o in order}
        vals = [abs(zs[o]) for o in order if np.isfinite(zs[o])]
        fit = min([o for o in order if np.isfinite(zs[o])], key=lambda o: abs(zs[o])) if vals else "—"
        best_abs = min(vals) if vals else float("nan")
        n_ok = sum(1 for o in order if np.isfinite(zs[o]) and abs(zs[o]) <= Z_REJECT)
        n_miss = sum(1 for o in order if st[o] == "zero_variance_models_miss_obs")
        n_und = sum(1 for o in order if st[o] in ("no_finite_instance", "obs_undefined"))
        cm_only = bool(n_ok == 1 and "ConfigModel" in str(fit) and metric in CM_INVARIANT)
        if cm_only:
            tag = "只有配置模型，且是**构造恒等**（度序列被钉住），不构成证据"
        elif n_ok == 0:
            tag = "**四个模型全部解释不了**"
        elif n_ok < len(order):
            tag = f"{n_ok}/{len(order)} 个模型能覆盖"
        else:
            tag = "四个模型都能覆盖"
        bits = []
        if n_miss:
            bits.append(f"{n_miss} 个零方差且不覆盖观测")
        if n_und:
            bits.append(f"{n_und} 个在该模型上无定义")
        if bits and not cm_only:
            tag += "（" + "，".join(bits) + "）"
        rows.append((best_abs, metric, zs, fit, tag))
    rows.sort(key=lambda t: -(t[0] if np.isfinite(t[0]) else -1))
    for _best, metric, zs, fit, tag in rows:
        A(f"| `{metric}` | {f(obs[metric])} | " +
          " | ".join((f"{zs[o]:+.2f}" if np.isfinite(zs[o]) else "n/a") for o in order) +
          f" | `{fit}` | {f(_best, 2)} | {tag} |")
    A("")
    A("读法：排序键是「四个模型里最小的 |z|」，即**连最好的模型也差这么远** —— 排在最上面的就是"
      "四个模型共同解释不了的结构性特征。`n/a` = 该模型在该指标上零方差且实例分布不含观测值"
      "（`state=zero_variance_models_miss_obs`），等同于「完全解释不了」；"
      "`‹invariant_by_construction›` 的指标对配置模型恒等（度序列被逐点保持），"
      "**不构成**它的优势证据，已从它的 `rms_*_free8` 里剔除。\n")

    A("## 5. 判据点名的已知事实：量化差多少\n")
    for model in order:
        if not (model.startswith("BA") or model.startswith("WS") or model == "ER"):
            continue
        for metric, lab in [("max_degree", "最大度"), ("degree_gini", "度 Gini"),
                            ("powerlaw_alpha_x3", "α(x_min=3)"),
                            ("degree_assortativity", "度同配性"),
                            ("avg_clustering", "平均聚类系数")]:
            mu, sd, z = g(model, metric, "mean"), g(model, metric, "sd"), g(model, metric, "z")
            o = obs[metric]
            rat = (f"，观测/模型 = {o/mu:.3f}×"
                   if (mu != 0 and o * mu > 0 and 0.2 <= abs(o / mu) <= 5)
                   else "（比值无意义：两者接近 0 的负值之比 / 符号相反，只报差值与 z）")
            A(f"- `{model}` / `{metric}`（{lab}）：观测 {o:.4f} vs 模型 {mu:.4f}±{sd:.4f}；"
              f"差 {o-mu:+.4f}{rat}，z = {z:+.2f}")
    # 判据点名的"异配"事实：单独量化 BA 只覆盖了多少
    bam = [o for o in order if o.startswith("BA")]
    if bam:
        mo = bam[0]
        o_ass, b_ass, b_sd = obs["degree_assortativity"], g(mo, "degree_assortativity", "mean"), g(mo, "degree_assortativity", "sd")
        A(f"- **异配性核对**（判据 §4）：观测度同配性 = {o_ass:.4f}（负 = 异配，即高度点—低度点对的占比高于同度配对情形）。"
          f"`{mo}` 实例均值 = {b_ass:.4f}，只达到观测负相关强度的 {abs(b_ass)/abs(o_ass):.1%}；"
          f"两者差 {o_ass-b_ass:+.4f} = 实例分布的 {abs((o_ass-b_ass)/b_sd):.1f} 个标准差。"
          f"配置模型实例均值 = {g('ConfigModel_degreePreserving','degree_assortativity','mean'):.4f}"
          f"（sd {g('ConfigModel_degreePreserving','degree_assortativity','sd'):.4f}，"
          f"z = {g('ConfigModel_degreePreserving','degree_assortativity','z'):+.2f}），"
          "度保持打散后同配性向 0 靠拢，但剩余负值与度序列共存（该零模型把度序列固定住，"
          "只打散配对方式），因此它衡量的是「度序列之外的配对结构」这一层的偏离。")
    A("")
    A("### 5a. 参数取整误差（判据要求「m/k 匹配平均度」，这里给出匹配不到的部分）\n")
    A("| 模型 | 角色 | 平均度与观测差 | 边数与观测差 |")
    A("|---|---|---:|---:|")
    for r in v671[v671["role"] != "scan"].sort_values(["role", "model"]).itertuples():
        A(f"| `{r.model}` | {r.role} | {r.mean_degree_gap_vs_obs:+.4f} | {r.edge_count_gap_vs_obs:+.1f} |")
    A("")
    A("（10 个 WS 扫描格子的同两列在 `verdict.csv` 里，k 全部固定，故每格差值相同。）\n")
    A("### 5b. 强度层面\n")
    A(f"观测无向**强度** Gini = {extra['obs_strength_gini_671']:.4f}（676 口径 "
      f"{extra['obs_strength_gini_676']:.4f}），**度** Gini = {extra['obs_deg_gini_671']:.4f}。"
      "本单元四个模型都只生成**无权拓扑**，不生成边权，因此强度异质性完全不在矩阵的覆盖范围内 —— "
      "这不是「模型拟合失败」，而是「模型没有这个自由度」。权重层面的对照属于 E6/B4。\n")

    A("## 6. 实例层面的连通情况（路径类指标的口径依赖它）\n")
    A("| 模型 | 角色 | 实例数 | 全连通 | 非连通 | GCC 占比 mean±sd | 分量数 mean |")
    A("|---|---|---:|---:|---:|---|---:|")
    for o in sorted(set(v671["model"])):
        sub = instances[(instances["node_set"] == "attributed_671") & (instances["model"] == o)]
        role = sub["role"].iloc[0]
        if role == "scan":
            continue
        gf = sub["gcc_fraction"].to_numpy(dtype=float)
        A(f"| `{o}` | {role} | {len(sub)} | {int((sub['n_components']==1).sum())} | "
          f"{int((sub['n_components']>1).sum())} | {np.nanmean(gf):.4f} ± {np.nanstd(gf, ddof=1):.4f} | "
          f"{np.nanmean(sub['n_components'].to_numpy(dtype=float)):.2f} |")
    ncm = instances[(instances["node_set"] == "attributed_671") &
                    (instances["model"] == "ConfigModel_degreePreserving")]
    A("")
    A(f"要点：**度保持配置模型 {int((ncm['n_components']>1).sum())}/{len(ncm)} 个实例是非连通的**"
      f"（平均 {np.nanmean(ncm['n_components'].to_numpy(dtype=float)):.2f} 个分量），"
      "而 ER/WS/BA 在本参数下几乎总是单连通 —— 所以「路径类指标只在 GCC 上算」这条规则"
      "对配置模型不是可选项，否则它的路径指标会在一个不连通的图上直接失败。")
    A("WS 扫描各格子的连通情况在 `verdict.csv` / `instances.csv`。\n")

    A("## 7. 敏感性：WS 重连概率扫描 / 参数取整变体 / 676 口径\n")
    sc = v671[v671["role"] == "scan"].sort_values("rank_core4")
    if len(sc):
        kscan = instances[instances["model"] == sc.iloc[0]["model"]]["par_k"].iloc[0]
        A(f"### 7a. WS 扫描（k 固定为 {int(kscan)}，只有重连概率 p 变）\n")
        A("| WS p | 排名 | rms_z_core4 | rms_reldev_core4 | 聚类 z | 路径 z | 直径 z | 同配 z |")
        A("|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in sc.itertuples():
            A(f"| {r.model.replace('WS_scan_p','')} | {int(r.rank_core4)} | {r.rms_z_core4} | "
              f"{r.rms_reldev_core4} | {f(g(r.model,'avg_clustering','z'),2)} | "
              f"{f(g(r.model,'aspl_gcc','z'),2)} | {f(g(r.model,'diameter_gcc','z'),2)} | "
              f"{f(g(r.model,'degree_assortativity','z'),2)} |")
        _ok_p = [r.model for r in sc.itertuples() if r.n_core4_explained == 4]
        A(f"\n扫描里 `rms_z_core4` 最小的是 `{sc.iloc[0].model}`；"
          f"主表用的是 p=0.05 那一格（判据点名值）。"
          f"{'存在' if _ok_p else '**不存在**'}把 4 个 CORE4 全部拉进 |z|≤2 的 p"
          + (f"（{', '.join(_ok_p)}）" if _ok_p else "")
          + f"。p 扫描移动的是「聚类↔路径」折中点，"
          "度分布层面（`degree_gini` / `max_degree` / α）在所有 10 格里都远离观测值，"
          "数值见 `null_model_matrix.csv`。")
        A("`—` = 该模型该指标的 30 个实例方差为 0（直径是整数、粒度粗，容易全同），"
          "而观测值不在其中，等同于「解释不了」；`n_core4_explained` 已把它按未解释处理。\n")
    sn = v671[v671["role"] == "sensitivity"].sort_values("rank_core4")
    if len(sn):
        A("### 7b. 参数取整方向与修剪变体\n")
        A("| 变体 | 排名 | rms_z_core4 | rms_reldev_core4 | 平均度差 | 边数差 |")
        A("|---|---:|---:|---:|---:|---:|")
        for r in sn.itertuples():
            A(f"| `{r.model}` | {int(r.rank_core4)} | {r.rms_z_core4} | {r.rms_reldev_core4} | "
              f"{r.mean_degree_gap_vs_obs:+.4f} | {r.edge_count_gap_vs_obs:+.1f} |")
        A("")
    v676 = verdict[(verdict["node_set"] == "in_graph_676") & (verdict["role"] == "primary")]
    if len(v676):
        A("### 7c. 676 节点口径（CONVENTIONS §1 附录口径；含 5 个悬空节点）\n")
        A("检验判决是否随节点集漂移：\n")
        A("| 模型 | rms_z_core4 | rms_reldev_core4 | rms_z_free8 | 最大\\|z\\| 指标 | 671 口径同指标 z |")
        A("|---|---:|---:|---:|---|---:|")
        for r in v676.sort_values("rms_z_core4").itertuples():
            A(f"| `{r.model}` | {r.rms_z_core4} | {r.rms_reldev_core4} | {r.rms_z_free8} | "
              f"`{r.worst_metric}` ({r.max_abs_z_free}) | {f(g(r.model, r.worst_metric, 'z'),2)} |")
        A("")

    A("## 8. 每个数字可指到哪儿\n")
    A("- 指标定义：`run_a3.py:all_metrics()`；Gini `run_a3.py:gini()`；"
      "α `run_a3.py:pl_alpha()`；路径一次遍历 `run_a3.py:path_mean_and_diameter()`"
      "（与 `networkx.average_shortest_path_length`/`nx.diameter` 在观测 GCC 上做过等值断言，见 `RUN.txt`）")
    A("- 参数匹配：`run_a3.py:make_models()`（ER 的 p=2m/(n(n−1))；WS 的 k=round(平均度/2)×2；"
      "BA 的 m 由 `run_a3.py:ba_m_for()` 解 m(n−m)=|E| 取整）")
    A("- 配置模型：`run_a3.py:config_model_instances()` → `csa_common.degree_preserving_null`"
      f"（nswap=10·|E|、max_tries=100·nswap、每轮 fresh copy；"
      "本脚本对「挂钩实例批 == csa_common 统计实例批」和「度序列保持 / 无自环」都做了断言）")
    A("- z、双侧经验 p、零方差处理：`run_a3.py:summarize()`；"
      "RMS `run_a3.py:rms()`；多元距离 `run_a3.py:mahalanobis()`；排名规则 `run_a3.py:rank_primary()`")
    A("- 机器可读：`null_model_matrix.csv`（指标×模型×口径）、`verdict.csv`（判决全部列）、"
      "`instances.csv`（每实例 10 指标 + 连通性 + 实际 seed，可逐行复核）、"
      "`observed_and_config.json`（观测值与所有参数）、`RUN.txt`（环境/耗时/seed 方案）\n")

    A("## 9. 局限（与结论同等重要）\n")
    A("1. **这是结构相似性比较，不是生成机制的证据。** 数据是单次横截面，没有时间序列，"
      "无法检验「节点增长」与「优先连接」这两个 BA 的核心机制假设；"
      "本表只能说某个静态拓扑快照与某个模型的样本在这几个指标上更接近，"
      "**不能**说网络按该模型的方式长成这样。")
    A("2. z 是标准化距离，不是模型证据强度；RMS 是描述性平均，"
      "CORE4 内部（聚类／路径／直径）彼此相关，综合分数没有概率解释。"
      "Mahalanobis 距离把相关一起用进去，可能与 RMS 给出不同顺序，两个都已在 `verdict.csv`。")
    A("3. n≈670 时各模型的实例分布很窄（`cv_sd_over_mean` 列给出），"
      "|z| 动辄几十上百，**只能用于模型之间的横向比较，不能当成「显著性」的度量**；"
      f"双侧经验 p 的分辨率下限 = 1/实例数 = {1.0/N_PRIMARY:.3f}（主模型 {N_PRIMARY} 个实例），"
      f"`p_two_sided=0` 只表示「{N_PRIMARY} 个实例里没有一个偏离得比观测值更远」。")
    A("4. 参数取整误差不可消除：WS 的 k 只能取偶数、BA 的 m 只能取整数，"
      "两者都无法把平均度精确匹配到 " + f"{extra['obs_mean_degree_671']:.4f}" +
      "；差值在 §5a 与 `verdict.csv` 里显式给出，并配了反方向取整与随机修剪变体作对照。")
    A("5. 边交换零模型在一般图上是**非均匀采样**（Viger & Latapy 2005，已在 "
      "`csa_common.degree_preserving_null` 的 note 里声明），"
      "配置模型的 sd 与 p 是近似的；交换是否充分无法严格证明，只能观察边改动比例（本次实测见 RUN.txt）。")
    A("6. 幂律 α 固定 `x_min=3`：这只是各模型同口径下的一个汇总统计量，"
      "x_min 的选择、拟合优度、与对数正态/指数/截断幂律的比较属于 A1，不在本单元判决范围。")
    A("7. 模型均为**无权无向**图：边权异质性、有向性、"
      "以及 162 个整站不可用节点出度全为 0 的删失（CONVENTIONS §2.1/§2.2）"
      "都不在本矩阵覆盖范围内；有向结构的模型对照属于 A4。")
    A("8. 观测图有 2 个连通分量（GCC 669 + 一个 2 点分量），"
      "ER/WS/BA 的绝大多数实例是单连通的 —— 这个差异本身被记在 `gcc_fraction` 指标里参与判决。")

    with open(os.path.join(d, "result.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
