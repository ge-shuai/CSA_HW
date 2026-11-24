# -*- coding: utf-8 -*-
"""F3_template_verdict — 判别力/功效分析：选题 A（模板相似 ↔ 超链接相邻）值不值得做.

口径：CONVENTIONS §4（每轮 fresh 重标号；双侧 p 按偏离 null mean 计）、§7（措辞）、§9（产出）。
输入：out/F1_mantel_final/sim_matrices.npz + mantel_grid.csv（同一套签名与邻近口径）
      out/F2_template_families/family_edge_enrichment.csv（离散家族判据，作为参照统计量）
全部结论来自**模拟**（把已知大小的效应种进邻近矩阵，再跑同一套置换检验数检出率），
不引用教科书功效公式。

效应投放模型（写死）：k 条边按 P(edge ∝ exp(β·z(S))) 无放回抽样（Gumbel top-k 技巧），
k 恒等于观测边数；reweight 版保持观测边集与权重多重集不变，只按倾斜重新分配权重。
β=0 即"无关联"，用来核对该设计下置换检验的实际第一类错误率。

产出：power_curve.csv detectable_effect.csv null_calibration.csv equivalence_bound.csv
      required_data.csv measurement_reliability.csv depth_attenuation.csv
      signature_degeneracy.csv verdict.csv key_numbers.csv RUN.txt
"""
from __future__ import annotations

import os
import re
import sys
import time
import json

import numpy as np
import pandas as pd

SEED = 20261113
HERE = os.path.dirname(os.path.abspath(__file__))
F1DIR = os.path.join(os.path.dirname(HERE), "F1_mantel_final")
F2DIR = os.path.join(os.path.dirname(HERE), "F2_template_families")
PIPE = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, F2DIR)
sys.path.insert(0, PIPE)
import csa_common as C  # noqa: E402
import url_signature as US  # noqa: E402

MOD = "F3_template_verdict"
N_SIM_REP = int(os.environ.get("F3_N_REP", "250"))    # 每个 (设计, 效应档) 的模拟重复数（契约 >=200）
N_PERM_SIM = int(os.environ.get("F3_N_PERM", "199"))  # 每个模拟重复内部的置换次数
BLOCK = 25             # 每 BLOCK 个重复刷新一次置换集合（§4：不复用轨迹）
DEPTH_REP = int(os.environ.get("F3_DEPTH_REP", "150"))
REQ_NREP = int(os.environ.get("F3_REQ_REP", "200"))
ALPHA = 0.05
CHUNK = 25
T0 = time.time()
LOG, KEYS = [], {}


def log(m):
    LOG.append(f"[{time.time() - T0:7.1f}s] {m}")
    print(LOG[-1], flush=True)


def line_map():
    src = open(os.path.abspath(__file__), encoding="utf-8").read().splitlines()
    return {mm.group(1): i for i, ln in enumerate(src, 1) for mm in [re.search(r"#\s*MARK:\s*([A-Za-z0-9_]+)", ln)] if mm}


LM = line_map()


KEYLINES = {}


def key(k, v):
    KEYS[k] = v
    KEYLINES[k] = sys._getframe(1).f_lineno  # 调用处行号 = 该数字的产生位置


def pear(a, bv):
    a = a - a.mean()
    bv = bv - bv.mean()
    da, db = float(a @ a), float(bv @ bv)
    if da == 0 or db == 0:
        return np.nan
    return float(a @ bv / np.sqrt(da * db))


def seed_for(tag):
    """确定性种子派生：与 PYTHONHASHSEED 无关（不用内置 hash）。"""
    h = 0
    for ch in tag:
        h = (h * 131 + ord(ch)) % 9973
    return SEED + h


# ------------------------------------------------------------- 0. 输入 + 地基
NPZ = os.path.join(F1DIR, "sim_matrices.npz")
assert os.path.exists(NPZ), "先跑 F1（需要 sim_matrices.npz）"
z = np.load(NPZ, allow_pickle=True)  # MARK: load_inputs
NODES = [str(x) for x in z["nodes"]]
ELIG = [str(x) for x in z["elig"]]
IDX = {n: i for i, n in enumerate(NODES)}
N = len(NODES)
A_bin, A_w = z["A_bin"], z["A_w"]
S1, S2, S3 = z["S_S1_tok_jaccard"], z["S_S2_ext_cosine"], z["S_S3_tmpl_jaccard"]
S3H = z["S_S3_tmpl_hamming"]
iu_r, iu_c = np.triu_indices(N, 1)
eidx = np.array([IDX[n_] for n_ in ELIG])
er, ec = np.triu_indices(len(ELIG), 1)
k_obs = int(A_bin[iu_r, iu_c].sum())
key("n_edges_observed", k_obs)
key("n_pairs_671", int(len(iu_r)))
key("n_pairs_394", int(len(er)))

b = C.build_all()
C.assert_anchors(b)  # MARK: anchors
nd = b["nodes"].set_index("node").loc[NODES]
mg = pd.read_csv(os.path.join(F1DIR, "mantel_grid.csv"))  # MARK: load_grid
NODE_SET_671 = "attributed671"
_sets_elig = [s for s in mg["node_set"].unique() if s != NODE_SET_671 and "elig" in str(s)]
assert len(_sets_elig) == 1, f"预期的 elig 口径唯一，实际 {_sets_elig}"
NODE_SET_ELIG = _sets_elig[0]
log("loaded F1 npz + mantel_grid.csv: N=%d elig=%d edges=%d" % (N, len(ELIG), k_obs))

# ------------------------------------------------------------- 1. 签名能否支撑 Mantel
deg_rows = []  # MARK: degeneracy
for nm, Smat, rows, cols in [("S1_firstseg_tokens|671", S1, iu_r, iu_c),
                             ("S2_extension_mix|671", S2, iu_r, iu_c),
                             ("S3_path_fingerprint|671", S3, iu_r, iu_c),
                             ("S3_tmpl_hamming|671", S3H, iu_r, iu_c),
                             ("S1_firstseg_tokens|394", S1[np.ix_(eidx, eidx)], er, ec),
                             ("S3_path_fingerprint|394", S3[np.ix_(eidx, eidx)], er, ec)]:
    x = Smat[rows, cols]
    vc = pd.Series(np.round(x, 6)).value_counts()
    p = x / max(x.sum(), 1e-12)
    deg_rows.append({"matrix": nm, "n_pairs": len(x), "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
                     "mode_value": float(vc.index[0]), "frac_at_mode": float(vc.iloc[0] / len(x)),
                     "n_distinct": int(len(vc)), "frac_zero": float((x == 0).mean()),
                     "frac_ge_0p5": float((x >= 0.5).mean()), "frac_ge_0p9": float((x >= 0.9).mean()),
                     "effective_n_pairs": float(1.0 / (p * p).sum()),
                     "top1pct_share_of_similarity_mass": float(np.sort(x)[::-1][: int(0.01 * len(x))].sum() / max(x.sum(), 1e-12))})
dfdeg = pd.DataFrame(deg_rows)
C.save_csv(dfdeg, MOD, "signature_degeneracy.csv")
# 主/副口径的符号翻转（同一种签名，两种相似度定义给出反号 r）
flip = mg[mg.duplicated(subset=["signature", "proximity", "node_set"], keep=False)] \
    .sort_values(["signature", "proximity", "node_set", "flavor"])
flips = []
for (sg, px, nsm), gsub in flip.groupby(["signature", "proximity", "node_set"]):
    vals = dict(zip(gsub.flavor, gsub.mantel_r))
    if "primary" in vals and "secondary" in vals:
        flips.append({"signature": sg, "proximity": px, "node_set": gsub.node_set.iloc[0],
                      "r_primary": vals["primary"], "r_secondary": vals["secondary"],
                      "sign_flip": bool(np.sign(vals["primary"]) != np.sign(vals["secondary"])),
                      "abs_gap": abs(vals["primary"] - vals["secondary"])})
C.save_csv(pd.DataFrame(flips), MOD, "sign_flip_audit.csv")
key("n_sign_flips", int(sum(1 for f in flips if f["sign_flip"])))

# ------------------------------------------------------------- 2. 功效模拟
def perm_cache(Smat, rm, cm, n_perm, rng):  # MARK: perm_cache
    """n_perm 个"行列同时重标号"的相似度向量（分块生成，控制内存）。"""
    n = Smat.shape[0]
    Xc = np.empty((n_perm, len(rm)), dtype=np.float32)
    CH = 16
    for st in range(0, n_perm, CH):
        en = min(st + CH, n_perm)
        rows = []
        for _ in range(en - st):
            p = rng.permutation(n)
            rows.append(Smat[p[rm], p[cm]])
        XX = np.stack(rows)
        Xc[st:en] = XX - XX.mean(axis=1, keepdims=True)
    return Xc, np.sqrt((Xc * Xc).sum(1))


def plant_binary(x_std, beta, k, rng):
    """k 条边按 exp(beta*z) 倾斜、无放回投放（Gumbel top-k）；β=0 = 全域均匀随机边集。"""
    keys = beta * x_std + rng.gumbel(size=len(x_std))  # MARK: plant
    sel = np.argpartition(-keys, k - 1)[:k]
    y = np.zeros(len(x_std))
    y[sel] = 1.0
    return y, sel


def make_pool(deg, npool, rng):
    """按观测度分布抽端点 → 候选无序点对（去自环/去重）。
    用来构造"保持枢纽结构的随机图"（配置模型口径），作为全域均匀随机边集的对照零模型。"""
    p = (deg.astype(np.float64) + 1.0)
    p /= p.sum()
    n = len(deg)
    seen = set()
    out = []
    while len(out) < npool:
        m = 3 * (npool - len(out)) + 64
        u = rng.choice(n, size=m, p=p)
        v = rng.choice(n, size=m, p=p)
        for a, b in zip(u.tolist(), v.tolist()):
            if a == b:
                continue
            key_ = (a, b) if a < b else (b, a)
            if key_ in seen:
                continue
            seen.add(key_)
            out.append(key_)
            if len(out) == npool:
                break
    return np.array(out, dtype=np.int64)


def plant_config(Smat, pool, beta, k, zstats, rng):
    """在候选池上按 exp(beta*z) 倾斜取前 k 条边（保持枢纽度结构；β=0 = 池内均匀）。"""
    z_all, mean_, sd_ = zstats
    g = (beta * (z_all - mean_) / sd_ if beta > 0 else 0.0) + rng.gumbel(size=len(pool))  # MARK: plant_config
    sel = np.argpartition(-g, k - 1)[:k] if k < len(pool) else np.arange(len(pool))
    n = Smat.shape[0]
    y = np.zeros(n * (n - 1) // 2)
    y[pair_to_flat(pool[sel, 0], pool[sel, 1], n)] = 1.0
    return y, sel


def pair_to_flat(r_, c_, n):
    """无序点对 (a<b) → np.triu_indices(n,1) 的一维下标（行优先上三角）。"""
    return r_ * (n - 1) - (r_ * (r_ - 1)) // 2 + (c_ - r_ - 1)  # MARK: pair_to_flat


def plant_weighted(x_std, w_desc, edge_idx, beta, rng):
    """边集不变，把观测权重多重集按倾斜分配（权重最大的边落在倾斜最强的对上）。"""
    keys = beta * x_std[edge_idx] + rng.gumbel(size=len(edge_idx))
    order = np.argsort(-keys, kind="stable")
    y = np.zeros(len(x_std))
    y[edge_idx[order]] = w_desc
    return y


def power_run(Smat, kind, rows, cols, mask, beta, n_rep=N_SIM_REP, seed=None,
              k_edges=None, w_desc=None, edge_idx=None, pool=None, zstats=None):
    """一个 (签名, 邻近设计, 效应档 β) 的检出率与可检效应。

    kind: er_binary   = 全域均匀随机边集 + exp(βz) 倾斜（β=0 → 拓扑完全打散）
          config      = 按观测度序列抽候选池再倾斜（β=0 → 保留枢纽结构的零模型）
          reweight    = 保留观测边集与权重多重集，只按倾斜重排权重（β=0 → 权重与签名无关）
    """
    rng = np.random.default_rng(seed if seed is not None else SEED)
    rm, cm = rows[mask], cols[mask]
    x = Smat[rm, cm].astype(np.float64)
    xs = (x - x.mean()) / x.std(ddof=1)
    xc0 = x - x.mean()
    dx = float(xc0 @ xc0)
    rr, ll, pp, nsd = [], [], [], []
    done = 0
    while done < n_rep:  # MARK: power_block_loop
        nb = min(BLOCK, n_rep - done)
        Xc, xn = perm_cache(Smat, rm, cm, N_PERM_SIM, rng)   # 每块刷新置换集合
        Y = np.empty((nb, len(x)))
        for j in range(nb):
            if kind == "er_binary":
                Y[j] = plant_binary(xs, beta, k_edges, rng)[0]
            elif kind == "config":
                Y[j] = plant_config(Smat, pool, beta, k_edges, zstats, rng)[0]
            else:
                Y[j] = plant_weighted(xs, w_desc, edge_idx, beta, rng)
        Ym = Y.mean(axis=1, keepdims=True)
        Yc = (Y - Ym)
        yn = np.sqrt((Yc * Yc).sum(1))
        Yc32 = Yc.astype(np.float32)
        null = np.empty((N_PERM_SIM, nb), dtype=np.float64)
        for st in range(0, N_PERM_SIM, CHUNK):  # MARK: null_dot
            en = min(st + CHUNK, N_PERM_SIM)
            null[st:en] = (Xc[st:en] @ Yc32.T) / np.outer(xn[st:en], yn)
        nm_, ns_ = null.mean(0), null.std(0, ddof=1)
        obsr = (Yc @ xc0) / np.sqrt((Yc * Yc).sum(1) * dx)
        pval = (np.abs(null - nm_) >= np.abs((obsr - nm_)[None, :])).mean(axis=0)  # §4 双侧
        for j in range(nb):
            if kind == "reweight":
                w = Y[j]
                lift = float(np.average(x, weights=w) - x.mean()) / x.std(ddof=1) if w.sum() else np.nan
            else:
                sel = np.nonzero(Y[j])[0]
                lift = float((x[sel].mean() - x.mean()) / x.std(ddof=1)) if len(sel) else np.nan
            rr.append(float(obsr[j])); ll.append(lift); pp.append(float(pval[j])); nsd.append(float(ns_[j]))
        done += nb
    pv = np.array(pp)
    rej = pv <= ALPHA
    rr_a = np.array(rr)
    return {"beta": beta, "n_rep": n_rep, "mean_r": float(np.mean(rr)), "sd_r": float(np.std(rr)),
            "median_r": float(np.median(rr)), "mean_lift": float(np.nanmean(ll)),
            "r_q05": float(np.percentile(rr_a, 5)), "r_q95": float(np.percentile(rr_a, 95)),
            "se_mean_r": float(np.std(rr, ddof=1) / np.sqrt(n_rep)),
            "detection_rate": float(rej.mean()), "n_reject": int(rej.sum()),
            "perm_null_sd_mean": float(np.mean(nsd)),
            "se_detection": float(np.sqrt(rej.mean() * (1 - rej.mean()) / n_rep)),
            "median_p": float(np.median(pv))}


BETAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]  # MARK: beta_grid
ones671 = np.ones(len(iu_r), bool)
ones394 = np.ones(len(er), bool)
edge_idx_bin = np.nonzero(A_bin[iu_r, iu_c] > 0)[0]
w_desc = np.sort(A_w[iu_r, iu_c][edge_idx_bin])[::-1].astype(float)
S3E = S3[np.ix_(eidx, eidx)]
S1E = S1[np.ix_(eidx, eidx)]
k_elig = int(A_bin[np.ix_(eidx, eidx)].sum() / 2)
edge_idx_e = np.nonzero(A_bin[np.ix_(eidx, eidx)][er, ec] > 0)[0]
w_desc_e = np.sort(A_w[np.ix_(eidx, eidx)][er, ec][edge_idx_e])[::-1].astype(float)
deg671 = A_bin.sum(1)                      # 无向度（仅用于零模型抽样，不是报告口径）
degE = A_bin[np.ix_(eidx, eidx)].sum(1)
prng = np.random.default_rng(SEED + 9)     # MARK: config_pool
POOL671 = make_pool(deg671, 60000, prng)   # 候选池：按度分布抽出的无序点对
POOLE = make_pool(degE, 40000, np.random.default_rng(SEED + 10))
log("config 候选池: 671 口径 %d 对 / elig 口径 %d 对（重复 %d / %d 次）"
    % (len(POOL671), len(POOLE), 60000 - len(POOL671) + len(np.unique(POOL671, axis=0)), 0))


def zstats_of(Smat, pool, mean_base):
    z = Smat[pool[:, 0], pool[:, 1]]
    return z, float(z.mean()), float(z.std(ddof=1))


ZS3_671 = zstats_of(S3, POOL671, None)
ZS1_671 = zstats_of(S1, POOL671, None)
ZS3_E = zstats_of(S3E, POOLE, None)
DESIGNS = [
    ("er_binary", "S3_path_fingerprint", S3, iu_r, iu_c, ones671, k_obs, None, None, None, None),
    ("er_binary", "S1_firstseg_tokens", S1, iu_r, iu_c, ones671, k_obs, None, None, None, None),
    ("config", "S3_path_fingerprint", S3, iu_r, iu_c, ones671, k_obs, None, None, POOL671, ZS3_671),
    ("config", "S1_firstseg_tokens", S1, iu_r, iu_c, ones671, k_obs, None, None, POOL671, ZS1_671),
    ("reweight", "S3_path_fingerprint", S3, iu_r, iu_c, ones671, None, w_desc, edge_idx_bin, None, None),
    ("reweight", "S1_firstseg_tokens", S1, iu_r, iu_c, ones671, None, w_desc, edge_idx_bin, None, None),
    ("er_binary", "S3_path_fingerprint|elig394", S3E, er, ec, ones394, k_elig, None, None, None, None),
    ("er_binary", "S1_firstseg_tokens|elig394", S1E, er, ec, ones394, k_elig, None, None, None, None),
    ("config", "S3_path_fingerprint|elig394", S3E, er, ec, ones394, k_elig, None, None, POOLE, ZS3_E),
]
pow_rows = []  # MARK: power_curve
for kind, sname, Smat, rows, cols, mask, kk, wv, eix, pl, zs in DESIGNS:
    for beta in BETAS:
        r = power_run(Smat, kind, rows, cols, mask, beta,
                      seed=seed_for(sname + kind) + int(beta * 1000),
                      k_edges=kk, w_desc=wv, edge_idx=eix, pool=pl, zstats=zs)
        r.update({"design": kind, "signature": sname, "node_set": "394" if "elig" in sname else "671",
                  "k_planted": kk if kind != "reweight" else len(eix), "n_pairs": int(mask.sum()),
                  "alpha": ALPHA, "n_perm_per_rep": N_PERM_SIM})
        pow_rows.append(r)
        log("POWER %-9s %-28s beta=%5.2f r=%+.4f lift=%.3f detect=%.3f+-%.3f"
            % (kind, sname, beta, r["mean_r"], r["mean_lift"], r["detection_rate"], r["se_detection"]))
dfp = pd.DataFrame(pow_rows)
C.save_csv(dfp, MOD, "power_curve.csv")


def interp_target(sub, col, target):  # MARK: detectable
    s = sub.sort_values("beta")
    xv, yv = s[col].to_numpy(float), s.detection_rate.to_numpy(float)
    if yv.max() < target:
        return np.nan
    i = int(np.argmax(yv >= target))
    if i == 0:
        return float(xv[0])
    lo, hi = yv[i - 1], yv[i]
    return float(xv[i - 1] + (target - lo) * (xv[i] - xv[i - 1]) / (hi - lo)) if hi > lo else float(xv[i])


target_rows = []
for (kind, sname), gsub in dfp.groupby(["design", "signature"]):
    target_rows.append({"design": kind, "signature": sname,
                        "r_at_80pct_power": interp_target(gsub, "mean_r", 0.80),
                        "r_at_50pct_power": interp_target(gsub, "mean_r", 0.50),
                        "lift_at_80pct_power": interp_target(gsub, "mean_lift", 0.80),
                        "lift_at_50pct_power": interp_target(gsub, "mean_lift", 0.50),
                        "max_r_at_max_beta": float(gsub[gsub.beta > 0].mean_r.max()),
                        "max_detection": float(gsub.detection_rate.max()),
                        "detection_at_beta0": float(gsub[gsub.beta == 0].detection_rate.iloc[0])})
dft = pd.DataFrame(target_rows)
C.save_csv(dft, MOD, "detectable_effect.csv")

cal = dfp[dfp.beta == 0.0][["design", "signature", "node_set", "k_planted", "mean_r", "sd_r",
                            "se_mean_r", "r_q05", "r_q95",
                            "perm_null_sd_mean", "detection_rate", "se_detection", "median_p",
                            "n_rep"]].copy()
cal["nominal_alpha"] = ALPHA
cal["within_2se_of_alpha"] = np.abs(cal.detection_rate - ALPHA) <= 2 * cal.se_detection
C.save_csv(cal, MOD, "null_calibration.csv")
key("typeI_max_over_designs", float(cal.detection_rate.max()))
key("typeI_min_over_designs", float(cal.detection_rate.min()))
log("null calibration at beta=0: reject rates " + ", ".join(f"{v:.3f}" for v in cal.detection_rate))

# --------------------------------------------------- 2b. 零模型与这张图是否匹配（枢纽混杂）
# 同一套置换检验，在"完全不含模板信号、但保持观测度序列"的随机图上跑，r 的均值就是
# 拓扑本身能产生的假信号基线；§4 的置换零模型把 S 重标号，会连"度↔签名流行度"的耦合一起打掉。
cal2 = cal.copy()
def _obs_r(sig, node_set):
    """从 F1 的 mantel_grid 取该 (signature, node_set) 的观测 r。

    FIX 2026-11-23 (lead)：F1 的 mantel_grid 并不覆盖 F3 跑过的每一个组合，
    原实现直接 `.iloc[0]` → `IndexError: single positional indexer is out-of-bounds`，
    在写 null_model_audit.csv 之前把整个阶段打断。缺组合时返回 NaN，
    让 null_model_audit.csv 里留空，而不是让整轮 21 分钟的计算作废。
    """
    sel = mg[(mg.signature == sig) & (mg.proximity == "adj_binary") & (mg.flavor == "primary")
             & (mg.node_set == ("attributed671" if node_set == "671" else NODE_SET_ELIG))]
    return float(sel["mantel_r"].iloc[0]) if len(sel) else float("nan")


cal2["observed_r_same_cell"] = [_obs_r(r.signature, r.node_set) for r in cal.itertuples()]
cal2["typeI_se"] = cal2.se_detection
cal2["obs_minus_h0mean"] = cal2.observed_r_same_cell - cal2.mean_r
cal2["obs_beyond_h0_q95"] = cal2.observed_r_same_cell > cal2.r_q95
cal2["sd_h0_over_permnull"] = cal2.sd_r / cal2.perm_null_sd_mean
C.save_csv(cal2, MOD, "null_model_audit.csv")
for r in cal2.itertuples():
    log("NULLAUDIT %-9s %-28s h0_r=%+.4f(%.4f) h0_sd=%.4f perm_sd=%.4f obs=%+.4f beyond_q95=%s"
        % (r.design, r.signature, r.mean_r, r.se_mean_r, r.sd_r, r.perm_null_sd_mean,
           r.observed_r_same_cell, r.obs_beyond_h0_q95))
cfg = cal2[cal2.design == "config"]
key("h0mean_config_max", float(cfg.mean_r.max()) if len(cfg) else np.nan)

# 度 ↔ 平均签名相似度（签名流行度混杂的直接证据）
from scipy.stats import spearmanr  # noqa: E402  # MARK: popularity
pop_rows = []
for sname, Smat, nn in [("S1_firstseg_tokens", S1, NODES), ("S2_extension_mix", S2, NODES),
                        ("S3_path_fingerprint", S3, NODES)]:
    m = Smat.copy()
    np.fill_diagonal(m, np.nan)
    mean_sim = np.nanmean(m, axis=1)
    dg = np.array([b["Gu"].degree(n) for n in nn], float)
    lu = nd.loc[nn, "listed_urls"].to_numpy(float)
    rho_d, p_d = spearmanr(dg, mean_sim)
    rho_u, p_u = spearmanr(lu, mean_sim)
    pop_rows.append({"signature": sname, "spearman_rho_degree_vs_meansim": float(rho_d),
                     "p_degree": float(p_d), "spearman_rho_listedurls_vs_meansim": float(rho_u),
                     "p_listedurls": float(p_u),
                     "mean_sim_top10deg": float(np.mean(np.sort(mean_sim)[-10:])),
                     "mean_sim_median": float(np.median(mean_sim))})
C.save_csv(pd.DataFrame(pop_rows), MOD, "signature_popularity_confound.csv")
key("rho_degree_vs_meansim_S1", float(pop_rows[0]["spearman_rho_degree_vs_meansim"]))
key("rho_degree_vs_meansim_S3", float(pop_rows[2]["spearman_rho_degree_vs_meansim"]))

# 条件（保住边集）检验：加权问题应该问"已存在的边里，越像的是不是权重越大"
cond_rows = []  # MARK: conditional_test
for sname, Smat in [("S1_firstseg_tokens", S1), ("S2_extension_mix", S2), ("S3_path_fingerprint", S3)]:
    xe = Smat[iu_r, iu_c][edge_idx_bin]
    we = A_w[iu_r, iu_c][edge_idx_bin]
    if len(xe) < 3 or xe.std() == 0 or we.std() == 0:
        continue
    r_obs = pear(xe, we)
    rng_c = np.random.default_rng(SEED + 555)
    nulls = np.empty(N_PERM_SIM)
    for j in range(N_PERM_SIM):
        nulls[j] = pear(xe, we[rng_c.permutation(len(we))])
    nm_, ns_ = float(nulls.mean()), float(nulls.std(ddof=1))
    cond_rows.append({"signature": sname, "n_edges": len(xe), "r_within_edges": r_obs,
                      "h0_mean": nm_, "h0_sd": ns_, "z": (r_obs - nm_) / ns_,
                      "p_two_sided_deviation": float((np.abs(nulls - nm_) >= abs(r_obs - nm_)).mean()),
                      "note": "只在观测到的 2493 条边上打乱权重（条件于边集）"})
C.save_csv(pd.DataFrame(cond_rows), MOD, "conditional_edge_test.csv")
for c_ in cond_rows:
    log("COND %-18s r_within_edges=%+.4f z=%+.2f p=%.4f"
        % (c_["signature"], c_["r_within_edges"], c_["z"], c_["p_two_sided_deviation"]))

# ------------------------------------------------------------- 3. 等价边界（能排除到什么程度）
eq_rows = []  # MARK: equivalence
def h0_base(sname, node_set):
    """保持观测度序列、不含任何模板信号的随机图，在同一置换检验下给出的 r 均值。"""
    row = cal2[(cal2.design == "config") & (cal2.signature == sname)
               & (cal2.node_set == ("671" if node_set == NODE_SET_671 else "394"))]
    return float(row.mean_r.iloc[0]) if len(row) else np.nan


for sname in ("S1_firstseg_tokens", "S2_extension_mix", "S3_path_fingerprint"):
    sub = mg[(mg.signature == sname) & (mg.proximity == "adj_binary")
             & (mg.flavor == "primary")]
    for _, r in sub.iterrows():
        lo, hi = float(r["boot_ci_lo"]), float(r["boot_ci_hi"])
        ceil = float(r["r_ceiling_pos"])
        b0 = h0_base(sname, r["node_set"])
        eq_rows.append({"signature": sname, "proximity": "adj_binary", "node_set": r["node_set"],
                        "observed_r": float(r["mantel_r"]), "p_two_sided": float(r["p_two_sided"]),
                        "h0_baseline_config_model": b0,
                        "obs_minus_h0_baseline": float(r["mantel_r"]) - b0,
                        "boot_ci_lo": lo, "boot_ci_hi": hi, "boot_se": float(r["boot_se"]),
                        "perm_null_sd": float(r["null_sd"]),
                        "equivalence_margin_TOST": float(max(abs(lo), abs(hi))),
                        "can_rule_out_abs_r_gt": float(max(abs(lo), abs(hi))),
                        "r_gt_0.02_excluded": bool(hi < 0.02 and lo > -0.02),
                        "r_gt_0.05_excluded": bool(hi < 0.05 and lo > -0.05),
                        "r_gt_0.10_excluded": bool(hi < 0.10 and lo > -0.10),
                        "ceiling": ceil, "obs_share_of_ceiling": float(r["mantel_r"] / ceil) if ceil else np.nan})
C.save_csv(pd.DataFrame(eq_rows), MOD, "equivalence_bound.csv")

# ------------------------------------------------------------- 4. 需要多少条边
req_rows = []  # MARK: required_data
for kk in (k_obs, 5000, 10000, 25000, 50000):
    for beta in (0.05, 0.1, 0.2, 0.35, 0.6, 1.0):
        r = power_run(S3, "er_binary", iu_r, iu_c, ones671, beta, n_rep=REQ_NREP,
                      seed=seed_for("req%d%.2f" % (kk, beta)), k_edges=kk)
        r.update({"signature": "S3_path_fingerprint", "design": "er_binary", "node_set": "671",
                  "k_planted": kk, "k_over_observed": round(kk / k_obs, 3),
                  "implied_mean_degree": round(2 * kk / N, 2), "n_pairs": int(ones671.sum())})
        req_rows.append(r)
        log("REQ k=%6d (%.1fx) beta=%.2f r=%+.4f lift=%.3f detect=%.3f"
            % (kk, kk / k_obs, beta, r["mean_r"], r["mean_lift"], r["detection_rate"]))
reqdf = pd.DataFrame(req_rows)
C.save_csv(reqdf, MOD, "required_data.csv")
# 每个边数档位的 80% 功效可检 r（对 mean_r 插值）
rq = []
for kk, gsub in reqdf.groupby("k_planted"):
    rq.append({"k_edges": int(kk), "k_over_observed": round(kk / k_obs, 3),
               "implied_mean_degree": round(2 * kk / N, 2),
               "r_at_80pct_power": interp_target(gsub, "mean_r", 0.80),
               "r_at_50pct_power": interp_target(gsub, "mean_r", 0.50),
               "max_detection": float(gsub.detection_rate.max()),
               "mean_r_at_max_beta": float(gsub.mean_r.max())})
C.save_csv(pd.DataFrame(rq), MOD, "required_data_summary.csv")
for x in rq:
    log("REQSUM k=%6d (%.1fx) 可检 r@80%%=%s" % (x["k_edges"], x["k_over_observed"],
                                                f"{x['r_at_80pct_power']:.5f}"
                                                if np.isfinite(x["r_at_80pct_power"]) else "不可达"))

# ------------------------------------------------------------- 5. 抓取深度：信度与衰减
full = US.build(check_against_f1=True, url_cap=None)  # MARK: split_half
EL = ELIG
urls = C.load_urls()
by_site = {s: g["url"].astype(str).tolist() for s, g in urls.groupby("subdomain")}
pv = full["tmpl_vocab"]
rng = np.random.default_rng(SEED)
A_parts, B_parts = {}, {}
for n_ in EL:  # 按 URL 行随机二分（不是按去重后的模板），再取各自的路径模板集合
    us = by_site.get(n_, [])
    lst = sorted(US.parse_url(u)["tmpl"] for u in us)
    pm = rng.permutation(len(lst))
    h = len(lst) // 2
    A_parts[n_] = {lst[i] for i in pm[:h]}
    B_parts[n_] = {lst[i] for i in pm[h:]}


def binm(parts, vocab):
    col = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((len(EL), len(vocab)), dtype=np.float32)
    for i, n_ in enumerate(EL):
        for it in parts[n_]:
            j = col.get(it)
            if j is not None:
                X[i, j] = 1.0
    return X


SA = US.sim_jaccard(binm(A_parts, pv))
SB = US.sim_jaccard(binm(B_parts, pv))
a, bb = SA[er, ec], SB[er, ec]
r_half = pear(a, bb)
r_sb = 2 * r_half / (1 + r_half)
rel_rows = [{"measure": "split_half 模板集合(394 站, 词表固定为全量词表)", "n_sites": len(EL),
             "mean_urls_per_site": float(np.mean([len(by_site.get(n_, [])) for n_ in EL])),
             "corr_between_halves": float(r_half),
             "spearman_brown_reliability_of_full": float(r_sb),
             "agreement_with_full_signature": np.nan,
             "frac_pairs_zero_in_both_halves": float(((a == 0) & (bb == 0)).mean())}]
S3E_full = S3[np.ix_(eidx, eidx)]
for cap in (2, 3, 5, 10, 15, 20):  # MARK: depth_reliability
    s2 = US.build(check_against_f1=False, url_cap=cap, sub_seed=SEED + 31)
    idx2 = [s2["idx"][n_] for n_ in EL]
    x2 = s2["SIMS"]["S3_tmpl_jaccard"][np.ix_(idx2, idx2)][er, ec]
    rel_rows.append({"measure": f"cap={cap} URL/站 重算签名", "n_sites": len(EL),
                     "mean_urls_per_site": float(np.mean([min(cap, len(by_site.get(n_, []))) for n_ in EL])),
                     "corr_between_halves": np.nan, "spearman_brown_reliability_of_full": np.nan,
                     "agreement_with_full_signature": float(pear(x2, S3E_full[er, ec])),
                     "frac_pairs_zero_in_both_halves": float((x2 == 0).mean())})
C.save_csv(pd.DataFrame(rel_rows), MOD, "measurement_reliability.csv")
key("split_half_r", round(float(r_half), 4))
key("spearman_brown_full", round(float(r_sb), 4))

dep_rows = []  # MARK: depth_power
# 衰减校正的参照：F1 实测的两档 |r|（最强的 S1×二值 与 同学口径的 S3×二值）
_obs = mg[(mg.proximity == "adj_binary") & (mg.flavor == "primary")
          & (mg.node_set == "attributed671")].set_index("signature")["mantel_r"]
REF_S1 = float(_obs["S1_firstseg_tokens"])
REF_S3 = float(_obs["S3_path_fingerprint"])
log("衰减校正参照 r：S1=%+.5f, S3=%+.5f" % (REF_S1, REF_S3))
for cap in (2, 3, 5, 10, 15, 20):
    s2 = US.build(check_against_f1=False, url_cap=cap, sub_seed=SEED + 77)
    idx2 = [s2["idx"][n_] for n_ in EL]
    Sdeg = s2["SIMS"]["S3_tmpl_jaccard"][np.ix_(idx2, idx2)][er, ec]
    xf = S3E_full[er, ec]
    xs = (xf - xf.mean()) / xf.std(ddof=1)
    rng2 = np.random.default_rng(SEED + 1234 + cap)
    rt, rd = [], []
    for _ in range(150):
        y, _sel = plant_binary(xs, 1.5, k_elig, rng2)
        rt.append(pear(xf, y)); rd.append(pear(Sdeg, y))
    att = float(np.mean(rd) / np.mean(rt))
    dep_rows.append({"url_cap": cap, "beta_planted": 1.5, "k_edges": k_elig, "n_rep": len(rt),
                     "mean_r_measured_with_full_20url": float(np.mean(rt)),
                     "mean_r_measured_with_cap_d": float(np.mean(rd)),
                     "attenuation_factor": att,
                     "true_r_implied_by_observed_S1_r": float(REF_S1 / att) if att else np.nan,
                     "true_r_implied_by_observed_S3_r": float(REF_S3 / att) if att else np.nan,
                     "obs_ref_S1_r": REF_S1, "obs_ref_S3_r": REF_S3})
    log("DEPTH cap=%2d atten=%.3f  r_true=%.4f -> r_meas=%.4f"
        % (cap, att, dep_rows[-1]["mean_r_measured_with_full_20url"], dep_rows[-1]["mean_r_measured_with_cap_d"]))
C.save_csv(pd.DataFrame(dep_rows), MOD, "depth_attenuation.csv")

# ------------------------------------------------------------- 6. 判决
verdict = []  # MARK: verdict
def dftrow(design, sig):
    sub = dft[(dft.design == design) & (dft.signature == sig)]
    return sub.iloc[0] if len(sub) else None


p3 = dftrow("er_binary", "S3_path_fingerprint")
p1 = dftrow("er_binary", "S1_firstseg_tokens")
p1e = dftrow("er_binary", "S1_firstseg_tokens|elig394")
p3c = dftrow("config", "S3_path_fingerprint")
p1c = dftrow("config", "S1_firstseg_tokens")
eqdf = pd.DataFrame(eq_rows)


def eqv(sname, node="attributed671", col="observed_r"):
    r = eqdf[(eqdf.signature == sname) & (eqdf.node_set == node)]
    return float(r[col].iloc[0])


r_s1_671 = eqv("S1_firstseg_tokens")
r_s1_394 = eqv("S1_firstseg_tokens", NODE_SET_ELIG)
r_s3_671 = eqv("S3_path_fingerprint")
ceil_s3 = eqv("S3_path_fingerprint", col="ceiling")
tost_s3 = eqv("S3_path_fingerprint", col="equivalence_margin_TOST")
enr_path = os.path.join(F2DIR, "family_edge_enrichment.csv")
lift_fam = np.nan
if os.path.exists(enr_path):
    enr = pd.read_csv(enr_path)
    lift_fam = float(enr.loc[enr["item"] == "lift (obs/exp)", "value"].iloc[0])

def vi(item, value, nd=5):
    """NaN 安全的 verdict 行（80% 功效不可达时写成人话，不写 nan）。"""
    try:
        f = float(value)
    except (TypeError, ValueError):
        verdict.append({"item": item, "value": str(value)})
        return
    verdict.append({"item": item,
                    "value": round(f, nd) if np.isfinite(f) else "不可达/未定义（见 detectable_effect.csv）"})


vi("判决（一句话）",
   "不做——按现在这份 20 条/站的抓取和这三种签名，Mantel 检验既没有能力确认、"
   "也没有能力排除\"共享模板与超链接相邻\"这条量级不大的关系，阴性/弱阳性都不携带信息")
vi("现有最强正相关（671 口径, S1×二值邻接）", r_s1_671)
vi("同一结果在 394 站（签名可靠区间）口径", r_s1_394)
vi("S3 路径指纹×二值邻接 r（671）", r_s3_671)
vi("该设计可表达的效应上界 r_ceiling", ceil_s3, 4)
vi("80% 功效最小可检 r（S3 671）", p3.r_at_80pct_power)
vi("80% 功效最小可检 lift（S3 671, sd 单位）", p3.lift_at_80pct_power, 4)
vi("80% 功效最小可检 r（S1 671）", p1.r_at_80pct_power)
vi("80% 功效最小可检 r（S1 394）", p1e.r_at_80pct_power)
vi("TOST 等价边际（S3 671，能排除 |r|>该值）", tost_s3, 4)
vi("β=0 时检验的实际第一类错误率范围",
   f"[{KEYS['typeI_min_over_designs']:.3f}, {KEYS['typeI_max_over_designs']:.3f}]")
vi("split-half 信度 / Spearman–Brown 全量信度", f"{r_half:.3f} / {r_sb:.3f}")
vi("F2 离散家族判据的边内富集 lift", lift_fam, 4)
vi("有向边数 / 无序点对数（671 口径能看见的比例 %）", 100 * k_obs / len(iu_r), 4)
C.save_csv(pd.DataFrame(verdict), MOD, "verdict.csv")

kn = []  # MARK: keys_save
for k_, v_ in KEYS.items():
    kn.append({"key": k_, "value": round(v_, 6) if isinstance(v_, float) else v_,
               "src_line": "run_f3.py:L%d" % KEYLINES[k_]})
for _, r in dft.iterrows():
    kn.append({"key": f"detectable|{r.design}|{r.signature}|r_at_80pct_power",
               "value": round(float(r.r_at_80pct_power), 5), "src_line": "run_f3.py:L%d" % LM["detectable"]})
    kn.append({"key": f"detectable|{r.design}|{r.signature}|lift_at_80pct_power",
               "value": round(float(r.lift_at_80pct_power), 4), "src_line": "run_f3.py:L%d" % LM["detectable"]})
C.save_csv(pd.DataFrame(kn), MOD, "key_numbers.csv")
C.save_csv(pd.DataFrame([{"mark": m, "line": LM[m]} for m in LM]), MOD, "mark_lines.csv")

wall = time.time() - T0
C.write_run_note(MOD, [
    "F3_template_verdict RUN.txt",
    "command: cd pipeline && PYTHONHASHSEED=0 python out/F3_template_verdict/run_f3.py",
    "versions: " + json.dumps(C.versions(), ensure_ascii=False),
    f"SEED = {SEED}; n_sim_rep = {N_SIM_REP}; n_perm_per_sim_rep = {N_PERM_SIM};",
    f"  置换集合每 {BLOCK} 个模拟重复重新生成一次（§4：独立重复，不在同一轨迹上取点）",
    f"planting: k 条边按 exp(beta*z(S)) 无放回投放（Gumbel top-k），k 固定为观测边数 {k_obs}（671）/ {k_elig}（394）",
    "inputs: F1 sim_matrices.npz, F1 mantel_grid.csv, F2 family_edge_enrichment.csv",
    "双侧 p = mean(|null-mean(null)| >= |obs-mean(null)|)   [CONVENTIONS 4]",
    f"wall time: {wall:.1f}s",
    "log:", *LOG,
])
log("done in %.1f s" % wall)
