"""D1 · health_gradient — 暴露变量十分位 × 失效率的健康梯度（校正前 / 校正后两套）。

口径全部来自 csa_common（CONVENTIONS §1/§2/§7/§8）：
  §2.1 失效相关分析主口径 = **入度**；无向总度只作对照，且必须并列给出（本模块 d1_exposure_contrast.csv）。
  §2.2 真失效 = A_conn_fail(111)+D_not_found(12)+F_server_error(6)=129；
       假失效 = B_auth_or_skip(18)+C_robots_blocked(14)=32 → 校正时从样本剔除；
       G_other_400(1) → 单列"待人工判断"。每份结果都给校正前/校正后两套。
  §7   单次横截面 → 只写"与…相关 / 在…上可预测"，不写机制。
  §8   SEED=20261113；pandas 3.0 只能用 pd.qcut（DataFrame.qcut 已移除）。

分箱规范（本模块自带"静默拆并列值"审计，禁止让 qcut 悄悄决定边界）：
  主口径   scheme=rank10   ：先按 (取值, FQDN) 稳定排序，再 pd.qcut(秩, 10) → **恰好 10 箱、每箱 67/68 人**。
                        代价：并列值会被切开，所以每行都给出该箱边界的 tie_straddle_* 计数，
                        读者能自己判断"这条梯度里有多少来自任意切点"。
  稳健口径 scheme=value_aware：同一取值的站点**永不拆开** → 箱数 ≤10、箱人数不等，边界唯一。
  直接 pd.qcut(原值, 10) 在本数据上会**报错**（bin edges not unique）；
  加 duplicates='drop' 则静默退化成 7/6/8/3 箱、首箱 279 人 —— 见 d1_binning_tie_audit.csv。
  单列     url_count / listed_urls 的堆积值（=1、=20、≈100 抓取预算）各自独立成层，见 cap_bins 表。

运行：
    cd pipeline && PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8 python -X utf8 out/D1_health_gradient/run_d1.py
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PIPELINE)
import csa_common as C  # noqa: E402

MODULE = "D1_health_gradient"
SEED = C.SEED  # 20261113
N_Q = 10
np.random.seed(SEED)                      # 全局兜底；本模块全部计算为确定性统计量
rng = np.random.default_rng(SEED)         # 显式 rng（同 SEED），当前无抽样调用

# ============================================================================ 数据
T0 = time.time()
b = C.build_all()
BUILD_LINE = inspect.currentframe().f_lineno
C.assert_anchors(b)                       # 31 项锚点
ANCHOR_LINE = inspect.currentframe().f_lineno - 1
nodes = b["nodes"]
VERSIONS = C.versions()

nodes = nodes.sort_values("node", kind="mergesort").reset_index(drop=True)
nodes["_removed"] = nodes["false_fail"] | nodes["ambiguous_fail"]     # 校正时剔除的 32+1
S_uncorr = nodes.copy()                                              # 校正前 671
S_corr = nodes.loc[~nodes["_removed"]].copy()                        # 校正后 638
N_UNCORR, N_CORR = len(S_uncorr), len(S_corr)

OUTCOMES = [  # (key, 中文, 分子掩码列, 校正归属)
    ("dead", "全挂率", "is_dead", "uncorr"),
    ("partial", "部分失效率", "is_partial", "uncorr"),
    ("true_fail", "真失效率", "true_fail", "corr"),
    ("clean_part", "干净部分失效率", "partial_clean", "corr"),
]
EXPOSURES = [
    ("degree_u", "无向总度"),
    ("in_degree", "入度"),
    ("url_count", "url_count(抓取器报告页面总数)"),
    ("listed_urls", "listed_urls(落库明细数,上限20)"),
]


# ============================================================================ 工具
def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple:
    """Wilson score 区间（不引第三方库）。"""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * float(np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (max(0.0, c - h), min(1.0, c + h))


def value_aware_bins(series: pd.Series, q: int = N_Q) -> pd.DataFrame:
    """值感知分箱：并列值永不拆开（箱数 ≤ q，人数不等）。"""
    vc = series.value_counts().sort_index()
    n = int(vc.sum())
    target = n / q
    groups, cur, acc = [], [], 0
    for v, c in vc.items():
        if cur and acc >= target:                 # 当前箱已达目标人数 → 封箱
            groups.append(cur)
            cur, acc = [], 0
        cur.append((v, int(c)))
        acc += int(c)
    if cur:
        groups.append(cur)
    rows = [{"bin": i + 1, "lo": g[0][0], "hi": g[-1][0], "n": sum(c for _, c in g),
             "n_unique": len(g)} for i, g in enumerate(groups)]
    return pd.DataFrame(rows)


def assign_bins_by_value(series: pd.Series, plan: pd.DataFrame) -> np.ndarray:
    lab = np.full(len(series), -1, dtype=int)
    arr = series.to_numpy()
    for _, r in plan.iterrows():
        lab[(arr >= r["lo"]) & (arr <= r["hi"])] = int(r["bin"])
    if (lab < 0).any():
        raise RuntimeError("值感知分箱未覆盖全部样本")
    return lab


def rank_decile_labels(df: pd.DataFrame, col: str, q: int = N_Q):
    """主口径分箱：pd.qcut 作用在"按 (取值, FQDN) 稳定排序后的秩"上 → 恰好 q 个等人数箱。

    返回 (labels, meta)；labels 与 df 索引对齐，meta 每箱一行，含边界并列人数。
    """
    ordered = df.sort_values([col, "node"], kind="mergesort")
    rank = pd.Series(np.arange(len(ordered), dtype=float), index=ordered.index)
    lab = pd.qcut(rank, q, labels=False) + 1                      # pandas 3.0: 只能用 pd.qcut
    lab = lab.astype(int).reindex(df.index)
    vals = df[col].to_numpy()
    meta = []
    for k in range(1, q + 1):
        sel = (lab.to_numpy() == k)
        vlo, vhi = int(vals[sel].min()), int(vals[sel].max())
        n_lo = int((vals == vlo).sum())                            # 该边界值在全样本的人数
        n_lo_in = int(((vals == vlo) & sel).sum())
        n_hi = int((vals == vhi).sum())
        n_hi_in = int(((vals == vhi) & sel).sum())
        meta.append({"bin": k, "lo": vlo, "hi": vhi, "n": int(sel.sum()),
                     "n_unique": int(len(set(vals[sel]))),
                     "tie_straddle_bottom_value": n_lo, "tie_straddle_bottom_in_bin": n_lo_in,
                     "tie_straddle_top_value": n_hi, "tie_straddle_top_in_bin": n_hi_in})
    return lab, pd.DataFrame(meta)


def bin_stats(sub: pd.DataFrame) -> dict:
    """一个箱（子样本）→ 4 个失效率的分子/分母/率/Wilson CI。

    分母口径 = 该箱在当前样本里的全部节点（671 表用 671 分母；638 表用 638 分母）。
    分子口径：dead/partial 用原始掩码；true_fail/clean_part 用校正后掩码。
    """
    n = len(sub)
    out = {"n_bin": n}
    for key, _cn, mask, _c in OUTCOMES:
        k = int(sub[mask].sum())
        lo, hi = wilson(k, n)
        out[f"{key}_n"] = k
        out[f"{key}_rate"] = round(k / n, 4) if n else np.nan
        out[f"{key}_ci_lo"] = round(lo, 4)
        out[f"{key}_ci_hi"] = round(hi, 4)
    return out


def make_table(sdf: pd.DataFrame, expo: str, expo_cn: str, sample_tag: str, correction: str,
               scheme: str) -> pd.DataFrame:
    if scheme == "rank10":
        lab, meta = rank_decile_labels(sdf, expo, N_Q)
    else:
        plan = value_aware_bins(sdf[expo].rename(expo), N_Q)
        lab = pd.Series(assign_bins_by_value(sdf[expo].rename(expo), plan), index=sdf.index)
        meta = plan.assign(tie_straddle_bottom_value=plan["n"], tie_straddle_bottom_in_bin=plan["n"],
                           tie_straddle_top_value=plan["n"], tie_straddle_top_in_bin=plan["n"])
    rows = []
    for _, m in meta.iterrows():
        sub = sdf[lab.to_numpy() == int(m["bin"])]
        r = {"exposure": expo, "exposure_cn": expo_cn, "sample_tag": sample_tag, "n_sample": len(sdf),
             "correction": correction, "scheme": scheme, "bin": int(m["bin"]),
             "bin_label": (f"[{m['lo']}, {m['hi']}]" if m["lo"] != m["hi"] else f"={m['lo']}"),
             "bin_lo": m["lo"], "bin_hi": m["hi"], "n_unique_values": int(m["n_unique"]),
             "tie_straddle_bottom_value": int(m["tie_straddle_bottom_value"]),
             "tie_straddle_bottom_in_bin": int(m["tie_straddle_bottom_in_bin"]),
             "tie_straddle_top_value": int(m["tie_straddle_top_value"]),
             "tie_straddle_top_in_bin": int(m["tie_straddle_top_in_bin"]),
             "bin_mean_exposure": round(float(sub[expo].mean()), 4),
             "share_of_true_fail_in_bin": (round(float(sub["true_fail"].sum() / sub["is_dead"].sum()), 4)
                                           if sub["is_dead"].sum() else np.nan)}
        r.update(bin_stats(sub))
        rows.append(r)
    return pd.DataFrame(rows)


def linear_binomial_lrt(cnt: np.ndarray, nn: np.ndarray, x: np.ndarray):
    """分箱计数上的线性趋势检验（Cochran–Armitage 的 GLM 版）：
    M0 logit(p)=a  vs  M1 logit(p)=a+b·x → (LR chi2, p, b)。"""
    cnt = cnt.astype(float)
    nn = nn.astype(float)

    def nll(par):
        a, bb = float(par[0]), float(par[1])
        eta = np.clip(a + bb * x, -35, 35)
        p = np.clip(1 / (1 + np.exp(-eta)), 1e-12, 1 - 1e-12)
        return -float(np.sum(cnt * np.log(p) + (nn - cnt) * np.log(1 - p)))

    m0 = minimize(lambda q: nll([q[0], 0.0]), [0.0, 0.0], method="Nelder-Mead",
                  options=dict(xatol=1e-10, fatol=1e-10, maxiter=40000, maxfev=40000))
    m1 = minimize(nll, [0.0, 0.0], method="Nelder-Mead",
                  options=dict(xatol=1e-10, fatol=1e-10, maxiter=80000, maxfev=80000))
    lr = max(2 * (m0.fun - m1.fun), 0.0)
    return round(float(lr), 4), float(stats.chi2.sf(lr, 1)), round(float(m1.x[1]), 4)


def steepness(tab: pd.DataFrame) -> list:
    """一张分箱表 → 4 个失效率各自的"梯度陡窄"量化。"""
    out = []
    t = tab.sort_values("bin")
    for key, cn, mask, corr in OUTCOMES:
        bot, top = t.iloc[0], t.iloc[-1]
        r_lo, r_hi = float(bot[f"{key}_rate"]), float(top[f"{key}_rate"])
        rho, p = stats.spearmanr(t["bin_mean_exposure"], t[f"{key}_rate"])
        lr, p_lr, slope = linear_binomial_lrt(t[f"{key}_n"].to_numpy(dtype=float),
                                             t["n_bin"].to_numpy(dtype=float),
                                             t["bin"].to_numpy(dtype=float))
        rr = (r_hi / r_lo) if r_lo > 0 else np.inf
        out.append({
            "exposure": t["exposure"].iloc[0], "sample_tag": t["sample_tag"].iloc[0],
            "scheme": t["scheme"].iloc[0], "correction_of_numerator": corr,
            "outcome": key, "outcome_cn": cn, "n_bins": len(t),
            "bottom_bin": int(bot["bin"]), "bottom_label": bot["bin_label"],
            "bottom_n": int(bot["n_bin"]), "bottom_rate": round(r_lo, 4),
            "bottom_ci": [bot[f"{key}_ci_lo"], bot[f"{key}_ci_hi"]],
            "top_bin": int(top["bin"]), "top_label": top["bin_label"],
            "top_n": int(top["n_bin"]), "top_rate": round(r_hi, 4),
            "top_ci": [top[f"{key}_ci_lo"], top[f"{key}_ci_hi"]],
            "rate_ratio_top_over_bottom": (round(float(rr), 3) if np.isfinite(rr) else "inf"),
            "rate_diff_pp": round((r_hi - r_lo) * 100, 2),
            "spearman_binlevel_rho": (round(float(rho), 4) if np.isfinite(rho) else np.nan),
            "spearman_binlevel_p": (float(p) if np.isfinite(p) else np.nan),
            "linear_trend_LRchi2": lr, "linear_trend_p": p_lr, "linear_trend_logit_b_per_bin": slope,
            "monotone_in_bin_order": bool(t[f"{key}_rate"].is_monotonic_increasing
                                          or t[f"{key}_rate"].is_monotonic_decreasing),
        })
    return out


def naive_qcut_audit(df: pd.DataFrame, col: str) -> dict:
    """审计：直接 pd.qcut(原值,10) 在本列上会怎样（报错 / duplicates='drop' 静默退化 / 拆并列）。"""
    v = df[col]
    try:
        nb, sizes = "error", None
        pd.qcut(v, N_Q, labels=False)
        nb = 10
    except ValueError as e:
        sizes = str(e).split("Index(")[1].split(")")[0] if "Index(" in str(e) else str(e)[:120]
    try:
        d = pd.qcut(v, N_Q, labels=False, duplicates="drop")
        nd, dsizes = int(d.nunique()), d.value_counts().sort_index().to_list()
    except ValueError:
        nd, dsizes = -1, []
    lab, meta = rank_decile_labels(df, col, N_Q)
    n_split = int(((meta["tie_straddle_bottom_value"] > meta["tie_straddle_bottom_in_bin"])
                   | (meta["tie_straddle_top_value"] > meta["tie_straddle_top_in_bin"])).sum())
    modal = v.mode().iloc[0]
    return {"exposure": col, "n": len(df), "n_unique": int(v.nunique()),
            "qcut_native_10": nb, "qcut_native_dup_edges": sizes,
            "qcut_duplicates_drop_bins": nd, "qcut_duplicates_drop_sizes": json.dumps(dsizes),
            "largest_bin_if_dropped": int(max(dsizes)) if dsizes else 0,
            "rank10_bins": N_Q, "rank10_bins_with_split_ties": n_split,
            "value_aware_bins": int(len(value_aware_bins(v.rename(col), N_Q))),
            "modal_value": int(modal), "modal_count": int((v == modal).sum()),
            "max_tiegroup_size": int(v.value_counts().max())}


# ============================================================================ 主表
tabs, summ = [], []
for expo, expo_cn in EXPOSURES:
    for sample_tag, correction, sdf in (("S_uncorr_671", "校正前", S_uncorr),
                                        ("S_corr_638", "校正后", S_corr)):
        for scheme in ("rank10", "value_aware"):
            tab = make_table(sdf, expo, expo_cn, sample_tag, correction, scheme)
            tabs.append(tab)
            summ.append(pd.DataFrame(steepness(tab)))
deciles = pd.concat(tabs, ignore_index=True)
summary = pd.concat(summ, ignore_index=True)

# 节点级 Spearman（连续口径 ok_ratio_pct + 各 0/1 结果）——对齐 §2.1 引用的 0.376 / 0.216
NODE_EXPOSURES = list(EXPOSURES) + [("out_degree", "出度(死站恒为0, 仅作删失诊断)"),
                                     ("strength_u", "无向强度(观察到的链接次数之和)")]
nl_rows = []
for expo, expo_cn in NODE_EXPOSURES:
    for ycol, ycn in [("is_dead", "全挂"), ("is_partial", "部分失效"), ("true_fail", "真失效"),
                      ("partial_clean", "干净部分失效"), ("ok_ratio_pct", "URL可用率(连续)")]:
        r1, p1 = stats.spearmanr(S_uncorr[expo], S_uncorr[ycol].astype(float))
        keep = ycol in ("is_dead", "is_partial", "true_fail", "partial_clean")
        r2, p2 = stats.spearmanr(S_corr[expo], S_corr[ycol].astype(float)) if keep else (np.nan, np.nan)
        nl_rows.append({"exposure": expo, "exposure_cn": expo_cn, "outcome": ycol, "outcome_cn": ycn,
                        "rho_s_uncorr_671": round(float(r1), 4), "p_uncorr": float(p1),
                        "rho_s_corr_638": (round(float(r2), 4) if np.isfinite(r2) else "NA"),
                        "p_corr": (float(p2) if np.isfinite(p2) else "NA"),
                        "n_corr": (N_CORR if keep else "NA")})
node_level = pd.DataFrame(nl_rows)

# 分箱方案敏感性：rank10 vs value_aware 的陡窄差
sens = []
for expo, expo_cn in EXPOSURES:
    for outcome in ("dead", "partial", "true_fail"):
        a = summary[(summary.exposure == expo) & (summary.outcome == outcome)
                    & (summary.scheme == "rank10") & (summary.sample_tag == "S_uncorr_671")].iloc[0]
        bb = summary[(summary.exposure == expo) & (summary.outcome == outcome)
                     & (summary.scheme == "value_aware") & (summary.sample_tag == "S_uncorr_671")].iloc[0]
        sens.append({"exposure": expo, "outcome": outcome, "outcome_cn": a["outcome_cn"],
                     "rank10_n_bins": a["n_bins"], "value_aware_n_bins": bb["n_bins"],
                     "rank10_rate_ratio": a["rate_ratio_top_over_bottom"],
                     "value_aware_rate_ratio": bb["rate_ratio_top_over_bottom"],
                     "rank10_rate_diff_pp": a["rate_diff_pp"], "value_aware_rate_diff_pp": bb["rate_diff_pp"],
                     "rank10_spearman": a["spearman_binlevel_rho"], "value_aware_spearman": bb["spearman_binlevel_rho"],
                     "abs_diff_rate_diff_pp": round(abs(float(a["rate_diff_pp"]) - float(bb["rate_diff_pp"])), 2),
                     "note": "rank10 会拆并列值；value_aware 不拆但箱人数不等"})
sens_df = pd.DataFrame(sens)

# 堆积值 / 上限单列
def strata_rows(col_label: str, preds: list) -> pd.DataFrame:
    rows = []
    for name, pred in preds:
        sub = S_uncorr.loc[pred(S_uncorr)]
        if not len(sub):
            continue
        r = {"variable": col_label, "stratum": name, "n": len(sub), "share_of_671": round(len(sub) / N_UNCORR, 4)}
        r.update(bin_stats(sub))
        r["mean_in_degree"] = round(float(sub["in_degree"].mean()), 3)
        r["mean_degree_u"] = round(float(sub["degree_u"].mean()), 3)
        r["mean_out_degree"] = round(float(sub["out_degree"].mean()), 3)
        r["n_dead_with_out0"] = int((sub["is_dead"] & (sub["out_degree"] == 0)).sum())
        rows.append(r)
    return pd.DataFrame(rows)


urlcap = strata_rows("url_count", [
    ("=1", lambda d: d.url_count == 1),
    ("2-9", lambda d: d.url_count.between(2, 9)),
    ("=20（题面所指『20 条上限』堆积处）", lambda d: d.url_count == 20),
    ("21-98", lambda d: d.url_count.between(21, 98)),
    ("99-105（≈100 抓取预算堆积）", lambda d: d.url_count.between(99, 105)),
    (">105", lambda d: d.url_count > 105),
])
listcap = strata_rows("listed_urls", [
    ("=1", lambda d: d.listed_urls == 1),
    ("2-19", lambda d: d.listed_urls.between(2, 19)),
    ("=20（真正的落库上限，ok_ratio_pct 的分母）", lambda d: d.listed_urls == 20),
])
trunc = strata_rows("url_count vs listed_urls", [
    ("url_count>listed_urls（确有未落库页面）", lambda d: d.url_count > d.listed_urls),
    ("url_count==listed_urls（明细已穷尽）", lambda d: d.url_count == d.listed_urls),
])

expo_dist = pd.DataFrame([{
    "exposure": c, "exposure_cn": cn, "n": N_UNCORR,
    "min": int(S_uncorr[c].min()), "max": int(S_uncorr[c].max()),
    "n_unique": int(S_uncorr[c].nunique()),
    "mode": int(S_uncorr[c].mode().iloc[0]), "mode_count": int((S_uncorr[c] == S_uncorr[c].mode().iloc[0]).sum()),
    "share_at_mode": round(float((S_uncorr[c] == S_uncorr[c].mode().iloc[0]).mean()), 4),
    "mean": round(float(S_uncorr[c].mean()), 3), "median": float(S_uncorr[c].median()),
    "p25": float(S_uncorr[c].quantile(.25)), "p75": float(S_uncorr[c].quantile(.75)),
    "p90": float(S_uncorr[c].quantile(.90)),
    "n_at_20": int((S_uncorr[c] == 20).sum()),
    "top_decile_cut_naive": float(S_uncorr[c].quantile(.9)),
} for c, cn in EXPOSURES])

tie_audit_df = pd.DataFrame([naive_qcut_audit(S_uncorr, c) for c, _ in EXPOSURES])
tie_audit_df["note"] = ("qcut_native_10='error' 表示直接十分位报错；"
                        "duplicates='drop' 会把并列值合成一个大箱（见 largest_bin_if_dropped）；"
                        "rank10_bins_with_split_ties = 有多少箱的边界值是跟邻箱共享的并列值")

# 总度 vs 入度 并排（§2.1 主口径对照）
Y2COL = {"dead": "is_dead", "partial": "is_partial", "true_fail": "true_fail", "clean_part": "partial_clean"}
nl_idx = node_level.set_index(["exposure", "outcome"])["rho_s_uncorr_671"].to_dict()
side = []
for ea_eb in [("degree_u", "in_degree", "总度 vs 入度"), ("url_count", "in_degree", "url_count vs 入度"),
              ("listed_urls", "in_degree", "listed_urls vs 入度")]:
    expo_a, expo_b, label = ea_eb
    for key, cn, mask, corr in OUTCOMES:
        ra = summary[(summary.exposure == expo_a) & (summary.outcome == key)
                     & (summary.scheme == "rank10") & (summary.sample_tag == "S_uncorr_671")]
        rb = summary[(summary.exposure == expo_b) & (summary.outcome == key)
                     & (summary.scheme == "rank10") & (summary.sample_tag == "S_uncorr_671")]
        if not len(ra) or not len(rb):
            continue
        ra, rb = ra.iloc[0], rb.iloc[0]
        rho_a, rho_b = float(nl_idx[(expo_a, Y2COL[key])]), float(nl_idx[(expo_b, Y2COL[key])])
        side.append({"contrast": label, "outcome": key, "outcome_cn": cn,
                     "A_exposure": expo_a, "A_rate_diff_pp": ra["rate_diff_pp"],
                     "A_bin_spearman": ra["spearman_binlevel_rho"], "A_node_rho": round(rho_a, 4),
                     "B_exposure": expo_b, "B_rate_diff_pp": rb["rate_diff_pp"],
                     "B_bin_spearman": rb["spearman_binlevel_rho"], "B_node_rho": round(rho_b, 4),
                     "node_rho_gap_A_minus_B": round(rho_a - rho_b, 4),
                     "rate_diff_gap_pp": round(float(ra["rate_diff_pp"]) - float(rb["rate_diff_pp"]), 2)})
side_df = pd.DataFrame(side)

# 死站在两种口径下的暴露位置：证明"总度梯度"里有多少是定义性的
dead_mask = S_uncorr["is_dead"]
closure = {
    "n_uncorr": N_UNCORR, "n_corr": N_CORR,
    "n_removed_false": int(S_uncorr["false_fail"].sum()),
    "n_removed_ambiguous": int(S_uncorr["ambiguous_fail"].sum()),
    "dead_rate_uncorr": round(float(S_uncorr["is_dead"].mean()), 4),
    "true_fail_rate_uncorr": round(float(S_uncorr["true_fail"].mean()), 4),
    "dead_rate_corr_equals_true_fail_rate": bool(
        np.allclose(S_corr["is_dead"].astype(float), S_corr["true_fail"].astype(float))),
    "degree_u_equals_in_degree_for_all_dead": bool(
        (S_uncorr.loc[dead_mask, "degree_u"] == S_uncorr.loc[dead_mask, "in_degree"]).all()),
    "dead_share_in_rank10_bottom_bin": {},
}
for expo, _ in EXPOSURES:
    lab, _ = rank_decile_labels(S_uncorr, expo, N_Q)
    bot = int(S_uncorr.loc[lab.to_numpy() == 1, "is_dead"].sum())
    top = int(S_uncorr.loc[lab.to_numpy() == N_Q, "is_dead"].sum())
    closure["dead_share_in_rank10_bottom_bin"][expo] = {
        "bottom_bin_dead_n": bot, "bottom_bin_n": int((lab == 1).sum()),
        "top_bin_dead_n": top, "top_bin_n": int((lab == N_Q).sum())}

# ============================================================================ 落盘
C.save_csv(deciles, MODULE, "d1_decile_table.csv")
C.save_csv(summary, MODULE, "d1_gradient_summary.csv")
C.save_csv(node_level, MODULE, "d1_nodelevel_spearman.csv")
C.save_csv(tie_audit_df, MODULE, "d1_binning_tie_audit.csv")
C.save_csv(sens_df, MODULE, "d1_binning_scheme_sensitivity.csv")
C.save_csv(expo_dist, MODULE, "d1_exposure_distribution.csv")
C.save_csv(urlcap, MODULE, "d1_url_count_cap_bins.csv")
C.save_csv(listcap, MODULE, "d1_listed_urls_cap_bins.csv")
C.save_csv(trunc, MODULE, "d1_page_truncation_strata.csv")
C.save_csv(side_df, MODULE, "d1_exposure_contrast.csv")
C.save_json(closure, MODULE, "d1_closure_checks.json")

ELAPSED = time.time() - T0
C.write_run_note(MODULE, [
    "D1_health_gradient run log",
    "command: cd pipeline && PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8 python -X utf8 out/D1_health_gradient/run_d1.py",
    f"versions: {json.dumps(VERSIONS, ensure_ascii=False)}",
    f"seed: {SEED} (np.random.seed + default_rng 均已显式传；本模块无抽样，全部确定性)",
    f"wall_time_sec: {ELAPSED:.2f}",
    f"anchors: csa_common.assert_anchors() 31/31 PASS  (run_d1.py:L{ANCHOR_LINE})",
    f"samples: 校正前 S_uncorr={N_UNCORR}（有属性节点，主表口径）; 校正后 S_corr={N_CORR}"
    f"（剔除假失效 {int(S_uncorr['false_fail'].sum())} + 待人工判断 {int(S_uncorr['ambiguous_fail'].sum())}）",
    "outcomes: 全挂率(is_dead=162) / 部分失效率(is_partial=149) / 真失效率(true_fail=A+D+F=129) / 干净部分失效率(partial_clean=80)",
    "bins: 主口径 rank10=pd.qcut(稳定秩,10) 恰好10箱等人数(会拆并列值,每行给 tie_straddle_* 计数)；"
    "稳健口径 value_aware(并列不拆,箱数<=10,人数不等)；堆积值/20上限单列",
    "note: 676 节点口径未用于主表（CONVENTIONS §1，主表必须 671）",
    "files: d1_decile_table.csv d1_gradient_summary.csv d1_nodelevel_spearman.csv d1_binning_tie_audit.csv "
    "d1_binning_scheme_sensitivity.csv d1_exposure_distribution.csv d1_url_count_cap_bins.csv "
    "d1_listed_urls_cap_bins.csv d1_page_truncation_strata.csv d1_exposure_contrast.csv d1_closure_checks.json",
])

pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 60)
print("== 31 anchors PASS ==")
print("== closure ==")
print(json.dumps(closure, ensure_ascii=False, indent=1))
print("== exposure distribution ==")
print(expo_dist.to_string(index=False))
print("== qcut tie audit ==")
print(tie_audit_df.to_string(index=False))
print("== decile table (rank10, S_uncorr_671) ==")
show = ["exposure", "bin", "bin_label", "n_bin", "dead_rate", "partial_rate", "true_fail_rate",
        "clean_part_rate", "tie_straddle_top_value", "tie_straddle_top_in_bin"]
print(deciles[(deciles.scheme == "rank10") & (deciles.sample_tag == "S_uncorr_671")][show].to_string(index=False))
print("== decile table (rank10, S_corr_638) ==")
print(deciles[(deciles.scheme == "rank10") & (deciles.sample_tag == "S_corr_638")][show].to_string(index=False))
print("== steepness summary ==")
print(summary[["exposure", "sample_tag", "scheme", "outcome", "n_bins", "bottom_label", "bottom_rate",
               "top_label", "top_rate", "rate_ratio_top_over_bottom", "rate_diff_pp",
               "spearman_binlevel_rho", "spearman_binlevel_p", "linear_trend_LRchi2", "linear_trend_p",
               "monotone_in_bin_order"]].to_string(index=False))
print("== node-level spearman ==")
print(node_level.to_string(index=False))
print("== url_count strata ==")
print(urlcap.to_string(index=False))
print("== listed_urls strata ==")
print(listcap.to_string(index=False))
print("== page truncation strata ==")
print(trunc.to_string(index=False))
print("== exposure contrast (总度 vs 入度) ==")
print(side_df.to_string(index=False))
print("== binning scheme sensitivity ==")
print(sens_df.to_string(index=False))
print(f"elapsed {ELAPSED:.2f}s")
