"""D3 · failure_predictors — 真失效的可预测性（逻辑回归 + 交叉验证 + 置换零模型）。

MANIFEST D3 指定的模型：`is_true_fail ~ 入度 + log(1+url_count) + 类别 + 认证站点标记`，
要回答的是"控制住站点规模之后，网络位置（入度）还有没有独立的预测力"。

本单元先做一件事：**逐项检查这四个自变量与结局的定义关系**，因为题面模型里
`类别`（抓取失败的标题没有关键词 → 全部落进"其他"）与 `认证标记`（B_auth_or_skip 按 §2.2
定义就不是真失效）都是结局的读数而不是独立信息，直接喂 logistic 会得到完全分离的拟合。
产出因此分三层：
  1) 题面模型的**可估计性诊断**（d3_covariate_leakage.csv / d3_model_selection.csv / d3_skipped_terms.csv）；
  2) 可估计的主口径模型 N2 = `log1p(url_count) + 入度`（入度独立效应的正式检验）；
  3) 活站内次口径（结局改为 is_partial / partial_clean，入度与类别都可估）。

契约条款落实（CONVENTIONS）：
  §1 图与属性一律 C.build_all()，开跑前 31 项锚点；主表 671 口径，676 不进模型。
  §2.1 失效分析主口径 = 入度；总度/出度一律不进模型（死站出度恒 0 是删失的同义反复）。
  §2.2 校正前（671）/校正后（638）两套都给。
  §4 随机过程独立重复 + 显式 seed；零模型报 mean/sd/z/双侧经验 p。
  §7 只写"与…相关 / 在…上可预测 / 情景模拟显示"，并主动声明反向与共现、删失两条局限。

运行：
    cd pipeline && PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8 python -X utf8 out/D3_failure_predictors/run_d3.py
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from sklearn import __version__ as SKLEARN_VERSION
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PIPELINE)
import csa_common as C  # noqa: E402

MODULE = "D3_failure_predictors"
SEED = C.SEED                                            # 20261113
np.random.seed(SEED)                                     # 兜底；每个随机过程另传显式 seed

N_BOOT = 2000          # OOF AUC 的配对自助重复数
N_PERM = 1000          # AUC 的结局标签置换次数
K_MAIN = 5             # 主交叉验证折数（每折正例 >= 25）
K_ALT = 10             # 稳健性折数
CV_C = 1.0             # 交叉验证用的 L2 强度（标准化设计；sklearn 1.8 默认 penalty='l2'）
RIDGE_LAM = 1.0        # 完全分离时的岭惩罚（标准化设计，截距不罚）
CAT_MIN_N = 30         # 类别保留规则：样本内 n >= 30 保留，其余并入"其他"
CAT_REF = "其他"       # 哑变量参照类
MLE_MAX_ITER = 300
SEP_BETA_GUARD = 25.0  # |beta| 超过即判定该项分离
THR_PRIMARY = 0.5      # 主判正阈值；另报"阈值=先验"与"OOF 上 F1 最优"两档
B6_PY = os.path.join(PIPELINE, "out", "B6_community_semantics", "run_b6.py")
B6_CSV = os.path.join(PIPELINE, "out", "B6_community_semantics", "title_labels.csv")

T0 = time.time()

# ============================================================================ 数据与口径
b = C.build_all()
C.assert_anchors(b)                                      # 31 项锚点
ANCHOR_LINE = inspect.currentframe().f_lineno - 1
nodes = b["nodes"].sort_values("node", kind="mergesort").reset_index(drop=True)
VERSIONS = C.versions()
nodes["_removed"] = nodes["false_fail"] | nodes["ambiguous_fail"]
S_uncorr = nodes.copy()                                  # 671 校正前
S_corr = nodes.loc[~nodes["_removed"]].copy()            # 638 校正后
N_UNCORR, N_CORR = len(S_uncorr), len(S_corr)
Y_EVENT = "true_fail"                                    # 结局：真失效 A+D+F = 129
SAMPLES = {"S_uncorr_671": S_uncorr, "S_corr_638": S_corr}

# ---------------------------------------------------------------- 类别：调 B6 的规则表，不复制词表
_spec = importlib.util.spec_from_file_location("run_b6_canonical", B6_PY)
B6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B6)                             # main 有 __name__ 保护，不会跑分析
T_IMPORT = time.time() - T0                              # build_all + sklearn/B6 加载的实际耗时


def classify(df: pd.DataFrame) -> pd.DataFrame:
    return B6.classify_series(df["title"], df["sub"], df["subdomain"])


_lab = classify(S_uncorr)
S_uncorr["category_raw"] = _lab["category"].to_numpy()
S_uncorr["title_kind"] = _lab["title_kind"].to_numpy()
S_uncorr["category_confidence"] = _lab["confidence"].to_numpy()
_lab2 = classify(S_corr)
S_corr["category_raw"] = _lab2["category"].to_numpy()
S_corr["title_kind"] = _lab2["title_kind"].to_numpy()
S_corr["category_confidence"] = _lab2["confidence"].to_numpy()
_ref = pd.read_csv(B6_CSV, encoding="utf-8-sig")[["node", "category"]]
_chk = S_uncorr[["node", "category_raw"]].merge(_ref, on="node", how="inner")
CAT_AGREEMENT = float((_chk["category_raw"] == _chk["category"]).mean())
CAT_AGREEMENT_N = int(len(_chk))


def collapse(df: pd.DataFrame) -> tuple[pd.Series, list]:
    """样本内 n>=CAT_MIN_N 的类保留，其余并入 CAT_REF；返回类别列 + 映射明细。"""
    sizes = df["category_raw"].value_counts()
    kept = [c for c in sizes.index if sizes[c] >= CAT_MIN_N]
    if CAT_REF not in kept:
        kept.append(CAT_REF)
    cat = df["category_raw"].where(df["category_raw"].isin(kept), CAT_REF).rename("category")
    rows = [{"category_raw": c, "n_raw": int(n), "kept_as_own_level": int(c in kept),
             "final_level": (c if c in kept else CAT_REF)} for c, n in sizes.items()]
    return cat, rows


cat_rows = []
for tag, sdf in SAMPLES.items():
    c, rows = collapse(sdf)
    sdf["category"] = c.to_numpy()
    for r in rows:
        r["sample"] = tag
    cat_rows.extend(rows)
category_map = pd.DataFrame(cat_rows)[["sample", "category_raw", "n_raw", "kept_as_own_level", "final_level"]]

# ---------------------------------------------------------------- 特征
for sdf in SAMPLES.values():
    sdf["log1p_url"] = np.log1p(sdf["url_count"].astype(float))
    sdf["in_degree_f"] = sdf["in_degree"].astype(float)
    sdf["auth_flag"] = (sdf["dead_cat"] == "B_auth_or_skip").astype(int)

# ============================================================ 自变量与结局的定义关系（泄漏诊断）
AUTH_VS_STATUSCODE = bool((S_uncorr["status_codes"].fillna("").astype(str).str.contains("-1")
                           == (S_uncorr["auth_flag"] == 1)).all())
_auth = S_uncorr.loc[S_uncorr["auth_flag"] == 1]
AUTH_TAB = pd.DataFrame([
    {"group": "auth_flag=1（dead_cat=B_auth_or_skip）", "n": int(len(_auth)),
     "n_true_fail": int(_auth[Y_EVENT].sum()), "n_false_fail": int(_auth["false_fail"].sum()),
     "n_alive": int((~_auth["is_dead"]).sum())},
    {"group": "auth_flag=0（其余）", "n": int((S_uncorr["auth_flag"] == 0).sum()),
     "n_true_fail": int(S_uncorr.loc[S_uncorr["auth_flag"] == 0, Y_EVENT].sum()),
     "n_false_fail": int(S_uncorr.loc[S_uncorr["auth_flag"] == 0, "false_fail"].sum()),
     "n_alive": int((~S_uncorr.loc[S_uncorr["auth_flag"] == 0, "is_dead"]).sum())},
    {"group": "S_corr_638 内 auth_flag=1", "n": int(S_corr["auth_flag"].sum()),
     "n_true_fail": 0, "n_false_fail": 0, "n_alive": 0},
])
FISHER = stats.fisher_exact([[int(AUTH_TAB.loc[0, "n_true_fail"]), int(AUTH_TAB.loc[0, "n"])],
                             [int(AUTH_TAB.loc[1, "n_true_fail"]), int(AUTH_TAB.loc[1, "n"] - AUTH_TAB.loc[1, "n_true_fail"])]])
FISHER_ODDS, FISHER_P = float(FISHER.statistic), float(FISHER.pvalue)

ctab_cat = pd.crosstab(S_uncorr["category"], S_uncorr[Y_EVENT])
ctab_kind = pd.crosstab(S_uncorr["title_kind"], S_uncorr[Y_EVENT])
leak_rows = []
for lvl, row in ctab_cat.iterrows():
    n0, n1 = int(row.get(False, 0)), int(row.get(True, 0))
    leak_rows.append({"covariate": "category（n>=30 保留后）", "level": lvl, "n": n0 + n1,
                      "n_events": n1, "n_non_events": n0, "event_rate": round(n1 / max(n0 + n1, 1), 4),
                      "separated": int(n1 == 0 or n0 == 0),
                      "mechanism": "抓取失败站点的 title 是『[请求失败] …』一类占位文本，没有任何关键词 → 全部落进"
                                   "『其他』；于是除『其他』外的每个类别的失效事件数恒为 0（完全分离）"})
for lvl, row in ctab_kind.iterrows():
    n0, n1 = int(row.get(False, 0)), int(row.get(True, 0))
    leak_rows.append({"covariate": "title_kind（B6 的标题信息量标记）", "level": lvl, "n": n0 + n1,
                      "n_events": n1, "n_non_events": n0, "event_rate": round(n1 / max(n0 + n1, 1), 4),
                      "separated": int(n1 == 0 or n0 == 0),
                      "mechanism": "同上：fail_placeholder 这一档就是失败占位标题本身"})
for col, lab in (("url_count", "url_count（抓取器报告的 URL 总数）"),
                 ("listed_urls", "listed_urls（实际落库明细条数，上限 20）")):
    ve = S_uncorr.loc[S_uncorr[Y_EVENT], col].astype(float)
    vn = S_uncorr.loc[~S_uncorr[Y_EVENT], col].astype(float)
    leak_rows.append({"covariate": lab, "level": "true_fail=1", "n": int(len(ve)),
                      "n_events": int((ve == 1).sum()), "n_non_events": np.nan,
                      "event_rate": round(float(ve.median()), 4), "separated": 0,
                      "mechanism": f"median={ve.median():.1f}, mean={ve.mean():.3f}, max={ve.max():.0f}; "
                                   f"其中 url/listed=1 的站点 {int((ve == 1).sum())}/{int(len(ve))}"
                                   f"（= 抓取只成功一次或零次）→ 这一列本身就是抓取产出的读数"})
    leak_rows.append({"covariate": lab, "level": "true_fail=0", "n": int(len(vn)),
                      "n_events": int((vn == 1).sum()), "n_non_events": np.nan,
                      "event_rate": round(float(vn.median()), 4), "separated": 0,
                      "mechanism": f"median={vn.median():.1f}, mean={vn.mean():.3f}, max={vn.max():.0f}; "
                                   f"=1 的站点 {int((vn == 1).sum())}/{int(len(vn))}"})
leak_rows.append({"covariate": "auth_flag（认证标记）", "level": "见 d3_auth_flag_evidence.csv",
                  "n": int(AUTH_TAB.loc[0, "n"]), "n_events": int(AUTH_TAB.loc[0, "n_true_fail"]),
                  "n_non_events": int(AUTH_TAB.loc[0, "n_false_fail"]), "event_rate": 0.0, "separated": 1,
                  "mechanism": "§2.2 把 B_auth_or_skip 定义为**假失效**，因此 auth_flag=1 与 true_fail=1 互斥是定义，"
                               "不是发现；Fisher 精确检验只是把这个定义写成数（odds=%.4g, p=%.4g）" % (FISHER_ODDS, FISHER_P)})
leakage = pd.DataFrame(leak_rows)

SEMANTIC_SUBSET_N = int((S_uncorr["title_kind"] == "informative").sum())
SEMANTIC_SUBSET_EVENTS = int(S_uncorr.loc[S_uncorr["title_kind"] == "informative", Y_EVENT].sum())


def dummy_frame(sdf: pd.DataFrame) -> tuple[np.ndarray, list]:
    lv = sorted([c for c in sdf["category"].unique() if c != CAT_REF])
    cols = [(sdf["category"] == c).to_numpy(dtype=float) for c in lv]
    X = np.column_stack(cols) if cols else np.empty((len(sdf), 0))
    return X, [f"category[{c}]" for c in lv]


def build_design(sdf: pd.DataFrame, spec: list) -> tuple[np.ndarray, list]:
    Xc, nc = dummy_frame(sdf)
    blocks, names = [], []
    for s in spec:
        if s == "cat":
            blocks.append(Xc)
            names.extend(nc)
        else:
            blocks.append(sdf[s].to_numpy(dtype=float).reshape(-1, 1))
            names.append(s)
    if not blocks:
        return np.empty((len(sdf), 0)), []
    return np.hstack(blocks), names


# ============================================================ 无惩罚 logistic 的 IRLS（Wald SE / 偏差 / 分离检测）
def _softplus(eta: np.ndarray) -> np.ndarray:
    return np.where(eta > 0, eta + np.log1p(np.exp(-eta)), np.log1p(np.exp(eta)))


def _sigma(eta: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-eta))


def _dev(eta: np.ndarray, y: np.ndarray) -> float:
    return -2.0 * float((y * eta - _softplus(eta)).sum())


def fit_mle(X: np.ndarray, y: np.ndarray, max_iter: int = MLE_MAX_ITER) -> dict:
    n, p = X.shape
    D = np.hstack([np.ones((n, 1)), X])
    beta = np.zeros(p + 1)
    dev = _dev(D @ beta, y)
    converged, singular, n_iter, oscillating = False, False, 0, False
    for n_iter in range(1, max_iter + 1):
        mu = _sigma(D @ beta)
        w = np.clip(mu * (1 - mu), 1e-14, None)
        H = D.T @ (D * w[:, None])
        g = D.T @ (y - mu)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            singular = True
            step = np.linalg.pinv(H) @ g
        t, ok, d_new = 1.0, False, dev
        for _ in range(60):                            # 步长折半：保证偏差不升
            cand = beta + t * step
            d_new = _dev(D @ cand, y)
            if d_new <= dev + 1e-12:
                ok = True
                break
            t *= 0.5
        if not ok:
            oscillating = True
            break
        improve, moved = dev - d_new, float(np.max(np.abs(t * step)))
        beta, dev = cand, d_new
        if improve < 1e-10 and moved < 1e-7:
            converged = True
            break
    mu = _sigma(D @ beta)
    w = np.clip(mu * (1 - mu), 1e-14, None)
    H = D.T @ (D * w[:, None])
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov, singular = np.linalg.pinv(H), True
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    diverged = [nm for nm, v in zip(np.arange(p), beta[1:]) if abs(v) > SEP_BETA_GUARD]
    return {"beta": beta, "se": se, "deviance": dev, "converged": converged, "singular_hessian": singular,
            "n_iter": n_iter, "hit_max_iter": bool(n_iter >= max_iter and not converged),
            "diverged_idx": diverged, "separated": bool(len(diverged) > 0 or not converged),
            "n_terms": p, "n": n}


def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float = RIDGE_LAM) -> dict:
    """岭惩罚 logistic（标准化设计、截距不罚）：分离情形下唯一稳定的点估计，不给 SE。"""
    sc = StandardScaler().fit(X)
    Z = sc.transform(X)
    D = np.hstack([np.ones((len(y), 1)), Z])
    P = np.eye(Z.shape[1] + 1) * lam
    P[0, 0] = 0.0
    beta = np.zeros(Z.shape[1] + 1)
    for _ in range(300):
        mu = _sigma(D @ beta)
        w = np.clip(mu * (1 - mu), 1e-14, None)
        g = D.T @ (y - mu) - P @ beta
        H = D.T @ (D * w[:, None]) + P
        step = np.linalg.solve(H, g)
        beta = beta + step
        if float(np.max(np.abs(step))) < 1e-11:
            break
    return {"beta_std": beta, "n_terms": Z.shape[1]}


# ============================================================ 模型集（嵌套 + 主口径 + 无泄漏上界）
MODEL_SPECS = [
    ("M0_intercept", [], None, "trivial"),
    ("M1_url_only", ["log1p_url"], "M0_intercept", "crawl_yield_proxy"),
    ("M2_url_category", ["log1p_url", "cat"], "M1_url_only", "含标题语义（与结局同渠道）"),
    ("M3_url_category_indegree", ["log1p_url", "cat", "in_degree_f"], "M2_url_category", "含标题语义（与结局同渠道）"),
    ("M4_full_with_auth", ["log1p_url", "cat", "in_degree_f", "auth_flag"], "M3_url_category_indegree",
     "题面指定完整模型（含两个定义性伴生项）"),
    ("N1_indegree_only", ["in_degree_f"], "M0_intercept", "无泄漏：只有网络位置"),
    ("N2_url_indegree_PRIMARY", ["log1p_url", "in_degree_f"], "M1_url_only",
     "主口径：入度独立效应的正式检验"),
]
TERMS = {m: s for m, s, _, _ in MODEL_SPECS}
PARENT = {m: pv for m, _, pv, _ in MODEL_SPECS}
TAG_OF = {m: tg for m, _, _, tg in MODEL_SPECS}
PRIMARY = "N2_url_indegree_PRIMARY"

coef_rows, sel_rows, fits, skipped = [], [], {}, []
for tag, sdf in SAMPLES.items():
    y = sdf[Y_EVENT].to_numpy(dtype=float)
    n, ev = len(sdf), int(y.sum())
    iqr_in = float(sdf["in_degree"].quantile(.75) - sdf["in_degree"].quantile(.25))
    for mname, spec, _pv, tg in MODEL_SPECS:
        X, names = build_design(sdf, spec)
        if "auth_flag" in names and int(sdf["auth_flag"].sum()) == 0:
            skipped.append({"sample": tag, "model": mname, "term": "auth_flag",
                            "reason": "638 口径里假失效已按 §2.2 剔除 → auth_flag 恒 0，无变异，系数不可估",
                            "consequence": "M4 与 M3 设计等价"})
            continue
        f = fit_mle(X, y)
        fits[(tag, mname)] = (f, names, X, y)
        chi2 = dfree = p_lr = np.nan
        pv = PARENT[mname]
        if pv is not None and (tag, pv) in fits:
            fp = fits[(tag, pv)][0]
            chi2 = fp["deviance"] - f["deviance"]
            dfree = f["n_terms"] - fp["n_terms"]
            if dfree > 0:
                p_lr = float(stats.chi2.sf(max(chi2, 0.0), int(dfree)))
        k = f["n_terms"]
        div_names = [names[i] for i in f["diverged_idx"]] if k else []
        sel_rows.append({
            "sample": tag, "model": mname, "model_tag": tg, "parent": pv, "n": n, "n_events": ev,
            "event_rate": round(ev / n, 4), "n_terms": k, "deviance": round(f["deviance"], 4),
            "aic": round(f["deviance"] + 2 * (k + 1), 4), "bic": round(f["deviance"] + np.log(n) * (k + 1), 4),
            "epv": round(ev / k, 2) if k else np.nan,
            "lrt_vs_parent_chi2": round(chi2, 4) if chi2 == chi2 else np.nan,
            "lrt_vs_parent_df": int(dfree) if dfree == dfree else np.nan,
            "lrt_vs_parent_p": p_lr, "lrt_valid_under_separation": int(not f["separated"]),
            "mle_converged": bool(f["converged"]), "hit_max_iter": bool(f["hit_max_iter"]),
            "separated": bool(f["separated"]), "singular_hessian": bool(f["singular_hessian"]),
            "n_iter": f["n_iter"], "diverged_terms": "|".join(div_names),
            "note": ("完全分离：这些项的系数发散到 ±inf、Wald SE 爆炸、LRT 分布非标准 → 该模型的系数与 p 不可解释"
                     if div_names else "")})
        for j, nm in enumerate(["(intercept)"] + names):
            bb, ss = float(f["beta"][j]), float(f["se"][j])
            term_sep = int(j > 0 and abs(bb) > SEP_BETA_GUARD)
            z = bb / ss if ss > 0 else np.nan
            pv_wald = float(2 * stats.norm.sf(abs(z))) if z == z else np.nan
            lo, hi = bb - 1.96 * ss, bb + 1.96 * ss
            scale = iqr_in if nm == "in_degree_f" else 1.0
            coef_rows.append({
                "sample": tag, "model": mname, "model_tag": tg, "term": nm,
                "term_kind": ("definitional_complement" if nm == "auth_flag" else
                              "network_position" if nm == "in_degree_f" else
                              "crawl_yield_proxy" if nm == "log1p_url" else
                              "title_semantics_proxy" if nm.startswith("category[") else "intercept"),
                "n": n, "n_events": ev, "coef": round(bb, 4), "se": round(ss, 4),
                "z": round(z, 3) if z == z else np.nan, "p": pv_wald,
                "odds_ratio": round(float(np.exp(np.clip(bb, -60, 60))), 6),
                "or_ci_lo": round(float(np.exp(np.clip(lo, -60, 60))), 6),
                "or_ci_hi": round(float(np.exp(np.clip(hi, -60, 60))), 6),
                "effect_scale": (f"per IQR(in_degree)={iqr_in:g}" if scale != 1.0 else "per unit"),
                "or_at_effect_scale": round(float(np.exp(np.clip(bb * scale, -60, 60))), 6),
                "or_at_scale_ci": f"[{np.exp(np.clip(lo * scale, -60, 60)):.6f}, "
                                  f"{np.exp(np.clip(hi * scale, -60, 60)):.6f}]",
                "or_clipped": int(max(abs(bb), abs(lo), abs(hi)) > 60),
                "term_separated": term_sep,
                "interpretation_allowed": int(not term_sep and not f["separated"]),
                "warning": ("该项发散（分离）→ 系数/SE/p 无意义" if term_sep else
                            ("模型内含分离项，本行的 SE 也不可信" if f["separated"] else ""))})
    # 岭惩罚版题面完整模型：给分离项一个有界读数（非 MLE，不用于推断）
    if int(sdf["auth_flag"].sum()) > 0:
        Xf, names_f = build_design(sdf, TERMS["M4_full_with_auth"])
        r = fit_ridge(Xf, y, RIDGE_LAM)
        for j, nm in enumerate(["(intercept)"] + names_f):
            coef_rows.append({"sample": tag, "model": "M4_ridge_penalized", "model_tag": "penalized_readout",
                              "term": nm, "term_kind": "intercept" if j == 0 else "ridge_std",
                              "n": n, "n_events": ev, "coef": round(float(r["beta_std"][j]), 4),
                              "se": np.nan, "z": np.nan, "p": np.nan,
                              "odds_ratio": round(float(np.exp(np.clip(r["beta_std"][j], -60, 60))), 6),
                              "or_ci_lo": np.nan, "or_ci_hi": np.nan,
                              "effect_scale": f"标准化设计上 1 SD（岭惩罚 lam={RIDGE_LAM}，截距不罚）",
                              "or_at_effect_scale": round(float(np.exp(np.clip(r["beta_std"][j], -60, 60))), 6),
                              "or_at_scale_ci": "n/a", "or_clipped": 0, "term_separated": 0,
                              "interpretation_allowed": 0,
                              "warning": "非 MLE：不可与 N2 的系数比较，只用于给分离项一个有界读数"})

coefficients = pd.DataFrame(coef_rows)
selection = pd.DataFrame(sel_rows)
skipped_models = pd.DataFrame(skipped)

# ============================================================ 交叉验证（OOF）+ 配对自助
def cv_oof(X: np.ndarray, y: np.ndarray, k: int, seed: int = SEED) -> np.ndarray:
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    if X.shape[1] == 0:
        for tr, te in skf.split(np.zeros(len(y)), y):     # 截距模型：折内先验
            oof[te] = y[tr].mean()
        return oof
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])                 # 缩放只在训练折上拟合（无泄漏）
        m = LogisticRegression(C=CV_C, max_iter=5000, solver="lbfgs")
        m.fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return oof


def boot_idx(n: int, r: int) -> np.ndarray:
    return np.random.default_rng(SEED + r).integers(0, n, n)   # 独立重复（§4）


def auc_safe(y: np.ndarray, s: np.ndarray) -> float:
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return float(roc_auc_score(y, s))


cv_rows, dl_rows, oof_store = [], [], {}
for tag, sdf in SAMPLES.items():
    y = sdf[Y_EVENT].to_numpy(dtype=float)
    n, prev = len(y), float(y.mean())
    models_here = [m for m, _, _, _ in MODEL_SPECS if (tag, m) in fits]
    for mname in models_here:
        X, names = build_design(sdf, TERMS[mname])
        for k in (K_MAIN, K_ALT):
            s = cv_oof(X, y, k)
            oof_store[(tag, mname, k)] = s
            a = auc_safe(y, s)
            cv_rows.append({"sample": tag, "model": mname, "model_tag": TAG_OF[mname], "k": k,
                            "n": n, "n_events": int(y.sum()), "n_features": len(names),
                            "cv_auc": round(a, 4), "auc_minus_0.5": round(a - 0.5, 4),
                            "pr_auc": round(float(average_precision_score(y, s)), 4),
                            "prior_event_rate": round(prev, 4),
                            "brier": round(float(brier_score_loss(y, s)), 5),
                            "auc_boot_ci": "n/a", "auc_boot_se": np.nan, "n_boot_valid": 0,
                            "cv_setting": f"StratifiedKFold(k={k}, shuffle=True, random_state={SEED})；"
                                          f"L2 C={CV_C}（sklearn 默认 penalty），缩放只在训练折拟合",
                            "leakage_flag": ("含『类别/认证标记』：真失效站点的标题是抓取失败占位文本，"
                                             "类别与 auth_flag 都是结局的读数（见 d3_covariate_leakage.csv）"
                                             if ("cat" in TERMS[mname] or "auth_flag" in TERMS[mname]) else
                                             "含 url_count：抓取产出本身就是结局的读数" if TERMS[mname] else
                                             "无自变量（平凡基线）")})
    scores = {m: oof_store[(tag, m, K_MAIN)] for m in models_here}
    aucs = {m: auc_safe(y, scores[m]) for m in models_here}
    boot = {m: [] for m in models_here}
    pairs = [(mm, pv) for mm in models_here for pv in models_here
             if len(TERMS[pv]) < len(TERMS[mm])]        # 只与"自变量更少"的模型配对比
    deltas = {pr: [] for pr in pairs}
    for r in range(N_BOOT):
        idx = boot_idx(n, r)
        yb = y[idx]
        ab = {m: auc_safe(yb, scores[m][idx]) for m in models_here}
        for m in models_here:
            if ab[m] == ab[m]:
                boot[m].append(ab[m])
        for (a_, b_) in pairs:
            if ab[a_] == ab[a_] and ab[b_] == ab[b_]:
                deltas[(a_, b_)].append(ab[a_] - ab[b_])
    for m in models_here:
        arr = np.array(boot[m])
        ci = f"[{np.quantile(arr, .025):.4f}, {np.quantile(arr, .975):.4f}]"
        for row in cv_rows:
            if row["sample"] == tag and row["model"] == m and row["k"] == K_MAIN:
                row.update({"auc_boot_ci": ci, "auc_boot_se": round(float(arr.std(ddof=1)), 4),
                            "n_boot_valid": int(len(arr))})
    for (a_, b_) in pairs:
        arr = np.array(deltas[(a_, b_)])
        if len(arr) == 0:
            continue
        dl_rows.append({"sample": tag, "model": a_, "baseline": b_, "k": K_MAIN,
                        "auc_model": round(aucs[a_], 4), "auc_baseline": round(aucs[b_], 4),
                        "delta_auc": round(aucs[a_] - aucs[b_], 4),
                        "delta_ci_lo": round(float(np.quantile(arr, .025)), 4),
                        "delta_ci_hi": round(float(np.quantile(arr, .975)), 4),
                        "ci_covers_zero": int(np.quantile(arr, .025) <= 0 <= np.quantile(arr, .975)),
                        "prob_model_gt_baseline": round(float((arr > 0).mean()), 4),
                        "n_boot_valid": int(len(arr)),
                        "note": "配对自助：同一批重采样下比较两个模型的 AUC 差，比各自 CI 更紧"})
cv_metrics = pd.DataFrame(cv_rows)
delta_auc = pd.DataFrame(dl_rows)

# ============================================================ 阈值上的精确率 / 召回率（类别不平衡 129/671）
def prf(tp, fp, fn):
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and (prec + rec) > 0 else np.nan
    return prec, rec, f1


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


thr_rows, f1_rows = [], []
for tag, sdf in SAMPLES.items():
    y = sdf[Y_EVENT].to_numpy(dtype=float)
    prev = float(y.mean())
    for mname in [m for m, _, _, _ in MODEL_SPECS if (tag, m, K_MAIN) in oof_store]:
        s = oof_store[(tag, mname, K_MAIN)]
        cand = np.unique(np.round(s, 4))
        grid = []
        for t in cand:
            tp = int(((s >= t) & (y == 1)).sum()); fp = int(((s >= t) & (y == 0)).sum())
            fn = int(((s < t) & (y == 1)).sum())
            grid.append((float(t), *prf(tp, fp, fn)))
        grid = [g for g in grid if g[3] == g[3]]
        t_star = max(grid, key=lambda z_: (z_[3], z_[2]))[0] if grid else np.nan
        f1_rows.append({"sample": tag, "model": mname, "k": K_MAIN,
                        "thr_maximising_f1_on_oof": round(float(t_star), 4) if t_star == t_star else np.nan,
                        "f1_at_that_thr": round(float(max(g[3] for g in grid)), 4) if grid else np.nan,
                        "caveat": "阈值是在同一批 OOF 分数上挑出来的 → 选择性乐观，只作参考，不作主口径"})
        for tname, t in (("thr_0.5", THR_PRIMARY), ("thr_prior", round(prev, 4)),
                         ("thr_f1_max_oof", float(t_star) if t_star == t_star else np.nan)):
            if t != t:
                continue
            tp = int(((s >= t) & (y == 1)).sum()); fp = int(((s >= t) & (y == 0)).sum())
            fn = int(((s < t) & (y == 1)).sum()); tn = int(((s < t) & (y == 0)).sum())
            prec, rec, f1 = prf(tp, fp, fn)
            lo, hi = wilson(tp, tp + fp) if tp + fp else (np.nan, np.nan)
            thr_rows.append({"sample": tag, "model": mname, "model_tag": TAG_OF[mname], "k": K_MAIN,
                             "threshold_rule": tname, "threshold": round(float(t), 4),
                             "n_predicted_positive": tp + fp, "predicted_positive_share": round((tp + fp) / len(y), 4),
                             "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                             "precision": round(prec, 4) if prec == prec else np.nan,
                             "precision_wilson_ci": (f"[{lo:.4f}, {hi:.4f}]" if lo == lo else "n/a"),
                             "recall": round(rec, 4), "f1": round(f1, 4) if f1 == f1 else np.nan,
                             "specificity": round(tn / (tn + fp), 4),
                             "imbalance_context": f"正例 {int(y.sum())}/{len(y)}={prev:.2%}；"
                                                  f"全判正时精确率上界={prev:.4f}"})
    thr_rows.append({"sample": tag, "model": "trivial_all_negative", "model_tag": "trivial", "k": "-",
                     "threshold_rule": "-", "threshold": np.nan, "n_predicted_positive": 0,
                     "predicted_positive_share": 0.0, "TP": 0, "FP": 0, "FN": int(y.sum()),
                     "TN": int(len(y) - int(y.sum())), "precision": np.nan, "precision_wilson_ci": "n/a",
                     "recall": 0.0, "f1": np.nan, "specificity": 1.0,
                     "imbalance_context": "类别不平衡下的地板：任何模型的召回都要与它比"})
thresholds = pd.DataFrame(thr_rows)
f1_choice = pd.DataFrame(f1_rows)

# ============================================================ AUC 的结局置换零模型（§4）
perm_rows = []
for tag, sdf in SAMPLES.items():
    y = sdf[Y_EVENT].to_numpy(dtype=float)
    n = len(y)
    for mname in ("N1_indegree_only", "N2_url_indegree_PRIMARY", "M4_full_with_auth"):
        if (tag, mname, K_MAIN) not in oof_store:
            continue
        s = oof_store[(tag, mname, K_MAIN)]
        obs = auc_safe(y, s)
        null = np.empty(N_PERM)
        for r in range(N_PERM):
            rng = np.random.default_rng(SEED + 50_000 + r)      # 独立重复
            null[r] = auc_safe(y[rng.permutation(n)], s)
        mu, sd = float(null.mean()), float(null.std(ddof=1))
        p_emp = float((1 + np.sum(np.abs(null - mu) >= abs(obs - mu))) / (N_PERM + 1))
        perm_rows.append({"sample": tag, "model": mname, "k": K_MAIN, "obs_auc": round(obs, 4),
                          "null_mean": round(mu, 4), "null_sd": round(sd, 4),
                          "z": round((obs - mu) / sd, 3), "p_two_sided_empirical": round(p_emp, 5),
                          "n_perm": N_PERM,
                          "note": "置换结局标签、OOF 分数固定；按偏离 null mean 的绝对值计（§4）。"
                                  "显著只说明分数与标签共现，不等于可用于机制推断"})
permutation = pd.DataFrame(perm_rows)

# ============================================================ OOF 校准（主口径 N2）
cal_rows = []
for tag, sdf in SAMPLES.items():
    y = sdf[Y_EVENT].to_numpy(dtype=float)
    mname = PRIMARY if (tag, PRIMARY, K_MAIN) in oof_store else "M1_url_only"
    s = oof_store[(tag, mname, K_MAIN)]
    d = pd.DataFrame({"y": y, "p": s}).sort_values("p", kind="mergesort").reset_index(drop=True)
    d["decile"] = pd.qcut(np.arange(len(d)) / len(d), 10, labels=False) + 1
    for bn, g in d.groupby("decile"):
        k_, n_ = int(g["y"].sum()), int(len(g))
        lo, hi = wilson(k_, n_)
        cal_rows.append({"sample": tag, "model": mname, "k": K_MAIN, "decile": int(bn), "n": n_,
                         "mean_predicted": round(float(g["p"].mean()), 4),
                         "min_predicted": round(float(g["p"].min()), 4),
                         "max_predicted": round(float(g["p"].max()), 4),
                         "observed_rate": round(float(g["y"].mean()), 4),
                         "wilson_lo": round(lo, 4), "wilson_hi": round(hi, 4),
                         "gap_obs_minus_pred": round(float(g["y"].mean() - g["p"].mean()), 4)})
calibration = pd.DataFrame(cal_rows)

# ============================================================ 特征与口径摘要
feat_rows = []
for tag, sdf in SAMPLES.items():
    y = sdf[Y_EVENT].to_numpy(dtype=float)
    iqr = float(sdf["in_degree"].quantile(.75) - sdf["in_degree"].quantile(.25))
    feat_rows.append({"sample": tag, "n": len(sdf), "n_events": int(y.sum()),
                      "event_rate": round(float(y.mean()), 4),
                      "in_degree_min": int(sdf["in_degree"].min()),
                      "in_degree_q1": float(sdf["in_degree"].quantile(.25)),
                      "in_degree_median": float(sdf["in_degree"].median()),
                      "in_degree_q3": float(sdf["in_degree"].quantile(.75)),
                      "in_degree_iqr": iqr, "in_degree_max": int(sdf["in_degree"].max()),
                      "in_degree_mean_events": round(float(sdf.loc[y == 1, "in_degree"].mean()), 4),
                      "in_degree_mean_non_events": round(float(sdf.loc[y == 0, "in_degree"].mean()), 4),
                      "in_degree_zero_n": int((sdf["in_degree"] == 0).sum()),
                      "spearman_in_degree_vs_event": round(float(stats.spearmanr(sdf["in_degree_f"], y).statistic), 4),
                      "spearman_log1p_url_vs_event": round(float(stats.spearmanr(sdf["log1p_url"], y).statistic), 4),
                      "url_count_median": float(sdf["url_count"].median()),
                      "url_count_median_events": float(sdf.loc[y == 1, "url_count"].median()),
                      "url_count_max": int(sdf["url_count"].max()),
                      "log1p_url_mean": round(float(sdf["log1p_url"].mean()), 4),
                      "n_category_levels": int(sdf["category"].nunique()),
                      "n_category_dummies": int(sdf["category"].nunique() - 1),
                      "n_category_levels_with_zero_events": int(((pd.crosstab(sdf["category"], sdf[Y_EVENT])[True] == 0).sum()),),
                      "n_auth_flag_ones": int(sdf["auth_flag"].sum()),
                      "degree_u_in_model": 0, "out_degree_in_model": 0})
feature_summary = pd.DataFrame(feat_rows)

# ============================================================ 次口径：只在 509 个活站上（结局不由"抓不到"定义）
alive = S_uncorr.loc[~S_uncorr["is_dead"]].copy()
ALIVE_SPECS = [("A1_url", ["log1p_url"]),
               ("A2_url_category", ["log1p_url", "cat"]),
               ("A3_url_category_indegree", ["log1p_url", "cat", "in_degree_f"]),
               ("A4_url_indegree", ["log1p_url", "in_degree_f"])]
alive_rows = []
for yname in ("is_partial", "partial_clean"):
    y = alive[yname].to_numpy(dtype=float)
    iqr = float(alive["in_degree"].quantile(.75) - alive["in_degree"].quantile(.25))
    holder = {}
    for mname, spec in ALIVE_SPECS:
        X, names = build_design(alive, spec)
        f = fit_mle(X, y)
        holder[mname] = (f, names, X)
        oof = cv_oof(X, y, K_MAIN)
        for j, nm in enumerate(["(intercept)"] + names):
            if nm not in ("in_degree_f", "log1p_url"):
                continue
            bb, ss = float(f["beta"][j]), float(f["se"][j])
            z = bb / ss if ss > 0 else np.nan
            alive_rows.append({"outcome": yname, "sample": "alive_509（不含任何全挂站点）", "model": mname,
                               "term": nm, "n": int(len(alive)), "n_events": int(y.sum()),
                               "event_rate": round(float(y.mean()), 4),
                               "coef": round(bb, 4), "se": round(ss, 4), "z": round(z, 3),
                               "p": float(2 * stats.norm.sf(abs(z))),
                               "effect_scale": (f"per IQR(in_degree)={iqr:g}" if nm == "in_degree_f" else "per unit"),
                               "or_at_effect_scale": round(float(np.exp(np.clip(bb * (iqr if nm == "in_degree_f" else 1), -60, 60))), 4),
                               "cv_auc": round(auc_safe(y, oof), 4),
                               "mle_converged": bool(f["converged"]), "separated": bool(f["separated"]),
                               "lrt_vs_parent_chi2": np.nan, "lrt_vs_parent_df": np.nan, "lrt_vs_parent_p": np.nan})
    for child, par in (("A2_url_category", "A1_url"), ("A3_url_category_indegree", "A2_url_category"),
                       ("A4_url_indegree", "A1_url")):
        fchild, _, _ = holder[child]
        fpar, _, _ = holder[par]
        chi2 = fpar["deviance"] - fchild["deviance"]
        dfree = fchild["n_terms"] - fpar["n_terms"]
        pv = float(stats.chi2.sf(max(chi2, 0.0), dfree)) if dfree > 0 else np.nan
        for r in alive_rows:
            if r["outcome"] == yname and r["model"] == child:
                r.update({"lrt_vs_parent_chi2": round(chi2, 4), "lrt_vs_parent_df": int(dfree),
                          "lrt_vs_parent_p": pv,
                          "lrt_parent": par})
alive_models = pd.DataFrame(alive_rows)
alive_models["role"] = "次口径（题面未要求）：结局换成'活站内部的部分失效'，此时入度/类别都不再是抓取失败的读数"

# ============================================================ 落盘
for df_, nm in [(coefficients, "d3_coefficients.csv"), (selection, "d3_model_selection.csv"),
                (skipped_models, "d3_skipped_terms.csv"), (cv_metrics, "d3_cv_metrics.csv"),
                (delta_auc, "d3_delta_auc_bootstrap.csv"), (thresholds, "d3_threshold_metrics.csv"),
                (f1_choice, "d3_f1_threshold_choice.csv"), (permutation, "d3_auc_permutation_null.csv"),
                (calibration, "d3_calibration_oof.csv"), (feature_summary, "d3_feature_summary.csv"),
                (category_map, "d3_category_collapse.csv"), (leakage, "d3_covariate_leakage.csv"),
                (alive_models, "d3_alive_only_secondary_models.csv"), (AUTH_TAB, "d3_auth_flag_evidence.csv")]:
    C.save_csv(df_, MODULE, nm)

key = {}
for tag in SAMPLES:
    for m in (PRIMARY, "N1_indegree_only", "M1_url_only", "M0_intercept"):
        if (tag, m) not in fits:
            continue
        f, names, _, _ = fits[(tag, m)]
        if "in_degree_f" in names:
            j = names.index("in_degree_f") + 1
            key[f"{tag}|{m}|in_degree"] = {"coef": round(float(f["beta"][j]), 4),
                                           "se": round(float(f["se"][j]), 4),
                                           "p": float(2 * stats.norm.sf(abs(f["beta"][j] / f["se"][j]))),
                                           "separated": bool(f["separated"]),
                                           "cv_auc": round(auc_safe(SAMPLES[tag][Y_EVENT].to_numpy(dtype=float),
                                                                     oof_store[(tag, m, K_MAIN)]), 4)}
        if m == "M1_url_only":
            j = names.index("log1p_url") + 1
            key[f"{tag}|{m}|log1p_url"] = {"coef": round(float(f["beta"][j]), 4),
                                           "se": round(float(f["se"][j]), 4),
                                           "p": float(2 * stats.norm.sf(abs(f["beta"][j] / f["se"][j])))}

summary = {
    "anchor_line": ANCHOR_LINE, "seed": SEED,
    "versions": dict(VERSIONS, sklearn=SKLEARN_VERSION, scipy=scipy.__version__),
    "n_uncorr": N_UNCORR, "n_corr": N_CORR,
    "events_uncorr": int(S_uncorr[Y_EVENT].sum()), "events_corr": int(S_corr[Y_EVENT].sum()),
    "category_source": "B6 run_b6.classify_series（运行时现场调用规则表，未复制词表）",
    "category_agreement_with_title_labels": CAT_AGREEMENT, "category_agreement_n": CAT_AGREEMENT_N,
    "category_rule": f"样本内 n>={CAT_MIN_N} 保留，其余并入 {CAT_REF}；参照类 {CAT_REF}",
    "auth_flag": {"definition": "dead_cat == 'B_auth_or_skip'（状态码含 -1；与 §2.2 的假失效同一集合）",
                  "equals_statuscode_minus1": int(AUTH_VS_STATUSCODE),
                  "n_auth": int(AUTH_TAB.loc[0, "n"]), "n_auth_true_fail": int(AUTH_TAB.loc[0, "n_true_fail"]),
                  "n_auth_in_638": int(S_corr["auth_flag"].sum()),
                  "fisher_exact_odds": FISHER_ODDS, "fisher_exact_p": FISHER_P,
                  "estimable": False, "reason": "与 true_fail 按定义互斥 → 完全分离；638 口径里恒 0"},
    "category_separation": {"levels_with_zero_events": int(leakage[leakage.covariate.str.startswith("category")]["separated"].sum()),
                            "all_events_in_level": CAT_REF,
                            "informative_title_subset_n": SEMANTIC_SUBSET_N,
                            "informative_title_subset_events": SEMANTIC_SUBSET_EVENTS},
    "models": [m for m, _, _, _ in MODEL_SPECS],
    "primary_model_for_indegree_inference": PRIMARY,
    "cv": {"k_main": K_MAIN, "k_alt": K_ALT, "random_state": SEED, "C_l2": CV_C,
           "n_boot": N_BOOT, "n_perm": N_PERM, "ridge_lam": RIDGE_LAM},
    "iqr_in_degree": {t: float(s["in_degree"].quantile(.75) - s["in_degree"].quantile(.25))
                      for t, s in SAMPLES.items()},
    "mle": {f"{t}|{m}": {"converged": bool(fits[(t, m)][0]["converged"]),
                         "separated": bool(fits[(t, m)][0]["separated"]),
                         "deviance": round(fits[(t, m)][0]["deviance"], 4),
                         "n_iter": fits[(t, m)][0]["n_iter"],
                         "diverged_terms": selection.loc[(selection["sample"] == t) & (selection["model"] == m), "diverged_terms"].iloc[0]}
            for (t, m) in fits},
    "key_estimates": key,
    "banned_columns_used": [],
    "wall_time_sec": round(time.time() - T0, 2),
}
C.save_json(summary, MODULE, "d3_summary.json")

C.write_run_note(MODULE, [
    "D3_failure_predictors run log",
    "command: cd pipeline && PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8 python -X utf8 out/D3_failure_predictors/run_d3.py",
    f"versions: {json.dumps(summary['versions'], ensure_ascii=False)}",
    f"seed: {SEED}; StratifiedKFold(k={K_MAIN}/{K_ALT}, shuffle=True, random_state={SEED})；"
    f"L2 C={CV_C}（sklearn 1.8 默认 penalty='l2'，缩放只在训练折拟合）；自助 B={N_BOOT}"
    f"（np.random.default_rng(SEED+r)，独立重复）；置换 n={N_PERM}（default_rng(SEED+50000+r)）；"
    f"岭惩罚 lam={RIDGE_LAM}；MLE 用自写 IRLS（无惩罚，步长折半，max_iter={MLE_MAX_ITER}）",
    f"wall_time_sec: {time.time() - T0:.2f}（其中 build_all + sklearn/B6 加载 ≈ {T_IMPORT:.2f}s；"
    f"本机 sklearn import 很慢，属环境成本，不影响数值）",
    f"anchors: csa_common.assert_anchors() 31/31 PASS (run_d3.py:L{ANCHOR_LINE})",
    f"category: B6.classify_series 现场计算，与 title_labels.csv 一致率 {CAT_AGREEMENT:.4f}（n={CAT_AGREEMENT_N}）；"
    f"保留规则 n>={CAT_MIN_N}",
    f"separation: 类别在真失效上完全分离（129/129 事件全在『其他』；informative 标题子样本 {SEMANTIC_SUBSET_N} 站 0 事件）；"
    f"auth_flag 18 站 0 事件（Fisher p={FISHER_P:.3g}）→ 题面 M4 不可解释",
    f"primary({PRIMARY}): " + json.dumps({k: v for k, v in key.items()}, ensure_ascii=False),
    "files: d3_coefficients.csv d3_model_selection.csv d3_skipped_terms.csv d3_cv_metrics.csv "
    "d3_delta_auc_bootstrap.csv d3_threshold_metrics.csv d3_f1_threshold_choice.csv "
    "d3_auc_permutation_null.csv d3_calibration_oof.csv d3_feature_summary.csv d3_category_collapse.csv "
    "d3_covariate_leakage.csv d3_alive_only_secondary_models.csv d3_auth_flag_evidence.csv d3_summary.json",
])

pd.set_option("display.width", 320)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 60)
print("== 31 anchors PASS ==")
print(json.dumps(summary["key_estimates"], ensure_ascii=False, indent=1))
print("== auth_flag evidence ==")
print(AUTH_TAB.to_string(index=False))
print("== covariate leakage crosstabs ==")
print(ctab_cat.to_string())
print(ctab_kind.to_string())
print(leakage[leakage.covariate.str.contains("url_count|listed_urls|auth_flag")].to_string(index=False))
print("== model selection ==")
print(selection[["sample", "model", "n_terms", "deviance", "bic", "epv", "lrt_vs_parent_chi2",
                 "lrt_vs_parent_df", "lrt_vs_parent_p", "mle_converged", "separated",
                 "diverged_terms"]].to_string(index=False))
print("== coefficients: numeric terms ==")
print(coefficients[coefficients["term"].isin(["in_degree_f", "log1p_url", "auth_flag"])][
    ["sample", "model", "term", "coef", "se", "z", "p", "or_at_effect_scale", "or_at_scale_ci",
     "term_separated", "interpretation_allowed"]].to_string(index=False))
print("== cv metrics ==")
print(cv_metrics[["sample", "model", "k", "cv_auc", "auc_boot_ci", "auc_boot_se", "pr_auc",
                  "brier", "leakage_flag"]].to_string(index=False))
print("== delta auc (paired bootstrap) ==")
print(delta_auc[delta_auc.model.isin([PRIMARY, "N1_indegree_only", "M4_full_with_auth",
                                      "M3_url_category_indegree"])].to_string(index=False))
print("== threshold metrics ==")
print(thresholds[thresholds.model.isin([PRIMARY, "M1_url_only", "M4_full_with_auth",
                                        "trivial_all_negative"])].to_string(index=False))
print("== permutation null ==")
print(permutation.to_string(index=False))
print("== calibration ==")
print(calibration.to_string(index=False))
print("== alive-only secondary ==")
print(alive_models.to_string(index=False))
print("== feature summary ==")
print(feature_summary.to_string(index=False))
print(f"elapsed {time.time() - T0:.2f}s")
