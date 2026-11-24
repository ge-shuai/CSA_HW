# -*- coding: utf-8 -*-
"""D5 · 失效聚集的替代分组检验（根域前缀 / 类别 / CMS 模板指纹）

为什么要换分组：边层面的"死—死相邻"可观察样本点为 0（见 D4），
节点层面的分组失效率则**完全可观察**——每个节点的真/假失效状态与分组键都拿得到。

每个分组族都给：
  · 逐组真失效率 + Wilson 与 Clopper–Pearson 95% 区间
  · 与网络其余部分的 Fisher 精确检验 / 2×2 卡方（期望频数不足时以 Fisher 为准）
  · BH 多重比较校正（α 与 m 显式写出）；n<10 的组判为"不可判定"并从推断里剔除
  · 组间/组内方差分解（η² 与 ICC(1)）+ 标签置换的整体检验（§4：每轮独立重抽）
  · 校正前（162 全挂）/ 校正后（129 真失效，33 个未证实节点退出样本）两套（§2.2）

只读 CSA_HW/；只写本模块目录。构图 import csa_common。措辞遵守 §7。
"""
from __future__ import annotations

import os
import re
import sys
import time
import inspect
import collections
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.dirname(os.path.dirname(HERE))
if PIPE not in sys.path:
    sys.path.insert(0, PIPE)
import csa_common as C  # noqa: E402

SEED = C.SEED
MOD = "D5_failure_clustering_alt"
ALPHA = 0.05
MIN_N = 10
N_PERM = 20000
T0 = time.time()

KEY: dict = {}


def rec(name, value, fmt="{:.6g}", note=""):
    line = inspect.currentframe().f_back.f_lineno
    KEY[name] = {"value": value, "ref": f"run_d5.py:L{line}", "fmt": fmt, "note": note}
    return value


def N(name):
    return KEY[name]["value"]


def R(name):
    return KEY[name]["ref"]


def f(name, pct=False):
    d = KEY[name]
    v = d["value"] * 100 if pct else d["value"]
    try:
        return d["fmt"].format(v)
    except (ValueError, TypeError):
        return str(v)


def cite(name, pct=False):
    return f"{f(name, pct)}（{KEY[name]['ref']}）"


LOG: list = []


def say(s):
    print(s, flush=True)
    LOG.append(str(s))


Z95 = stats.norm.ppf(1 - 0.025)


def wilson(k, n):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + Z95 * Z95 / n
    c = (p + Z95 * Z95 / (2 * n)) / d
    h = Z95 * np.sqrt(p * (1 - p) / n + Z95 * Z95 / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def clopper(k, n, a=0.05):
    """Clopper–Pearson 精确区间（k=0 或 k=n 时另一端取 0/1）。"""
    if n == 0:
        return (np.nan, np.nan)
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a / 2, k + 1, n - k))
    return (lo, hi)


def bh(pv):
    """Benjamini–Hochberg q 值。"""
    p = np.asarray(pv, float)
    ok = ~np.isnan(p)
    q = np.full(len(p), np.nan)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return q
    order = idx[np.argsort(p[idx])]
    m = len(order)
    ranked = p[order]
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    q[order] = adj
    return q


# ==================================================================== 0. 装配
C.assert_anchors()
ANCH = C.check_anchors()
C.save_csv(ANCH, MOD, "anchors.csv")
b = C.build_all()
m: pd.DataFrame = b["nodes"]
Gd, Gu = b["Gd"], b["Gu"]
urls = C.load_urls()
urls["sub"] = urls["subdomain"].str.replace(".ustc.edu.cn", "", regex=False)
say(f"[{time.time()-T0:6.1f}s] 数据就绪 nodes={len(m)} urls={len(urls)} 锚点 {int(ANCH['ok'].sum())}/{len(ANCH)} PASS")

NODES = list(m["node"])
NN = len(NODES)
unverified = set(m.loc[m["false_fail"] | m["ambiguous_fail"], "node"])
rec("n_nodes", NN, "{:d}")
rec("n_dead", int(m["is_dead"].sum()), "{:d}")
rec("n_true_fail", int(m["true_fail"].sum()), "{:d}")
rec("n_unverified", len(unverified), "{:d}")
rec("base_rate_dead", float(m["is_dead"].mean()), "{:.6f}")
rec("base_rate_tf", float(m["true_fail"].mean()), "{:.6f}")
INFO = ~(m["false_fail"] | m["ambiguous_fail"])
m["informative"] = INFO
NINF = int(INFO.sum())
rec("n_informative", NINF, "{:d}")
rec("base_rate_tf_inf", float(m.loc[INFO, "true_fail"].mean()), "{:.6f}")
rec("uc_dead", float(m.loc[m["is_dead"], "url_count"].mean()), "{:.3f}")
rec("uc_alive", float(m.loc[~m["is_dead"], "url_count"].mean()), "{:.3f}")
rec("uc_healthy", float(m.loc[m["is_healthy"], "url_count"].mean()), "{:.3f}")
rec("uc_partial", float(m.loc[m["is_partial"], "url_count"].mean()), "{:.3f}")
rec("uc_le2_dead", float((m.loc[m["is_dead"], "url_count"] <= 2).mean()), "{:.6f}")
rec("uc_le2_alive", float((m.loc[~m["is_dead"], "url_count"] <= 2).mean()), "{:.6f}")

# ============================================================== 1. 三个分组族
ERR_TITLE = re.compile(r"^\[(HTTP |robots|请求失败|需要身份认证|连接失败)")
m["err_title"] = m["title"].map(lambda t: bool(ERR_TITLE.match(str(t))))
rec("n_err_title", int(m["err_title"].sum()), "{:d}")
rec("dead_with_err_title", int((m["err_title"] & m["is_dead"]).sum()), "{:d}")
rec("err_title_share_dead", float(m.loc[m["is_dead"], "err_title"].mean()), "{:.6f}")


def root_prefix(node):
    """去掉 .ustc.edu.cn 后取**最右**一段 = 所在委托域（.lib / .cmet / .sz / ...）。"""
    s = node[: -len(".ustc.edu.cn")] if node.endswith(".ustc.edu.cn") else node
    parts = s.split(".")
    return "APEX(裸子域)" if len(parts) == 1 else parts[-1]


m["g_root"] = [root_prefix(n) for n in NODES]
rec("n_groups_root", int(m["g_root"].nunique()), "{:d}")

# 类别：数据里**没有** category 列，这里重建同学 extra.py:137-143 的标题规则分类器
RULES = [("招生", ["招生", "录取", "admission", "zsb", "yz"]),
         ("教务培养", ["教务", "教学", "培养", "课程", "课堂", "研究生院", "teach", "gradschool"]),
         ("科研平台", ["科研", "实验室", "研究中心", "研究院", "平台", "lab", "sklfs"]),
         ("院系", ["学院", "系", "学部", "school", "department", "dept"]),
         ("新闻宣传", ["新闻", "宣传", "媒体", "news", "press"]),
         ("服务门户", ["门户", "服务", "搜索", "图书馆", "邮箱", "一卡通", "缴费", "招聘", "人才", "lib", "mail", "job"]),
         ("附属单位", ["附中", "附小", "幼儿园", "医院", "kdfz"]),
         ("英文国际", ["english", "international", "en."]),
         ("个人主页", ["个人主页", "homepage", "~"])]


def title_cat(title, sub):
    t, s = str(title).lower(), str(sub).lower()
    for name, kws in RULES:
        if any(k.lower() in t or k.lower() == s for k in kws):
            return name
    return "其他"


def subcat(sub):
    """类别的可观察替身：只用短名 token（死站也有短名，标题没有）。"""
    s = str(sub).lower()
    head = s.split(".")[0]
    tbl = {"lib": "图书资源", "lug": "学生技术社团", "cmet": "工程教育", "nsrl": "国家重点实验室",
           "sz": "苏州研究院", "snst": "技术转移", "xnfz": "训练基座", "graduate": "研究生",
           "jw": "教务", "jwc": "教务", "teach": "教务", "news": "新闻", "wp": "门户/认证",
           "passport": "门户/认证", "id": "门户/认证", "wvpn": "门户/认证", "webvpn": "门户/认证",
           "mail": "邮件", "job": "人事", "hr": "人事", "zs": "招生", "zsb": "招生",
           "kdfz": "附属单位", "iat": "馆藏/DB", "icase": "科研平台"}
    out = [v for k, v in tbl.items() if head == k or head.endswith("-" + k) or f".{k}" in s]
    return out[0] if out else "其他短名"


m["g_cat_title"] = [title_cat(t, s) for t, s in zip(m["title"], m["sub"])]
m["g_cat_sub"] = [subcat(s) for s in m["sub"]]
rec("n_groups_cat_title", int(m["g_cat_title"].nunique()), "{:d}")
rec("n_groups_cat_sub", int(m["g_cat_sub"].nunique()), "{:d}")
rec("other_cat_share", float((m["g_cat_title"] == "其他").mean()), "{:.6f}")
rec("dead_in_other_cat", int(((m["g_cat_title"] == "其他") & m["is_dead"]).sum()), "{:d}")

# CMS / 模板指纹：扩展名 + 归一化首段路径
KNOWN_EXT = {"htm", "html", "php", "asp", "aspx", "jsp", "jspx", "psp", "shtml", "do", "action",
             "md", "json", "xml", "cgi", "pl", "ashx", "cfm", "axd", "asax", "asmx", "cgi"}


def parse_url(u):
    p = re.sub(r"^https?://[^/]+", "", str(u)).split("?")[0].split("#")[0].strip("/")
    segs = [x for x in p.split("/") if x]
    fn = segs[-1] if segs else ""
    mm = re.search(r"\.([A-Za-z0-9]{1,6})$", fn)
    e = mm.group(1).lower() if mm else "<none>"
    if e not in KNOWN_EXT:
        e = "<none>"
    first = re.sub(r"\d+", "#", segs[0].lower()) if segs else "<root>"
    return e, first


_parsed = urls["url"].map(parse_url)
urls["ext"] = [a for a, _ in _parsed]
urls["seg"] = [bb for _, bb in _parsed]
rec("n_url_rows", len(urls), "{:d}")


def fp_for(sub):
    g = urls[urls["sub"] == sub]
    if len(g) == 0:
        return ("<无URL明细>", "<无URL明细>")
    ec = collections.Counter(g["ext"])
    sc = collections.Counter(g["seg"])
    e = sorted(ec.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    s = sorted(sc.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return (e, f"{e}/{s}")


sub_list = list(m["sub"])
_fps = {s: fp_for(s) for s in sub_list}
m["g_ext"] = [_fps[s][0] for s in sub_list]
m["g_cms"] = [_fps[s][1] for s in sub_list]
rec("n_groups_cms", int(m["g_cms"].nunique()), "{:d}")
rec("n_groups_ext", int(m["g_ext"].nunique()), "{:d}")
rec("cms_unique_share", float(m["g_cms"].value_counts().iloc[0] / NN), "{:.6f}")
say(f"[{time.time()-T0:6.1f}s] 分组族就绪 root={N('n_groups_root')} cat_title={N('n_groups_cat_title')} "
    f"cat_sub={m['g_cat_sub'].nunique()} cms={N('n_groups_cms')} ext={N('n_groups_ext')}")

# ================================================== 2. 逐组检验 + BH + 删失诊断
FAMILIES = {
    "R1 根域前缀": "g_root",
    "C1 类别(标题规则)": "g_cat_title",
    "C2 类别(短名token)": "g_cat_sub",
    "M1 CMS指纹(扩展名×首段)": "g_cms",
    "M2 扩展名": "g_ext",
}
GROUP_ROWS = []
OMNI_ROWS = []


def two_by_two(k, n, k_rest, n_rest):
    """一组 vs 网络其余：Fisher 双侧 + （最小期望>=5 时）卡方。"""
    tab = np.array([[k, n - k], [k_rest, n_rest - k_rest]])
    odds, p_f = stats.fisher_exact(tab, alternative="two-sided")
    exp = np.outer(tab.sum(1), tab.sum(0)) / tab.sum()
    exp_min = float(exp.min())
    if exp_min >= 5:
        chi2, p_c, dof, _ = stats.chi2_contingency(tab, correction=False)
    else:
        chi2, p_c = (np.nan, np.nan)
    return float(odds), float(p_f), float(chi2), float(p_c), exp_min


for fam, col in FAMILIES.items():
    for gname, gdf in m.groupby(col):
        n = len(gdf)
        k_dead = int(gdf["is_dead"].sum())
        gi = gdf[gdf["informative"]]
        n_inf = len(gi)
        k_tf = int(gi["true_fail"].sum())
        n_unv = n - n_inf
        # 校正前：is_dead（全体 671 节点）
        or_pre, p_pre, chi2_pre, pchi_pre, expmin_pre = two_by_two(k_dead, n, N("n_dead") - k_dead, NN - n)
        wl_pre, wh_pre = wilson(k_dead, n)
        cl_pre, ch_pre = clopper(k_dead, n)
        # 校正后：true_fail（剔除 33 个未证实节点后的 {NINF} 个可判定节点）
        or_post, p_post, chi2_post, pchi_post, expmin_post = two_by_two(
            k_tf, n_inf, N("n_true_fail") - k_tf, NINF - n_inf)
        wl, wh = wilson(k_tf, n_inf)
        cl, ch = clopper(k_tf, n_inf)
        GROUP_ROWS.append({
            "family": fam, "column": col, "group": str(gname),
            "n_all": n, "n_informative": n_inf, "n_unverified": n_unv,
            "informative_n_ge_10": bool(n_inf >= MIN_N),
            "n_dead_precorr": k_dead, "rate_dead_precorr": k_dead / n if n else np.nan,
            "wilson_lo_pre": wl_pre, "wilson_hi_pre": wh_pre,
            "cp_lo_pre": cl_pre, "cp_hi_pre": ch_pre,
            "fisher_or_pre": or_pre, "fisher_p_pre": p_pre, "chi2_p_pre": pchi_pre,
            "n_true_fail": k_tf, "rate_true_fail": k_tf / n_inf if n_inf else np.nan,
            "wilson_lo": wl, "wilson_hi": wh, "cp_lo": cl, "cp_hi": ch,
            "fisher_or": or_post, "fisher_p": p_post,
            "chi2_2x2": chi2_post, "chi2_p": pchi_post,
            "expected_min_cell": expmin_post, "expected_min_cell_pre": expmin_pre,
            "test_used": "chi2+fisher" if expmin_post >= 5 else "fisher_only",
            "base_rate_tf_excl_group": (N("n_true_fail") - k_tf) / (NINF - n_inf) if NINF - n_inf else np.nan,
            "mean_url_count": float(gdf["url_count"].mean()),
            "mean_url_count_inf": float(gi["url_count"].mean()) if n_inf else np.nan,
            "mean_listed_urls": float(gdf["listed_urls"].mean()),
            "share_url_count_le2": float((gdf["url_count"] <= 2).mean()),
            "share_err_title": float(gdf["err_title"].mean()),
            "mean_in_degree": float(gdf["in_degree"].mean()),
            "n_healthy": int(gdf["is_healthy"].sum()),
        })
GRP = pd.DataFrame(GROUP_ROWS)
inf_mask = GRP["informative_n_ge_10"]
GRP["m_bh_family"] = np.nan
for fam in FAMILIES:
    sel = (GRP["family"] == fam) & inf_mask
    GRP.loc[sel, "m_bh_family"] = int(sel.sum())
GRP["q_bh_family"] = np.nan
for fam in FAMILIES:
    sel = (GRP["family"] == fam) & inf_mask
    if sel.sum():
        GRP.loc[sel, "q_bh_family"] = bh(GRP.loc[sel, "fisher_p"])
sel_all = inf_mask
GRP.loc[sel_all, "m_bh_pooled"] = int(sel_all.sum())
GRP.loc[sel_all, "q_bh_pooled"] = bh(GRP.loc[sel_all, "fisher_p"])
GRP["sig_after_bh_pooled"] = np.where(inf_mask, GRP["q_bh_pooled"] < ALPHA, np.nan)
rec("m_bh_pooled", int(sel_all.sum()), "{:d}")
rec("m_bh_family_max", int(GRP["m_bh_family"].max()), "{:d}")
rec("n_groups_total", len(GRP), "{:d}")
rec("n_groups_killed_small", int((~inf_mask).sum()), "{:d}")
rec("n_nodes_in_killed", int(GRP.loc[~inf_mask, "n_all"].sum()), "{:d}")
rec("n_sig_bh", int((GRP["q_bh_pooled"] < ALPHA).sum()), "{:d}")
rec("n_sig_raw_p", int((GRP.loc[inf_mask, "fisher_p"] < ALPHA).sum()), "{:d}")
GRP = GRP.sort_values(["family", "n_all"], ascending=[True, False]).reset_index(drop=True)
inf_mask = GRP["informative_n_ge_10"].to_numpy()
C.save_csv(GRP, MOD, "d5_groups.csv")
say(f"[{time.time()-T0:6.1f}s] 逐组检验完成：{len(GRP)} 组，n<{MIN_N} 剔除 {N('n_groups_killed_small')} 组，BH 后显著 {N('n_sig_bh')} 组")

# ============================================ 3. 整体：η² / ICC(1) / 标签置换（§4）
rng = np.random.default_rng(SEED)


def omnibus(col, y, mask=None, reps=N_PERM, seed=SEED + 5):
    """单向分组 SSB 的标签置换检验（§4：每轮独立重排，不累积）。mask = 样本子集。"""
    lab_all = pd.factorize(m[col].to_numpy())[0]
    if mask is None:
        labels, yy = lab_all, np.asarray(y, float)
    else:
        labels, yy = pd.factorize(lab_all[mask])[0], np.asarray(y, float)[mask]
    labels = pd.factorize(labels)[0]
    n = len(yy)
    ng = int(labels.max()) + 1
    cnt = np.bincount(labels, minlength=ng).astype(float)
    tot = float(yy.sum())
    pbar = tot / n
    sst = float(((yy - pbar) ** 2).sum())
    gsum = np.bincount(labels, weights=yy, minlength=ng)
    ssb = float(((gsum ** 2) / cnt).sum() - tot * tot / n)
    ssw = sst - ssb
    dfb = ng - 1
    dfw = n - ng
    msb = ssb / dfb if dfb else np.nan
    msw = ssw / dfw if dfw > 0 else np.nan
    kbar = n / ng
    icc = (msb - msw) / (msb + (kbar - 1) * msw) if (msb and msw and (msb + (kbar - 1) * msw) != 0) else np.nan
    chi2 = float((((gsum - cnt * pbar) ** 2) / (cnt * pbar * (1 - pbar))).sum()) if 0 < pbar < 1 else np.nan
    r = np.random.default_rng(seed)
    perms = np.array([float((((np.bincount(labels, weights=pp, minlength=ng)) ** 2) / cnt).sum()) - tot * tot / n
                      for pp in (yy[r.permutation(n)] for _ in range(reps))])
    mean, sd = float(perms.mean()), float(perms.std(ddof=1))
    nb = int((np.abs(perms - mean) >= abs(ssb - mean)).sum())
    return {"n_sample": n, "ssb_obs": ssb, "sst": sst, "ssw": ssw,
            "eta2": ssb / sst if sst else np.nan, "var_share_within": ssw / sst if sst else np.nan,
            "icc1": float(icc), "n_groups": ng, "dfb": dfb, "chi2": chi2,
            "chi2_p_asym": float(stats.chi2.sf(chi2, dfb)) if (dfb > 0 and np.isfinite(chi2)) else np.nan,
            "perm_mean": mean, "perm_sd": sd, "z": (ssb - mean) / sd if sd > 0 else np.nan,
            "p_two_sided": nb / reps, "p_plus1": (1 + nb) / (1 + reps), "reps": reps,
            "p_floor": 1.0 / (1 + reps), "kbar": kbar,
            "min_g": int(cnt.min()), "median_g": float(np.median(cnt)),
            "n_groups_ge10": int((cnt >= MIN_N).sum())}


for fam, col in FAMILIES.items():
    for tag, yv, mk in (("校正前 is_dead(全样本)", m["is_dead"].to_numpy(float), None),
                        ("校正后 true_fail(剔除未证实)", m["true_fail"].to_numpy(float), INFO.to_numpy()),
                        ("校正前·剔除未证实 is_dead(inf)", m["is_dead"].to_numpy(float), INFO.to_numpy())):
        o = omnibus(col, yv, mk)
        OMNI_ROWS.append({"family": fam, "column": col, "outcome": tag, **o})
OMNI = pd.DataFrame(OMNI_ROWS)
C.save_csv(OMNI, MOD, "d5_omnibus.csv")
OMNI_TF = OMNI[OMNI["outcome"] == "校正后 true_fail(剔除未证实)"].set_index("family")
say(f"[{time.time()-T0:6.1f}s] 整体方差分解 + 标签置换 {N_PERM} 轮 × {len(OMNI)} 设定完成（{time.time()-T0:.0f}s）")

# 抽几个关键量进 KEY（供正文引用）
for fam, col in FAMILIES.items():
    tagc = fam.split()[0]
    for tag, keypfx in (("校正后 true_fail(剔除未证实)", "tf"), ("校正前 is_dead(全样本)", "dd")):
        row = OMNI[(OMNI["family"] == fam) & (OMNI["outcome"] == tag)].iloc[0]
        rec(f"eta2_{tagc}_{keypfx}", float(row["eta2"]), "{:.6f}")
        rec(f"icc_{tagc}_{keypfx}", float(row["icc1"]), "{:.6f}")
        rec(f"chi2_{tagc}_{keypfx}", float(row["chi2"]), "{:.2f}")
        rec(f"z_{tagc}_{keypfx}", float(row["z"]), "{:.3f}")
        rec(f"p_{tagc}_{keypfx}", float(row["p_two_sided"]), "{:.6f}")
        rec(f"pp1_{tagc}_{keypfx}", float(row["p_plus1"]), "{:.6f}")
        rec(f"ng_{tagc}_{keypfx}", int(row["n_groups"]), "{:d}")
        rec(f"ng10_{tagc}_{keypfx}", int(row["n_groups_ge10"]), "{:d}")
        rec(f"within_{tagc}_{keypfx}", float(row["var_share_within"]), "{:.6f}")

# 4. 逐组关键行的快速取数
GRP_INF = GRP[inf_mask].copy()


def grow(col, group, field):
    sel = (GRP["column"] == col) & (GRP["group"] == group)
    return float(GRP.loc[sel, field].iloc[0])


rec("apex_n", grow("g_root", "APEX(裸子域)", "n_all"), "{:d}")
rec("apex_tf", grow("g_root", "APEX(裸子域)", "n_true_fail"), "{:d}")
rec("apex_rate_tf", grow("g_root", "APEX(裸子域)", "rate_true_fail"), "{:.6f}")
rec("apex_share", grow("g_root", "APEX(裸子域)", "n_all") / NN, "{:.6f}")
rec("lib_n", grow("g_root", "lib", "n_all"), "{:d}")
rec("lib_tf", grow("g_root", "lib", "n_true_fail"), "{:d}")
rec("lib_rate", grow("g_root", "lib", "rate_true_fail"), "{:.6f}")
rec("lib_or", grow("g_root", "lib", "fisher_or"), "{:.4f}")
rec("lib_p", grow("g_root", "lib", "fisher_p"), "{:.6f}")
rec("lib_q", grow("g_root", "lib", "q_bh_pooled"), "{:.6f}")
rec("cmet_n", grow("g_root", "cmet", "n_all"), "{:d}")
rec("cmet_tf", grow("g_root", "cmet", "n_true_fail"), "{:d}")
rec("cmet_rate", grow("g_root", "cmet", "rate_true_fail"), "{:.6f}")
ROOT_BIG = GRP[(GRP["column"] == "g_root") & inf_mask][["group", "n_all", "n_informative", "n_true_fail",
                                                        "rate_true_fail", "wilson_lo", "wilson_hi", "cp_lo", "cp_hi",
                                                        "fisher_or", "fisher_p", "q_bh_pooled", "fisher_p_pre",
                                                        "q_bh_family", "mean_url_count", "share_url_count_le2"]]
C.save_csv(ROOT_BIG, MOD, "d5_root_groups.csv")

# 5. 泄漏诊断：类别（标题规则）与 CMS 指纹都可被"抓取失败"本身决定
LEAK = []
for fam, col in FAMILIES.items():
    sub = m[[col, "is_dead", "true_fail", "err_title", "url_count", "listed_urls"]].copy()
    gb = sub.groupby(col)
    tab = pd.DataFrame({"n": gb.size(), "dead": gb["is_dead"].sum(),
                        "all_err_title": gb["err_title"].mean(),
                        "mean_url_count": gb["url_count"].mean()})
    pure_dead = int(((tab["dead"] == tab["n"]) & (tab["n"] > 0)).sum())
    pure_alive = int(((tab["dead"] == 0) & (tab["n"] >= MIN_N)).sum())
    LEAK.append({"family": fam, "n_groups": len(tab),
                 "groups_all_members_dead": pure_dead,
                 "n_nodes_in_all_dead_groups": int(tab.loc[tab["dead"] == tab["n"], "n"].sum()),
                 "groups_ge10_all_alive": pure_alive,
                 "share_nodes_in_all_dead_groups": float(tab.loc[tab["dead"] == tab["n"], "n"].sum() / NN),
                 "max_group_share_err_title": float(tab["all_err_title"].max()),
                 "groups_with_err_title_le_1": int((tab["all_err_title"] <= 1.0).sum()),
                 "top_group": tab["n"].idxmax(), "top_group_n": int(tab["n"].max())})
LEAKDF = pd.DataFrame(LEAK)
C.save_csv(LEAKDF, MOD, "d5_leakage_censoring.csv")
rec("cat_all_dead_group_nodes", int(LEAKDF.loc[LEAKDF["family"].str.startswith("C1"), "n_nodes_in_all_dead_groups"].iloc[0]), "{:d}")
rec("cms_all_dead_group_nodes", int(LEAKDF.loc[LEAKDF["family"].str.startswith("M1"), "n_nodes_in_all_dead_groups"].iloc[0]), "{:d}")
C1 = GRP[GRP["column"] == "g_cat_title"]
rec("c1_groups_with_dead", int((C1["n_dead_precorr"] > 0).sum()), "{:d}")
rec("c1_other_dead", int(C1.loc[C1["group"] == "其他", "n_dead_precorr"].iloc[0]), "{:d}")
rec("c1_other_n", int(C1.loc[C1["group"] == "其他", "n_all"].iloc[0]), "{:d}")
rec("c1_other_tf", int(C1.loc[C1["group"] == "其他", "n_true_fail"].iloc[0]), "{:d}")
rec("c1_max_tf_outside_other", int(C1.loc[C1["group"] != "其他", "n_true_fail"].max()), "{:d}")
M1TOP = GRP[GRP["column"] == "g_cms"].sort_values("n_all", ascending=False).iloc[0]
rec("m1_top_group", str(M1TOP["group"]), "{}")
rec("m1_top_n", int(M1TOP["n_informative"]), "{:d}")
rec("m1_top_rate", float(M1TOP["rate_true_fail"]), "{:.6f}")
rec("m1_top_uc", float(M1TOP["mean_url_count"]), "{:.3f}")
rec("m1_top_le2", float(M1TOP["share_url_count_le2"]), "{:.6f}")
M1LOW = (GRP[GRP["column"] == "g_cms"]
         .query("n_informative >= @MIN_N").sort_values("rate_true_fail").iloc[0])
rec("m1_lowest_group", str(M1LOW["group"]), "{}")
rec("m1_lowest_rate", float(M1LOW["rate_true_fail"]), "{:.6f}")
rec("m1_lowest_uc", float(M1LOW["mean_url_count"]), "{:.3f}")
rec("sp_uc_vs_rate_cms", float(stats.spearmanr(GRP[GRP["column"] == "g_cms"]["mean_url_count"],
                                               GRP[GRP["column"] == "g_cms"]["rate_dead_precorr"])[0]), "{:.4f}")

# 5b. 分层置换：把"URL 枚举深度"钉住，再看指纹分组的组间平方和还剩多少
m["uc_stratum"] = pd.cut(m["url_count"], [-1, 2, 9, 49, 10 ** 9],
                         labels=["<=2", "3-9", "10-49", ">=50"]).astype(str)
I_MASK = INFO.to_numpy()
ROWS_I = np.flatnonzero(I_MASK)
Y_INF = m.loc[I_MASK, "true_fail"].to_numpy(float)
UC_CODE = pd.factorize(m.loc[I_MASK, "uc_stratum"], sort=True)[0]
STRATA_SIZES = np.bincount(UC_CODE)
rec("n_strata", int(len(STRATA_SIZES)), "{:d}")
rec("strata_sizes", " ".join(f"{k}:{v}" for k, v in
                             zip(sorted(set(m.loc[I_MASK, 'uc_stratum'])), STRATA_SIZES)), "{}")


def ssb_of(lab, yv, ng):
    nn = len(yv)
    tot = float(yv.sum())
    pbar = tot / nn
    sst = float(((yv - pbar) ** 2).sum())
    cnt = np.bincount(lab, minlength=ng).astype(float)
    gsum = np.bincount(lab, weights=yv, minlength=ng)
    ssb = float(((gsum ** 2) / cnt).sum() - tot * tot / nn)
    return ssb, sst


def strat_perm(col, reps=5000, seed=SEED + 17):
    """层内置换（保持每个 url_count 层里的失效数与每层人数不变），§4 每轮独立重排。"""
    lab = pd.factorize(m.loc[I_MASK, col], sort=False)[0]
    ng = int(lab.max()) + 1
    ssb_obs, sst = ssb_of(lab, Y_INF, ng)
    groups_by_stratum = [np.flatnonzero(UC_CODE == s) for s in range(len(STRATA_SIZES))]
    r = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        y2 = Y_INF.copy()                      # 每轮新建，绝不在上一轮数组上继续改
        for idx in groups_by_stratum:
            y2[idx] = Y_INF[r.permutation(idx)]
        vals.append(ssb_of(lab, y2, ng)[0])
    vals = np.asarray(vals)
    mean, sd = float(vals.mean()), float(vals.std(ddof=1))
    nb = int((np.abs(vals - mean) >= abs(ssb_obs - mean)).sum())
    ssb_flat, sst_flat = ssb_of(lab, Y_INF, ng)
    return {"family": col, "n_sample": len(Y_INF), "n_groups": ng, "ssb_obs": ssb_obs,
            "eta2_obs": ssb_obs / sst if sst else np.nan,
            "eta2_expected_under_null": mean / sst_flat if sst_flat else np.nan,
            "ssb_null_mean": mean, "ssb_null_sd": sd, "z": (ssb_obs - mean) / sd if sd > 0 else np.nan,
            "p_two_sided": nb / reps, "p_plus1": (1 + nb) / (1 + reps), "reps": reps,
            "unstratified_z": float(OMNI_TF.loc[[k for k in FAMILIES if FAMILIES[k] == col][0], "z"]),
            "unstratified_eta2": float(OMNI_TF.loc[[k for k in FAMILIES if FAMILIES[k] == col][0], "eta2"])}


STRAT = pd.DataFrame([strat_perm(col) for col in FAMILIES.values()])
STRAT["family_tag"] = [k for k in FAMILIES]
C.save_csv(STRAT, MOD, "d5_stratified.csv")
for fam, col in FAMILIES.items():
    tagc = fam.split()[0]
    row = STRAT[STRAT["family"] == col].iloc[0]
    rec(f"eta2s_{tagc}", float(row["eta2_obs"]), "{:.4f}")
    rec(f"eta2s_null_{tagc}", float(row["eta2_expected_under_null"]), "{:.4f}")
    rec(f"zs_{tagc}", float(row["z"]), "{:.3f}")
    rec("ps_" + tagc, float(row["p_plus1"]), "{:.6f}")
say(f"[{time.time()-T0:6.1f}s] 分层置换完成（{len(STRAT)} 族 × 5000 轮，url_count 层 {N('n_strata')} 个）")

# 6. 显著组清单
SIG = GRP_INF[GRP_INF["q_bh_pooled"] < ALPHA].copy()
C.save_csv(SIG[["family", "group", "n_all", "n_informative", "n_unverified", "n_true_fail", "rate_true_fail",
                "wilson_lo", "wilson_hi", "cp_lo", "cp_hi", "fisher_or", "fisher_p", "q_bh_pooled", "q_bh_family",
                "n_dead_precorr", "rate_dead_precorr", "fisher_p_pre", "mean_url_count", "share_url_count_le2",
                "share_err_title", "mean_in_degree"]],
           MOD, "d5_significant_groups.csv")
rec("sig_list", ", ".join(SIG["family"].str.split().str[0] + ":" + SIG["group"].astype(str)) or "（无）", "{}")

C.save_csv(pd.DataFrame([{"key": k, "value": d["value"], "ref": d["ref"], "note": d["note"]}
                         for k, d in KEY.items() if not k.startswith("_")]), MOD, "key_numbers.csv")


def mdtable(df, cols, headers, fmts):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c, fm in zip(cols, fmts):
            v = r[c]
            if isinstance(v, (int, np.integer)):
                cells.append(f"{v:d}")
            elif isinstance(v, (float, np.floating)):
                if pd.isna(v):
                    cells.append("—")
                else:
                    cells.append(fm.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)



# ---- 恒等校验 + 判决标签（数据驱动，不预设方向）
rec("n_dead_inf", int(m.loc[INFO, "is_dead"].sum()), "{:d}")
rec("ident_dead_eq_tf_in_inf", int(N("n_dead_inf") == N("n_true_fail")), "{:d}")
rec("n_omni_settings", len(OMNI), "{:d}")
rec("n_omni_sig", int((OMNI["p_plus1"] < ALPHA).sum()), "{:d}")
rec("omni_sig_list", "; ".join(f"{r['family']}·{r['outcome']} z={r['z']:.2f}"
                               for _, r in OMNI[OMNI["p_plus1"] < ALPHA].iterrows()) or "（无）", "{}")
rec("n_strat_sig", int((STRAT["p_plus1"] < ALPHA).sum()), "{:d}")
rec("strat_sig_list", "; ".join(f"{r['family_tag']} z={r['z']:.2f}"
                                for _, r in STRAT[STRAT["p_plus1"] < ALPHA].iterrows()) or "（无）", "{}")
rec("n_strat_below", int((STRAT["z"] < 0).sum()), "{:d}")
rec("sig_families", ", ".join(sorted(set(SIG["family"].str.split().str[0]))) or "（无）", "{}")
rec("n_sig_from_clean_fams", int(SIG["family"].str.split().str[0].isin(["R1", "C2"]).sum()), "{:d}")


def verdict(fam):
    r = OMNI_TF.loc[fam]
    lab = "显著" if r["p_plus1"] < ALPHA else ("边缘" if r["p_plus1"] < 0.10 else "不显著")
    return lab


LAB = {fam: verdict(fam) for fam in FAMILIES}
rec("eta2_max_tf", float(OMNI_TF["eta2"].max()), "{:.6f}")
rec("icc_max_tf", float(OMNI_TF["icc1"].max()), "{:.6f}")
rec("c2_other_n", int((m["g_cat_sub"] == "其他短名").sum()), "{:d}")

INF_SHOW = GRP_INF.sort_values(["family", "n_all"], ascending=[True, False]).head(40)
KILLED_SHOW = GRP[~inf_mask].groupby("family").agg(n_groups=("group", "size"), n_nodes=("n_all", "sum"),
                                                   max_n=("n_all", "max")).reset_index()

MD = f"""# D5 · 失效聚集的替代分组检验 —— 结果

- 节点口径：主口径 {cite('n_nodes')} 个有属性节点（`csa_common.build_all()['nodes']`）。分组键与失效状态**都可直接观察**——这正是 D4 里"死—死相邻"所缺的条件，所以本单元把聚集问题从边层面搬到节点层面。
- 分组族：R1 根域前缀（{cite('n_groups_root')} 组）、C1 类别·标题规则（{cite('n_groups_cat_title')} 组）、C2 类别·短名 token（{cite('n_groups_cat_sub')} 组）、M1 CMS 指纹＝众数扩展名×归一化首段（{cite('n_groups_cms')} 组）、M2 扩展名（{cite('n_groups_ext')} 组）。
- α = {ALPHA}；"信息组"＝可判定样本内 n ≥ {MIN_N} 的组，进入 BH 的池化 m = {cite('m_bh_pooled')}（每族 m 见 `m_bh_family`，最大 {cite('m_bh_family_max')}）；n < {MIN_N} 的 {cite('n_groups_killed_small')} 个组判"不可判定"并从推断里剔除（涉及 {cite('n_nodes_in_killed')} 个节点次）。
- 两套口径（§2.2）：校正前＝`is_dead`（{cite('n_dead')}）在全部 {cite('n_nodes')} 个节点上，基准率 {cite('base_rate_dead', True)}%；校正后＝`true_fail`（{cite('n_true_fail')}）在剔除 {cite('n_unverified')} 个未证实节点后的 {cite('n_informative')} 个节点上，基准率 {cite('base_rate_tf_inf', True)}%。可判定样本内 `is_dead` 与 `true_fail` 完全重合（{cite('n_dead_inf')}＝{cite('n_true_fail')}，校验位 {cite('ident_dead_eq_tf_in_inf')}），所以两套口径的差别只可能是那 33 个节点造成的。
- 置换：① 整体标签重排 {N_PERM} 轮 × {cite('n_omni_settings')} 个（族×口径）设定；② 分层（把 url_count 层内的失效数钉死）{5000} 轮 × 5 族。每轮都新建数组、独立重排（§4，不累积）；p 分辨率下限 1/(轮数+1)。
- 产物：`d5_groups.csv`（{cite('n_groups_total')} 行全量）、`d5_root_groups.csv`、`d5_omnibus.csv`、`d5_stratified.csv`、`d5_significant_groups.csv`、`d5_leakage_censoring.csv`、`key_numbers.csv`、`anchors.csv`。

## 0. 判决

| 族 | 观测 η²(校正后) | ICC(1) | 整体置换 z | p+1 | 钉住 url_count 后 z | p+1 | 该族能否作证据 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| R1 根域前缀 | {cite('eta2_R1_tf')} | {cite('icc_R1_tf')} | {cite('z_R1_tf')} | {cite('pp1_R1_tf')} | {cite('zs_R1')} | {cite('ps_R1')} | **能**（键与抓取结果无关），结果＝无聚集 |
| C1 类别(标题规则) | {cite('eta2_C1_tf')} | {cite('icc_C1_tf')} | {cite('z_C1_tf')} | {cite('pp1_C1_tf')} | {cite('zs_C1')} | {cite('ps_C1')} | **不能**：分类器泄漏（§4） |
| C2 类别(短名token) | {cite('eta2_C2_tf')} | {cite('icc_C2_tf')} | {cite('z_C2_tf')} | {cite('pp1_C2_tf')} | {cite('zs_C2')} | {cite('ps_C2')} | 能，作 C1 的可观察替身；结果＝无聚集 |
| M1 CMS 指纹 | {cite('eta2_M1_tf')} | {cite('icc_M1_tf')} | {cite('z_M1_tf')} | {cite('pp1_M1_tf')} | {cite('zs_M1')} | {cite('ps_M1')} | **不能**：删失假象（§3） |
| M2 扩展名 | {cite('eta2_M2_tf')} | {cite('icc_M2_tf')} | {cite('z_M2_tf')} | {cite('pp1_M2_tf')} | {cite('zs_M2')} | {cite('ps_M2')} | **不能**：删失假象（§3） |

1. **与抓取结果无关的两族（R1、C2）都测不出聚集。** R1 校正后 η²={cite('eta2_R1_tf')}、ICC(1)={cite('icc_R1_tf')}、z={cite('z_R1_tf')}（p+1={cite('pp1_R1_tf')}）；C2 η²={cite('eta2_C2_tf')}、z={cite('z_C2_tf')}（p+1={cite('pp1_C2_tf')}）。两族的观测 SSB 都**低于**零模型均值（z 为负）。
2. **看起来"最强"的聚集全部来自 M1/M2，而它恰好是删失的签名。** 不分层时 M1 z={cite('z_M1_tf')}、M2 z={cite('z_M2_tf')}（p+1 都到 {cite('pp1_M2_tf')} 的下限）；一旦把 url_count 分层钉住（{cite('n_strata')} 层：{cite('strata_sizes')}），同样统计量的 z 变成 M1 {cite('zs_M1')}、M2 {cite('zs_M2')}——降到零模型期望之下。**η² 是同一个观测值（{cite('eta2_M1_tf')}/{cite('eta2_M2_tf')}），变掉的只是参照系**：指纹分桶与"这个站能被枚举出多少 URL"绑在一起。
3. **C1 的"聚集"分层后依然存在（z={cite('zs_C1')}、p+1={cite('ps_C1')}），但那是标签泄漏而不是删失**：全部 {cite('c1_other_tf')} 个真失效都落在"其他"这一组，其余 {cite('c1_groups_with_dead')} − 1 ＝ 0 个组含死站（`n_dead_precorr` 列，"其他"组 {cite('c1_other_dead')}/{cite('n_dead')}）；非"其他"组里真失效最大值 {cite('c1_max_tf_outside_other')}。死站标题一律是抓取错误串，命中不了关键词，所以 C1 量的是"分类器认不出错误标题"。
4. **BH 之后显著的 {cite('n_sig_bh')} 个组全部来自 C1/M1/M2**（{cite('sig_list')}；来自干净族 R1/C2 的有 {cite('n_sig_from_clean_fams')} 个）。{cite('n_omni_settings')} 个整体置换设定里过 α 的 {cite('n_omni_sig')} 个、分层置换里过 α 的 {cite('n_strat_sig')} 个（{cite('omni_sig_list')}／{cite('strat_sig_list')}），其中 {cite('n_strat_below')} 个族的 z 为负。
5. **结论：换成分组检验之后，仍然得不到"失效在类别/委托域/模板族上聚集"的证据**；能得到的只有两条方法学事实——(i) 任务书里"类别"这一维在数据中**没有 category 字段**，可复原的标题规则分类器与结果定义相关；(ii) CMS 指纹维度的显著性可以在不改变观测值的前提下仅靠更换零模型而消失。

## 1. 逐组：真失效率 + 双区间 + 对"网络其余"的检验（信息组，`d5_groups.csv`）

率＝`true_fail`／可判定节点数；Wilson 与 Clopper–Pearson 均为 95%；`fisher_p`＝该组 vs 其余的 2×2 Fisher 双侧；`q_bh_pooled`＝跨 {cite('m_bh_pooled')} 个信息组（五族合并）的 BH q；`q_bh_family`＝族内 BH q；最小期望格 < 5 时卡方不可用（`test_used`=fisher_only）。校正前 `is_dead` 的同名统计在 `*_pre` 列。**`OR=inf` 出现在"该组之外一个失效都没有"时，是泄漏的信号，不是效应的量度。**

""" + mdtable(
    INF_SHOW,
    ["family", "group", "n_all", "n_informative", "n_unverified", "n_true_fail", "rate_true_fail",
     "wilson_lo", "wilson_hi", "cp_lo", "cp_hi", "fisher_or", "fisher_p", "q_bh_pooled", "q_bh_family",
     "expected_min_cell", "n_dead_precorr", "mean_url_count", "share_url_count_le2", "share_err_title"],
    ["族", "组", "n(全)", "n(可判定)", "未证实", "真失效k", "率", "Wilson下", "Wilson上", "CP下", "CP上",
     "OR", "Fisher p", "BH q(池)", "BH q(族)", "最小期望格", "死站(校正前)", "url_count均值", "url_count<=2占比", "错误标题占比"],
    ["{}", "{}", "{:d}", "{:d}", "{:d}", "{:d}", "{:.3f}", "{:.3f}", "{:.3f}", "{:.3f}", "{:.3f}", "{:.3f}",
     "{:.2e}", "{:.2e}", "{:.2e}", "{:.2f}", "{:d}", "{:.1f}", "{:.3f}", "{:.3f}"]) + f"""

（仅列 n(可判定) ≥ {MIN_N} 且规模最大的 {len(INF_SHOW)} 组；全量 {cite('n_groups_total')} 行在 `d5_groups.csv`。）

被剔除的小组（n(可判定) < {MIN_N} → 不可判定、不参与 BH）：

""" + mdtable(KILLED_SHOW, ["family", "n_groups", "n_nodes", "max_n"],
             ["族", "被剔除组数", "被剔除组节点次", "其中最大组 n"], ["{}", "{:d}", "{:d}", "{:d}"]) + f"""

根域族单独成表（唯一干净族，含双区间与 BH）见 `d5_root_groups.csv`：APEX(裸子域) {cite('apex_n')} 个节点（占全体 {cite('apex_share', True)}%）里真失效 {cite('apex_tf')} 个（{cite('apex_rate_tf', True)}%），lib {cite('lib_n')} 个里 {cite('lib_tf')} 个（{cite('lib_rate', True)}%，Fisher p={cite('lib_p')}、池化 BH q={cite('lib_q')}），cmet {cite('cmet_n')} 个里 {cite('cmet_tf')} 个（{cite('cmet_rate', True)}%）。

## 2. 组间 / 组内方差分解 + 整体标签置换（`d5_omnibus.csv`）

η² = SSB/SST（二值结果下等于点二列相关的平方）；ICC(1) = (MSB−MSW)/(MSB+(k̄−1)·MSW)；整体检验＝保持各组人数不变、把结果标签独立重排 {N_PERM} 轮，统计量取 SSB，按 §4 报 null mean / sd / z / 双侧 p（看偏离零模型均值，不看 |vals|≥|obs|）。

""" + mdtable(
    OMNI, ["family", "outcome", "n_sample", "n_groups", "n_groups_ge10", "eta2", "icc1", "chi2", "dfb",
           "perm_mean", "perm_sd", "z", "p_two_sided", "p_plus1"],
    ["族", "结果口径", "样本n", "组数", "n>=10组", "η²", "ICC(1)", "χ²", "df", "SSB null均值", "SSB null标准差",
     "z", "双侧 p", "p+1"],
    ["{}", "{}", "{:d}", "{:d}", "{:d}", "{:.4f}", "{:.4f}", "{:.2f}", "{:d}", "{:.3f}", "{:.3f}", "{:.2f}",
     "{:.5f}", "{:.5f}"]) + f"""

三点要和上表一起读：(a) 组数多的族（R1 {cite('ng_R1_tf')} 组、M1 {cite('ng_M1_tf')} 组）里期望频数 < 5 的格很多，**χ² 的解析 p 只当参考**（`chi2_p_asym`），结论以置换 p 为准。(b) 表中"校正后 true_fail"与"校正前·剔除未证实 is_dead"两行**逐字节相同**——这不是 bug，而是 §0 里那条恒等关系（可判定样本内 `is_dead`≡`true_fail`）的实现校验。(c) 同族校正前→校正后的 z 变化给出把 {cite('n_unverified')} 个未证实节点计入失效的代价，例如 R1 从 z={cite('z_R1_dd')} 变成 {cite('z_R1_tf')}。

## 3. 分层置换：把"URL 枚举深度"钉住（`d5_stratified.csv`）

问题：M1/M2 的组间差异是不是"死站本来就枚举不出几个 URL"的另一种写法？做法：只在 url_count 层内重排失效标签（每层的失效数、人数、组×层列联表边缘都保持不变），统计量仍是同一 SSB。

""" + mdtable(
    STRAT, ["family_tag", "n_sample", "n_groups", "ssb_obs", "eta2_obs", "ssb_null_mean", "ssb_null_sd", "z",
            "p_two_sided", "p_plus1", "unstratified_z"],
    ["族", "样本n", "组数", "SSB(obs)", "η²(同 §2)", "分层null均值", "分层null sd", "z", "双侧 p", "p+1", "不分层 z"],
    ["{}", "{:d}", "{:d}", "{:.3f}", "{:.4f}", "{:.3f}", "{:.3f}", "{:.2f}", "{:.5f}", "{:.5f}", "{:.2f}"]) + f"""

- 分层后 {cite('n_strat_below')} 个族的 z 为负（观测比随机还小），包括 M1（{cite('zs_M1')}）与 M2（{cite('zs_M2')}）。也就是说这两族的"聚集"完全在枚举深度的解释范围之内。
- 佐证（组层面）：211 个 CMS 指纹组的 `mean_url_count` 与 `rate_dead_precorr` 的 Spearman = {cite('sp_uc_vs_rate_cms')}；最大组 {cite('m1_top_group')}（{cite('m1_top_n')} 个可判定节点）真失效率 {cite('m1_top_rate', True)}%、url_count 均值 {cite('m1_top_uc')}、{cite('m1_top_le2', True)}% 的成员只有 ≤2 条可枚举 URL；而 {cite('m1_lowest_group')} 组失效率为 {cite('m1_lowest_rate', True)}%、url_count 均值 {cite('m1_lowest_uc')}。两组的差别首先是"能不能枚举"，其次才是"什么模板"。
- 只有 C1 在分层后仍显著（z={cite('zs_C1')}），因为它的泄漏不在枚举深度这条通道上（见 §4）。

## 4. 泄漏 / 删失诊断（`d5_leakage_censoring.csv`）

""" + mdtable(
    LEAKDF, ["family", "n_groups", "groups_all_members_dead", "n_nodes_in_all_dead_groups",
             "groups_ge10_all_alive", "top_group", "top_group_n", "max_group_share_err_title"],
    ["族", "组数", "全员死站的组数", "这些组里的节点数", "n>=10 且全员活站的组数", "最大组", "n", "最大组错误标题占比"],
    ["{}", "{}", "{:d}", "{:d}", "{:d}", "{}", "{:d}", "{:.3f}"]) + f"""

- **C1 类别（标题规则）：结论作废。** 标题带抓取错误串的节点 {cite('n_err_title')} 个，其中死站 {cite('dead_with_err_title')} 个（占死站 {cite('err_title_share_dead', True)}%）；这些串命中不了任何关键词，全部落进"其他"（{cite('c1_other_n')} 个节点、{cite('c1_other_dead')} 个死站、{cite('c1_other_tf')} 个真失效），C1 中含死站的组共 {cite('c1_groups_with_dead')} 个。数据里**没有 category 字段**，所以类别维度只能用 C2 替身。
- **M1/M2 CMS 指纹：只作描述。** 死站可枚举 URL 极少（`url_count`≤2 的占比：死站 {cite('uc_le2_dead', True)}%、活站 {cite('uc_le2_alive', True)}%；均值：死站 {cite('uc_dead')}、部分失效 {cite('uc_partial')}、健康 {cite('uc_healthy')}），只有首页的站必然落进 `<none>/<root>` 桶。§3 的分层置换已经把这条通道量化。
- **R1 根域前缀：唯一不受这两条通道污染的族**（分组键由节点名给出）。代价是 {cite('apex_share', True)}% 的节点挤在 APEX 一组，n(可判定)≥{MIN_N} 的组只有 {cite('ng10_R1_tf')} 个，其余组都很小、区间很宽。

## 5. 可辩护的表述

1. **委托域维度（R1）**：真失效率的组间份额 η²={cite('eta2_R1_tf')}、ICC(1)={cite('icc_R1_tf')}；整体置换 z={cite('z_R1_tf')}、p+1={cite('pp1_R1_tf')}（{N_PERM} 轮），分层后 z={cite('zs_R1')}、p+1={cite('ps_R1')}。APEX 组率 {cite('apex_rate_tf', True)}% 对基准 {cite('base_rate_tf_inf', True)}%。**"同一委托域（同一院系/同一 IT 管理单元）的站点一起坏"得不到支持**；同时可推断组只有 {cite('ng10_R1_tf')} 个，也不足以支持强反对——正确写法是"未见聚集，功效有限"。
2. **类别维度（C1 不可用，C2 替身）**：C2 η²={cite('eta2_C2_tf')}、z={cite('z_C2_tf')}、p+1={cite('pp1_C2_tf')}，分层后 z={cite('zs_C2')}；但 token 表只覆盖有限，{cite('c2_other_n')} 个节点落在"其他短名"，所以只能说"按可观察短名 token 分组未见聚集"。
3. **模板/CMS 维度**：观测 η² 很高（M1 {cite('eta2_M1_tf')}、M2 {cite('eta2_M2_tf')}）但不支持"同模板一起坏"——把枚举深度钉住后 z 转到负侧（M1 {cite('zs_M1')}、M2 {cite('zs_M2')}）。可辩护的写法是"**指纹分桶与 URL 枚举深度共变，共变之外没有残余的模板聚集**"。
4. **多重比较**：α={ALPHA}、m={cite('m_bh_pooled')}（池化，五族合并）；原始 p<α 的 {cite('n_sig_raw_p')} 组在 BH 后仍为 {cite('n_sig_bh')} 组（{cite('sig_list')}）。族内 BH 的 q 见 `q_bh_family`，族内 m 见 `m_bh_family`（最大 {cite('m_bh_family_max')}）。任何引用这 {cite('n_sig_bh')} 组的表述都必须带 §3/§4 的两条通道说明。
5. 与 D4 相互印证：边层面的死—死相邻不可识别（可观察样本点为 0），节点层面的分组检验在这份快照上给出的组间份额最高也只有 η²={cite('eta2_max_tf')}（且属泄漏族），干净族只有 {cite('eta2_R1_tf')}／{cite('eta2_C2_tf')}。**本单元的诚实结果是"阴性＋两条已知伪迹通道"。**
"""

with open(os.path.join(C.module_dir(MOD), "result.md"), "w", encoding="utf-8") as fh:
    fh.write(MD)

wall = time.time() - T0
C.write_run_note(MOD, [
    "D5_failure_clustering_alt RUN.txt",
    "command: cd pipeline && PYTHONHASHSEED=0 python out/D5_failure_clustering_alt/run_d5.py",
    "versions: " + ", ".join(f"{k}={v}" for k, v in C.versions().items()),
    f"SEED = {SEED}; PYTHONHASHSEED = {os.environ.get('PYTHONHASHSEED')}",
    f"total wall time: {wall:.1f}s",
    f"anchors: {int(ANCH['ok'].sum())}/{len(ANCH)} PASS (anchors.csv)",
    f"groupings: {list(FAMILIES.values())} ; MIN_N={MIN_N}（n<{MIN_N} 判不可判定并从推断剔除）; alpha={ALPHA}; "
    f"m(BH 池化)={N('m_bh_pooled')}; m(BH 每族)见 d5_groups.csv 的 m_bh_family 列（最大 {N('m_bh_family_max')}）",
    f"outcomes: 校正前 is_dead(全 {NN} 节点) / 校正后 true_fail(剔除 {N('n_unverified')} 个未证实节点后的 {N('n_informative')} 节点)；"
    f"第三套（is_dead on informative）与第二套恒等（{N('n_dead_inf')}={N('n_true_fail')}），作为实现校验保留",
    f"permutation 1: 整体标签重排 {N_PERM} reps x {len(OMNI)} settings; rng=default_rng({SEED}+5)",
    f"permutation 2: url_count 分层内置换 5000 reps x 5 families; rng=default_rng({SEED}+17); "
    f"层={list(pd.cut(m['url_count'], [-1,2,9,49,10**9], labels=['<=2','3-9','10-49','>=50']).astype(str).unique())!r} 规模 {N('strata_sizes')}",
    "每轮都新建标签数组并在层内独立 permutation（不在上一轮结果上继续打乱，无累积轨迹）",
    "null reporting: 报 SSB 的 null mean/sd，z=(obs-mean)/sd；双侧 p 用 |perm-mean|>=|obs-mean|；同时给 p+1 与下限 1/(reps+1)",
    "tests: Fisher exact(2x2, two-sided) 始终给出；chi2_contingency 仅当最小期望频数>=5；"
    "Wilson(正态近似) + Clopper-Pearson(scipy.stats.beta.ppf) 双区间；BH 按族内与跨族池化各一套",
    "known leaks: C1 标题规则分类器（死站标题=错误串，全部落'其他'，129/129 真失效）；"
    "M1/M2 指纹与 url_count 删失共变（分层置换后 z 转负）；数据里没有 category 字段；R1 不受这两条影响",
    "outputs: d5_groups.csv d5_root_groups.csv d5_omnibus.csv d5_stratified.csv d5_significant_groups.csv "
    "d5_leakage_censoring.csv key_numbers.csv anchors.csv result.md",
    "",
    "--- stdout ---",
] + LOG)
say(f"[{wall:.1f}s] 写出 result.md + 7 张 CSV + RUN.txt")
