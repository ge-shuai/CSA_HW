# -*- coding: utf-8 -*-
"""D6 · 149 个部分失效站点的解剖 + 失效率与结构位置的同口径相关

任务书要求：
  1) 失败 URL 的状态码 / 扩展名 / 路径签名解剖（共现表，不作断言）；
  2) **强制**把 149 个站点拆成 69 个"含认证类状态码(-1/-2/403)"与 80 个"干净"，分开报告；
  3) 失效率与结构位置（入度、url_count）的相关，口径与 D1 一致（失效分析主口径＝入度）。

两条必须先讲清的口径（CONVENTIONS §2.1/§2.2）：
  · `nodes_full.csv` 的 ok/bad/ok_ratio_pct 只在**已枚举的 URL 样本**（`listed_urls`，上限 20）上算，
    不是全站比例；`url_count` 是"发现到的 URL 条数"，两者可差数倍。
  · 状态码 -1（需认证/本轮跳过）与 -2（robots 禁止）**不是失效**；0 里混着"连接失败"和
    "连接失败-本运行跳过"两类标题，本文按标题把它们拆开报告。

只读 CSA_HW/；只写本模块目录；构图 import csa_common；措辞遵守 §7。
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
MOD = "D6_partial_anatomy"
N_BOOT = 5000
T0 = time.time()

KEY: dict = {}


def rec(name, value, fmt="{:.6g}", note=""):
    line = inspect.currentframe().f_back.f_lineno
    KEY[name] = {"value": value, "ref": f"run_d6.py:L{line}", "fmt": fmt, "note": note}
    return value


def N(name):
    return KEY[name]["value"]


def cite(name, pct=False):
    d = KEY[name]
    v = d["value"] * 100 if pct else d["value"]
    try:
        s = d["fmt"].format(v)
    except (ValueError, TypeError):
        s = str(v)
    return f"{s}（{d['ref']}）"


LOG: list = []


def say(s):
    print(s, flush=True)
    LOG.append(str(s))


# ==================================================================== 0. 装配与一致性
C.assert_anchors()
ANCH = C.check_anchors()
C.save_csv(ANCH, MOD, "anchors.csv")
b = C.build_all()
m: pd.DataFrame = b["nodes"]
Gd, Gu = b["Gd"], b["Gu"]
urls = C.load_urls()
sets = b["sets"]
say(f"[{time.time()-T0:6.1f}s] 建图/锚点 {int(ANCH['ok'].sum())}/{len(ANCH)} PASS nodes={len(m)} urls={len(urls)}")

rec("n_nodes", len(m), "{:d}")
rec("n_urls", len(urls), "{:d}")
rec("n_partial", int(m["is_partial"].sum()), "{:d}")
rec("n_auth_like", int(m["partial_has_auth_like"].sum()), "{:d}")
rec("n_clean_partial", int(m["partial_clean"].sum()), "{:d}")
rec("n_healthy", int(m["is_healthy"].sum()), "{:d}")
rec("n_dead", int(m["is_dead"].sum()), "{:d}")
rec("n_true_fail", int(m["true_fail"].sum()), "{:d}")
rec("n_unverified", int((m["false_fail"] | m["ambiguous_fail"]).sum()), "{:d}")

OK_STATUSES = {200, 201}
urls["okrow"] = urls["status"].isin(OK_STATUSES)
urls["failrow"] = ~urls["okrow"]
PERIOD_RE = re.compile(r"\.html?$|\.shtml$|\.php$|\.aspx?$|\.jsp$|\.psp$|\.do$|\.action$")
KNOWN_EXT = {"htm", "html", "php", "asp", "aspx", "jsp", "jspx", "psp", "shtml", "do", "action",
             "md", "json", "xml", "cgi", "pl", "ashx", "cfm", "axd", "asax", "asmx", "pdf",
             "doc", "docx", "xls", "xlsx", "ppt", "pptx", "zip", "rar", "jpg", "png", "gif"}


def parse_url(u):
    p = re.sub(r"^https?://[^/]+", "", str(u)).split("?")[0].split("#")[0].strip("/")
    segs = [x for x in p.split("/") if x]
    fn = segs[-1] if segs else ""
    mm = re.search(r"\.([A-Za-z0-9]{1,6})$", fn)
    e = mm.group(1).lower() if mm else "<none>"
    if e not in KNOWN_EXT:
        e = "<other>"
    first = re.sub(r"\d+", "#", segs[0].lower()) if segs else "<root>"
    return e, first, segs


_p = urls["url"].map(parse_url)
urls["ext"] = [a for a, _, _ in _p]
urls["seg"] = [bb for _, bb, _ in _p]
urls["nsegs"] = [len(ss) for _, _, ss in _p]


def path_sig(segs):
    if not segs:
        return "首页/裸根"
    last = segs[-1].lower()
    if re.match(r"^list\d*\.(html?|php|aspx?|jsp)$", last):
        return "列表页 list.htm"
    if re.match(r"^(info|content|show|view|news|detail)\d*\.(html?|php|aspx?|jsp)$", last):
        return "详情页"
    if re.search(r"/(download|files?|upload|attachment)/", "/" + "/".join(segs).lower()):
        return "下载/附件"
    if last in ("index.htm", "index.html", "default.aspx"):
        return "目录首页 index"
    if re.match(r"^\d+\.(html?|php|aspx?)$", last):
        return "数字命名页"
    return "其他路径"


urls["psig"] = [path_sig(ss) for _, _, ss in _p]


def code_family(s):
    if s in OK_STATUSES:
        return "OK(2xx)"
    if s == -1:
        return "-1 需认证/本轮跳过"
    if s == -2:
        return "-2 robots 禁止"
    if s == 0:
        return "0 连接层失败"
    if s == 403:
        return "403 ACL/禁目录"
    if s in (404, 410):
        return "404/410 服务器应答缺失"
    if s == 400:
        return "400 请求异常"
    if 500 <= s < 600:
        return "5xx 服务器错误"
    return f"其他 {s}"


urls["fam"] = urls["status"].map(code_family)
AUTH_LIKE_FAMS = {"-1 需认证/本轮跳过", "-2 robots 禁止", "403 ACL/禁目录"}
NONFAIL_FAMS = {"-1 需认证/本轮跳过", "-2 robots 禁止"}
urls["auth_like_row"] = urls["fam"].isin(AUTH_LIKE_FAMS)
urls["nonfail_row"] = urls["fam"].isin(NONFAIL_FAMS)
# 0 里面按标题再拆：连接失败 vs 本轮跳过
ZERO = urls["status"] == 0
urls["zero_skip"] = ZERO & urls["title"].astype(str).str.contains("本运行跳过", na=False)
urls["zero_conn"] = ZERO & ~urls["zero_skip"]
rec("n_zero_rows", int(ZERO.sum()), "{:d}")
rec("n_zero_skip", int(urls["zero_skip"].sum()), "{:d}")
rec("n_zero_conn", int(urls["zero_conn"].sum()), "{:d}")

# 一致性：nodes 的 ok/bad 是否等于 urls 里该站的行数
row_counts = urls.groupby("subdomain").size().rename("n_rows")
chk = m.set_index("subdomain").join(row_counts, how="left")
rec("n_partial_with_urlrows", int(chk.loc[chk["is_partial"], "n_rows"].notna().sum()), "{:d}")
rec("mismatch_ok_bad", int((chk["ok"] + chk["bad"] != chk["n_rows"]).sum()), "{:d}")
rec("mismatch_urlcount_listed", int((chk["url_count"] != chk["n_rows"]).sum()), "{:d}")
rec("mismatch_listed_nrows", int((chk["listed_urls"] != chk["n_rows"]).sum()), "{:d}")
rec("listed_cap", float(chk["listed_urls"].max()), "{:d}")
rec("n_listed_at_cap", int((chk["listed_urls"] == chk["listed_urls"].max()).sum()), "{:d}")
say(f"[{time.time()-T0:6.1f}s] 一致性：ok+bad≠行数 {N('mismatch_ok_bad')}，url_count≠行数 {N('mismatch_urlcount_listed')}，"
    f"listed_urls≠行数 {N('mismatch_listed_nrows')}（listed 上限 {N('listed_cap')}，{N('n_listed_at_cap')} 个站打在上限）")

P = m[m["is_partial"]].copy()
PA = P[P["partial_has_auth_like"]].copy()
PC = P[P["partial_clean"]].copy()
PURLS = urls[urls["subdomain"].isin(set(P["subdomain"]))].copy()
rec("n_partial_urlrows", len(PURLS), "{:d}")
rec("n_partial_failrows", int(PURLS["failrow"].sum()), "{:d}")
rec("partial_fail_share", float(PURLS["failrow"].mean()), "{:.6f}")

# ==================================================== 1. 逐站指标（含替代口径失效率）
def site_metrics(sub_df):
    g = PURLS[PURLS["subdomain"].isin(set(sub_df["subdomain"]))]
    per = g.groupby("subdomain").agg(n_rows=("status", "size"),
                                     n_fail=("failrow", "sum"),
                                     n_auth=("auth_like_row", "sum"),
                                     n_nonfail=("nonfail_row", "sum"),
                                     n_zero=("status", lambda s: int((s == 0).sum())),
                                     n_404=("status", lambda s: int(s.isin([404, 410]).sum())),
                                     n_5xx=("status", lambda s: int(((s >= 500) & (s < 600)).sum())))
    out = sub_df.set_index("subdomain").join(per)
    out["f"] = out["bad"] / out["n_rows"]                       # 原口径（=bad/listed_urls）
    out["f_strict"] = (out["n_fail"] - out["n_auth"]) / (out["n_rows"] - out["n_auth"])  # 去掉 -1/-2/403
    out["f_nonfail"] = (out["n_fail"] - out["n_nonfail"]) / (out["n_rows"] - out["n_nonfail"])  # 只去 -1/-2
    return out


SIT = site_metrics(P)
SITA = site_metrics(PA)
SITC = site_metrics(PC)
for tag, d in (("all", SIT), ("auth", SITA), ("clean", SITC)):
    rec(f"mean_f_{tag}", float(d["f"].mean()), "{:.6f}")
    rec("median_f_" + tag, float(d["f"].median()), "{:.6f}")
    rec("mean_f_strict_" + tag, float(d["f_strict"].mean()), "{:.6f}")
    rec("mean_f_nonfail_" + tag, float(d["f_nonfail"].mean()), "{:.6f}")
rec("n_flip_healthy_strict", int((SIT["f_strict"] <= 0).sum()), "{:d}")
rec("n_flip_healthy_nonfail", int((SIT["f_nonfail"] <= 0).sum()), "{:d}")
rec("n_flip_healthy_strict_clean", int((SITC["f_strict"] <= 0).sum()), "{:d}")
rec("share_zero_rows_in_partial", float(SIT["n_zero"].sum() / SIT["n_rows"].sum()), "{:.6f}")
rec("share_auth_rows_in_partial", float(SIT["n_auth"].sum() / SIT["n_rows"].sum()), "{:.6f}")

out_cols = ["subdomain", "title", "url_count", "listed_urls", "n_rows", "ok", "bad", "ok_ratio_pct",
            "status_codes", "f", "n_auth", "n_nonfail", "f_strict", "f_nonfail", "n_zero", "n_404", "n_5xx",
            "partial_has_auth_like", "in_degree", "out_degree", "degree_u", "health_state"]
SIT.reset_index()[out_cols].to_csv(os.path.join(C.module_dir(MOD), "d6_partial_sites.csv"),
                                   index=False, encoding="utf-8-sig")

# ============================================ 2. 失败 URL 解剖：状态码 × 扩展名 × 路径签名
F = PURLS[PURLS["failrow"]].copy()
rec("n_fail_urls_partial", len(F), "{:d}")
rec("mean_nsegs_fail", float(F["nsegs"].mean()), "{:.3f}")
rec("mean_nsegs_ok", float(PURLS.loc[~PURLS["failrow"], "nsegs"].mean()), "{:.3f}")
rec("share_root_fail", float((F["seg"] == "<root>").mean()), "{:.6f}")
rec("share_root_ok", float((PURLS.loc[~PURLS["failrow"], "seg"] == "<root>").mean()), "{:.6f}")

rows = []
for fam, g in F.groupby("fam"):
    exts = g["ext"].value_counts()
    sigs = g["psig"].value_counts()
    rows.append({
        "code_family": fam, "n_urls": len(g), "share_of_failed": len(g) / len(F),
        "n_sites": g["subdomain"].nunique(), "share_of_partial_sites": g["subdomain"].nunique() / len(P),
        "fail_rate_of_that_family_within_site": float(g.groupby("subdomain").size().mean()),
        "top_ext": exts.index[0], "top_ext_share": float(exts.iloc[0] / len(g)),
        "top_path_sig": sigs.index[0], "top_path_sig_share": float(sigs.iloc[0] / len(g)),
        "share_ext_none": float((g["ext"] == "<none>").mean()),
        "share_path_depth_ge2": float((g["nsegs"] >= 2).mean()),
        "is_failure_by_convention": "否（§2.2：-1/-2 不是失效）" if fam in NONFAIL_FAMS else
                                    ("半（403＝服务器应答的访问控制）" if fam == "403 ACL/禁目录" else "是"),
        "example_url": g["url"].iloc[0],
        "title_evidence": "; ".join(f"{k}:{v}" for k, v in g["title"].astype(str).value_counts().head(3).items()),
    })
CODETAB = pd.DataFrame(rows).sort_values("n_urls", ascending=False)
C.save_csv(CODETAB, MOD, "d6_code_anatomy.csv")

# 共现表（只报频数，不作断言）
COX0 = pd.crosstab(F["fam"], F["ext"])
rec("n_coext_cells", int((COX0 > 0).sum().sum()), "{:d}")
COX = COX0.copy()
COX["合计"] = COX.sum(axis=1)
COX = COX.sort_values("合计", ascending=False).drop(columns=["合计"]).join(COX[["合计"]])
COX.reset_index().to_csv(os.path.join(C.module_dir(MOD), "d6_code_ext_cooccurrence.csv"),
                        index=False, encoding="utf-8-sig")
COP = pd.crosstab(F["fam"], F["psig"])
COP["合计"] = COP.sum(axis=1)
COP.reset_index().to_csv(os.path.join(C.module_dir(MOD), "d6_code_pathsig_cooccurrence.csv"),
                        index=False, encoding="utf-8-sig")
rec("n_coext_cells", int((COX > 0).sum().sum() - len(COX)), "{:d}")

# 站点层面的码族共现（同站同时出现两类）
fam_site = pd.crosstab(PURLS["subdomain"], PURLS["fam"]) > 0
co = fam_site.T.dot(fam_site).astype(int)
COSITE = pd.DataFrame({"A": [], "B": [], "both": []})
fams = [c for c in co.columns if c != "OK(2xx)"]
pairs = []
for i1 in range(len(fams)):
    for j1 in range(i1 + 1, len(fams)):
        a, bb = fams[i1], fams[j1]
        pairs.append({"A": a, "B": bb, "n_A": int(fam_site[a].sum()), "n_B": int(fam_site[bb].sum()),
                      "both": int(((fam_site[a]) & (fam_site[bb])).sum()),
                      "jaccard": float(((fam_site[a]) & (fam_site[bb])).sum() /
                                       ((fam_site[a]) | (fam_site[bb])).sum())})
COSITE = pd.DataFrame(pairs).sort_values("both", ascending=False)
COSITE.reset_index(drop=True).to_csv(os.path.join(C.module_dir(MOD), "d6_site_code_cooccurrence.csv"),
                                     index=False, encoding="utf-8-sig")
rec("n_site_pairs", len(COSITE), "{:d}")
rec("top_pair", f"{COSITE.iloc[0]['A']} × {COSITE.iloc[0]['B']}", "{}")
rec("top_pair_both", int(COSITE.iloc[0]["both"]), "{:d}")
say(f"[{time.time()-T0:6.1f}s] 解剖完成：失败 URL {N('n_fail_urls_partial')} 条，码族 {len(CODETAB)} 类，站点共现对 {N('n_site_pairs')}")

# ==================================== 3. 69 / 80 分组各自解剖
SPLIT = []
for tag, d, lab in (("partial_has_auth_like(69)", PA, "含 -1/-2/403 的部分失效站"),
                    ("partial_clean(80)", PC, "干净的部分失效站")):
    sub = PURLS[PURLS["subdomain"].isin(set(d["subdomain"]))]
    fl = sub[sub["failrow"]]
    sel = SIT.index.isin(set(d["subdomain"]))
    SPLIT.append({"group": tag, "label": lab, "n_sites": len(d),
                  "share_of_149": len(d) / len(P),
                  "n_url_rows": len(sub), "n_failed_rows": len(fl),
                  "fail_share_rows": float(sub["failrow"].mean()),
                  "mean_site_f": float(SIT.loc[sel, "f"].mean()),
                  "mean_f_strict": float(SIT.loc[SIT.index.isin(set(d["subdomain"])), "f_strict"].mean()),
                  "n_minus1": int((fl["fam"] == "-1 需认证/本轮跳过").sum()),
                  "n_minus2": int((fl["fam"] == "-2 robots 禁止").sum()),
                  "n_403": int((fl["fam"] == "403 ACL/禁目录").sum()),
                  "n_0": int((fl["fam"] == "0 连接层失败").sum()),
                  "n_0_skip": int(fl["zero_skip"].sum()),
                  "n_0_conn": int(fl["zero_conn"].sum()),
                  "n_404_410": int((fl["fam"] == "404/410 服务器应答缺失").sum()),
                  "n_5xx": int((fl["fam"] == "5xx 服务器错误").sum()),
                  "mean_in_degree": float(d["in_degree"].mean()),
                  "mean_degree_u": float(d["degree_u"].mean()),
                  "mean_url_count": float(d["url_count"].mean()),
                  "mean_listed_urls": float(d["listed_urls"].mean()),
                  "top_ext": fl["ext"].value_counts().index[0],
                  "top_path_sig": fl["psig"].value_counts().index[0] if len(fl) else "—",
                  "n_flip_healthy_strict": int((SIT.loc[SIT.index.isin(set(d["subdomain"])), "f_strict"] <= 0).sum()),
                  })
SPLITDF = pd.DataFrame(SPLIT)
C.save_csv(SPLITDF, MOD, "d6_authlike_vs_clean.csv")
rec("flip69", int(SPLITDF.loc[0, "n_flip_healthy_strict"]), "{:d}")
rec("flip80", int(SPLITDF.loc[1, "n_flip_healthy_strict"]), "{:d}")
rec("mean_f_69", float(SPLITDF.loc[0, "mean_site_f"]), "{:.6f}")
rec("mean_f_80", float(SPLITDF.loc[1, "mean_site_f"]), "{:.6f}")
rec("idg_69", float(SPLITDF.loc[0, "mean_in_degree"]), "{:.4f}")
rec("idg_80", float(SPLITDF.loc[1, "mean_in_degree"]), "{:.4f}")
rec("uc_69", float(SPLITDF.loc[0, "mean_url_count"]), "{:.3f}")
rec("uc_80", float(SPLITDF.loc[1, "mean_url_count"]), "{:.3f}")
rec("idg_healthy", float(m.loc[m["is_healthy"], "in_degree"].mean()), "{:.4f}")
rec("idg_partial", float(P["in_degree"].mean()), "{:.4f}")
rec("idg_dead", float(m.loc[m["is_dead"], "in_degree"].mean()), "{:.4f}")

# ============================ 4. 失效率 × 结构位置（与 D1 同口径：主口径＝入度）
def boot_spearman(x, y, reps=N_BOOT, seed=None):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    n = len(x)
    rho, p = stats.spearmanr(x, y)
    r = np.random.default_rng(seed)
    vals = np.empty(reps)
    for k in range(reps):
        idx = r.integers(0, n, n)          # 每轮独立重抽（不累积）
        if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2:
            vals[k] = np.nan
        else:
            vals[k] = stats.spearmanr(x[idx], y[idx]).statistic
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    zf = np.arctanh(np.clip(rho, -0.999999, 0.999999)) / np.sqrt(1 / (n - 3))
    fz_lo, fz_hi = np.tanh(zf - stats.norm.ppf(0.975) / np.sqrt(n - 3)), np.tanh(zf + stats.norm.ppf(0.975) / np.sqrt(n - 3))
    tau, pt = stats.kendalltau(x, y)
    return {"n": n, "rho": float(rho), "p_rho": float(p), "tau_b": float(tau), "p_tau": float(pt),
            "boot_lo": float(lo), "boot_hi": float(hi), "boot_reps": reps,
            "boot_nans": int(np.isnan(vals).sum()),
            "fisher_lo": float(fz_lo), "fisher_hi": float(fz_hi),
            "spearman_x_exact": float(stats.spearmanr(x, y).statistic)}


CORR = []
EXPS = [("in_degree", "入度（§2.1 失效分析主口径）"), ("degree_u", "无向总度（跨口径影子，仅对照）"),
        ("url_count", "发现的 URL 条数（抓取侧暴露量，被删失）"), ("listed_urls", "已枚举 URL 条数（上限 20）")]
for scope, d in (("全部 149 个部分失效站", SIT),
                 ("含认证类 69 站", SIT.loc[SIT.index.isin(set(PA["subdomain"]))]),
                 ("干净 80 站", SIT.loc[SIT.index.isin(set(PC["subdomain"]))])):
    for col, lab in EXPS:
        res = boot_spearman(d["f"], d[col], seed=SEED + hash(scope) % 1000)
        res2 = boot_spearman(d["f_strict"], d[col], seed=SEED + 7 + hash(scope) % 1000)
        CORR.append({"scope": scope, "outcome": "f=bad/listed（原口径）", "exposure": col, "exposure_label": lab, **res})
        CORR.append({"scope": scope, "outcome": "f_strict（剔除 -1/-2/403）", "exposure": col, "exposure_label": lab, **res2})
CORRDF = pd.DataFrame(CORR)
C.save_csv(CORRDF, MOD, "d6_failure_fraction_correlations.csv")
key_rho = {}
for _, r in CORRDF.iterrows():
    key_rho[(r["scope"], r["outcome"], r["exposure"])] = r


def kr(scope, outcome, exposure, field):
    r = key_rho[(scope, outcome, exposure)]
    return float(r[field])


rec("rho_main", kr("全部 149 个部分失效站", "f=bad/listed（原口径）", "in_degree", "rho"), "{:.4f}")
rec("p_main", kr("全部 149 个部分失效站", "f=bad/listed（原口径）", "in_degree", "p_rho"), "{:.3e}")
rec("ci_main", f"[{kr('全部 149 个部分失效站', 'f=bad/listed（原口径）', 'in_degree', 'boot_lo'):.4f}, "
                f"{kr('全部 149 个部分失效站', 'f=bad/listed（原口径）', 'in_degree', 'boot_hi'):.4f}]", "{}")
rec("rho_uc", kr("全部 149 个部分失效站", "f=bad/listed（原口径）", "url_count", "rho"), "{:.4f}")
rec("ci_uc", f"[{kr('全部 149 个部分失效站', 'f=bad/listed（原口径）', 'url_count', 'boot_lo'):.4f}, "
              f"{kr('全部 149 个部分失效站', 'f=bad/listed（原口径）', 'url_count', 'boot_hi'):.4f}]", "{}")
rec("rho_strict_in", kr("全部 149 个部分失效站", "f_strict（剔除 -1/-2/403）", "in_degree", "rho"), "{:.4f}")
rec("rho_strict_in_p", kr("全部 149 个部分失效站", "f_strict（剔除 -1/-2/403）", "in_degree", "p_rho"), "{:.3e}")
rec("rho_69_in", kr("含认证类 69 站", "f=bad/listed（原口径）", "in_degree", "rho"), "{:.4f}")
rec("rho_80_in", kr("干净 80 站", "f=bad/listed（原口径）", "in_degree", "rho"), "{:.4f}")
rec("rho_69_strict_in", kr("含认证类 69 站", "f_strict（剔除 -1/-2/403）", "in_degree", "rho"), "{:.4f}")
rec("rho_80_strict_in", kr("干净 80 站", "f_strict（剔除 -1/-2/403）", "in_degree", "rho"), "{:.4f}")
rec("rho_80_strict_p_in", kr("干净 80 站", "f_strict（剔除 -1/-2/403）", "in_degree", "p_rho"), "{:.3e}")
rec("rho_du", kr("全部 149 个部分失效站", "f=bad/listed（原口径）", "degree_u", "rho"), "{:.4f}")
rec("rho_listed", kr("全部 149 个部分失效站", "f=bad/listed（原口径）", "listed_urls", "rho"), "{:.4f}")
say(f"[{time.time()-T0:6.1f}s] 相关族完成（{len(CORRDF)} 个设定，{N_BOOT} 轮 bootstrap/设定）")

# 分组间的位置差异（非参数，不作因果表述）
rec("mwu_69_80_in", float(stats.mannwhitneyu(PA["in_degree"], PC["in_degree"], alternative="two-sided")[1]), "{:.3e}")
rec("mwu_69_80_uc", float(stats.mannwhitneyu(PA["url_count"], PC["url_count"], alternative="two-sided")[1]), "{:.3e}")
rec("mwu_69_80_listed", float(stats.mannwhitneyu(PA["listed_urls"], PC["listed_urls"], alternative="two-sided")[1]), "{:.3e}")
# 部分失效 vs 健康：入度对照（同口径）
rec("mwu_partial_healthy_in", float(stats.mannwhitneyu(P["in_degree"], m.loc[m["is_healthy"], "in_degree"],
                                                       alternative="two-sided")[1]), "{:.3e}")

# 顶/底分位的对照（与 D1 的分箱风格一致，但只用秩中位数）
rec("f_q1", float(SIT["f"].quantile(0.25)), "{:.4f}")
rec("f_q3", float(SIT["f"].quantile(0.75)), "{:.4f}")
top = SIT.nlargest(30, "f")
bot = SIT.nsmallest(30, "f")
rec("top30_mean_in", float(top["in_degree"].mean()), "{:.4f}")
rec("bot30_mean_in", float(bot["in_degree"].mean()), "{:.4f}")
rec("top30_mean_uc", float(top["url_count"].mean()), "{:.3f}")
rec("bot30_mean_uc", float(bot["url_count"].mean()), "{:.3f}")

C.save_csv(pd.DataFrame([{"key": k, "value": d["value"], "ref": d["ref"], "note": d["note"]}
                         for k, d in KEY.items()]), MOD, "key_numbers.csv")


def mdtable(df, cols, headers, fmts):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c, fm in zip(cols, fmts):
            v = r[c]
            if isinstance(v, (int, np.integer)):
                cells.append(f"{v:d}")
            elif isinstance(v, (float, np.floating)):
                cells.append("—" if pd.isna(v) else fm.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


CT = CODETAB.copy()
CT["share_of_failed"] = CT["share_of_failed"]
CORR_SHOW = CORRDF[(CORRDF["outcome"] == "f=bad/listed（原口径）") & (CORRDF["exposure"].isin(["in_degree", "url_count"]))]
CORR_STRICT = CORRDF[(CORRDF["outcome"] == "f_strict（剔除 -1/-2/403）") & (CORRDF["exposure"].isin(["in_degree", "url_count"]))]

MD = f"""# D6 · 149 个部分失效站点的解剖 —— 结果

- 口径：主口径 {cite('n_nodes')} 个有属性节点里的部分失效站 {cite('n_partial')}（`ok_ratio_pct`∈(0,100)），锚点 {int(ANCH['ok'].sum())}/{len(ANCH)} PASS（`anchors.csv`）。它们的 URL 明细 {cite('n_partial_urlrows')} 行，其中失败行 {cite('n_partial_failrows')}（占 {cite('partial_fail_share', True)}%）。
- **强制拆分**：含认证类状态码（-1/-2/403）的 {cite('n_auth_like')} 站 vs 干净的 {cite('n_clean_partial')} 站（`csa_common.add_health` 给的 `partial_has_auth_like` / `partial_clean`，两者互斥且并起来正好 {cite('n_partial')}）。
- 两条口径提醒（§2.1/§2.2）：`ok/bad/ok_ratio_pct` 的分母是**已枚举 URL**（`listed_urls`，上限 {N('listed_cap')}），不是全站；`url_count` 是"发现到的条数"，与枚举行数不一致的站有 {cite('mismatch_urlcount_listed')} 个。状态码 -1 / -2 不是失效，403 是服务器应答的访问控制。
- 一致性自检（`key_numbers.csv`）：`ok+bad` 与 URL 行数不符的站 {cite('mismatch_ok_bad')} 个；`listed_urls` 与行数不符 {cite('mismatch_listed_nrows')} 个；打在枚举上限上的站 {cite('n_listed_at_cap')} 个。
- 产物：`d6_partial_sites.csv`（{cite('n_partial')} 行逐站）、`d6_code_anatomy.csv`、`d6_code_ext_cooccurrence.csv`、`d6_code_pathsig_cooccurrence.csv`、`d6_site_code_cooccurrence.csv`、`d6_authlike_vs_clean.csv`、`d6_failure_fraction_correlations.csv`、`key_numbers.csv`、`anchors.csv`。

## 0. 判决

1. **部分失效站的"失败"里有一半不是失效。** {cite('n_fail_urls_partial')} 条失败 URL 中，-1（需认证/本轮跳过）+ -2（robots）+ 403（ACL）共占 §2 表的前三行；按 §2.2 只把 -1/-2 视为"非失效"时，剔除后 {cite('n_flip_healthy_nonfail')} 个站变成"全枚举都成功"；再连 403 一起剔除，{cite('n_flip_healthy_strict')} 个站（{cite('n_auth_like')} 个里 {cite('flip69')} 个、{cite('n_clean_partial')} 个里 {cite('flip80')} 个）从"部分失效"变成健康。**"149"这个数是口径的产物，不是站点的属性。**
2. **状态码 0 内部还是混的**：全部 URL 里 0 有 {cite('n_zero_rows')} 条，标题写明"连接失败-本运行跳过"的 {cite('n_zero_skip')} 条、真连接错误（ConnectionError/Timeout/SSLError）{cite('n_zero_conn')} 条。同一码位承载了"我没抓"和"它不通"两种意义（D7 逐个看）。
3. **失败 URL 的路径签名高度结构化**（描述，不作断言）：失败行里 {cite('share_root_fail', True)}% 落在站点根路径，成功行只有 {cite('share_root_ok', True)}%；失败行平均路径深度 {cite('mean_nsegs_fail')}，成功行 {cite('mean_nsegs_ok')}。共现表见 §3。
4. **失效率与结构位置：主口径（入度）看不到梯度。** ρ(f, 入度) = {cite('rho_main')}（p={cite('p_main')}，bootstrap {N_BOOT} 轮 95% CI {cite('ci_main')}）；换成剔除认证类之后的 f_strict 是 ρ={cite('rho_strict_in')}（p={cite('rho_strict_in_p')}）。对照口径 ρ(f, url_count) = {cite('rho_uc')}（CI {cite('ci_uc')}）——抓取侧暴露量与失效率的关联比结构位置更强，而 url_count 本身受删失，所以这条只能当"共变"读。

## 1. 69 vs 80：必须分开看的两组

""" + mdtable(
    SPLITDF, ["group", "label", "n_sites", "n_url_rows", "n_failed_rows", "fail_share_rows", "mean_site_f",
              "mean_f_strict", "n_minus1", "n_minus2", "n_403", "n_0", "n_0_skip", "n_0_conn", "n_404_410",
              "n_5xx", "mean_in_degree", "mean_url_count", "mean_listed_urls", "top_ext", "top_path_sig",
              "n_flip_healthy_strict"],
    ["组", "含义", "站数", "URL行", "失败行", "失败行占比", "站内失效率均值", "剔除认证类后失效率", "-1", "-2", "403",
     "0", "0=本轮跳过", "0=真连接失败", "404/410", "5xx", "入度均值", "url_count均值", "listed均值", "众数扩展名",
     "众数路径签名", "剔除后变健康站数"],
    ["{}", "{}", "{:d}", "{:d}", "{:d}", "{:.3f}", "{:.3f}", "{:.3f}", "{:d}", "{:d}", "{:d}", "{:d}", "{:d}",
     "{:d}", "{:d}", "{:d}", "{:.2f}", "{:.1f}", "{:.2f}", "{}", "{}", "{:d}"]) + f"""

- 两组的结构位置差别不大：入度均值 {cite('idg_69')} vs {cite('idg_80')}（Mann–Whitney p={cite('mwu_69_80_in')}），但抓取侧差得远：url_count 均值 {cite('uc_69')} vs {cite('uc_80')}（p={cite('mwu_69_80_uc')}）、枚举行数 p={cite('mwu_69_80_listed')}。**"含认证类"这一标签与"这个站能被枚举出多少 URL"共变**，所以两组不能混在一个失效率里报（混了就等于把 {cite('share_auth_rows_in_partial', True)}% 的枚举行当成失败）。
- 参照系：健康站入度均值 {cite('idg_healthy')}、部分失效 {cite('idg_partial')}、整站不可用 {cite('idg_dead')}（同口径＝入度；总度是跨口径，见 §2.1）。部分失效 vs 健康的入度分布差异 p={cite('mwu_partial_healthy_in')}。

## 2. 失败 URL 的解剖（`d6_code_anatomy.csv`）

""" + mdtable(
    CT, ["code_family", "n_urls", "share_of_failed", "n_sites", "share_of_partial_sites", "top_ext",
         "top_ext_share", "top_path_sig", "top_path_sig_share", "share_ext_none", "is_failure_by_convention",
         "title_evidence"],
    ["状态码族", "条数", "占失败", "涉及站数", "占 149 站", "众数扩展名", "占比", "众数路径签名", "占比",
     "无扩展名占比", "按 §2.2 算失效吗", "标题证据串"],
    ["{}", "{:d}", "{:.3f}", "{:d}", "{:.3f}", "{}", "{:.3f}", "{}", "{:.3f}", "{:.3f}", "{}", "{}"]) + f"""

- 表里 `涉及站数` 会大于失败行数之和的分母意义：一个站可同时贡献多类码，所以这是"哪些码族在哪些站同时出现"的描述，不是分解。
- `is_failure_by_convention` 列把 §2.2 的判据逐行标出：{cite('n_fail_urls_partial')} 条失败 URL 里，被判"否/半"的那几族（-1、-2、403）合计 {cite('share_auth_rows_in_partial', True)}% 的枚举行。

## 3. 共现（只报频数，不作断言）

- 码族 × 扩展名：`d6_code_ext_cooccurrence.csv`（非零单元 {cite('n_coext_cells')} 个）。
- 码族 × 路径签名：`d6_code_pathsig_cooccurrence.csv`。
- 站点层面"同站共现"的码族对：`d6_site_code_cooccurrence.csv`，{cite('n_site_pairs')} 对里共现最多的是 {cite('top_pair')}（{cite('top_pair_both')} 个站同时出现）。
- 判读边界：共现表说明"什么和什么一起出现"，不能说明"为什么一起出现"——三类通道都无法从这份单次横截面里排除：爬虫的枚举顺序、认证网关（CAS/VPN）的覆盖范围、服务器端重构。

## 4. 失效率 × 结构位置（`d6_failure_fraction_correlations.csv`，Spearman + {N_BOOT} 轮 bootstrap CI）

""" + mdtable(
    pd.concat([CORR_SHOW, CORR_STRICT]),
    ["scope", "outcome", "exposure", "n", "rho", "p_rho", "tau_b", "boot_lo", "boot_hi", "fisher_lo", "fisher_hi"],
    ["样本", "失效率口径", "结构/暴露量", "n", "ρ", "p", "τ-b", "bootstrap CI下", "上", "Fisher CI下", "上"],
    ["{}", "{}", "{}", "{:d}", "{:.4f}", "{:.3e}", "{:.4f}", "{:.4f}", "{:.4f}", "{:.4f}", "{:.4f}"]) + f"""

- 分位对照（避免把 ρ 当唯一证据）：失效率最高的 30 站入度均值 {cite('top30_mean_in')}、最低 30 站 {cite('bot30_mean_in')}；url_count 均值 {cite('top30_mean_uc')} vs {cite('bot30_mean_uc')}。f 的四分位 {cite('f_q1')} / {cite('f_q3')}。
- 口径纪律（§2.1）：主口径是**入度**；表里 `degree_u` 的行只作跨口径影子（ρ(f, degree_u)={cite('rho_du')}），`listed_urls` 那行（ρ={cite('rho_listed')}）量的是枚举深度而非网络位置。
- 69/80 分组内的 ρ：原口径 {cite('rho_69_in')} vs {cite('rho_80_in')}；剔除认证类后 {cite('rho_69_strict_in')} vs {cite('rho_80_strict_in')}（后者 p={cite('rho_80_strict_p_in')}）——**样本砍到 80 之后，任何显著性都来自小样本噪声，不足以支撑"位置决定部分失效"这种写法**。

## 5. 可辩护的表述

1. "部分失效"={cite('n_partial')} 个站是在"已枚举 URL 里存在非 2xx 行"这一判据下的产物；其中 {cite('n_auth_like')} 个站的失败行里含 -1/-2/403（按 §2.2 不算失效），把这几族剔除后 {cite('n_flip_healthy_strict')} 个站与"部分失效"的标签不一致。**报告 149 时必须同时报这两个数。**
2. 部分失效站的失败 URL 有明显的**路径形状**：根路径占比 {cite('share_root_fail', True)}（成功行 {cite('share_root_ok', True)}）、平均深度 {cite('mean_nsegs_fail')}（成功行 {cite('mean_nsegs_ok')}）。这与"枚举到第几层"的抓取过程共同变化，本文不区分二者。
3. 失效率与网络位置的关联**在主口径（入度）上不存在**（ρ={cite('rho_main')}，CI {cite('ci_main')}），与抓取侧暴露量的关联更强但后者受删失污染（§2.1、D2、D5）。可写的句子是"部分失效的深度与站点可枚举性共变，与入度不共变"。
4. 与 D4/D5 接口：D4 说边层面的死—死相邻不可识别；D5 说分组检验里 CMS/类别族有两条伪迹通道；本单元给出同一现象在**站内部**的形态——同一站内哪些 URL 会失败，主要由这条 URL 在第几层、要不要认证决定，而不是由这个站在网络里的位置决定。
"""
with open(os.path.join(C.module_dir(MOD), "result.md"), "w", encoding="utf-8") as fh:
    fh.write(MD)

wall = time.time() - T0
C.write_run_note(MOD, [
    "D6_partial_anatomy RUN.txt",
    "command: cd pipeline && PYTHONHASHSEED=0 python out/D6_partial_anatomy/run_d6.py",
    "versions: " + ", ".join(f"{k}={v}" for k, v in C.versions().items()),
    f"SEED = {SEED}; PYTHONHASHSEED = {os.environ.get('PYTHONHASHSEED')}",
    f"total wall time: {wall:.1f}s",
    f"anchors: {int(ANCH['ok'].sum())}/{len(ANCH)} PASS (anchors.csv)",
    f"sample: 部分失效 {N('n_partial')} 站（{N('n_auth_like')} 含 -1/-2/403 / {N('n_clean_partial')} 干净），"
    f"URL 明细 {N('n_partial_urlrows')} 行，失败 {N('n_partial_failrows')} 行",
    f"bootstrap: Spearman {N_BOOT} 轮/设定，独立 r.integers 重抽（每轮新样本，不累积），"
    f"seed = SEED + 偏移（每组设定不同偏移）；同时给 Fisher-z 与 tau-b 对照",
    "definitions: f = bad/listed_urls（原口径，来自 nodes_full 的 ok/bad）；"
    "f_nonfail = 剔除 -1/-2 行；f_strict = 剔除 -1/-2/403 行；OK = status in {200,201}",
    "口径自检: ok+bad≠urls 行数 的站数=" + str(N("mismatch_ok_bad")) + "; url_count≠行数=" + str(N("mismatch_urlcount_listed"))
    + "; listed_urls≠行数=" + str(N("mismatch_listed_nrows")) + "; listed_urls 上限=" + str(int(N("listed_cap")))
    + "; 打在上限的站=" + str(N("n_listed_at_cap")),
    "status semantics (CONVENTIONS §2.2 / DATA_DICTIONARY): 0=连接层失败(标题里混有'本运行跳过'与 ConnectionError/Timeout/SSLError)，"
    "-1=需认证或本轮跳过(非失效)，-2=robots 禁止(非失效)，403=服务器应答的 ACL，404/410=服务器应答缺失，5xx=服务器错误",
    "no causal wording (§7); co-occurrence tables are descriptive only",
    "outputs: d6_partial_sites.csv d6_code_anatomy.csv d6_code_ext_cooccurrence.csv "
    "d6_code_pathsig_cooccurrence.csv d6_site_code_cooccurrence.csv d6_authlike_vs_clean.csv "
    "d6_failure_fraction_correlations.csv key_numbers.csv anchors.csv result.md",
    "",
    "--- stdout ---",
] + LOG)
say(f"[{wall:.1f}s] 写出 result.md + 8 张 CSV + RUN.txt")
