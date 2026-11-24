# -*- coding: utf-8 -*-
"""F1_mantel_final — 三种 URL 签名 × 三种邻近矩阵的 Mantel 检验（终版）.

口径来源：pipeline/CONVENTIONS.md §4（零模型/置换规范）、§7（措辞）、§9（产出）。
建图只用 csa_common.build_all()，本脚本不重建图。

签名（全部由 URL 结构构造，无文本挖掘、无分词库）：
  S1_firstseg_tokens  首个路径段 token 的 0/1 集合 → Jaccard（主）/ Cosine（副）
  S2_extension_mix    扩展名配比向量 → Cosine（主）/ 中心化后的 Pearson（副）
  S3_path_fingerprint 数字归一后的完整路径模板集合 → Jaccard（主）/ 归一化 Hamming（副）
邻近矩阵：adj_binary、adj_weighted（无向，权重=两方向之和，与 Gu 一致）、spd_hops（仅 GCC 内有限）。

产出（全部写在本目录）：
  mantel_grid.csv        主结果表 r / null mean / sd / z / p / bootstrap CI
  bootstrap_vs_perm.csv  置换 sd 与自助 sd 的对比（Mantel 推断有效性的关键）
  informative_pairs.csv  可见点对占比、r 的理论上界（ceiling）
  structure_visibility.csv 稀疏性/删失/枢纽集中度对"检验能看见什么"的约束
  signature_audit.csv    相似度矩阵自检（NaN/对称/对角/取值集中度/退化程度）
  signature_features.csv 每站签名特征
  cms_marker_audit.csv   常见 CMS 路径标记在本数据里的命中量
  sim_matrices.npz       矩阵本体，供 F2/F3 复用（保证三单元同一签名）
  key_numbers.csv        关键数字 + 产生它的源码行号
"""
from __future__ import annotations

import os
import re
import sys
import time
import json
import urllib.parse as up

import numpy as np
import pandas as pd
import networkx as nx

SEED = 20261113
HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PIPE)
import csa_common as C  # noqa: E402

N_PERM = int(os.environ.get("F1_N_PERM", "1999"))   # §4 要求 >=999，默认 1999（F1_N_PERM 只用于冒烟测试）
N_BOOT = int(os.environ.get("F1_N_BOOT", "1000"))
VOCAB_MIN_SITES = 3     # 词表规则：token/模板至少出现在 3 个站点
T0 = time.time()
LOG: list = []
KEYS: dict = {}


def log(msg):
    LOG.append(f"[{time.time() - T0:7.1f}s] {msg}")
    print(LOG[-1], flush=True)


def line_map():
    """源码里的 `# MARK: name` → 行号，供 result.md 引用（CONVENTIONS §9）。"""
    src = open(os.path.abspath(__file__), encoding="utf-8").read().splitlines()
    out = {}
    for i, ln in enumerate(src, 1):
        m = re.search(r"#\s*MARK:\s*([A-Za-z0-9_]+)", ln)
        if m:
            out[m.group(1)] = i
    return out


LM = line_map()
KEYS["line_map"] = json.dumps(LM)


def key(name, value):
    KEYS[name] = value


# ------------------------------------------------------------------ 0. 地基
b = C.build_all()
C.assert_anchors(b)  # MARK: anchors
log("anchors 31/31 PASS; Gu=%d nodes / %d edges" % (b["Gu"].number_of_nodes(), b["Gu"].number_of_edges()))

urls = C.load_urls()  # MARK: load_urls
key("n_url_rows", len(urls))

NODES = sorted(b["nodes"]["node"].tolist())  # MARK: node_order
IDX = {n: i for i, n in enumerate(NODES)}
N = len(NODES)
key("n_nodes", N)
nd = b["nodes"].set_index("node").loc[NODES]
dead = set(nd.index[nd["is_dead"]])  # MARK: dead_alive_sets
alive = set(NODES) - dead
key("n_dead", len(dead))
key("n_alive", len(alive))
elig = sorted(nd.index[nd["listed_urls"] >= 10])  # MARK: eligible
key("n_eligible_listed_ge10", len(elig))
key("n_sites_hit_20url_cap", int((nd["listed_urls"] == 20).sum()))

URLS_BY = {s: g["url"].astype(str).tolist() for s, g in urls.groupby("subdomain")}  # MARK: urls_by

# ------------------------------------------------------------------ 1. 签名特征
EXT_MAP = {"htm", "html", "php", "asp", "aspx", "jsp", "jspx", "psp", "shtml", "cfm",
           "do", "action", "ashx", "ascx", "xml", "json"}  # MARK: ext_map
EXT_LIST = sorted(EXT_MAP) + ["other", "none"]
CMS_MARKERS = ["/plus/", "/e/", "/d/", "/static/", "/templets/", "/dede/", "/data/",
               "/admin/", "/wp-content/", "/wp-json/", "/sites/all/", "/_next/",
               "/assets/", "/index.php", "/index.jsp", "/list.htm", "/list.jsp",
               "/info/", "/content/", "/website/", "/moreinfo", "/shownews",
               "/module/", "/_web/cms/", "/gbs/"]  # MARK: cms_markers
NUM = re.compile(r"\d+")


def parse_url(u):
    """URL → 结构特征（小写、数字串→#、查询串丢弃、端口不进签名）。"""
    sp = up.urlsplit(u.lower())
    path = sp.path or "/"
    segs = [NUM.sub("#", s) for s in path.split("/") if s]
    first = segs[0] if segs else ""
    last = segs[-1] if segs else ""
    ext = "none"
    if "." in last:
        tail = last.rsplit(".", 1)[1]
        if tail in EXT_MAP:
            ext = tail
            last = last.rsplit(".", 1)[0] + "." + tail
        elif re.fullmatch(r"[a-z0-9]{1,5}", tail) and "@" not in last:
            ext = "other"
    tmpl = "/" + "/".join(segs) if segs else "/"
    return {"depth": len(segs), "first": first, "ext": ext, "tmpl": tmpl,
            "has_num": bool(NUM.search(last)), "has_query": bool(sp.query)}


FEAT = {}
ALL_PATHS = []
for n in NODES:  # MARK: feat_built
    rows = [parse_url(u) for u in URLS_BY.get(n, [])]
    extc = {e: 0 for e in EXT_LIST}
    for r in rows:
        extc[r["ext"]] += 1
    tot = max(1, len(rows))
    FEAT[n] = {"n": len(rows),
               "tokens": {r["first"] for r in rows if r["first"]},
               "templates": {r["tmpl"] for r in rows},
               "ext_mix": {k: v / tot for k, v in extc.items()},
               "depths": [r["depth"] for r in rows],
               "has_num": float(np.mean([r["has_num"] for r in rows])) if rows else 0.0,
               "has_query": float(np.mean([r["has_query"] for r in rows])) if rows else 0.0}
    ALL_PATHS.extend([r["tmpl"] for r in rows])
key("n_root_only_sites", int(sum(1 for n in NODES if not FEAT[n]["tokens"])))

cms_hits = pd.Series({m: int(sum(p.count(m) > 0 for p in ALL_PATHS)) for m in CMS_MARKERS})  # MARK: cms_hits
key("cms_marker_max_hits", int(cms_hits.max()))


def build_vocab(items, min_sites=VOCAB_MIN_SITES):
    cnt = {}
    for n in NODES:
        for it in items[n]:
            cnt[it] = cnt.get(it, 0) + 1
    return sorted([k for k, v in cnt.items() if v >= min_sites]), cnt


tok_vocab, tok_cnt = build_vocab({n: FEAT[n]["tokens"] for n in NODES})  # MARK: tok_vocab
tp_vocab, tp_cnt = build_vocab({n: FEAT[n]["templates"] for n in NODES})  # MARK: tp_vocab
key("n_token_vocab_raw", len(tok_cnt))
key("n_token_vocab_kept", len(tok_vocab))
key("n_tmpl_vocab_raw", len(tp_cnt))
key("n_tmpl_vocab_kept", len(tp_vocab))


def bin_matrix(items, vocab):
    col = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((N, len(vocab)), dtype=np.float32)
    for i, n in enumerate(NODES):
        for it in items[n]:
            j = col.get(it)
            if j is not None:
                X[i, j] = 1.0
    return X


X_tok = bin_matrix({n: FEAT[n]["tokens"] for n in NODES}, tok_vocab)  # MARK: X_tok
X_tmpl = bin_matrix({n: FEAT[n]["templates"] for n in NODES}, tp_vocab)  # MARK: X_tmpl
E_mix = np.array([[FEAT[n]["ext_mix"][e] for e in EXT_LIST] for n in NODES])  # MARK: E_mix


def sim_jaccard(X):
    inter = (X @ X.T).astype(np.float64)
    s = X.sum(1).astype(np.float64)
    den = s[:, None] + s[None, :] - inter
    return np.where(den > 0, inter / np.where(den > 0, den, 1.0), 0.0)


def sim_cosine(X):
    Xd = X.astype(np.float64)
    nr = np.sqrt((Xd * Xd).sum(1))
    den = np.outer(nr, nr)
    return np.where(den > 0, (Xd @ Xd.T) / np.where(den > 0, den, 1.0), 0.0)


def sim_hamming(X):
    V = float(X.shape[1])
    inter = (X @ X.T).astype(np.float64)
    s = X.sum(1).astype(np.float64)
    return (V - s[:, None] - s[None, :] + inter) / V


def sim_centered_pearson(X):
    Xc = X - X.mean(0, keepdims=True)
    nr = np.sqrt((Xc * Xc).sum(1))
    nr[nr == 0] = np.inf
    return (Xc @ Xc.T) / np.outer(nr, nr)


SIMS = {"S1_tok_jaccard": sim_jaccard(X_tok),        # MARK: sims
        "S1_tok_cosine": sim_cosine(X_tok),
        "S2_ext_cosine": sim_cosine(E_mix),
        "S2_ext_centered": sim_centered_pearson(E_mix),
        "S3_tmpl_jaccard": sim_jaccard(X_tmpl),
        "S3_tmpl_hamming": sim_hamming(X_tmpl)}
PRIMARY = {"S1_firstseg_tokens": "S1_tok_jaccard",
           "S2_extension_mix": "S2_ext_cosine",
           "S3_path_fingerprint": "S3_tmpl_jaccard"}
SECONDARY = {"S1_firstseg_tokens": "S1_tok_cosine",
             "S2_extension_mix": "S2_ext_centered",
             "S3_path_fingerprint": "S3_tmpl_hamming"}

# ------------------------------------------------------------------ 2. 邻近矩阵
A_bin = np.zeros((N, N))  # MARK: adjacency
A_w = np.zeros((N, N))
for u, v, d in b["Gu"].edges(data=True):
    if u in IDX and v in IDX:
        i, j = IDX[u], IDX[v]
        A_bin[i, j] = A_bin[j, i] = 1.0
        A_w[i, j] = A_w[j, i] = float(d.get("weight", 1))
key("n_edges_gu", b["Gu"].number_of_edges())
key("n_edges_within_attributed", int(A_bin.sum() / 2))
hubn = C.hub_nodes(b["Gu"])  # MARK: hub_nodes

GCC = sorted(b["sets"]["gcc"] & set(NODES))  # MARK: gcc_nodes
key("n_gcc_attributed", len(GCC))
Gsub = b["Gb"].subgraph(GCC)
gidx = {n: i for i, n in enumerate(GCC)}
Ng = len(GCC)
D_spd = np.full((Ng, Ng), np.nan)  # MARK: spd_build
for s_i, s in enumerate(GCC):
    for t, dd in nx.single_source_shortest_path_length(Gsub, s).items():
        D_spd[s_i, gidx[t]] = float(dd)
key("n_pairs_no_path_in_gcc", int((np.isnan(D_spd).sum() - Ng) / 2))
np.fill_diagonal(D_spd, np.nan)

P = {"adj_binary": A_bin, "adj_weighted": A_w}  # MARK: proximity
iu_r, iu_c = np.triu_indices(N, 1)  # MARK: pair_index


# ------------------------------------------------------------------ 3. Mantel 机器
def pear(a, bv):
    a = a - a.mean()
    bv = bv - bv.mean()
    da, db = float(a @ a), float(bv @ bv)
    if da == 0 or db == 0:
        return np.nan
    return float(a @ bv / np.sqrt(da * db))


def make_mask(Pmat, rows, cols):
    """只用有限元（spd 在分量外为 NaN）。"""
    v = Pmat[rows, cols]
    return np.isfinite(v)


def mantel(Smat, Pmat, n, rows, cols, mask, n_perm=N_PERM, seed=SEED):
    """行/列同时重标号的置换检验；每轮独立 fresh shuffle（§4）。
    双侧 p = mean(|null - mean(null)| >= |obs - mean(null)|)（§4 规定口径）。"""
    rm, cm = rows[mask], cols[mask]  # MARK: mantel_masked_idx
    x = Smat[rm, cm]
    y = Pmat[rm, cm]
    obs = pear(x, y)
    rng = np.random.default_rng(seed)
    yc = y - y.mean()
    dby = float(yc @ yc)
    nulls = np.empty(n_perm)
    CH = 32  # MARK: mantel_chunk
    for st in range(0, n_perm, CH):
        en = min(st + CH, n_perm)
        perm = np.stack([rng.permutation(n) for _ in range(en - st)])
        XX = Smat[perm[:, rm], perm[:, cm]]
        Xc = XX - XX.mean(axis=1, keepdims=True)
        nulls[st:en] = (Xc @ yc) / np.sqrt((Xc * Xc).sum(1) * dby)
    nm, ns = float(nulls.mean()), float(nulls.std(ddof=1))
    z = (obs - nm) / ns if ns > 0 else np.nan
    p = float((np.abs(nulls - nm) >= abs(obs - nm)).mean())  # MARK: p_two_sided
    return obs, nm, ns, z, p, float(nulls.min()), float(nulls.max())


def boot_ci_r(Smat, Pmat, n, rows, cols, mask, B=N_BOOT, seed=SEED + 17):
    """节点重抽样自助：一次抽样同时决定 S 与 P 的点对，保留点对间的依赖结构。"""
    rng = np.random.default_rng(seed)
    rm, cm = rows[mask], cols[mask]
    x0, y0 = Smat[rm, cm], Pmat[rm, cm]
    vals = []
    CH = 32
    for st in range(0, B, CH):
        en = min(st + CH, B)
        BS = np.stack([rng.integers(0, n, size=n) for _ in range(en - st)])
        RR, CC = BS[:, rm], BS[:, cm]
        keep = RR != CC
        XX = np.where(keep, Smat[RR, CC], np.nan)
        YY = np.where(keep, Pmat[RR, CC], np.nan)
        for k in range(XX.shape[0]):
            fin = np.isfinite(XX[k]) & np.isfinite(YY[k])
            rr = pear(XX[k][fin], YY[k][fin])
            if np.isfinite(rr):
                vals.append(rr)
    v = np.array(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), float(v.std(ddof=1)), len(v)


def ceiling_r(Smat, Pmat, rows, cols, mask):
    """重排不等式上界：把 P 的取值按降序摆在相似度最高的点对上（r_max），
    升序摆放给 r_min。这是该 (签名, 邻近口径, 节点集) 下 Mantel r **能达到**的范围；
    观测 r 与上界之比才是"这个设计能表达多少效应"的度量。"""
    x = Smat[rows, cols][mask]
    y = Pmat[rows, cols][mask]
    rank = np.argsort(-x, kind="stable")  # MARK: ceiling
    ybest = np.empty(len(y))
    ybest[rank] = np.sort(y)[::-1]
    yworst = np.empty(len(y))
    yworst[rank] = np.sort(y)
    return pear(x, ybest), pear(x, yworst), float(np.median(x[rank][:int((y > 0).sum()) or 1]))


# ------------------------------------------------------------------ 4. 主网格
rows_main, rows_boot = [], []
full_spd = np.full((N, N), np.nan)  # MARK: spd_to_full
gi = np.array([IDX[n_] for n_ in GCC])
full_spd[np.ix_(gi, gi)] = D_spd
P["spd_hops"] = full_spd
PROX_ORDER = ["adj_binary", "adj_weighted", "spd_hops"]

for sig_name in PRIMARY:  # MARK: main_grid
    for flavor in ("primary", "secondary"):
        sname = PRIMARY[sig_name] if flavor == "primary" else SECONDARY[sig_name]
        Smat = SIMS[sname]
        for pname in PROX_ORDER:
            Pmat = P[pname]
            mask = make_mask(Pmat, iu_r, iu_c)
            pv = Pmat[iu_r, iu_c][mask]
            k = int((pv > 0).sum())
            obs, nm, ns, z, p, nmin, nmax = mantel(Smat, Pmat, N, iu_r, iu_c, mask)
            lo, hi, se, nb = boot_ci_r(Smat, Pmat, N, iu_r, iu_c, mask)
            r_max, r_min, med_top = ceiling_r(Smat, Pmat, iu_r, iu_c, mask)
            absceil = max(abs(r_max), abs(r_min))
            rows_main.append({
                "signature": sig_name, "similarity": sname, "flavor": flavor,
                "proximity": pname, "node_set": "attributed671", "n_pairs_used": int(mask.sum()),
                "k_nonzero_proximity": k, "mantel_r": obs, "null_mean": nm, "null_sd": ns,
                "null_min": nmin, "null_max": nmax, "z": z, "p_two_sided": p, "n_perm": N_PERM,
                "boot_ci_lo": lo, "boot_ci_hi": hi, "boot_se": se, "n_boot": nb,
                "r_ceiling_pos": r_max, "r_ceiling_neg": r_min,
                "r_over_ceiling": obs / absceil if absceil else np.nan,
                "mean_sim_on_nonzero_prox": float(Smat[iu_r, iu_c][mask][pv > 0].mean()) if k else np.nan,
                "mean_sim_all_pairs": float(Smat[iu_r, iu_c][mask].mean()),
            })
            rows_boot.append({"signature": sig_name, "similarity": sname, "proximity": pname,
                              "perm_null_sd": ns, "boot_se": se,
                              "ratio_boot_over_perm": se / ns if ns else np.nan,
                              "boot_ci_lo": lo, "boot_ci_hi": hi})
            log("Mantel %-18s %-9s %-12s r=%+.5f p=%.4f nullsd=%.5f bootse=%.5f ceil=%+.4f/%+.4f"
                % (sig_name, sname, pname, obs, p, ns, se, r_max, r_min))

# 敏感性：只用 listed_urls>=10 的站点
eidx = np.array([IDX[n] for n in elig])  # MARK: elig_setup
er, ec = np.triu_indices(len(elig), 1)
key("n_edges_within_eligible", int(A_bin[np.ix_(eidx, eidx)].sum() / 2))
for sig_name in PRIMARY:  # MARK: elig_grid
    Smat = SIMS[PRIMARY[sig_name]][np.ix_(eidx, eidx)]
    for pname in ("adj_binary", "adj_weighted"):
        Pmat = P[pname][np.ix_(eidx, eidx)]
        mask = make_mask(Pmat, er, ec)
        pv = Pmat[er, ec][mask]
        k = int((pv > 0).sum())
        obs, nm, ns, z, p, nmin, nmax = mantel(Smat, Pmat, len(elig), er, ec, mask, seed=SEED + 5)
        lo, hi, se, nb = boot_ci_r(Smat, Pmat, len(elig), er, ec, mask, seed=SEED + 23)
        r_max, r_min, _ = ceiling_r(Smat, Pmat, er, ec, mask)
        absceil = max(abs(r_max), abs(r_min))
        rows_main.append({"signature": sig_name, "similarity": PRIMARY[sig_name], "flavor": "primary",
                          "proximity": pname, "node_set": f"eligible{len(elig)}",
                          "n_pairs_used": int(mask.sum()), "k_nonzero_proximity": k,
                          "mantel_r": obs, "null_mean": nm, "null_sd": ns, "null_min": nmin,
                          "null_max": nmax, "z": z, "p_two_sided": p, "n_perm": N_PERM,
                          "boot_ci_lo": lo, "boot_ci_hi": hi, "boot_se": se, "n_boot": nb,
                          "r_ceiling_pos": r_max, "r_ceiling_neg": r_min,
                          "r_over_ceiling": obs / absceil if absceil else np.nan,
                          "mean_sim_on_nonzero_prox": float(Smat[er, ec][mask][pv > 0].mean()) if k else np.nan,
                          "mean_sim_all_pairs": float(Smat[er, ec][mask].mean())})
        log("[elig%d] Mantel %-18s %-12s r=%+.5f p=%.4f bootse=%.5f ceil=%+.4f/%+.4f"
            % (len(elig), sig_name, pname, obs, p, se, r_max, r_min))

dfm = pd.DataFrame(rows_main)
C.save_csv(dfm, "F1_mantel_final", "mantel_grid.csv")
C.save_csv(pd.DataFrame(rows_boot), "F1_mantel_final", "bootstrap_vs_perm.csv")

# ------------------------------------------------------------------ 4b. 诊断：效应是谁造的
Gu_nohub = C.strip_hub_edges(b["Gu"])  # MARK: diag_grid
A_bin_nh = np.zeros((N, N))
A_w_nh = np.zeros((N, N))
for u_, v_, d_ in Gu_nohub.edges(data=True):
    if u_ in IDX and v_ in IDX:
        i_, j_ = IDX[u_], IDX[v_]
        A_bin_nh[i_, j_] = A_bin_nh[j_, i_] = 1.0
        A_w_nh[i_, j_] = A_w_nh[j_, i_] = float(d_.get("weight", 1))
key("n_edges_after_hub_strip", int(A_bin_nh[iu_r, iu_c].sum() / 2))

alive_mask_nodes = np.array([n in alive for n in NODES])  # 双端都活着 → 两个方向的边都可能被观测
pair_alive = alive_mask_nodes[iu_r] & alive_mask_nodes[iu_c]
big_nodes = np.array([n in set(elig) for n in NODES])      # listed_urls>=10
pair_big = big_nodes[iu_r] & big_nodes[iu_c]
pair_small = (~big_nodes[iu_r]) & (~big_nodes[iu_c])
pair_mixed = pair_big ^ pair_small
ONES = np.ones(len(iu_r), bool)
DIAGS = [  # (标签, 邻近矩阵, 点对子集, 参照口径)
    ("adj_binary_nohub_edges", A_bin_nh, ONES, "adj_binary"),
    ("adj_weighted_nohub_edges", A_w_nh, ONES, "adj_weighted"),
    ("adj_binary_alive_pairs_only", A_bin, pair_alive, "adj_binary"),
    ("adj_weighted_alive_pairs_only", A_w, pair_alive, "adj_weighted"),
    ("adj_binary_pairs_both_lt10urls", A_bin, pair_small, "adj_binary"),
    ("adj_binary_pairs_both_ge10urls", A_bin, pair_big, "adj_binary"),
    ("adj_binary_pairs_mixed_depth", A_bin, pair_mixed, "adj_binary"),
]
diag_rows = []
for sig_name in PRIMARY:
    Smat = SIMS[PRIMARY[sig_name]]
    for dname, Pmat, pmask, refname in DIAGS:
        base_mask = make_mask(Pmat, iu_r, iu_c)
        mask = base_mask & pmask
        obs, nm, ns, z, p, _, _ = mantel(Smat, Pmat, N, iu_r, iu_c, mask)
        lo, hi, se, _ = boot_ci_r(Smat, Pmat, N, iu_r, iu_c, mask)
        ref = dfm[(dfm.signature == sig_name) & (dfm.proximity == refname)
                  & (dfm.node_set == "attributed671")]["mantel_r"].iloc[0]
        r_max, r_min, _ = ceiling_r(Smat, Pmat, iu_r, iu_c, mask)
        diag_rows.append({"signature": sig_name, "similarity": PRIMARY[sig_name], "diagnostic": dname,
                          "n_pairs_used": int(mask.sum()),
                          "k_nonzero_proximity": int((Pmat[iu_r, iu_c][mask] > 0).sum()),
                          "mantel_r": obs, "null_sd": ns, "z": z, "p_two_sided": p,
                          "boot_ci_lo": lo, "boot_ci_hi": hi, "boot_se": se,
                          "mantel_r_reference_full": ref, "delta_vs_full": obs - ref,
                          "r_ceiling_pos": r_max, "r_over_ceiling": obs / max(abs(r_max), abs(r_min))})
        log("DIAG %-18s %-28s pairs=%d k=%d r=%+.5f (full=%+.5f) p=%.4f ceil=%.4f"
            % (sig_name, dname, int(mask.sum()), int((Pmat[iu_r, iu_c][mask] > 0).sum()),
               obs, ref, p, r_max))

# 协方差贡献分解：r 的分子是 sum (x-x̄)(y-ȳ)，看它对少数点对/少数节点的依赖度
contrib_rows = []
xbar_y = {}  # MARK: contrib
for sig_name in PRIMARY:
    Smat = SIMS[PRIMARY[sig_name]]
    x = Smat[iu_r, iu_c]
    y = A_bin[iu_r, iu_c]
    xc, yc_ = x - x.mean(), y - y.mean()
    prod = xc * yc_
    tot = prod.sum()
    per_node = np.zeros(N)
    np.add.at(per_node, iu_r, prod)
    np.add.at(per_node, iu_c, prod)
    o = np.argsort(-np.abs(prod), kind="stable")
    hub_arr = np.array([n in hubn for n in NODES])
    pair_hub = hub_arr[iu_r] | hub_arr[iu_c]
    contrib_rows.append({"signature": sig_name, "total_cov_numer": float(tot),
                         "share_top1pct_pairs": float(prod[o[: int(0.01 * len(prod))]].sum() / tot),
                         "share_top0p1pct_pairs": float(prod[o[: int(0.001 * len(prod))]].sum() / tot),
                         "share_pairs_touching_hub": float(prod[pair_hub].sum() / tot),
                         "share_pairs_alive_alive": float(prod[pair_alive].sum() / tot),
                         "n_pairs_positive_contrib": int((prod > 0).sum()),
                         "n_pairs_negative_contrib": int((prod < 0).sum()),
                         "top10_node_share": float(np.sort(np.abs(per_node))[::-1][:10].sum() / tot),
                         "top_node": NODES[int(np.argmax(per_node))],
                         "top_node_share": float(per_node.max() / tot)})
C.save_csv(pd.DataFrame(contrib_rows), "F1_mantel_final", "covariance_decomposition.csv")
C.save_csv(pd.DataFrame(diag_rows), "F1_mantel_final", "mantel_diagnostics.csv")

# ------------------------------------------------------------------ 5. 可见点对 / 结构约束
cov = []  # MARK: coverage_table
ones = np.ones(len(iu_r), bool)
for sig_name in PRIMARY:
    Smat = SIMS[PRIMARY[sig_name]]
    x = Smat[iu_r, iu_c]
    k = int(A_bin[iu_r, iu_c].sum())
    r_max, r_min, med_top = ceiling_r(Smat, A_bin, iu_r, iu_c, ones)
    cov.append({"signature": sig_name, "node_set": "attributed671", "n_pairs": len(x),
                "sim_mean": float(x.mean()), "sim_sd": float(x.std(ddof=1)),
                "sim_median": float(np.median(x)), "sim_p90": float(np.percentile(x, 90)),
                "sim_p99": float(np.percentile(x, 99)),
                "frac_pairs_zero_sim": float((x == 0).mean()),
                "n_distinct_sim": int(len(np.unique(np.round(x, 12)))),
                "k_edges_binary": k, "frac_pairs_nonzero_proximity": k / len(x),
                "r_ceiling_binary_pos": r_max, "r_ceiling_binary_neg": r_min,
                "median_sim_of_top_k_pairs": med_top,
                "expected_sim_on_edges_if_independent": float(x.mean()),
                "observed_sim_on_edges": float(x[A_bin[iu_r, iu_c] > 0].mean()),
                "n_edges_with_positive_sim": int((x[A_bin[iu_r, iu_c] > 0] > 0).sum())})
C.save_csv(pd.DataFrame(cov), "F1_mantel_final", "informative_pairs.csv")

ndead = len(dead)  # MARK: censoring_block
pairs_all = N * (N - 1) // 2
pairs_dd = ndead * (ndead - 1) // 2
pairs_ad = ndead * (N - ndead)
hub_edges = sum(1 for u, v in b["Gu"].edges() if u in hubn or v in hubn)
edges_touch_dead = sum(1 for u, v in b["Gu"].edges() if (u in dead) ^ (v in dead))
w_all = float(A_w[iu_r, iu_c].sum())
w_top1pct = float(np.sort(A_w[iu_r, iu_c])[::-1][: int(0.01 * len(iu_r))].sum())
rows_struct = [
    {"item": "unordered node pairs (671 attributed)", "value": pairs_all},
    {"item": "observed undirected edges within 671", "value": int(A_bin[iu_r, iu_c].sum())},
    {"item": "edge-pair share %", "value": round(100 * A_bin[iu_r, iu_c].sum() / pairs_all, 4)},
    {"item": "dead sites", "value": ndead},
    {"item": "pairs dead-dead (both directions unobservable)", "value": pairs_dd},
    {"item": "pairs dead-dead %", "value": round(100 * pairs_dd / pairs_all, 2)},
    {"item": "pairs dead-alive (dead-side out-edge censored)", "value": pairs_ad},
    {"item": "pairs with >=1 dead endpoint %", "value": round(100 * (pairs_dd + pairs_ad) / pairs_all, 2)},
    {"item": "observed edges with >=1 dead endpoint", "value": edges_touch_dead},
    {"item": "edges touching 6 nav hubs (C.hub_nodes)", "value": hub_edges},
    {"item": "edges touching 6 nav hubs %", "value": round(100 * hub_edges / b["Gu"].number_of_edges(), 2)},
    {"item": "weight mass in top-1% pairs", "value": round(100 * w_top1pct / w_all, 2)},
    {"item": "sites at 20-URL cap", "value": int((nd["listed_urls"] == 20).sum())},
    {"item": "sites with <10 listed URLs", "value": int((nd["listed_urls"] < 10).sum())},
    {"item": "eligible sites (listed_urls>=10)", "value": len(elig)},
    {"item": "edges within eligible pairs", "value": int(A_bin[np.ix_(eidx, eidx)].sum() / 2)},
]
C.save_csv(pd.DataFrame(rows_struct), "F1_mantel_final", "structure_visibility.csv")

# ------------------------------------------------------------------ 6. 签名审计
aud = []  # MARK: audit
for name, Smat in SIMS.items():
    x = Smat[iu_r, iu_c]
    xs = np.sort(x)[::-1]
    aud.append({"matrix": name, "shape": "%dx%d" % Smat.shape, "n_nan": int(np.isnan(Smat).sum()),
                "n_inf": int(np.isinf(Smat).sum()), "max_asym": float(np.abs(Smat - Smat.T).max()),
                "diag_min": float(np.diag(Smat).min()), "diag_max": float(np.diag(Smat).max()),
                "pair_mean": float(x.mean()), "pair_sd": float(x.std(ddof=1)),
                "frac_zero": float((x == 0).mean()), "frac_gt_090": float((x > 0.9).mean()),
                "n_distinct": int(len(np.unique(np.round(x, 12)))),
                "share_of_positive_mass_in_top1pct_pairs": float(xs[: int(0.01 * len(xs))].sum() / max(x.sum(), 1e-12)),
                "empty_vector_sites": int(sum(1 for n_ in NODES if not FEAT[n_]["tokens"]))
                if name.startswith(("S1", "S3")) else int(sum(1 for n_ in NODES if FEAT[n_]["n"] == 0))})
C.save_csv(pd.DataFrame(aud), "F1_mantel_final", "signature_audit.csv")

feat_rows = []  # MARK: feat_table
for i, n in enumerate(NODES):
    f = FEAT[n]
    feat_rows.append({"node": n, "listed_urls": f["n"], "n_tokens": len(f["tokens"]),
                      "n_tokens_in_vocab": int(X_tok[i].sum()), "n_templates": len(f["templates"]),
                      "n_templates_in_vocab": int(X_tmpl[i].sum()),
                      "depth_mean": float(np.mean(f["depths"])) if f["depths"] else 0.0,
                      "ext_top": max(f["ext_mix"], key=f["ext_mix"].get),
                      "ext_htm_share": f["ext_mix"]["htm"], "ext_php_share": f["ext_mix"]["php"],
                      "ext_none_share": f["ext_mix"]["none"], "ext_other_share": f["ext_mix"]["other"],
                      "has_num_share": f["has_num"], "has_query_share": f["has_query"],
                      "root_only_site": int(len(f["tokens"]) == 0)})
C.save_csv(pd.DataFrame(feat_rows), "F1_mantel_final", "signature_features.csv")

C.save_csv(pd.DataFrame({"marker": cms_hits.index, "n_paths_hit": cms_hits.values,
                         "share_of_all_paths": (cms_hits.values / max(len(ALL_PATHS), 1))}),
           "F1_mantel_final", "cms_marker_audit.csv")  # MARK: cms_save

np.savez_compressed(os.path.join(C.module_dir("F1_mantel_final"), "sim_matrices.npz"),
                    nodes=np.array(NODES), elig=np.array(elig), A_bin=A_bin, A_w=A_w,
                    D_spd_full=P["spd_hops"], X_tok=X_tok, X_tmpl=X_tmpl, E_mix=E_mix,
                    tok_vocab=np.array(tok_vocab, dtype=object),
                    **{f"S_{k}": v for k, v in SIMS.items()})  # MARK: npz_save

# ------------------------------------------------------------------ 7. key_numbers / RUN
kn = []  # MARK: keys_save
MARK_FOR_KEY = {"n_url_rows": "load_urls", "n_nodes": "node_order", "n_eligible_listed_ge10": "eligible",
                "n_edges_within_attributed": "adjacency", "n_gcc_attributed": "gcc_nodes",
                "n_token_vocab_kept": "tok_vocab", "n_tmpl_vocab_kept": "tp_vocab",
                "n_root_only_sites": "feat_built", "cms_marker_max_hits": "cms_hits",
                "n_edges_within_eligible": "elig_setup"}
for k_, v_ in KEYS.items():
    if k_ == "line_map":
        continue
    kn.append({"key": k_, "value": v_, "src_line": "run_f1.py:L%d" % LM.get(MARK_FOR_KEY.get(k_, ""), 0)})
kn.append({"key": "n_perm", "value": N_PERM, "src_line": "run_f1.py:L%d" % LM["mantel_chunk"]})
kn.append({"key": "n_boot", "value": N_BOOT, "src_line": "run_f1.py:L%d" % LM["main_grid"]})
for _, r in dfm.iterrows():
    src_line = LM["main_grid"] if r["node_set"] == "attributed671" else LM["elig_grid"]
    kn.append({"key": "mantel|%s|%s|%s|%s" % (r["signature"], r["similarity"], r["proximity"], r["node_set"]),
               "value": "r=%+.5f p=%.4f nullsd=%.5f z=%.2f bootCI=[%.4f,%.4f] ceiling=%+.4f/%+.4f"
                        % (r["mantel_r"], r["p_two_sided"], r["null_sd"], r["z"],
                           r["boot_ci_lo"], r["boot_ci_hi"], r["r_ceiling_pos"], r["r_ceiling_neg"]),
               "src_line": "run_f1.py:L%d" % src_line})
for _, r in pd.DataFrame(diag_rows).iterrows():
    kn.append({"key": "diag|%s|%s" % (r["signature"], r["diagnostic"]),
               "value": "r=%+.5f (full %+.5f, delta %+.5f) p=%.4f ceil=%+.4f"
                        % (r["mantel_r"], r["mantel_r_reference_full"], r["delta_vs_full"],
                           r["p_two_sided"], r["r_ceiling_pos"]),
               "src_line": "run_f1.py:L%d" % LM["diag_grid"]})
for _, r in pd.DataFrame(contrib_rows).iterrows():
    kn.append({"key": "contrib|%s|share_top1pct_pairs" % r["signature"],
               "value": round(r["share_top1pct_pairs"], 4), "src_line": "run_f1.py:L%d" % LM["contrib"]})
for c in cov:
    kn.append({"key": "coverage|%s|frac_pairs_nonzero_proximity" % c["signature"],
               "value": round(c["frac_pairs_nonzero_proximity"], 5),
               "src_line": "run_f1.py:L%d" % LM["coverage_table"]})
    kn.append({"key": "coverage|%s|r_ceiling_binary_pos" % c["signature"],
               "value": round(c["r_ceiling_binary_pos"], 4), "src_line": "run_f1.py:L%d" % LM["ceiling"]})
for s in rows_struct:
    kn.append({"key": "struct|%s" % s["item"], "value": s["value"],
               "src_line": "run_f1.py:L%d" % LM["censoring_block"]})
C.save_csv(pd.DataFrame(kn), "F1_mantel_final", "key_numbers.csv")

wall = time.time() - T0
C.write_run_note("F1_mantel_final", [
    "F1_mantel_final RUN.txt",
    "command: cd pipeline && PYTHONHASHSEED=0 python out/F1_mantel_final/run_f1.py",
    "versions: " + json.dumps(C.versions(), ensure_ascii=False),
    "SEED = %d ; n_perm = %d (fresh simultaneous row/col relabelling per replicate, CONVENTIONS 4) ; n_boot = %d" % (SEED, N_PERM, N_BOOT),
    "anchors: csa_common.assert_anchors() 31/31 PASS",
    "two-sided p = mean(|null-mean(null)| >= |obs-mean(null)|)   [CONVENTIONS 4]",
    "node sets: attributed671 (primary), eligible listed_urls>=10 = %d (sensitivity)" % len(elig),
    "proximity: adj_binary / adj_weighted (=Gu 两方向权重之和) / spd_hops (GCC %d 节点内有限)" % Ng,
    "wall time: %.1f s" % wall,
    "outputs: mantel_grid.csv bootstrap_vs_perm.csv informative_pairs.csv structure_visibility.csv",
    "           signature_audit.csv signature_features.csv cms_marker_audit.csv sim_matrices.npz key_numbers.csv RUN.txt",
    "log:", *LOG,
])
log("done in %.1f s" % wall)
