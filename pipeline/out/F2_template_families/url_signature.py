# -*- coding: utf-8 -*-
"""url_signature — F 组共用的 URL 结构签名构造器（F2/F3 的唯一入口）.

规则与 out/F1_mantel_final/run_f1.py §1 完全一致（F1 是口径的原始定义处）；
本模块每次调用都会和 F1 落盘的 sim_matrices.npz 做数值比对，
若矩阵不一致就抛错——这样三个单元用的是同一套签名，漂移会被立刻发现。

只用字符串/路径结构：小写、查询串丢弃、数字串归一为 #、扩展名白名单、
词表过滤（一个 token/模板至少出现在 3 个站点）。无分词库、无文本挖掘。
"""
from __future__ import annotations

import os
import re
import sys
import urllib.parse as up

import numpy as np
import networkx as nx

SEED = 20261113
HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PIPE)
import csa_common as C  # noqa: E402

F1_DIR = os.path.join(C.OUT_ROOT, "F1_mantel_final")
NPZ = os.path.join(F1_DIR, "sim_matrices.npz")

EXT_MAP = {"htm", "html", "php", "asp", "aspx", "jsp", "jspx", "psp", "shtml", "cfm",
           "do", "action", "ashx", "ascx", "xml", "json"}
EXT_LIST = sorted(EXT_MAP) + ["other", "none"]
NUM = re.compile(r"\d+")
VOCAB_MIN_SITES = 3


def parse_url(u):
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
        elif re.fullmatch(r"[a-z0-9]{1,5}", tail) and "@" not in last:
            ext = "other"
    tmpl = "/" + "/".join(segs) if segs else "/"
    return {"depth": len(segs), "first": first, "ext": ext, "tmpl": tmpl,
            "has_num": bool(NUM.search(last)), "has_query": bool(sp.query)}


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


def build(node_subset=None, check_against_f1=True, url_cap=None, sub_seed=SEED):
    """返回签名矩阵 + 节点表。node_subset=None → 全部 671 个有属性节点（字典序）。

    url_cap=d：每站只随机保留 min(d, 实际条数) 条 URL（Spearman–Brown/信度分析用，
    F3 用它量化 20 条上限造成的签名衰减）。None = 全量（与 F1 一致，可比对）。
    """
    b = C.build_all()
    C.assert_anchors(b)
    urls = C.load_urls()
    nodes_all = sorted(b["nodes"]["node"].tolist())
    if node_subset is None:
        NODES = nodes_all
    else:
        NODES = sorted(set(node_subset))
        missing = set(NODES) - set(nodes_all)
        if missing:
            raise ValueError(f"未知节点: {sorted(missing)[:5]}")
    IDX = {n: i for i, n in enumerate(NODES)}
    N = len(NODES)
    URLs_BY = {s: g["url"].astype(str).tolist() for s, g in urls.groupby("subdomain")}
    srng = np.random.default_rng(sub_seed) if url_cap is not None else None

    feats = {"tokens": [], "templates": [], "ext_mix": [], "depths": [], "paths": []}
    for n in NODES:
        us = URLs_BY.get(n, [])
        if srng is not None and len(us) > url_cap:
            us = [us[i] for i in sorted(srng.choice(len(us), size=url_cap, replace=False))]
        rows = [parse_url(u) for u in us]
        extc = {e: 0 for e in EXT_LIST}
        for r in rows:
            extc[r["ext"]] += 1
        tot = max(1, len(rows))
        feats["tokens"].append({r["first"] for r in rows if r["first"]})
        feats["templates"].append({r["tmpl"] for r in rows})
        feats["ext_mix"].append({k: v / tot for k, v in extc.items()})
        feats["depths"].append([r["depth"] for r in rows])
        feats["paths"].append([r["tmpl"] for r in rows])

    def vocab(items):
        cnt = {}
        for s in items:
            for it in s:
                cnt[it] = cnt.get(it, 0) + 1
        return sorted([k for k, v in cnt.items() if v >= VOCAB_MIN_SITES]), cnt

    tv, tcnt = vocab(feats["tokens"])
    pv, pcnt = vocab(feats["templates"])

    def binmat(items, voc):
        col = {w: i for i, w in enumerate(voc)}
        X = np.zeros((N, len(voc)), dtype=np.float32)
        for i, s in enumerate(items):
            for it in s:
                j = col.get(it)
                if j is not None:
                    X[i, j] = 1.0
        return X

    X_tok = binmat(feats["tokens"], tv)
    X_tmpl = binmat(feats["templates"], pv)
    E_mix = np.array([[f[e] for e in EXT_LIST] for f in feats["ext_mix"]])

    SIMS = {"S1_tok_jaccard": sim_jaccard(X_tok), "S1_tok_cosine": sim_cosine(X_tok),
            "S3_tmpl_jaccard": sim_jaccard(X_tmpl), "S3_tmpl_hamming": sim_hamming(X_tmpl),
            "S2_ext_cosine": sim_cosine(E_mix)}

    if check_against_f1 and node_subset is None and os.path.exists(NPZ):
        z = np.load(NPZ, allow_pickle=True)
        for k, v in (("S1_tok_jaccard", SIMS["S1_tok_jaccard"]), ("S3_tmpl_jaccard", SIMS["S3_tmpl_jaccard"]),
                     ("S2_ext_cosine", SIMS["S2_ext_cosine"])):
            d = float(np.abs(z["S_" + k] - v).max())
            if d > 1e-9:
                raise AssertionError(f"签名与 F1 落盘矩阵不一致：{k} 最大差 {d:g}（口径漂移，停）")
    return {"b": b, "nodes": NODES, "idx": IDX, "N": N, "X_tok": X_tok, "X_tmpl": X_tmpl,
            "E_mix": E_mix, "SIMS": SIMS, "tok_vocab": tv, "tmpl_vocab": pv,
            "tok_cnt": tcnt, "tmpl_cnt": pcnt, "feats": feats}


def adjacency(S, which="binary"):
    """从 csa_common 的 Gu 取邻接矩阵（不重建图）。"""
    Gu = S["b"]["Gu"]
    idx, N = S["idx"], S["N"]
    A = np.zeros((N, N))
    for u, v, d in Gu.edges(data=True):
        if u in idx and v in idx:
            i, j = idx[u], idx[v]
            A[i, j] = A[j, i] = (1.0 if which == "binary" else float(d.get("weight", 1)))
    return A


def pair_indices(N):
    return np.triu_indices(N, 1)


def pear(a, bv):
    a = a - a.mean()
    bv = bv - bv.mean()
    da, db = float(a @ a), float(bv @ bv)
    if da == 0 or db == 0:
        return np.nan
    return float(a @ bv / np.sqrt(da * db))


def mantel_perm_p(Smat, Pmat, n, rows, cols, mask, n_perm, seed):
    """§4 口径的 Mantel 置换检验（每轮 fresh 重标号；双侧 p 按偏离 null mean 计）。"""
    rm, cm = rows[mask], cols[mask]
    x, y = Smat[rm, cm], Pmat[rm, cm]
    obs = pear(x, y)
    rng = np.random.default_rng(seed)
    yc = y - y.mean()
    dby = float(yc @ yc)
    nulls = np.empty(n_perm)
    CH = 32
    for st in range(0, n_perm, CH):
        en = min(st + CH, n_perm)
        perm = np.stack([rng.permutation(n) for _ in range(en - st)])
        XX = Smat[perm[:, rm], perm[:, cm]]
        Xc = XX - XX.mean(axis=1, keepdims=True)
        nulls[st:en] = (Xc @ yc) / np.sqrt((Xc * Xc).sum(1) * dby)
    nm, ns = float(nulls.mean()), float(nulls.std(ddof=1))
    p = float((np.abs(nulls - nm) >= abs(obs - nm)).mean())
    return obs, nm, ns, (obs - nm) / ns if ns else np.nan, p, nulls
