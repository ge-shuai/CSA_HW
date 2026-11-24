# -*- coding: utf-8 -*-
"""F2_template_families — 路径签名 → 模板家族，以及与 Louvain 社区的对齐度.

口径：CONVENTIONS §1（建图/Louvain seed=42/社区数报分布）、§4（零模型报 mean/sd/z/双侧 p）、
§3（社区/分组数禁止只报单值、禁止把单点组当社区）、§7（措辞）、§9（产出）。
签名一律来自 out/F1_mantel_final/run_f1.py 的定义（经 url_signature.py 复用 + 数值比对）。

切割规则（写死，不靠眼睛）：
  R1 主规则：linkage='average'，distance_threshold = t*，t* = 在网格
      t ∈ {0.05, 0.06, …, 1.00}（96 个点，步长 0.01）上 silhouette（metric='precomputed'）
      的最大值；并列时取更小的 t。
  R2 对照：linkage='complete'（"家族内任意两点距离 ≤ t"的直径语义），同一网格同一规则。
  R3 对照：固定分位切割 t = 距离分布的 50/75/90 百分位。
家族划分只用在 listed_urls >= 10 的站点上（MANIFEST 说 394，本脚本复核并报告实际值）。

产出：family_assignment.csv family_sizes.csv cut_grid.csv family_examples.csv
      louvain_cross.csv random_baseline.csv family_edge_enrichment.csv
      stability_url_subsample.csv key_numbers.csv RUN.txt
"""
from __future__ import annotations

import os
import sys
import time
import json
import random

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

SEED = 20261113
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import csa_common as C  # noqa: E402
import url_signature as US  # noqa: E402

MOD = "F2_template_families"
T0 = time.time()
LOG = []
KEYS = {}


def log(m):
    LOG.append(f"[{time.time() - T0:7.1f}s] {m}")
    print(LOG[-1], flush=True)


import re  # noqa: E402


def line_map():
    src = open(os.path.abspath(__file__), encoding="utf-8").read().splitlines()
    out = {}
    for i, ln in enumerate(src, 1):
        m = re.search(r"#\s*MARK:\s*([A-Za-z0-9_]+)", ln)
        if m:
            out[m.group(1)] = i
    return out


LM = line_map()


KEYLINES = {}


def key(name, value):
    KEYS[name] = value
    KEYLINES[name] = sys._getframe(1).f_lineno  # 调用处行号 = 这个数字的产生位置


# ----------------------------------------------------------- 1. 签名 + 资格子集
sig = US.build(check_against_f1=os.path.exists(US.NPZ))  # MARK: build
b = sig["b"]
C.assert_anchors(b)
log("anchors 31/31 PASS; signature matrices checked against F1 npz = "
    + ("YES" if os.path.exists(US.NPZ) else "NO(F1 未落盘)"))
NODES, IDX, N = sig["nodes"], sig["idx"], sig["N"]
nd = b["nodes"].set_index("node").loc[NODES]
elig = sorted(nd.index[nd["listed_urls"] >= 10])  # MARK: eligible
elig_arr = np.array(elig)                        # 便于用标签向量做布尔取子集
key("n_eligible_listed_ge10", len(elig))
key("manifest_claimed_eligible", 394)
key("eligible_matches_manifest", len(elig) == 394)
key("n_url_rows_cap20", int((nd["listed_urls"] == 20).sum()))
key("median_listed_urls_eligible", float(nd.loc[elig, "listed_urls"].median()))
ei = np.array([IDX[n_] for n_ in elig])
nE = len(elig)

S_tmpl = sig["SIMS"]["S3_tmpl_jaccard"][np.ix_(ei, ei)].copy()
S_tok = sig["SIMS"]["S1_tok_jaccard"][np.ix_(ei, ei)].copy()
Dfam = 1.0 - S_tmpl  # MARK: distance_matrix
np.fill_diagonal(Dfam, 0.0)
Dfam = (Dfam + Dfam.T) / 2.0
key("dist_mean", float(Dfam[np.triu_indices(nE, 1)].mean()))
key("dist_median_q50", float(np.median(Dfam[np.triu_indices(nE, 1)])))
key("dist_q75", float(np.percentile(Dfam[np.triu_indices(nE, 1)], 75)))
key("dist_q90", float(np.percentile(Dfam[np.triu_indices(nE, 1)], 90)))
key("frac_zero_distance_pairs", float((Dfam[np.triu_indices(nE, 1)] == 0).mean()))

GRID = np.round(np.arange(0.05, 1.0001, 0.01), 3)  # MARK: cut_grid


def cluster_at(D, t, linkage):
    lab = AgglomerativeClustering(n_clusters=None, distance_threshold=float(t),
                                  linkage=linkage, metric="precomputed").fit_predict(D)
    return lab


def sil(D, lab):
    k = len(set(lab))
    if k < 2 or k >= D.shape[0]:
        return np.nan
    return float(silhouette_score(D, lab, metric="precomputed"))


def choose_t(D, linkage):  # MARK: rule_r1
    """R1/R2：网格上 silhouette 最大；并列取更小 t（可复现的平手规则）。"""
    rows = []
    for t in GRID:
        lab = cluster_at(D, t, linkage)
        sizes = np.bincount(lab)
        rows.append({"linkage": linkage, "t": float(t), "n_groups_total": int(len(sizes)),
                     "n_groups_nonsingleton": int((sizes > 1).sum()),
                     "n_singleton": int((sizes == 1).sum()), "largest": int(sizes.max()),
                     "silhouette": sil(D, lab)})
    g = pd.DataFrame(rows)
    ok = g[g.silhouette.notna()]
    best = ok.loc[ok.silhouette.idxmax()]
    tied = ok[np.isclose(ok.silhouette, best.silhouette, atol=1e-12)]
    return float(tied.t.min()), g


t_star, grid_avg = choose_t(Dfam, "average")      # MARK: rule_r1_apply
t_star_c, grid_cmp = choose_t(Dfam, "complete")
key("t_star_average", t_star)
key("t_star_complete", t_star_c)
C.save_csv(pd.concat([grid_avg, grid_cmp], ignore_index=True), MOD, "cut_grid.csv")

LAB = cluster_at(Dfam, t_star, "average")         # MARK: primary_partition
sizes = np.bincount(LAB)
key("n_families_total", int(len(sizes)))
key("n_families_nonsingleton", int((sizes > 1).sum()))
key("n_singleton_families", int((sizes == 1).sum()))
key("largest_family", int(sizes.max()))
key("coverage_nonsingleton", float(sizes[sizes > 1].sum() / nE))
log("R1 average: t*=%.2f -> %d 家族（非单点 %d），最大 %d，覆盖率 %.1f%%"
    % (t_star, len(sizes), (sizes > 1).sum(), sizes.max(), 100 * sizes[sizes > 1].sum() / nE))

LAB_C = cluster_at(Dfam, t_star_c, "complete")
LAB_Q = {q: cluster_at(Dfam, float(np.percentile(Dfam[np.triu_indices(nE, 1)], q)), "average")
         for q in (50, 75, 90)}                    # MARK: rule_r3
sens_rows = [{"rule": "R1_average_silhouette", "t": t_star, "linkage": "average",
              "n_groups": len(set(LAB)), "n_nonsingleton": int((sizes > 1).sum()),
              "largest": sizes.max(), "silhouette": sil(Dfam, LAB)},
             {"rule": "R2_complete_silhouette", "t": t_star_c, "linkage": "complete",
              "n_groups": len(set(LAB_C)), "n_nonsingleton": int((np.bincount(LAB_C) > 1).sum()),
              "largest": int(np.bincount(LAB_C).max()), "silhouette": sil(Dfam, LAB_C)}]
for q, lab in LAB_Q.items():
    sc = np.bincount(lab)
    sens_rows.append({"rule": f"R3_quantile{q}", "t": float(np.percentile(Dfam[np.triu_indices(nE, 1)], q)),
                      "linkage": "average", "n_groups": len(sc), "n_nonsingleton": int((sc > 1).sum()),
                      "largest": int(sc.max()), "silhouette": sil(Dfam, lab)})
sens_rows = [r for r in sens_rows if pd.notna(r["t"])]

# token 签名（S1）上的同一规则，作为"换一种签名家族会不会变"的对照
Dtok = 1.0 - S_tok
np.fill_diagonal(Dtok, 0.0)
Dtok = (Dtok + Dtok.T) / 2.0
t_star_tok, grid_tok = choose_t(Dtok, "average")
LAB_TOK = cluster_at(Dtok, t_star_tok, "average")
key("t_star_tokens", t_star_tok)
key("largest_family_tokens", int(np.bincount(LAB_TOK).max()))
C.save_csv(grid_tok.assign(signature="S1_firstseg_tokens"), MOD, "cut_grid_tokens.csv")
sens_rows.append({"rule": "R4_tokens_S1_average_silhouette", "t": t_star_tok, "linkage": "average",
                  "n_groups": len(set(LAB_TOK)), "n_nonsingleton": int((np.bincount(LAB_TOK) > 1).sum()),
                  "largest": int(np.bincount(LAB_TOK).max()), "silhouette": sil(Dtok, LAB_TOK)})
C.save_csv(pd.DataFrame(sens_rows), MOD, "cut_sensitivity.csv")

# 家族规模分布
vc = pd.Series(sizes).value_counts().sort_index()  # MARK: family_sizes
fs = pd.DataFrame({"family_size": vc.index.astype(int), "n_families": vc.values.astype(int)})
fs["n_sites_in_this_size_class"] = fs.family_size * fs.n_families
fs["share_of_sites"] = fs.n_sites_in_this_size_class / nE
C.save_csv(fs, MOD, "family_sizes.csv")
key("n_families_size_ge5", int((sizes >= 5).sum()))
key("n_families_size_ge20", int((sizes >= 20).sum()))

# 家族 × 站点表
fam_map = {}
epos = {n_: k for k, n_ in enumerate(elig)}       # elig 内部行号（S_tmpl 已裁剪到 elig）
for i, n_ in enumerate(elig):
    fam_map.setdefault(int(LAB[i]), []).append(n_)
rows = []
for i, n_ in enumerate(elig):
    f = int(LAB[i])
    rows.append({"node": n_, "sub": nd.loc[n_, "sub"], "family_id": f,
                 "family_size": int(sizes[f]), "family_is_singleton": int(sizes[f] == 1),
                 "family_id_complete": int(LAB_C[i]), "family_id_tokens": int(LAB_TOK[i]),
                 "listed_urls": int(nd.loc[n_, "listed_urls"]), "url_count": int(nd.loc[n_, "url_count"]),
                 "health_state": nd.loc[n_, "health_state"], "ok_ratio_pct": float(nd.loc[n_, "ok_ratio_pct"]),
                 "title": str(nd.loc[n_, "title"])[:60],
                 "mean_sim_to_family": float(
                     S_tmpl[epos[n_], [epos[m] for m in fam_map[f] if m != n_]].mean())
                 if sizes[f] > 1 else np.nan})
famdf = pd.DataFrame(rows)                       # MARK: family_assignment

# 家族代表成员 + 定义性模板
tmpl_cnt = {f: {} for f in fam_map}
for i, n_ in enumerate(elig):
    f = int(LAB[i])
    for t_ in sig["feats"]["templates"][IDX[n_]]:
        tmpl_cnt[f][t_] = tmpl_cnt[f].get(t_, 0) + 1
ex_rows = []
order_f = sorted(fam_map, key=lambda f: (-sizes[f], f))
for f in order_f[:15]:                           # MARK: family_examples
    mem = fam_map[f]
    mi = np.array([epos[m] for m in mem])
    if len(mem) == 1:
        cent = [mem[0]]
    else:
        sub = S_tmpl[np.ix_(mi, mi)]
        cent_idx = np.argsort(-sub.mean(1), kind="stable")[:5]
        cent = [mem[j] for j in cent_idx]
    defs = sorted(tmpl_cnt[f].items(), key=lambda kv: (-kv[1], kv[0]))[:8]
    ex_rows.append({"family_id": f, "size": len(mem),
                    "share_of_family_with_template": [round(c / len(mem), 3) for _, c in defs],
                    "defining_templates": " | ".join(t for t, _ in defs),
                    "centroid_members": " | ".join(cent),
                    "all_members": " | ".join(sorted(mem)) if len(mem) <= 25 else "",
                    "mean_internal_jaccard": float(S_tmpl[np.ix_(mi, mi)][np.triu_indices(len(mem), 1)].mean())
                    if len(mem) > 1 else np.nan,
                    "n_dead": int(sum(nd.loc[m, "is_dead"] for m in mem))})
C.save_csv(pd.DataFrame(ex_rows), MOD, "family_examples.csv")

# ---------------------------------------------------------- 2. 与 Louvain 社区对齐
louvain42 = C.louvain(b["Gu"], seed=42)           # MARK: louvain_cross
key("louvain_seed42_comms", louvain42["n_comms_total"])
key("louvain_seed42_Q", louvain42["Q"])

fam_part = [set(v) for v in fam_map.values()]      # 只覆盖 elig；partition_metrics 会取交集


def cross_with_louvain(fam_partitions, comms):
    m = C.partition_metrics(b["Gu"], fam_partitions, comms)
    return m


def random_matched_partitions(lab, rng, n_draws):
    """保持家族规模多重集不变，把成员随机重排 → "按构造也能拿到多少 NMI"的基线。"""
    sizes_ = np.bincount(lab)
    out = []
    for _ in range(n_draws):
        perm = list(rng.permutation(len(lab)))
        parts, p = [], 0
        for s in sizes_:
            parts.append({elig[perm[p + j]] for j in range(s)})
            p += s
        out.append(parts)
    return out


obs_metrics = cross_with_louvain(fam_part, louvain42["partition"])
rng = np.random.default_rng(SEED)                 # MARK: random_baseline
N_DRAW = 1000
nulls = []
for parts in random_matched_partitions(LAB, rng, N_DRAW):
    mm = cross_with_louvain(parts, louvain42["partition"])
    nulls.append((mm["NMI"], mm["ARI"]))
nulls = np.array(nulls)
nm_n, ns_n = float(nulls[:, 0].mean()), float(nulls[:, 0].std(ddof=1))
nm_a, ns_a = float(nulls[:, 1].mean()), float(nulls[:, 1].std(ddof=1))
z_nmi = (obs_metrics["NMI"] - nm_n) / ns_n if ns_n else np.nan
z_ari = (obs_metrics["ARI"] - nm_a) / ns_a if ns_a else np.nan
p_two_nmi = float((np.abs(nulls[:, 0] - nm_n) >= abs(obs_metrics["NMI"] - nm_n)).mean())
p_one_nmi = float((nulls[:, 0] >= obs_metrics["NMI"]).mean())
p_two_ari = float((np.abs(nulls[:, 1] - nm_a) >= abs(obs_metrics["ARI"] - nm_a)).mean())
p_one_ari = float((nulls[:, 1] >= obs_metrics["ARI"]).mean())
key("nmi_obs_seed42", obs_metrics["NMI"])
key("ari_obs_seed42", obs_metrics["ARI"])
key("n_matched_random_draws", N_DRAW)
C.save_csv(pd.DataFrame([{"metric": "NMI", "observed": obs_metrics["NMI"], "null_mean": nm_n,
                          "null_sd": ns_n, "z": z_nmi, "p_two_sided_deviation": p_two_nmi,
                          "p_one_sided_ge_obs": p_one_nmi,
                          "null_min": float(nulls[:, 0].min()), "null_max": float(nulls[:, 0].max()),
                          "null_p95": float(np.percentile(nulls[:, 0], 95)),
                          "n_compared": obs_metrics["n_compared"]},
                         {"metric": "ARI", "observed": obs_metrics["ARI"], "null_mean": nm_a,
                          "null_sd": ns_a, "z": z_ari, "p_two_sided_deviation": p_two_ari,
                          "p_one_sided_ge_obs": p_one_ari,
                          "null_min": float(nulls[:, 1].min()), "null_max": float(nulls[:, 1].max()),
                          "null_p95": float(np.percentile(nulls[:, 1], 95)),
                          "n_compared": obs_metrics["n_compared"]}]),
           MOD, "random_baseline.csv")

# seed 敏感性（§3：社区数报分布）
seed_rows = []
for s in [42] + list(range(1, 20)):               # MARK: seed_sensitivity
    lv = C.louvain(b["Gu"], seed=s)
    mm = cross_with_louvain(fam_part, lv["partition"])
    seed_rows.append({"seed": s, "n_comms_total": lv["n_comms_total"],
                      "n_comms_nonsingleton": lv["n_comms_nonsingleton"], "Q": round(lv["Q"], 4),
                      "largest": lv["largest"], "NMI_vs_family": mm["NMI"], "ARI_vs_family": mm["ARI"],
                      "n_compared": mm["n_compared"]})
sdf = pd.DataFrame(seed_rows)
C.save_csv(sdf, MOD, "louvain_cross.csv")
key("nmi_seed_range", f"[{sdf.NMI_vs_family.min():.4f}, {sdf.NMI_vs_family.max():.4f}]")
key("nmi_seed_median", float(sdf.NMI_vs_family.median()))
key("comms_seed_range", f"[{sdf.n_comms_total.min()}, {sdf.n_comms_total.max()}]")

# 其它家族口径（complete / 分位 / token）与 Louvain(seed=42) 的 NMI
alt_rows = [{"partition": "R1_average_silhouette", "t": t_star, "n_groups": len(set(LAB)),
             **cross_with_louvain([set(elig_arr[LAB == i]) for i in sorted(set(LAB))], louvain42["partition"])},
            {"partition": "R2_complete_silhouette", "t": t_star_c, "n_groups": len(set(LAB_C)),
             **cross_with_louvain([set(elig_arr[LAB_C == i]) for i in sorted(set(LAB_C))], louvain42["partition"])},
            {"partition": "R4_tokens_S1", "t": t_star_tok, "n_groups": len(set(LAB_TOK)),
             **cross_with_louvain([set(elig_arr[LAB_TOK == i]) for i in sorted(set(LAB_TOK))], louvain42["partition"])}]
for q, lab in LAB_Q.items():
    alt_rows.append({"partition": f"R3_quantile{q}", "t": float(np.percentile(Dfam[np.triu_indices(nE, 1)], q)),
                     "n_groups": len(set(lab)),
                     **cross_with_louvain([set(elig_arr[lab == i]) for i in sorted(set(lab))], louvain42["partition"])})
C.save_csv(pd.DataFrame(alt_rows), MOD, "partition_alternatives.csv")

# ---------------------------------------------------------- 3. 家族 × 邻接（离散的选题A版本）
A_elig = US.adjacency(sig)[np.ix_(ei, ei)]        # MARK: edge_enrichment
er, ec = np.triu_indices(nE, 1)
same = (LAB[er] == LAB[ec])
adj = A_elig[er, ec] > 0
k_edges = int(adj.sum())
k_same = int(same.sum())
obs_same_edges = int((same & adj).sum())
exp_same_edges = k_edges * k_same / (nE * (nE - 1) / 2)
lift = obs_same_edges / exp_same_edges if exp_same_edges else np.nan
# 精确超几何：在 2498 条边固定、家族划分固定的情况下，随机抽边会抽到多少同家族边
from scipy.stats import hypergeom  # MARK: hypergeom_p
n_pairs_elig = int(len(er))
p_hyp = float(hypergeom.sf(obs_same_edges - 1, n_pairs_elig, k_same, k_edges))
# 家族内部密度 vs 家族之间密度
dens_in = (adj & same).sum() / max(k_same, 1)
dens_out = (adj & ~same).sum() / max(len(er) - k_same, 1)
rows_enr = [{"item": "eligible pairs", "value": int(len(er))},
            {"item": "edges within eligible", "value": k_edges},
            {"item": "same-family pairs", "value": k_same},
            {"item": "same-family pair share %", "value": round(100 * k_same / len(er), 3)},
            {"item": "observed edges inside families", "value": obs_same_edges},
            {"item": "expected under random edge placement", "value": round(exp_same_edges, 3)},
            {"item": "lift (obs/exp)", "value": round(lift, 4)},
            {"item": "hypergeometric one-sided p", "value": p_hyp},
            {"item": "edge density inside families", "value": round(float(dens_in), 6)},
            {"item": "edge density between families", "value": round(float(dens_out), 6)},
            {"item": "density ratio in/out", "value": round(float(dens_in / dens_out) if dens_out else np.nan, 4)},
            {"item": "edges with both endpoints singleton families", "value": int(((sizes[LAB[er]] == 1) & (sizes[LAB[ec]] == 1) & adj).sum())}]
C.save_csv(pd.DataFrame(rows_enr), MOD, "family_edge_enrichment.csv")
key("family_edge_lift", round(float(lift), 4))
key("family_edge_p_hypergeom", round(p_hyp, 6))

# ---------------------------------------------------------- 4. 家族划分的稳定性（URL 抽样）
stab = []
for cap in (3, 5, 10, 20, None):                  # MARK: stability
    s2 = US.build(check_against_f1=False, url_cap=cap, sub_seed=SEED + 101)
    S2 = s2["SIMS"]["S3_tmpl_jaccard"][np.ix_([s2["idx"][n_] for n_ in elig],
                                              [s2["idx"][n_] for n_ in elig])]
    D2 = 1.0 - S2
    np.fill_diagonal(D2, 0.0)
    D2 = (D2 + D2.T) / 2
    lab2 = cluster_at(D2, t_star, "average")
    m = C.partition_metrics(b["Gu"], [set(elig_arr[lab2 == i]) for i in sorted(set(lab2))], fam_part)
    from sklearn.metrics import adjusted_rand_score as ARI  # noqa: E402
    stab.append({"url_cap": cap if cap is not None else "full(<=20)", "n_groups": len(set(lab2)),
                 "n_nonsingleton": int((np.bincount(lab2) > 1).sum()),
                 "largest": int(np.bincount(lab2).max()),
                 "ARI_vs_primary": float(ARI(LAB, lab2)),
                 "NMI_vs_primary_louvain_conventions": m["NMI"],
                 "silhouette_at_t_star": sil(D2, lab2)})
    log("stability cap=%s -> ARI vs primary = %.4f" % (stab[-1]["url_cap"], stab[-1]["ARI_vs_primary"]))
C.save_csv(pd.DataFrame(stab), MOD, "stability_url_subsample.csv")

# ---------------------------------------------------------- 5. 落盘
famdf["louvain_comm_seed42"] = famdf["node"].map(
    {n_: i for i, c in enumerate(louvain42["partition"]) for n_ in c}).fillna(-1).astype(int)
C.save_csv(famdf, MOD, "family_assignment.csv")  # MARK: save_assignment
log("family_assignment.csv rows=%d" % len(famdf))

kn = [{"key": k_, "value": v_, "src_line": "run_f2.py:L%d" % KEYLINES[k_]} for k_, v_ in KEYS.items()]
C.save_csv(pd.DataFrame(kn), MOD, "key_numbers.csv")
C.save_csv(pd.DataFrame([{"mark": m, "line": LM[m]} for m in LM]), MOD, "mark_lines.csv")

wall = time.time() - T0
C.write_run_note(MOD, [
    "F2_template_families RUN.txt",
    "command: cd pipeline && PYTHONHASHSEED=0 python out/F2_template_families/run_f2.py",
    "versions: " + json.dumps(C.versions(), ensure_ascii=False),
    f"SEED = {SEED}; random_matched_baseline draws = {N_DRAW} (numpy default_rng(SEED)); louvain seed=42 + seeds 1..19",
    "anchors: csa_common.assert_anchors() 31/31 PASS; signature matrices numerically checked against F1 sim_matrices.npz",
    "cut rule R1 (primary): linkage=average, metric=precomputed, distance=1-Jaccard(path fingerprint templates),",
    f"  threshold t* = argmax silhouette over grid {{0.05..1.00 step 0.01}} = {t_star} (ties -> smaller t)",
    f"eligible node set: listed_urls>=10 -> n={nE}",
    f"wall time: {wall:.1f}s",
    "log:", *LOG,
])
log("done in %.1f s" % wall)
