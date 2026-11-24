"""00_build_data — 生成全项目共用的清洗产物与锚点表。

跑法：python 00_build_data.py
输出：pipeline/data/{clean_nodes.csv, clean_edges.csv, dead_categories.csv,
                     anchors.csv, censoring_facts.csv, DATA_DICTIONARY.md}
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import networkx as nx

import csa_common as C

MOD = "00_build_data"


def main():
    os.makedirs(C.CLEAN_DIR, exist_ok=True)
    log = [f"00_build_data run at pipeline seed={C.SEED}", str(C.versions())]

    # 1) 先用原始 CSV 建一遍，验锚点
    b_raw = C.build_all(use_clean=False)
    anchors = C.check_anchors(b_raw)
    anchors.to_csv(os.path.join(C.CLEAN_DIR, "anchors.csv"), index=False, encoding="utf-8-sig")
    if not anchors["ok"].all():
        raise SystemExit("原始数据锚点未全过，停止：\n" + anchors[~anchors.ok].to_string(index=False))
    log.append(f"anchors(raw): {int(anchors.ok.sum())}/{len(anchors)} PASS")

    m, Gd, Gu, dc = b_raw["nodes"], b_raw["Gd"], b_raw["Gu"], b_raw["dead_cat"]
    sets = b_raw["sets"]

    # 2) clean_nodes：原始列 + 派生列（度、健康、失效语义）
    keep_orig = ["subdomain", "sub", "title", "url_count", "listed_urls",
                 "ok", "bad", "ok_ratio_pct", "status_codes", "color", "root_url"]
    derived = ["node", "in_degree", "out_degree", "degree_u", "strength_u", "in_strength",
               "out_strength", "is_dead", "is_partial", "is_healthy", "health_state",
               "dead_cat", "true_fail", "false_fail", "ambiguous_fail",
               "partial_has_auth_like", "partial_clean"]
    cn = m[keep_orig + derived].copy()
    cn["in_gcc"] = cn["node"].isin(sets["gcc"])
    cn["is_dangling_ref"] = False
    cn = cn.sort_values("node").reset_index(drop=True)
    cn.to_csv(os.path.join(C.CLEAN_DIR, "clean_nodes.csv"), index=False, encoding="utf-8-sig")
    log.append(f"clean_nodes.csv: {len(cn)} rows x {cn.shape[1]} cols")

    # 3) clean_edges：有向原边 + 无向聚合边两套口径
    und_w = {(u, v) if u <= v else (v, u): d.get("weight", 1) for u, v, d in Gu.edges(data=True)}
    dead_set = set(m.loc[m["is_dead"], "node"])
    le = []
    recip = set(Gd.edges())
    for order, s, t, w in b_raw["links"][["edge_order", "source", "target", "weight"]].itertuples(index=False):
        k = (s, t) if s <= t else (t, s)
        le.append({
            "edge_order": int(order), "source": s, "target": t, "weight": int(w),
            "undirected_weight": int(und_w[k]),
            "is_reciprocal": (t, s) in recip,
            "source_is_dead": s in dead_set,
            "target_is_dead": t in dead_set,
        })
    ce = pd.DataFrame(le).sort_values("edge_order").reset_index(drop=True)
    ce.to_csv(os.path.join(C.CLEAN_DIR, "clean_edges.csv"), index=False, encoding="utf-8-sig")
    log.append(f"clean_edges.csv: {len(ce)} directed rows; "
               f"reciprocal pairs={int(ce.is_reciprocal.sum())}; "
               f"undirected edges={Gu.number_of_edges()}")

    # 4) dead_categories + 语义标签
    dcs = dc.copy()
    dcs["is_true_fail"] = dcs["cat"].isin(C.TRUE_FAIL_CATS)
    dcs["is_false_fail"] = dcs["cat"].isin(C.FALSE_FAIL_CATS)
    dcs["is_ambiguous"] = dcs["cat"].isin(C.AMBIGUOUS_CATS)
    dcs = dcs[["sub", "node", "cat", "codes", "url_count",
               "is_true_fail", "is_false_fail", "is_ambiguous"]].sort_values("sub")
    dcs.to_csv(os.path.join(C.CLEAN_DIR, "dead_categories.csv"), index=False, encoding="utf-8-sig")
    log.append("dead_categories.csv: " + dcs.cat.value_counts().to_dict().__repr__())

    # 5) 删失事实表（本项目最关键的一张口径表）
    dead = set(m.loc[m.is_dead, "node"])
    in_deg_dead = m.loc[m.is_dead, "in_degree"]
    facts = [
        ("整站不可用节点数", len(dead), ""),
        ("其中出度==0", sum(1 for n, d in Gd.out_degree(dead) if d == 0),
         "抓取器只能从抓到的页面提链接 → 死站出边**全部不可观测**"),
        ("以死站为起点的有向边", sum(1 for s, _ in Gd.edges() if s in dead),
         "= 0，所以'死—死相邻'在数据里是构造性不可识别，不是'没有发生'"),
        ("以死站为终点的有向边", sum(1 for _, t in Gd.edges() if t in dead), ""),
        ("死站平均无向度", round(float(m.loc[m.is_dead, "degree_u"].mean()), 3), "== 入度，因为出度恒为 0"),
        ("健康站平均无向度", round(float(m.loc[m.is_healthy, "degree_u"].mean()), 3), "含大量不可观测出边"),
        ("同口径比入度：死 vs 健康",
         f"{round(float(in_deg_dead.mean()), 3)} vs {round(float(m.loc[m.is_healthy, 'in_degree'].mean()), 3)}",
         "倍数远小于跨口径的 5.2×"),
        ("真失效 (A/D/F)", int(m.true_fail.sum()), "站点确实坏了"),
        ("假失效 (B/C)", int(m.false_fail.sum()), "认证跳转/robots 拦截，不是故障"),
        ("待定 (G_other_400)", int(m.ambiguous_fail.sum()), "须人工抽查"),
        ("含 -1/-2/403 的部分失效", int(m.partial_has_auth_like.sum()),
         "不能直接算'故障'"),
        ("干净的部分失效", int(m.partial_clean.sum()), ""),
        ("color 与 ok_ratio 不一致", int(((m.ok_ratio_pct == 0) & (m.color == "blue")).sum()),
         "32 个可用率 0 的站点 color 标为 blue → 健康分层禁用 color"),
    ]
    cf = pd.DataFrame(facts, columns=["fact", "value", "why_it_matters"])
    cf.to_csv(os.path.join(C.CLEAN_DIR, "censoring_facts.csv"), index=False, encoding="utf-8-sig")
    log.append("censoring_facts.csv written")

    # 6) 悬空节点清单
    dang = pd.DataFrame({"node": sorted(sets["dangling"])})
    dang["degree_u"] = dang["node"].map(dict(Gu.degree()))
    dang["in_degree"] = dang["node"].map(dict(Gd.in_degree()))
    dang["out_degree"] = dang["node"].map(dict(Gd.out_degree()))
    dang.to_csv(os.path.join(C.CLEAN_DIR, "dangling_nodes.csv"), index=False, encoding="utf-8-sig")

    # 7) 数据字典
    write_dictionary(cn, ce, dcs)

    # 8) 回读清洗产物，确认与原始口径一致
    b_clean = C.build_all(use_clean=True)
    a2 = C.check_anchors(b_clean)
    if not a2["ok"].all():
        raise SystemExit("清洗产物回读后锚点不一致：\n" + a2[~a2.ok].to_string(index=False))
    log.append(f"anchors(clean reload): {int(a2.ok.sum())}/{len(a2)} PASS")

    C.write_run_note(MOD, log)
    print("\n".join(log))
    print("\nALL ANCHORS PASS — 下游模块可以开跑")


def write_dictionary(cn, ce, dcs):
    rows = []
    for col in cn.columns:
        rows.append(("clean_nodes.csv", col, cn[col].dtype.name,
                     int(cn[col].isna().sum()), str(cn[col].dropna().iloc[0])[:40]))
    for col in ce.columns:
        rows.append(("clean_edges.csv", col, ce[col].dtype.name,
                     int(ce[col].isna().sum()), str(ce[col].dropna().iloc[0])[:40]))
    for col in dcs.columns:
        rows.append(("dead_categories.csv", col, dcs[col].dtype.name,
                     int(dcs[col].isna().sum()), str(dcs[col].dropna().iloc[0])[:40]))
    dd = pd.DataFrame(rows, columns=["file", "column", "dtype", "n_na", "example"])
    notes = {
        "clean_nodes.csv": {
            "subdomain": "FQDN，= 图节点主键 node",
            "sub": "短名，= dead_sites_category.csv 的连接键",
            "url_count": "抓取器报告的该子域 URL 总数",
            "listed_urls": "实际落库的明细条数，**上限 20**（352 个站点正好 20）",
            "ok/bad": "明细里状态 200 / 非 200 的条数",
            "ok_ratio_pct": "ok/listed_urls 百分数；健康分层唯一依据（100=healthy, 0=dead, 中间=partial）",
            "status_codes": "'码:次数' 串；0=连接失败 -1=需认证/跳过 -2=robots 禁止",
            "color": "**已弃用**：与 ok_ratio_pct 冲突（32 个可用率 0 的站点标 blue），仅存档用",
            "in_degree/out_degree/degree_u": "有向入度/出度/无向度（度必须从 links.csv 重算，nodes_full 的 degree 列作废）",
            "strength_u/in_strength/out_strength": "边权=观察到的链接次数，不是流量",
            "is_dead/is_partial/is_healthy": "健康三分",
            "dead_cat": "仅整站不可用节点有值；A/D/F=真失效，B/C=假失效，G=待定",
            "true_fail/false_fail/ambiguous_fail": "失效语义校正后的标签",
            "partial_has_auth_like/partial_clean": "部分失效里含/不含 -1,-2,403",
            "in_gcc": "是否属于无向图最大连通分量(674)",
        },
        "clean_edges.csv": {
            "source/target": "FQDN",
            "weight": "该有向边上观察到的链接次数",
            "undirected_weight": "反向边权重之和（无向加权图用的就是这个）",
            "is_reciprocal": "是否存在反向边",
            "source_is_dead": "**恒为 False**：死站的出边不可观测，这是删失证据不是结论",
            "target_is_dead": "有健康站指向它，入边可观测",
        },
        "dead_categories.csv": {
            "cat": "失败原因分类", "codes": "触发该分类的状态码串",
            "is_true_fail": "A_conn_fail / D_not_found / F_server_error",
            "is_false_fail": "B_auth_or_skip / C_robots_blocked",
        },
    }
    dd["meaning"] = dd.apply(lambda r: notes.get(r["file"], {}).get(r["column"], ""), axis=1)
    p = os.path.join(C.CLEAN_DIR, "DATA_DICTIONARY.md")
    cols = ["file", "column", "dtype", "n_na", "example", "meaning"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in dd[cols].itertuples(index=False):
        cells = [str(getattr(r, c)).replace("|", "\\|").replace("\n", " ") for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    with open(p, "w", encoding="utf-8") as f:
        f.write("# 数据字典（由 00_build_data.py 自动生成，勿手改）\n\n")
        f.write(f"版本：{C.versions()}\n\n")
        f.write("\n".join(lines) + "\n")
    dd.to_csv(os.path.join(C.CLEAN_DIR, "DATA_DICTIONARY.csv"), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
