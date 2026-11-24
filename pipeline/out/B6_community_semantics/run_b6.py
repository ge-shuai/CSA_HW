"""B6 · community_semantics — 社区划分与组织语义类别的一致性检验。

核心问题：把 Louvain 社区读作"院系/职能单位"是否成立？
主判据：NMI(Louvain 社区, 标题类别) 相对"类别标签随机置换"零分布的经验 p 值。

运行：
    cd pipeline/out/B6_community_semantics
    PYTHONIOENCODING=utf-8 python -X utf8 run_b6.py                  # 全量
    PYTHONIOENCODING=utf-8 python -X utf8 run_b6.py --stage classifier  # 只看分类器质量

口径全部来自 csa_common（CONVENTIONS §1/§3/§8）；本脚本不重新建图、不改清洗。
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE = os.path.dirname(os.path.dirname(HERE))  # out/<module> -> out -> pipeline
sys.path.insert(0, PIPELINE)
import csa_common as C  # noqa: E402

MODULE = "B6_community_semantics"
HERE = os.path.dirname(os.path.abspath(__file__))
SEED = C.SEED  # 20261113

# ==============================================================================
# 1. 关键词规则表（10 类，全部写死在这里，可直接阅读）
# ------------------------------------------------------------------------------
# 每类两组词：core（领域专名，命中记 3 分）、weak（泛指/边缘词，命中记 1 分）。
# 得分最高的类胜出；平分时按 rank（小者优先）。
# 以 "re:" 开头的是正则（大小写不敏感，作用在整条 title 上）；其余按子串匹配。
# 词表是在通读全部 671 条 title 之后写定的，写表时未看过任何社区统计量。
# ==============================================================================

RULES_RAW = {
    "个人主页": dict(rank=1, core=[
        "个人主页", "教师个人主页", "教授主页", "老师主页", "re:个人首页",
        "re:\\bhomepage\\b", "re:\\bprof\\.", "re:教授个人",
    ], weak=[
        "re:\\bDr\\.", "re:\\bAssociate Professor\\b", "re:\\bPhD\\b", "院士",
    ]),
    "图书馆": dict(rank=2, core=[
        "图书馆", "查收查引", "科技查新", "查新", "文献", "情报", "馆藏", "入馆",
        "参考咨询", "期刊中心", "杂志社", "数据库", "资源库", "知识管理", "方略",
        "re:\\blibrary\\b", "re:\\binformation commons\\b", "视频教育资源",
    ], weak=[
        "出版", "期刊", "检索", "知识", "资源", "视界", "培训库", "re:\\bopac\\b",
    ]),
    "招生": dict(rank=3, core=[
        "招生", "报考", "简章", "录取", "迎新", "网上报名", "历年分数", "生源",
        "考生", "推免", "re:\\badmission\\b",
    ], weak=[
        "报名", "面试", "EMBA", "MBA", "MPA", "英才班", "申请系统",
    ]),
    "新闻宣传": dict(rank=4, core=[
        "新闻", "宣传", "记者", "校报", "融媒", "新媒体", "专题网", "专题", "报道",
        "科大报", "re:\\bnews\\b",
    ], weak=[
        "传播", "科普", "博览", "公众科学日", "科技活动周", "展示", "影像",
    ]),
    "院系": dict(rank=5, core=[
        "学院", "学部", "re:学系", "re:系(?!统|结|列)", "re:\\bschool\\b",
        "re:\\bdepartment\\b", "re:\\bdivision\\b", "re:\\bfaculty\\b",
        "re:\\bcollege\\b", "re:\\bdept\\.",
    ], weak=[
        "系", "教研室", "学科",
    ]),
    "教务教学": dict(rank=6, core=[
        "教务", "本科生院", "教学", "课堂", "课程", "学籍", "学位", "成绩", "选课",
        "培养", "导师", "实验教学", "示范中心", "虚拟仿真", "考试", "答辩", "继续教育",
        "培训", "教师发展", "通识教育", "双学位", "辅修", "毕业设计", "re:\\bmooc",
    ], weak=[
        "本科生", "研究生", "学生", "教室", "教育", "学习", "体育", "外语", "英才班",
    ]),
    "附属单位": dict(rank=7, core=[
        "医院", "附中", "幼儿园", "小学", "实验学校", "出版社", "苏州高等研究院",
        "上海研究院", "先进技术研究院", "先研院", "高等研究院", "附属", "re:科大附",
    ], weak=[
        "校区", "集团", "招待所", "re:北京",
    ]),
    "科研平台": dict(rank=8, core=[
        "实验室", "研究中心", "研究所", "研究组", "课题组", "研究团队", "团队",
        "试验设施", "重点实验室", "工程研究中心", "国家工程实验室", "野外科学观测",
        "观测研究站", "工程实验室", "研究部", "仪器共享", "科研设施", "科研仪器",
        "计算中心", "研究院",
        "re:\\blab\\b", "re:\\blabs\\b", "re:\\blaborator", "re:\\bgroup\\b",
        "re:\\bresearch\\b", "re:\\binstitute\\b", "re:\\bcenter for\\b",
    ], weak=[
        "科研", "研究", "中心", "前沿", "工程", "基地", "专项", "重点", "试验",
        "科学", "re:lab\\b", "re:lab(?=[\\-@|_:/.])", "re:group", "re:team",
        "技术", "开发",
    ]),
    "管理服务": dict(rank=9, core=[
        "办公室", "管理系统", "信息系统", "人事", "财务", "采购", "资产", "后勤",
        "审计", "纪律", "监察", "巡察", "党建", "党委", "工会", "离退休", "保卫",
        "保密", "邮件", "网络信息", "一卡通", "缴费", "档案", "签署", "预约", "会议",
        "招聘", "就业", "校友", "基金会", "捐赠", "信息公开", "招标", "印章",
        "网上办理", "服务大厅", "注册中心", "信息化", "运维", "正版化", "基建",
        "基本建设", "综合管理", "学工", "团学", "资助", "心理健康", "物业", "饮食",
        "车辆", "节能", "校舍", "搜索", "站群", "仪器设备",
        "re:\\bOA\\b", "re:\\bmail\\b", "re:\\bgitlab\\b",
    ], weak=[
        "系统", "平台", "管理", "服务", "门户", "登录", "网站", "办公", "监督",
        "建设", "信息", "网络", "大厅", "综合", "re:\\bportal\\b",
    ]),
    # 其他：不设关键词，纯兜底
    "其他": dict(rank=10, core=[], weak=[]),
}

CATS = list(RULES_RAW.keys())
# 语义大类合并（类别数敏感性用）：把 10 类折叠成 4 / 6 类
COLLAPSE6 = {"招生": "教学培养", "教务教学": "教学培养", "院系": "院系组织",
             "科研平台": "科研单元", "图书馆": "支撑服务", "管理服务": "支撑服务",
             "附属单位": "附属单位", "个人主页": "个人/其他", "新闻宣传": "个人/其他",
             "其他": "个人/其他"}
COLLAPSE4 = {"招生": "教学培养", "教务教学": "教学培养", "院系": "院系组织",
             "科研平台": "科研单元", "图书馆": "支撑服务", "管理服务": "支撑服务",
             "附属单位": "支撑服务", "个人主页": "其他", "新闻宣传": "其他",
             "其他": "其他"}

N_PERM_MAIN = 1000   # 类别标签置换次数（主判据）
N_PERM_PUR = 1000    # 每社区纯度的置换次数
N_PERM_ALT = 500     # 替代解释置换次数
N_SEEDS = 24         # Louvain 复核 seed 数（>=20）

# ---------------------------------------------------------------------------
# title 信息量标记：抓取失败/HTTP 错误/robots 的标题不含任何语义，
# 通用导航标题（登录/首页）与"标题=子域回显"同样不构成组织语义证据。
# ---------------------------------------------------------------------------
PAT_FAIL = re.compile(
    r"^\s*[\[(（]?\s*(请求失败|连接失败|需要身份认证|robots\.txt|HTTP\s*\d"
    r"|ConnectTimeout|ReadTimeout|ConnectionError|SSL|Timeout)", re.I)
PAT_NAV = re.compile(
    r"^\s*(登录|登入页面|首页|默认站点|机构简介|跳转中|Loading|Home\b|Welcome\b"
    r"|IIS7|paper|sum|404)\s*$", re.I)


def compile_rules():
    out = {}
    for cat, d in RULES_RAW.items():
        for kind in ("core", "weak"):
            out[(cat, kind)] = [(kw, re.compile(kw[3:], re.I) if kw.lower().startswith("re:") else None)
                                for kw in d[kind]]
    return out


CRULES = compile_rules()


def score_all(title):
    """{cat: (score, [命中词])}"""
    t = str(title)
    tl = t.lower()
    res = {}
    for cat in CATS:
        hits, sc = [], 0
        for kind, w in (("core", 3), ("weak", 1)):
            for raw, rx in CRULES[(cat, kind)]:
                if (rx.search(t) if rx is not None else raw.lower() in tl):
                    hits.append(raw)
                    sc += w
        res[cat] = (sc, hits)
    return res


def title_kind(title, sub, subdomain):
    t = str(title).strip()
    if PAT_FAIL.search(t):
        return "fail_placeholder"
    if PAT_NAV.search(t):
        return "generic_nav"
    low = t.lower()
    if len(low) <= 24 and low in {str(sub).lower(), str(subdomain).lower(),
                                  str(sub).lower().replace(".", "")}:
        return "subdomain_echo"
    return "informative"


def classify_series(titles, subs, subdomains):
    """逐节点打分分类 → DataFrame（类别 / 命中词 / 置信标记）。"""
    rows = []
    for t, s, sd in zip(titles, subs, subdomains):
        sc = score_all(t)
        ranked = sorted(((sc[c][0], RULES_RAW[c]["rank"], c) for c in CATS),
                        key=lambda x: (-x[0], x[1]))
        best_score, _, best = ranked[0]
        pos = [r for r in ranked if r[0] > 0]
        second = pos[1][0] if len(pos) > 1 else 0
        if best_score == 0:
            best = "其他"          # 没有任何关键词命中 → 兜底类
            conf = "no_hit"
        elif len(pos) == 1:
            conf = "solo"
        elif best_score - second >= 3:
            conf = "clear"
        else:
            conf = "competing"
        rows.append({
            "title_raw": t, "category": best, "score": best_score,
            "runner_up_score": second, "margin": best_score - second,
            "matched_keywords": "|".join(sc[best][1]),
            "matched_categories": "|".join(f"{c}:{sc[c][0]}" for c in CATS if sc[c][0] > 0),
            "confidence": conf, "title_kind": title_kind(t, s, sd),
        })
    return pd.DataFrame(rows)


def classify_subset(rules_subset, titles):
    """只用给定子规则表分类（不重叠的两套），平分时同样按 rank。"""
    comp = {}
    for cat, kws in rules_subset.items():
        comp[cat] = [(k, re.compile(k[3:], re.I) if k.lower().startswith("re:") else None) for k in kws]
    out = []
    for t in titles:
        ts = str(t)
        tl = ts.lower()
        scores = {}
        for cat, kws in comp.items():
            sc = 0
            for raw, rx in kws:
                if (rx.search(ts) if rx is not None else raw.lower() in tl):
                    sc += 3 if raw in RULES_RAW[cat]["core"] else 1
            scores[cat] = sc
        ranked = sorted(((scores[c], RULES_RAW[c]["rank"], c) for c in scores),
                        key=lambda x: (-x[0], x[1]))
        out.append(ranked[0][2] if ranked[0][0] > 0 else "其他")
    return pd.Series(out, index=range(len(titles)))


def split_rules():
    """A/B 两套严格不重叠的关键词表。
    每类的 core 词表和 weak 词表各自按索引奇偶分半，两套各拿一半：
    A = core[0::2] + weak[0::2]，B = core[1::2] + weak[1::2]。
    交集为空，并集 = 全表，且两半在 core/weak 配比上平衡。
    """
    A, B = {}, {}
    for cat, d in RULES_RAW.items():
        A[cat] = list(d["core"][0::2]) + list(d["weak"][0::2])
        B[cat] = list(d["core"][1::2]) + list(d["weak"][1::2])
    return A, B


def cohens_kappa(a, b):
    labels = sorted(set(a) | set(b))
    idx = {l: i for i, l in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b):
        M[idx[x], idx[y]] += 1
    n = M.sum()
    po = np.trace(M) / n
    pe = float((M.sum(0) * M.sum(1)).sum() / n ** 2)
    return (float((po - pe) / (1 - pe)) if pe < 1 else float("nan"), float(po), float(pe))


# --------------------------------------------------------------- 节点特征（替代解释）

def attach_flags(df):
    ru = df["root_url"].fillna("").astype(str).str.lower()
    df["protocol"] = np.where(ru.str.startswith("https"), "https",
                              np.where(ru.str.startswith("http"), "http", "other"))
    df["sub_depth"] = df["sub"].fillna("").astype(str).str.count(r"\.") + 1
    df["is_en_sub"] = df["sub"].astype(str).str.match(r"^en[._]|.*[._]en$|^en$")
    ascii_only = df["title"].astype(str).str.match(r"^[\x00-\x7F]+$")
    df["title_ascii_only"] = ascii_only
    df["title_en_flag"] = df["is_en_sub"] | (ascii_only & (df["title_kind"] != "fail_placeholder"))
    st = df["status_codes"].fillna("").astype(str)
    df["status_family"] = np.select(
        [st.str.contains(r"(?:^|[\s,])-2:", regex=True),
         st.str.contains(r"(?:^|[\s,])-1:", regex=True),
         st.str.contains(r"(?:^|[\s,])403:", regex=True),
         st.str.contains(r"(?:^|[\s,])30[12]:", regex=True),
         st.str.contains(r"(?:^|[\s,])404:", regex=True),
         st.str.contains(r"(?:^|[\s,])5\d\d:", regex=True),
         st.str.contains(r"(?:^|[\s,])0:", regex=True)],
        ["robots_blocked", "auth_or_skip", "acl_403", "redirect", "notfound_404",
         "server_5xx", "conn_fail_0"], default="only_200")
    df["single_url_site"] = df["url_count"] <= 1
    df["url_capped"] = df["url_count"] >= 100
    tt = df["title"].astype(str).str.strip().str.lower()
    dup = tt.value_counts()
    dup = set(dup[dup > 1].index)
    df["title_dup_group"] = np.where(tt.isin(dup), "dup:" + tt, "unique_title")
    return df


# --------------------------------------------------------------- 统计工具

def emp_p_two(vals, obs, include_obs=False):
    """CONVENTIONS §4：双侧经验 p 按"偏离 null mean 的绝对值"计。

    include_obs=True → 保守 p = (1 + #{|T* - m| >= |obs - m|}) / (N+1)，
    观测值本身并入样本（这样 p 永远 > 0，可避免"p=0"的过度宣称）。
    """
    v = np.asarray(vals, float)
    if include_obs:
        allv = np.concatenate([v, [float(obs)]])
        m = allv.mean()
        thr = abs(float(obs) - m) - 1e-12
        return float((np.abs(allv - m) >= thr).sum() / (len(v) + 1))
    m = v.mean()
    return float((np.abs(v - m) >= abs(float(obs) - m) - 1e-12).mean())


def cat_assort(codes_u, codes_v, K, weights=None):
    """Newman (2002/2003) 离散类型 assortativity。
    输入：每条边两端的类别编码向量（同长）、类别数 K、可选权重。
    返回 (总体 r, 每类 r_j, 归一化 e_jk, 边缘 a_j)。
    e_jk 按无向边计两次（对称），行和 a_j = 端点落在 j 类的边的份额。"""
    e = np.zeros((K, K))
    w = np.ones(len(codes_u)) if weights is None else np.asarray(weights, float)
    np.add.at(e, (codes_u, codes_v), w)
    np.add.at(e, (codes_v, codes_u), w)
    tot = e.sum()
    if tot == 0:
        return float("nan"), np.full(K, np.nan), e, np.zeros(K)
    e = e / tot
    a = e.sum(1)
    s2 = float((a ** 2).sum())
    denom = 1 - s2
    r_all = float((np.trace(e) - s2) / denom) if denom > 0 else float("nan")
    r_each = np.full(K, np.nan)
    for j in range(K):
        d = a[j] * (1 - a[j])
        if d > 0:
            r_each[j] = (e[j, j] - a[j] ** 2) / d
    return r_all, r_each, e, a


def bh_adjust(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out


def partitions_from_labels(df, col, node_col="node"):
    return [set(g[node_col]) for _, g in df.groupby(col) if len(g)]


# --------------------------------------------------------------- 主流程

def main(stage="all"):
    t0 = time.time()
    C.assert_anchors()
    b = C.build_all()
    nodes = b["nodes"].copy()
    assert len(nodes) == C.ANCHORS["n_attributed"]

    # ---------------- 1. 标题分类 ----------------
    cls = classify_series(nodes["title"], nodes["sub"], nodes["subdomain"])
    labels_df = pd.concat([nodes[["node", "sub", "subdomain", "title"]].reset_index(drop=True),
                          cls.reset_index(drop=True).drop(columns=["title_raw"])], axis=1)
    keep_cols = ["node", "sub", "subdomain", "title", "category", "matched_keywords",
                 "matched_categories", "score", "runner_up_score", "margin",
                 "confidence", "title_kind"]
    labels_df = labels_df[keep_cols]
    nodes = nodes.reset_index(drop=True)
    nodes["category"] = cls["category"].values
    nodes["confidence"] = cls["confidence"].values
    nodes["title_kind"] = cls["title_kind"].values
    nodes = attach_flags(nodes)

    # ---------------- 2. 分类器自评 ----------------
    A, B = split_rules()
    labA = classify_subset(A, nodes["title"])
    labB = classify_subset(B, nodes["title"])
    labCore = classify_subset({c: RULES_RAW[c]["core"] for c in CATS}, nodes["title"])
    labWeak = classify_subset({c: RULES_RAW[c]["weak"] for c in CATS}, nodes["title"])
    Ts = nodes["title"].astype(str)
    cut = ((Ts.str.len() + 1) // 2).tolist()
    front = pd.Series([t[:h] for t, h in zip(Ts, cut)])
    back = pd.Series([t[h:] for t, h in zip(Ts, cut)])
    labFront = classify_subset(ALL_KW := {c: RULES_RAW[c]["core"] + RULES_RAW[c]["weak"]
                                          for c in CATS}, front)
    labBack = classify_subset(ALL_KW, back)
    full_lab = nodes["category"].reset_index(drop=True)

    kappa_rows = []
    for name, x, y in [("split_half_keywords", labA, labB),
                       ("core_vs_weak", labCore, labWeak),
                       ("title_front_vs_back", labFront, labBack)]:
        k_all, po, pe = cohens_kappa(x, y)
        msk = (nodes["title_kind"] == "informative").values
        k_inf, po_i, _ = cohens_kappa(x[msk], y[msk])
        # 主标签与半套标签各自的一致性（更贴切的"规则内部一致性"）
        kA, _, _ = cohens_kappa(x, full_lab)
        kB, _, _ = cohens_kappa(y, full_lab)
        kappa_rows.append({"split": name, "kappa_all": round(k_all, 4),
                           "po_all": round(po, 4), "pe_all": round(pe, 4),
                           "kappa_informative_only": round(k_inf, 4),
                           "po_informative": round(po_i, 4),
                           "n_informative": int(msk.sum()),
                           "kappa_vs_main_label": round(float(np.mean([kA, kB])), 4)})
    kap = pd.DataFrame(kappa_rows)

    qual = (nodes.groupby("category").size().rename("n_nodes").reset_index())
    qual["share"] = (qual["n_nodes"] / len(nodes)).round(4)
    kind_xt = pd.crosstab(nodes["category"], nodes["title_kind"])
    for k in ["fail_placeholder", "generic_nav", "subdomain_echo", "informative"]:
        qual[f"kind_{k}"] = qual["category"].map(kind_xt[k] if k in kind_xt.columns else 0)
    conf_xt = pd.crosstab(nodes["category"], nodes["confidence"])
    for c in ["solo", "clear", "competing", "no_hit"]:
        qual[f"conf_{c}"] = qual["category"].map(conf_xt[c] if c in conf_xt.columns else 0)
    qual = qual.sort_values("n_nodes", ascending=False).reset_index(drop=True)
    qual.insert(0, "row_type", "category")
    kappa_out = kap.assign(row_type="kappa", n_nodes="", share="", )
    classifier_quality = pd.concat([qual, kappa_out], axis=0, ignore_index=True)

    if stage == "classifier":
        C.save_csv(labels_df, MODULE, "title_labels.csv")
        C.save_csv(classifier_quality, MODULE, "classifier_quality.csv")
        print(cls["category"].value_counts().to_string())
        print("\n其他 share =", round((cls["category"] == "其他").mean(), 4))
        print("\ntitle_kind:\n", cls["title_kind"].value_counts().to_string())
        print("\n其他 内部 kind:\n",
              cls.loc[cls["category"] == "其他", "title_kind"].value_counts().to_string())
        print("\nkappa:\n", kap.to_string(index=False))
        print("\nconf x kind:\n", pd.crosstab(cls["confidence"], cls["title_kind"]).to_string())
        return None

    # ---------------- 3. 社区 × 类别 ----------------
    lou42 = C.louvain(b["Gu"], seed=42)
    nodes["comm42"] = C.to_labels(lou42["partition"], nodes["node"])
    robust = C.louvain_robust(b["Gu"], seeds=list(range(1, N_SEEDS + 1)))
    G671 = C.induced(b["Gu"], nodes["node"])           # 671 主分析节点
    edge_pairs = list(G671.edges())

    comms42 = lou42["partition"]
    cat_comms = partitions_from_labels(nodes, "category")
    base = C.partition_metrics(b["Gu"], comms42, cat_comms)

    # 列联表
    ct = pd.crosstab(nodes["comm42"], nodes["category"])
    ct = ct.reindex(columns=[c for c in CATS if c in ct.columns])
    ct.index.name = "community_seed42"
    ct_out = ct.reset_index()
    ct_out.insert(1, "comm_size", ct.sum(1).values)

    # 纯度（向量化：community × category 列联矩阵 → 每行最大值 / 行和）
    rng = np.random.default_rng(SEED)
    lab_arr = nodes["category"].to_numpy()
    comm_arr = nodes["comm42"].to_numpy()
    comm_levels = np.unique(comm_arr)
    ci = pd.Categorical(pd.Series(comm_arr), categories=comm_levels).codes
    kl = pd.Categorical(pd.Series(lab_arr), categories=[c for c in CATS if c in set(lab_arr)]).codes
    NC, NK = len(comm_levels), kl.max() + 1

    def cont_of(kcodes):
        M = np.zeros((NC, NK))
        np.add.at(M, (ci, kcodes), 1.0)
        return M

    def purity_of(kcodes):
        M = cont_of(kcodes)
        with np.errstate(invalid="ignore", divide="ignore"):
            pur = np.where(M.sum(1) > 0, M.max(1) / M.sum(1), np.nan)
        return pur, M

    pur, gtab = purity_of(kl)
    null_pur = np.empty((N_PERM_PUR, NC))
    for i in range(N_PERM_PUR):
        # 直接重排类别编码向量 = 重排类别标签（编码是双射），保持社区尺寸不变
        null_pur[i], _ = purity_of(kl[rng.permutation(len(kl))])
    null_mean_pur = null_pur.mean(0)
    null_sd_pur = null_pur.std(0, ddof=1)
    null_p_pur = np.array([emp_p_two(null_pur[:, j], pur[j]) if np.isfinite(pur[j]) else np.nan
                           for j in range(NC)])
    null_p_pur_obs = np.array([emp_p_two(null_pur[:, j], pur[j], include_obs=True)
                               if np.isfinite(pur[j]) else np.nan for j in range(NC)])

    comm_size = np.bincount(ci, minlength=NC).astype(float)
    max_count = gtab.max(1)
    max_kidx = gtab.argmax(1)
    cat_names = [c for c in CATS if c in set(lab_arr)]
    purity_df = pd.DataFrame({
        "community_seed42": comm_levels, "comm_size": comm_size.astype(int),
        "max_category": [cat_names[k] for k in max_kidx],
        "max_count": max_count.astype(int), "purity": np.round(pur, 4),
        "n_categories_present": (gtab > 0).sum(1),
        "null_purity_mean": np.round(null_mean_pur, 4),
        "null_purity_sd": np.round(null_sd_pur, 4),
        "purity_lift": np.round(pur - null_mean_pur, 4),
        "purity_z": np.round((pur - null_mean_pur) / np.where(null_sd_pur > 0, null_sd_pur, np.nan), 3),
        "purity_perm_p": np.round(null_p_pur, 5),
        "purity_perm_p_incl_obs": np.round(null_p_pur_obs, 5),
    })
    purity_df["purity_bh_p"] = np.round(bh_adjust(purity_df["purity_perm_p"].tolist()), 5)
    purity_df = purity_df.sort_values("comm_size", ascending=False).reset_index(drop=True)
    purity_df["is_singleton"] = purity_df["comm_size"] <= 1
    ns = ~purity_df["is_singleton"].to_numpy()
    size_w = purity_df["comm_size"] / purity_df["comm_size"].sum()
    purity_wt = float((purity_df["purity"] * size_w).sum())
    purity_wt_ns = float((purity_df.loc[ns, "purity"] *
                          (purity_df.loc[ns, "comm_size"] / purity_df.loc[ns, "comm_size"].sum())).sum())
    null_wt = float((null_mean_pur * comm_size / comm_size.sum()).sum())
    null_wt_ns = float((null_mean_pur[ns] * comm_size[ns] / comm_size[ns].sum()).sum())
    # 规模加权纯度的置换 p（把每社区的 null 纯度按同一置换重算，保留社区尺寸结构）
    null_wt_draws = np.array([(null_pur[i] * comm_size / comm_size.sum()).sum()
                              for i in range(N_PERM_PUR)])
    purity_p_wt = emp_p_two(null_wt_draws, purity_wt)
    purity_p_wt_ns = emp_p_two(
        np.array([(null_pur[i][ns] * comm_size[ns] / comm_size[ns].sum()).sum()
                  for i in range(N_PERM_PUR)]), purity_wt_ns)

    # 置换零分布 of NMI（本单元核心数字）：seed42 为主，另 24 个 seed 复核。
    # 社区标签向量按 seed 预先缓存，置换只重排类别向量 → 每次置换仅一次 NMI 调用。
    def nmi_pair(comm_vec, labels_vec):
        return float(normalized_mutual_info_score(comm_vec, labels_vec))

    seed_list = [42] + list(range(1, N_SEEDS + 1))
    runs, comm_vecs = {}, {}
    for sd_ in seed_list:
        lr = lou42 if sd_ == 42 else C.louvain(b["Gu"], seed=sd_)
        runs[sd_] = lr
        comm_vecs[sd_] = np.asarray(C.to_labels(lr["partition"], nodes["node"]))
        assert (comm_vecs[sd_] >= 0).all(), f"seed {sd_}: 有未覆盖节点"

    cv42 = comm_vecs[42]
    assert (cv42 == nodes["comm42"].to_numpy()).all()
    obs_nmi42 = nmi_pair(cv42, lab_arr)
    null_nmi42 = np.empty(N_PERM_MAIN)
    for i in range(N_PERM_MAIN):
        null_nmi42[i] = nmi_pair(cv42, lab_arr[rng.permutation(len(lab_arr))])

    perm_rows = []
    for sd_ in seed_list:
        meta, cv = runs[sd_], comm_vecs[sd_]
        obs = nmi_pair(cv, lab_arr)
        nv = np.empty(N_PERM_MAIN)
        for i in range(N_PERM_MAIN):
            nv[i] = nmi_pair(cv, lab_arr[rng.permutation(len(lab_arr))])
        perm_rows.append({
            "seed": sd_,
            "n_comms_total": meta["n_comms_total"],
            "n_comms_nonsingleton": meta["n_comms_nonsingleton"],
            "n_singleton": meta["n_singleton"],
            "largest": meta["largest"],
            "Q": round(meta["Q"], 4),
            "NMI_obs": round(obs, 4), "null_NMI_mean": round(float(nv.mean()), 4),
            "null_NMI_sd": round(float(nv.std(ddof=1)), 4),
            "null_NMI_q95": round(float(np.quantile(nv, 0.95)), 4),
            "z": round(float((obs - nv.mean()) / nv.std(ddof=1)), 3),
            "p_emp_two_sided": round(emp_p_two(nv, obs), 5),
            "p_emp_incl_obs": round(emp_p_two(nv, obs, include_obs=True), 5),
            "ARI_obs": round(float(adjusted_rand_score(cv, lab_arr)), 4),
            "n_perm": N_PERM_MAIN,
        })
    nmi_perm = pd.DataFrame(perm_rows)

    # 类别数敏感性：10 → 6 → 4 类 + 仅 informative 子集（全部报告，不挑口径）
    sens_rows = [{"scheme": "10类(主口径)", "n_categories": nodes["category"].nunique(),
                  "n_nodes": len(nodes),
                  "other_share": round(float((lab_arr == "其他").mean()), 4),
                  "NMI_obs": round(obs_nmi42, 4),
                  "null_mean": round(float(null_nmi42.mean()), 4),
                  "null_sd": round(float(null_nmi42.std(ddof=1)), 4),
                  "z": round(float((obs_nmi42 - null_nmi42.mean()) / null_nmi42.std(ddof=1)), 3),
                  "p_emp": round(emp_p_two(null_nmi42, obs_nmi42), 5),
                  "p_emp_incl_obs": round(emp_p_two(null_nmi42, obs_nmi42, include_obs=True), 5),
                  "ARI_obs": base["ARI"]}]
    rng2 = np.random.default_rng(SEED + 1)
    for name, col in [("6类合并", nodes["category"].map(COLLAPSE6)),
                      ("4类合并", nodes["category"].map(COLLAPSE4))]:
        arr = col.to_numpy()
        obs = nmi_pair(cv42, arr)
        nv = np.array([nmi_pair(cv42, arr[rng2.permutation(len(arr))])
                       for _ in range(N_PERM_MAIN)])
        sens_rows.append({"scheme": name, "n_categories": int(pd.Series(arr).nunique()),
                          "n_nodes": len(arr),
                          "other_share": round(float((pd.Series(arr) == "其他").mean()), 4),
                          "NMI_obs": round(float(obs), 4),
                          "null_mean": round(float(nv.mean()), 4),
                          "null_sd": round(float(nv.std(ddof=1)), 4),
                          "z": round(float((obs - nv.mean()) / nv.std(ddof=1)), 3),
                          "p_emp": round(emp_p_two(nv, obs), 5),
                          "p_emp_incl_obs": round(emp_p_two(nv, obs, include_obs=True), 5),
                          "ARI_obs": round(float(C.partition_metrics(
                              b["Gu"], comms42, partitions_from_labels(
                                  nodes.assign(cat6=col), "cat6"))["ARI"]), 4)})
    inf = nodes[nodes["title_kind"] == "informative"].reset_index(drop=True)
    arr = inf["category"].to_numpy()
    part_inf = [set(c & set(inf["node"])) for c in comms42]
    part_inf = [c for c in part_inf if c]
    comm_inf = np.asarray(C.to_labels(part_inf, inf["node"]))
    obs = nmi_pair(comm_inf, arr)
    nv = np.array([nmi_pair(comm_inf, arr[rng2.permutation(len(arr))])
                   for _ in range(N_PERM_MAIN)])
    sens_rows.append({"scheme": "10类·仅 informative 标题子集", "n_categories": int(len(set(arr))),
                      "n_nodes": len(arr),
                      "other_share": round(float((arr == "其他").mean()), 4),
                      "NMI_obs": round(float(obs), 4),
                      "null_mean": round(float(nv.mean()), 4),
                      "null_sd": round(float(nv.std(ddof=1)), 4),
                      "z": round(float((obs - nv.mean()) / nv.std(ddof=1)), 3),
                      "p_emp": round(emp_p_two(nv, obs), 5),
                      "p_emp_incl_obs": round(emp_p_two(nv, obs, include_obs=True), 5),
                      "ARI_obs": round(float(C.partition_metrics(
                          G671, [set(g["node"]) for _, g in inf.groupby("category")],
                          part_inf)["ARI"]), 4)})
    sensitivity = pd.DataFrame(sens_rows)

    # ---------------- 4. 属性同配性 ----------------
    cat_list = [c for c in CATS if c in set(nodes["category"])]
    code = {c: i for i, c in enumerate(cat_list)}
    Kc = len(cat_list)
    codes = nodes["category"].map(code).to_numpy()
    node_pos = {n: i for i, n in enumerate(nodes["node"])}
    eu = np.array([node_pos[u] for u, v in edge_pairs], dtype=int)
    ev = np.array([node_pos[v] for u, v in edge_pairs], dtype=int)
    ew = np.array([G671[u][v]["weight"] for u, v in edge_pairs], dtype=float)
    assert len(eu) == G671.number_of_edges()
    cu, cv_ = codes[eu], codes[ev]          # 每条边两端的类别编码

    r_all, r_each, emat, a = cat_assort(cu, cv_, Kc)
    r_all_w, r_each_w, emat_w, a_w = cat_assort(cu, cv_, Kc, weights=ew)

    # 置换零模型：类别规模不变的跨类重排（= "同类内部重排"的唯一有信息量版本，见下）
    null_all = np.empty(N_PERM_MAIN)
    null_each = np.empty((N_PERM_MAIN, Kc))
    null_all_w = np.empty(N_PERM_MAIN)
    null_each_w = np.empty((N_PERM_MAIN, Kc))
    for i in range(N_PERM_MAIN):
        cp = codes[rng.permutation(len(codes))]
        ra, re_, _, _ = cat_assort(cp[eu], cp[ev], Kc)
        raw, rew, _, _ = cat_assort(cp[eu], cp[ev], Kc, weights=ew)
        null_all[i], null_each[i] = ra, re_
        null_all_w[i], null_each_w[i] = raw, rew

    # 严格"只在同类内部重排"对 (e_jk, r) 是恒等变换（不改变任何一条边的端点类别），
    # 因此它只能当 sanity check，不能当零模型 —— 两个数都记录下来。
    grp_idx = {c: np.where(codes == code[c])[0] for c in cat_list}
    within_r = []
    for _ in range(20):
        nc = codes.copy()
        for c, ix in grp_idx.items():
            nc[rng.permutation(ix)] = nc[ix]     # 置换位置、值仍是同类编码 → 集合不变
        within_r.append(cat_assort(nc[eu], nc[ev], Kc)[0])
    id_check = r_all
    within_perm_r = float(np.mean(within_r))
    assert abs(within_perm_r - id_check) < 1e-12, "同类内部重排应给出同一 r（恒等变换）"

    H = G671.copy()
    nx.set_node_attributes(H, {n: int(codes[node_pos[n]]) for n in H.nodes()}, "cat_code")
    r_nx_pearson = float(nx.attribute_assortativity_coefficient(H, "cat_code"))

    rows = []
    for j, c in enumerate(cat_list):
        nv, nvw = null_each[:, j], null_each_w[:, j]
        obs, obs_w = r_each[j], r_each_w[j]
        rows.append({"category": c, "n_nodes": int((codes == j).sum()),
                     "q_j_nodes": round(float((codes == j).mean()), 4),
                     "a_j_edge_share": round(float(a[j]), 4),
                     "a_j_edge_share_weighted": round(float(a_w[j]), 4),
                     "e_jj": round(float(emat[j, j]), 5),
                     "e_jj_minus_qj_qk": round(float(emat[j, j] - a[j] ** 2), 5),
                     "same_class_share_of_its_edges": round(float(emat[j, j] / a[j]) if a[j] else np.nan, 4),
                     "assort_r_j": round(float(obs), 4),
                     "null_mean": round(float(nv.mean()), 4),
                     "null_sd": round(float(nv.std(ddof=1)), 4),
                     "z": round(float((obs - nv.mean()) / nv.std(ddof=1)), 3) if nv.std(ddof=1) > 0 else np.nan,
                     "p_raw": round(emp_p_two(nv, obs), 5),
                     "p_raw_incl_obs": round(emp_p_two(nv, obs, include_obs=True), 5),
                     "assort_r_j_weighted": round(float(obs_w), 4),
                     "null_mean_weighted": round(float(nvw.mean()), 4),
                     "null_sd_weighted": round(float(nvw.std(ddof=1)), 4),
                     "z_weighted": round(float((obs_w - nvw.mean()) / nvw.std(ddof=1)), 3) if nvw.std(ddof=1) > 0 else np.nan,
                     "p_raw_weighted": round(emp_p_two(nvw, obs_w), 5)})
    assort = pd.DataFrame(rows)
    # BH 校正：每类一个假设（10 个）；加权口径单独一批，不混在一起校正
    assort["p_bh"] = np.round(bh_adjust(assort["p_raw"].tolist()), 5)
    assort["q_bh_0.05"] = assort["p_bh"] < 0.05
    assort["p_bh_weighted"] = np.round(bh_adjust(assort["p_raw_weighted"].tolist()), 5)
    assort["q_bh_0.05_weighted"] = assort["p_bh_weighted"] < 0.05
    overall = {"category": "__ALL__(Newman 离散总体 r)", "n_nodes": len(nodes),
               "q_j_nodes": 1.0, "a_j_edge_share": 1.0, "a_j_edge_share_weighted": 1.0,
               "e_jj": round(float(np.trace(emat)), 4),
               "e_jj_minus_qj_qk": round(float(np.trace(emat) - (a ** 2).sum()), 5),
               "same_class_share_of_its_edges": round(float(np.trace(emat)), 4),
               "assort_r_j": round(float(r_all), 4),
               "null_mean": round(float(null_all.mean()), 4),
               "null_sd": round(float(null_all.std(ddof=1)), 4),
               "z": round(float((r_all - null_all.mean()) / null_all.std(ddof=1)), 3),
               "p_raw": round(emp_p_two(null_all, r_all), 5),
               "p_raw_incl_obs": round(emp_p_two(null_all, r_all, include_obs=True), 5),
               "assort_r_j_weighted": round(float(r_all_w), 4),
               "null_mean_weighted": round(float(null_all_w.mean()), 4),
               "null_sd_weighted": round(float(null_all_w.std(ddof=1)), 4),
               "z_weighted": round(float((r_all_w - null_all_w.mean()) / null_all_w.std(ddof=1)), 3),
               "p_raw_weighted": round(emp_p_two(null_all_w, r_all_w), 5),
               "p_bh": "", "q_bh_0.05": "", "p_bh_weighted": "", "q_bh_0.05_weighted": ""}
    assort_out = pd.concat([assort, pd.DataFrame([overall])], ignore_index=True)

    e_df = pd.DataFrame(emat, index=cat_list, columns=cat_list)
    e_df.index.name = "e_jk_from\\to"
    e_df.to_csv(os.path.join(C.module_dir(MODULE), "mixing_matrix.csv"), encoding="utf-8-sig")
    ew_df = pd.DataFrame(emat_w, index=cat_list, columns=cat_list)
    ew_df.index.name = "e_jk_weighted_from\\to"
    ew_df.to_csv(os.path.join(C.module_dir(MODULE), "mixing_matrix_weighted.csv"),
                 encoding="utf-8-sig")

    # ---------------- 5. 替代解释 ----------------
    alt_specs = [
        ("protocol(https/http)", "protocol"),
        ("sub 前缀层级(标签数)", "sub_depth"),
        ("是否英文站(en./纯ASCII标题)", "title_en_flag"),
        ("标题信息量类型", "title_kind"),
        ("同名标题组(模板/CMS 代理)", "title_dup_group"),
        ("status_codes 指纹", "status_family"),
        ("url_count<=1 单页站", "single_url_site"),
        ("url_count>=100 抓满", "url_capped"),
        ("health_state", "health_state"),
    ]
    alt_rows = []
    for name, col in alt_specs:
        v = nodes[col].astype(str).to_numpy()
        obs = nmi_pair(cv42, v)
        nv = np.array([nmi_pair(cv42, v[rng2.permutation(len(v))])
                       for _ in range(N_PERM_ALT)])
        pm = partitions_from_labels(nodes.assign(_g=v.astype(str)), "_g")
        ari = C.partition_metrics(b["Gu"], comms42, pm)["ARI"]
        alt_rows.append({"feature": name, "n_levels": int(pd.Series(v).nunique()),
                         "NMI_vs_comm_seed42": round(float(obs), 4),
                         "null_NMI_mean": round(float(nv.mean()), 4),
                         "null_NMI_sd": round(float(nv.std(ddof=1)), 4),
                         "null_NMI_q95": round(float(np.quantile(nv, 0.95)), 4),
                         "z": round(float((obs - nv.mean()) / nv.std(ddof=1)), 3),
                         "p_emp": round(emp_p_two(nv, obs), 5),
                         "p_emp_incl_obs": round(emp_p_two(nv, obs, include_obs=True), 5),
                         "ARI": round(float(ari), 4), "n_perm": N_PERM_ALT})
    alt_rows.append({"feature": "__baseline__: 标题类别(10类)", "n_levels": len(cat_list),
                     "NMI_vs_comm_seed42": round(obs_nmi42, 4),
                     "null_NMI_mean": round(float(null_nmi42.mean()), 4),
                     "null_NMI_sd": round(float(null_nmi42.std(ddof=1)), 4),
                     "null_NMI_q95": round(float(np.quantile(null_nmi42, 0.95)), 4),
                     "z": round(float((obs_nmi42 - null_nmi42.mean()) / null_nmi42.std(ddof=1)), 3),
                     "p_emp": round(emp_p_two(null_nmi42, obs_nmi42), 5),
                     "p_emp_incl_obs": round(emp_p_two(null_nmi42, obs_nmi42, include_obs=True), 5),
                     "ARI": float(base["ARI"]), "n_perm": N_PERM_MAIN})
    alt = pd.DataFrame(alt_rows).sort_values("NMI_vs_comm_seed42", ascending=False)
    alt.to_csv(os.path.join(C.module_dir(MODULE), "alternative_features.csv"),
               index=False, encoding="utf-8-sig")

    # ---------------- 存盘 ----------------
    labels_out = labels_df.merge(nodes[["node", "comm42"]], on="node", how="left")
    labels_out = labels_out.rename(columns={"comm42": "community_seed42"})
    labels_out["community_size_seed42"] = labels_out["community_seed42"].map(
        nodes.groupby("comm42").size())
    for nm, s in [("label_splitA", labA), ("label_splitB", labB),
                  ("label_core", labCore), ("label_weak", labWeak),
                  ("label_front", labFront), ("label_back", labBack)]:
        labels_out[nm] = s.values
    C.save_csv(labels_out, MODULE, "title_labels.csv")
    C.save_csv(classifier_quality, MODULE, "classifier_quality.csv")
    C.save_csv(ct_out, MODULE, "contingency.csv")
    C.save_csv(purity_df, MODULE, "purity_by_community.csv")
    C.save_csv(nmi_perm, MODULE, "nmi_permutation.csv")
    C.save_csv(sensitivity, MODULE, "category_scheme_sensitivity.csv")
    C.save_csv(assort_out, MODULE, "assortativity_by_category.csv")

    summary = {
        "anchors": "31/31 PASS",
        "louvain_seed42": {k: lou42[k] for k in
                           ["Q", "n_comms_total", "n_comms_nonsingleton", "n_singleton",
                            "n_covered", "largest"]},
        "louvain_robust": {k: robust[k] for k in
                           ["n_seeds", "comms_total", "comms_nonsingleton", "Q", "largest"]},
        "n_categories": len(cat_list),
        "category_sizes": nodes["category"].value_counts().to_dict(),
        "other_share": round(float((nodes["category"] == "其他").mean()), 4),
        "title_kind": nodes["title_kind"].value_counts().to_dict(),
        "kappa": kappa_rows,
        "NMI_obs_seed42": base,
        "NMI_obs_direct": round(obs_nmi42, 4),
        "NMI_null_mean": round(float(null_nmi42.mean()), 4),
        "NMI_null_sd": round(float(null_nmi42.std(ddof=1)), 4),
        "NMI_null_q95": round(float(np.quantile(null_nmi42, 0.95)), 4),
        "NMI_z": round(float((obs_nmi42 - null_nmi42.mean()) / null_nmi42.std(ddof=1)), 3),
        "NMI_p_emp": round(emp_p_two(null_nmi42, obs_nmi42), 5),
        "NMI_p_emp_incl_obs": round(emp_p_two(null_nmi42, obs_nmi42, include_obs=True), 5),
        "NMI_across_seeds": {"min": float(nmi_perm["NMI_obs"].min()),
                             "median": float(nmi_perm["NMI_obs"].median()),
                             "max": float(nmi_perm["NMI_obs"].max()),
                             "n_ge_095_null": int((nmi_perm["NMI_obs"] >=
                                                   nmi_perm["null_NMI_q95"]).sum()),
                             "p_min": float(nmi_perm["p_emp_two_sided"].min()),
                             "p_max": float(nmi_perm["p_emp_two_sided"].max())},
        "purity": {"mean_unweighted": round(float(np.nanmean(pur)), 4),
                   "median_unweighted": round(float(np.nanmedian(pur)), 4),
                   "min": round(float(np.nanmin(pur)), 4), "max": round(float(np.nanmax(pur)), 4),
                   "size_weighted": round(purity_wt, 4),
                   "size_weighted_nonsingleton": round(purity_wt_ns, 4),
                   "null_size_weighted_mean": round(null_wt, 4),
                   "null_size_weighted_nonsingleton_mean": round(null_wt_ns, 4),
                   "lift_size_weighted": round(purity_wt - null_wt, 4),
                   "p_size_weighted": round(purity_p_wt, 5),
                   "p_size_weighted_nonsingleton": round(purity_p_wt_ns, 5),
                   "n_comms_purity_ge_0_5": int((purity_df["purity"] >= 0.5).sum()),
                   "n_comms_nonsingleton": int(ns.sum())},
        "same_class_edge_share_overall": round(float(np.trace(emat)), 4),
        "assort_overall": overall,
        "assort_overall_weighted": {k: overall[k] for k in
                                    ["assort_r_j_weighted", "null_mean_weighted",
                                     "null_sd_weighted", "z_weighted", "p_raw_weighted"]},
        "assort_nx_pearson_on_codes": round(r_nx_pearson, 4),
        "assort_within_class_perm_is_identity": [round(id_check, 6), round(within_perm_r, 6)],
        "n_sig_bh_05": int(assort["q_bh_0.05"].sum()),
        "category_scheme_sensitivity": sensitivity.to_dict(orient="records"),
        "alternative_features_top": alt[["feature", "NMI_vs_comm_seed42", "z", "p_emp", "ARI"]]
                                        .head(4).to_dict(orient="records"),
        "n_perm_main": N_PERM_MAIN, "n_perm_purity": N_PERM_PUR, "n_perm_alt": N_PERM_ALT,
        "n_seeds": len(seed_list),
        "runtime_sec": round(time.time() - t0, 1),
        "versions": b["versions"],
    }
    C.save_json(summary, MODULE, "summary.json")

    lines = [
        "B6_community_semantics RUN log",
        f"command: PYTHONIOENCODING=utf-8 python -X utf8 run_b6.py --stage all",
        f"started/finished local: {time.strftime('%F %T')}",
        f"runtime_sec: {summary['runtime_sec']}",
        f"SEED={SEED}  PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')}",
        f"python={b['versions']['python']} networkx={b['versions']['networkx']} "
        f"pandas={b['versions']['pandas']} numpy={b['versions']['numpy']}",
        "anchors: C.assert_anchors() 31/31 PASS (no exception raised)",
        f"louvain seed42: comms={lou42['n_comms_total']} nonsingleton={lou42['n_comms_nonsingleton']}"
        f" Q={round(lou42['Q'],4)} largest={lou42['largest']} covered={lou42['n_covered']}",
        f"louvain robust({robust['n_seeds']} seeds 1..{N_SEEDS}): "
        f"comms {robust['comms_total']} nonsingleton {robust['comms_nonsingleton']} "
        f"Q {robust['Q']} largest {robust['largest']}",
        f"N_PERM_MAIN={N_PERM_MAIN} N_PERM_PUR={N_PERM_PUR} N_PERM_ALT={N_PERM_ALT} "
        f"n_louvain_runs={len(seed_list)}",
        f"NMI obs(seed42)={base['NMI']} ARI={base['ARI']} on n_compared={base['n_compared']}",
        f"NMI null mean={summary['NMI_null_mean']} sd={summary['NMI_null_sd']} "
        f"q95={summary['NMI_null_q95']} z={summary['NMI_z']} p={summary['NMI_p_emp']} "
        f"(p_incl_obs={summary['NMI_p_emp_incl_obs']})",
        f"NMI across {len(seed_list)} seeds: obs {summary['NMI_across_seeds']['min']}-"
        f"{summary['NMI_across_seeds']['max']} (median {summary['NMI_across_seeds']['median']}), "
        f"p {summary['NMI_across_seeds']['p_min']}-{summary['NMI_across_seeds']['p_max']}, "
        f"seeds above null q95 = {summary['NMI_across_seeds']['n_ge_095_null']}",
        f"purity size_weighted={summary['purity']['size_weighted']} "
        f"null={summary['purity']['null_size_weighted_mean']} "
        f"lift={summary['purity']['lift_size_weighted']} p={summary['purity']['p_size_weighted']}",
        f"purity nonsingleton size_weighted={summary['purity']['size_weighted_nonsingleton']} "
        f"null={summary['purity']['null_size_weighted_nonsingleton_mean']} "
        f"p={summary['purity']['p_size_weighted_nonsingleton']}",
        f"assort overall r={overall['assort_r_j']} p={overall['p_raw']} "
        f"(weighted r={overall['assort_r_j_weighted']} p={overall['p_raw_weighted']}; "
        f"nx Pearson on int codes={summary['assort_nx_pearson_on_codes']})",
        f"assort same-class edge share={summary['same_class_edge_share_overall']}; "
        f"BH-significant classes={summary['n_sig_bh_05']}/{Kc}",
        f"within-class rearrangement is identity: {summary['assort_within_class_perm_is_identity']}",
        f"other_share={summary['other_share']} kappa_rows="
        + "; ".join(f"{r['split']}:{r['kappa_all']}/{r['kappa_informative_only']}" for r in kappa_rows),
        "files: title_labels.csv classifier_quality.csv contingency.csv "
        "purity_by_community.csv nmi_permutation.csv category_scheme_sensitivity.csv "
        "assortativity_by_category.csv mixing_matrix.csv mixing_matrix_weighted.csv "
        "alternative_features.csv summary.json result.md",
    ]
    C.write_run_note(MODULE, lines)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "classifier"])
    args = ap.parse_args()
    main(args.stage)
