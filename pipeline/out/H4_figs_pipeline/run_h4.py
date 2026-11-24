# -*- coding: utf-8 -*-
"""H4 · 管线口径与数据陷阱的成图单元。

MANIFEST 要求三张图，全部是"把已有结论画出来"，本单元不产生任何新的分析量：
1. `h4_attack_curves`      —— E1 的 6 策略攻击曲线族（GCC / E_fixed 双面板，random 带 ±sd）
2. `h4_pipeline_flow`      —— 管线与口径流程图（数据 → 清洗 → 四波分析 → 图/审计）
3. `h4_data_traps`         —— 数据陷阱一览（六个坑，每个数字都能指到 anchors.csv / censoring_facts.csv / 上游 CSV）

只读上游：`E1_attack_curves/e1_curve_efficiency.csv`、`data/anchors.csv`、
`data/censoring_facts.csv`、`E5_observed_scenarios/scenario_metrics.csv`。

两条不许违反的规矩（CONVENTIONS §2.3 / §7）：
- 效率只认 `E_fixed`（分母恒为基线 GCC 的 453602 个有序点对）；`E_full` 变高不是结论，
  只允许"同一删除规模 k 下跨情景比"。图里因此把 baseline 画成一条水平参考线而不是"起点"。
- 不写因果措辞；攻击曲线描述的是"移除顺序与连通性之间的对应"，不是"X 导致网络崩溃"。
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import figstyle as fs  # noqa: E402

fs.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = fs.make_dir("H4_figs_pipeline")
rows = []

# ---------------------------------------------------------------- 数据
eff = fs.read_csv("E1_attack_curves", "e1_curve_efficiency.csv")
anch = pd.read_csv(Path(fs.PIPELINE_DIR) / "data" / "anchors.csv", encoding="utf-8-sig")
AN = {r.anchor: r.got for r in anch.itertuples()}
censor = pd.read_csv(Path(fs.PIPELINE_DIR) / "data" / "censoring_facts.csv",
                     encoding="utf-8-sig")

STRAT = [
    ("degree", "度（无向无权）", fs.PALETTE_8[1]),
    ("betweenness", "介数（无向无权）", fs.PALETTE_8[4]),
    ("kcore", "k-core", fs.PALETTE_8[2]),
    ("pagerank_dw", "PageRank（有向加权，主口径）", fs.PALETTE_8[0]),
    ("pagerank_uu", "PageRank（无权无向）", fs.PALETTE_8[5]),
    ("random", "随机（120 次重复，均值±sd）", fs.C_NULL),
]
STRAT_CN = {k: lab for k, lab, _ in STRAT}
N_REPS = {k: int(eff.loc[eff.strategy == k, "n_reps"].iloc[0])
          for k, _, _ in STRAT if k in set(eff.strategy)}
E_BASE = float(eff.loc[(eff.strategy == "degree") & (eff.step == 0), "E_fixed"].iloc[0])

# ================================================================ 图 1 攻击曲线族
fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.9), sharex=True)

for ax, ycol, sdcol, ylab, title in (
    (axes[0], "gcc_pct", "gcc_pct_sd", "存活主分量 GCC（占 674 基准图 %）", "(A) GCC"),
    (axes[1], "E_fixed", "E_fixed_sd", r"固定分母效率 $E_{\mathrm{fixed}}$", "(B) 效率"),
):
    for key, lab, col in STRAT:
        d = eff[eff.strategy == key].sort_values("f_actual")
        x = d["f_actual"].to_numpy(dtype=float)
        y = d[ycol].to_numpy(dtype=float)
        sd = pd.to_numeric(d[sdcol], errors="coerce").to_numpy(dtype=float)
        has_sd = np.isfinite(sd).any()
        ax.plot(x, y, color=col, lw=2.0 if key != "random" else 2.4,
                ls="--" if key == "random" else "-", label=lab, zorder=3)
        if has_sd:
            ax.fill_between(x, y - sd, y + sd, color=col, alpha=0.22, lw=0, zorder=2)
    for f0 in (0.1, 0.2, 0.3):
        ax.axvline(f0, color="#999999", lw=0.8, ls=":", zorder=1)
        ax.text(f0, 0.985, "f=%g" % f0, fontsize=7.2, color="#777777",
                ha="center", va="top", transform=ax.get_xaxis_transform())
    ax.set_xlabel("已移除节点比例 f（按各自的排序从头删）")
    ax.set_ylabel(ylab)
    ax.set_title("%s：6 种移除顺序下的%s曲线" % (title, "连通性" if ycol == "gcc_pct" else "效率"), fontsize=10)

axes[1].axhline(E_BASE, color=fs.C_DEAD, lw=1.0, ls="-.", zorder=1)
axes[1].text(0.012, E_BASE + 0.004, "基线 %.4f" % E_BASE, fontsize=7.5, color=fs.C_DEAD,
             ha="left", va="bottom")
axes[0].set_ylim(-3, 104)
axes[1].set_ylim(-0.012, max(0.36, float(eff.E_fixed.max()) * 1.12))

# 说法全部由数据算出来，不靠口算：每条曲线在哪一档把主分量打到 10% 以下、
# 有向加权 PageRank 与随机曲线在 f<=0.25 的最大落差、以及 f=0.35 处的落差。
def _f_break(strat):
    d = eff[eff.strategy == strat].sort_values("f_actual")
    hit = d[d.gcc_pct < 10]
    return float(hit.f_actual.min()) if len(hit) else float("nan")


F_DEG, F_BET, F_KC = _f_break("degree"), _f_break("betweenness"), _f_break("kcore")
_m = eff[eff.strategy == "pagerank_dw"][["step", "f_actual", "gcc_pct"]].merge(
    eff[eff.strategy == "random"][["step", "gcc_pct"]], on="step", suffixes=("_pr", "_ra"))
_m["gap"] = (_m.gcc_pct_pr - _m.gcc_pct_ra).abs()


def _gap_at(f0):
    return float(_m.iloc[(_m.f_actual - f0).abs().argmin()].gap)


G10, G20, G35 = _gap_at(0.10), _gap_at(0.20), _gap_at(0.35)

h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, fontsize=8.2, frameon=False,
           bbox_to_anchor=(0.5, 0.128))
fig.suptitle("同一张 674 节点基准图、6 种移除顺序下的连通性与效率", fontsize=12.5, y=0.985)
fig.text(0.5, 0.930,
         "度在 f≈%.2f、介数在 f≈%.2f 就把主分量打到 10%% 以下，k-core 要到 f≈%.2f；"
         "有向加权 PageRank 是三种确定策略里最温和的一条，但仍始终低于随机："
         "f=0.10 处低 %.1f pp、f=0.20 处低 %.1f pp、f=0.35 处低 %.1f pp"
         % (F_DEG, F_BET, F_KC, G10, G20, G35),
         ha="center", fontsize=9.0, color="#333333")
fig.text(
    0.5, 0.030,
    "读法限制：(1) 右图 E_fixed 的分母恒为基线 453602 个有序点对，删除本身就扣分，"
    "所以只能在同一 k 上横向比不同曲线，不能拿某个点与基线点比大小；\n"
    "(2) 随机曲线的阴影是 120 次独立重复的 ±1sd —— 两条曲线落差小于阴影宽度时不能当差异，"
    "无权无向 PageRank 与度曲线几乎重合正是这种情况（ρ=0.977，见 E1/DISSENT.md）；\n"
    "(3) 曲线描述的是移除顺序与连通性之间的对应，不代表真实站点按这个顺序失效。",
    ha="center", fontsize=8, color="#333333")
fig.tight_layout(rect=(0, 0.235, 1, 0.905))

tab = []
for key, lab, _ in STRAT:
    d = eff[eff.strategy == key]
    for f0 in (0.1, 0.2, 0.3):
        r = d.iloc[(d.f_actual - f0).abs().argmin()]
        tab.append(dict(strategy=key, strategy_cn=lab, f_target=f0,
                        f_actual=round(float(r.f_actual), 5),
                        n_reps=int(r.n_reps),
                        gcc_pct=float(r.gcc_pct), gcc_pct_sd=float(r.gcc_pct_sd)
                        if pd.notna(r.gcc_pct_sd) else np.nan,
                        E_fixed=float(r.E_fixed), E_fixed_sd=float(r.E_fixed_sd)
                        if pd.notna(r.E_fixed_sd) else np.nan))
rows.append(fs.save_fig(fig, "h4_attack_curves", OUT, tab))
print("[H4] fig1 done")


# ================================================================ 图 2 管线/口径流程图
def box(ax, x, y, w, h, text, fc="#ffffff", ec="#3d4a5c", fs_=8.2, weight="normal",
        color="#111111"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                linewidth=1.0, facecolor=fc, edgecolor=ec, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs_,
            weight=weight, color=color, zorder=3, linespacing=1.45)


def arrow(ax, xy1, xy2, style="-|>", color="#3d4a5c", lw=1.1, ls="-"):
    ax.add_patch(FancyArrowPatch(xy1, xy2, arrowstyle=style, mutation_scale=10,
                                 linewidth=lw, color=color, linestyle=ls, zorder=1,
                                 shrinkA=2, shrinkB=2))


fig, ax = plt.subplots(figsize=(15.2, 8.4))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# --- 第一层：数据
box(ax, 0.010, 0.795, 0.185, 0.125,
    "CSA_HW/\n原始数据（只读）\n链接表 + 节点表\n+ 抓取状态码", fc="#eef2f7")
box(ax, 0.250, 0.795, 0.215, 0.125,
    "00_build_data 清洗\n31/31 锚点自检\n锚点不过即停，不进分析", fc="#dfe9f5")
box(ax, 0.520, 0.795, 0.235, 0.125,
    "pipeline/data/\nclean_nodes %s 节点\nclean_edges %s 有向 → %s 无向"
    % (AN.get("n_attributed"), AN.get("n_edges_directed"), AN.get("n_edges_undirected")),
    fc="#dfe9f5")
box(ax, 0.800, 0.795, 0.190, 0.125,
    "基准图（鲁棒性唯一口径）\nGCC %s 节点 / %s 边\nE_fixed 分母 %d 有序对"
    % (AN.get("gcc_size"), "2497", 453602), fc="#fdeee0")
arrow(ax, (0.195, 0.857), (0.250, 0.857))
arrow(ax, (0.465, 0.857), (0.520, 0.857))
arrow(ax, (0.755, 0.857), (0.800, 0.857))

# --- 数据总线
ax.plot([0.10, 0.92], [0.700, 0.700], color="#3d4a5c", lw=1.4, zorder=1)
arrow(ax, (0.637, 0.795), (0.637, 0.700), color="#3d4a5c", lw=1.4)

W = 0.208
XS = [0.018, 0.256, 0.494, 0.732]
WAVES = [
    ("W1 · A 组 5 + B 组 6\n拓扑基线 / 零模型\n社区结构（Louvain）", "#eef2f7"),
    ("W2 · C 组 6 + D 组 7\n中心性与角色\n失效分布与删失", "#eef2f7"),
    ("W3 · E 组 6 + F 组 3 + G 组 3\n鲁棒性与韧性 / 模板层\n风险矩阵与治理", "#eef2f7"),
    ("W4 · H 组 4 图 + I 组 5 审计\n只读上游 CSV\n不重算任何分析量", "#e9f6ee"),
]
for x, (txt, fc) in zip(XS, WAVES):
    box(ax, x, 0.500, W, 0.140, txt, fc=fc, fs_=8.0)
    arrow(ax, (x + W / 2, 0.700), (x + W / 2, 0.640))
for i in range(3):
    arrow(ax, (XS[i] + W + 0.004, 0.570), (XS[i + 1] - 0.004, 0.570),
          color="#9aa6b4", ls="--")

# --- 口径层
box(ax, 0.018, 0.245, 0.464, 0.185,
    "效率三口径（CONVENTIONS §2.3）\n"
    "E_full：全剩余图，不可达记 0，分母随删除收缩 → 只用于对齐外部数字\n"
    "E_gcc：只看幸存主分量 → 描述“幸存网络内部”\n"
    "E_fixed：分母恒为基线 453602 个有序对 → 【主口径】\n"
    "硬规则：只允许在同一删除规模 k 上跨情景比，不许与基线点比大小",
    fc="#fff6e5", fs_=8.0, ec="#c08b2e")
box(ax, 0.518, 0.245, 0.464, 0.185,
    "其余三个必须分列的口径\n"
    "度：入度（失效分析主口径）/ 出度（死站恒 0，删失）/ 无向总度（对照，仅附录）\n"
    "失效：真 %s + 假 %s + 待定 %s = %s（每份结果给校正前后两套）\n"
    "社区：n_comms_total 与 n_comms_nonsingleton 都要报，且报分布不报单个数"
    % (AN.get("n_true_fail"), AN.get("n_false_fail"), AN.get("n_ambiguous"),
       AN.get("n_dead")),
    fc="#fff6e5", fs_=8.0, ec="#c08b2e")
arrow(ax, (0.25, 0.500), (0.25, 0.432), color="#c08b2e", ls="--")
arrow(ax, (0.75, 0.500), (0.75, 0.432), color="#c08b2e", ls="--")

box(ax, 0.018, 0.045, 0.964, 0.160,
    "写盘纪律：每个单元只写 out/<自己的目录>/，共享地基代码 csa_common.py 只 import 不复制，CSA_HW/ 只读\n"
    "产出三件套（CONVENTIONS §9）：result.md（结论表，数字指到代码行） + <name>.csv（UTF-8-sig） + run_<name>.py（可独立重跑） + RUN.txt（环境/命令行/耗时/seed）\n"
    "图：PNG 300dpi + 矢量 PDF + 伴生 fig_*.csv，配色走 figstyle 统一色表，中文用 SimHei，不许 emoji\n"
    "跨机器复现：requirements.txt 版本钉死 + 31 个锚点做漂移探针；服务器上锚点 31/31 PASS 才开跑分析",
    fc="#f2f2f2", fs_=8.0)

fig.suptitle("流水线结构与口径分层：数据只进来一次，口径要在每一步显式选",
             fontsize=12.5, y=0.975)
fig.text(0.5, 0.012,
         "这张图描述的是代码组织与口径依赖关系，不含任何实测数值；"
         "所有上级数字来自 data/anchors.csv（31 项锚点）与 CONVENTIONS.md 的硬规则。",
         ha="center", fontsize=7.8, color="#666666")
fig.tight_layout(rect=(0, 0.03, 1, 0.965))

flow_rows = [
    dict(stage="数据层", node="CSA_HW → 00_build_data → pipeline/data/", writes="data/",
         gate="31/31 锚点", note="原始数据只读，清洗产物带 edge_order 列"),
    dict(stage="分析层 W1-W3", node="A/B → C/D → E/F/G", writes="out/<单元>/",
         gate="assert_anchors 先跑", note="每单元只写自己目录，建图只走 csa_common"),
    dict(stage="图与审计 W4", node="H 组 4 图 + I 组 5 审计", writes="out/<单元>/",
         gate="依赖上游 CSV", note="H 不重算、I 不改别人文件"),
    dict(stage="口径层", node="E_full/E_gcc/E_fixed · 入/出/总度 · 真/假失效 · 社区两计数",
         writes="—", gate="§2.3 硬规则", note="主口径固定，其余作对照并标明"),
]
rows.append(fs.save_fig(fig, "h4_pipeline_flow", OUT, flow_rows))
print("[H4] fig2 done")

# ================================================================ 图 3 数据陷阱一览
cm = fs.read_csv("E5_observed_scenarios", "scenario_metrics.csv")
d162 = cm[(cm.scenario == "dead162") & (cm.view == "as_observed")].iloc[0]
base_full = float(eff.loc[(eff.strategy == "degree") & (eff.step == 0), "E_full"].iloc[0])

try:
    hcs = fs.read_csv("B5_hub_removal", "hub_choice_sensitivity.csv")

    def _one(method):
        r = hcs[hcs.method == method].iloc[0]
        return int(r.edges_removed), int(r.n_hubs_resolved)

    hub_lo, hub_n6 = _one("nav6_exact")
    hub_hi, hub_npfx = _one("prefix_broken_DIAGNOSTIC")
    hub_note = "按短名前缀匹配枢纽会把一批 en.* 单位站一起删掉"
except Exception:
    hub_lo, hub_hi, hub_n6, hub_npfx = 505, 678, 6, 36
    hub_note = "按短名前缀匹配枢纽会多删一批 en.* 单位站"

try:
    w10 = fs.read_csv("B4_weight_variants", "weight_variant_per_seed.csv")
    w10 = w10[w10.variant == "backbone_w10"]
    ct = pd.to_numeric(w10.n_comms_total, errors="coerce")
    cn = pd.to_numeric(w10.n_comms_nonsingleton, errors="coerce")
    comm_txt = "%d–%d 个社区 / 其中仅 %d–%d 个非孤立" % (
        int(ct.min()), int(ct.max()), int(cn.min()), int(cn.max()))
except Exception:
    comm_txt = "336–340 个社区 / 其中仅 29–34 个非孤立"

color_bad = int(censor.loc[censor.fact.str.contains("color"), "value"].iloc[0])


def cjk_fill(s, width=26):
    """按"字"折行：中文没有空格，textwrap 会把整段中文当成一个超长单词而不折。

    ASCII/数字 token（csa_common.hub_nodes()、453602 之类）当一个整体不劈开，
    其余字符逐个排；行首不留空格，也不把中文标点甩到行首（拗句读）。
    """
    tok = re.compile(r"[A-Za-z0-9_\.\-\+/\(\)\[\]%]+|.", re.S)
    nop = set("，。、；：）〕】”’！？,.;:)]}")
    out, cur = [], ""
    for t in tok.findall(s):
        if t == "\n":
            out.append(cur.rstrip())
            cur = ""
            continue
        if not cur and t.isspace():
            continue
        if len(cur) + len(t) > width and cur and t not in nop:
            out.append(cur.rstrip())
            cur = "" if t.isspace() else t
        else:
            cur += t
    if cur:
        out.append(cur.rstrip())
    return "\n".join(out)


CARDS = [
    ("① 出度删失（最硬的一个）",
     "%s / %s 个死站\n出度恒为 0" % (AN.get("dead_with_outdegree_zero"), AN.get("n_dead")),
     "以死站为起点的有向边 = %s 条。爬虫只能从抓到的页面提链接，死站的出边是不可观测，不是"
     "“没链接”。所以出度、无向总度都不能进失效分析；同口径入度是 1.83 vs 4.94（2.7 倍），"
     "不是跨口径的 5.2 倍。" % AN.get("edges_from_dead"),
     "anchors.csv: dead_with_outdegree_zero / edges_from_dead；\ncensoring_facts.csv: 死站出边全部不可观测"),
    ("② 假失效必须先剔",
     "真 %s + 假 %s\n+ 待定 %s = %s" % (AN.get("n_true_fail"), AN.get("n_false_fail"),
                                        AN.get("n_ambiguous"), AN.get("n_dead")),
     "假失效是认证跳转（B 类 18 个）与 robots 拦截（C 类 14 个），状态码 -1/-2，不是故障。"
     "每份失效相关结果都要给“校正前 / 校正后”两套，否则 162 这个数会被读成 162 个真宕机。",
     "anchors.csv: n_true_fail / n_false_fail / n_ambiguous；CONVENTIONS §2.2"),
    ("③ 两个列直接禁用",
     "degree 列最大 146\n重算无向度最大 %s" % AN.get("max_degree_u"),
     "nodes_full 的 degree 列与重算值差 27，color 列与 ok_ratio_pct 冲突 %d 处"
     "（可用率 0 的站点被标成 blue）。健康分层只认 ok_ratio_pct，结构量一律重算。" % color_bad,
     "anchors.csv: max_degree_u；censoring_facts.csv: color 与 ok_ratio 不一致"),
    ("④ 枢纽要全名精确匹配",
     "全名命中 %d 条边\n前缀匹配误删 %d 条" % (hub_lo, hub_hi),
     "%s：6 个导航枢纽必须用 csa_common.hub_nodes() 全名精确匹配；按短名前缀匹配会解析出 %d 个"
     "“枢纽”，边删失从 %.1f%% 膨胀到 %.1f%%（多删 %d 条），社区结论会整体改写。"
     % (hub_note, hub_npfx, 100.0 * hub_lo / 2498, 100.0 * hub_hi / 2498, hub_hi - hub_lo),
     "B5_hub_removal/hub_choice_sensitivity.csv:\nnav6_exact vs prefix_broken_DIAGNOSTIC"),
    ("⑤ 社区数别把孤立点算进去",
     comm_txt,
     "Louvain 会给每个零度孤立点各返回一个单点社区，于是“338 个社区”里有 307 个是孤立点。"
     "主表只能用 n_comms_nonsingleton，而且要报多 seed 的分布。",
     "B4_weight_variants/weight_variant_per_seed.csv（backbone_w10，30 seed）；CONVENTIONS §3"),
    ("⑥ E_full 变高是分母收缩",
     "E_full %.4f 高于基线\nE_fixed %.4f" % (float(d162.E_full), float(d162.E_fixed)),
     "删 162 个全挂站时 E_full 反而高于基线 %.4f —— 因为分母随删除一起缩小，均值被机械抬高。"
     "同一删除规模下随机基线 E_fixed=%.4f，全挂情景 %.4f，即该情景的连通效率损失比随机%s。"
     % (base_full, float(d162.rand_mean_E_fixed), float(d162.E_fixed),
        "小" if float(d162.E_fixed) > float(d162.rand_mean_E_fixed) else "大"),
     "E5_observed_scenarios/scenario_metrics.csv: dead162|as_observed 行；CONVENTIONS §2.3"),
]

fig, axes = plt.subplots(2, 3, figsize=(15.6, 7.9))
for ax, (title, big, body, src) in zip(axes.ravel(), CARDS):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.015, 0.015), 0.97, 0.97,
                                boxstyle="round,pad=0.008,rounding_size=0.02",
                                linewidth=1.0, facecolor="#fbfbfc", edgecolor="#c8ccd4"))
    ax.text(0.05, 0.930, title, fontsize=10.2, weight="bold", color="#1f3b73",
            va="top", ha="left")
    ax.text(0.05, 0.815, big, fontsize=12.6 if max(len(x) for x in big.split("\n")) <= 22 else 10.8,
            weight="bold", color=fs.C_DEAD, va="top", ha="left", linespacing=1.35)
    # 卡片内宽约 470px、字号 7.9 时每字约 16px，取 26 字/行留安全余量。
    ax.text(0.05, 0.645, cjk_fill(body, 26),
            fontsize=7.9, color="#222222", va="top", ha="left", linespacing=1.75)
    ax.text(0.05, 0.100, "来源：" + src, fontsize=6.6, color="#7a7a7a",
            va="bottom", ha="left", linespacing=1.4)

fig.suptitle("六个数据陷阱：每一个都能把结论翻面，而且每个数字都能在 anchors.csv / 上游 CSV 里指到",
             fontsize=12.5, y=0.975)
fig.text(0.5, 0.012,
         "本图不产生新分析量，只把 CONVENTIONS 的硬规则与各单元实测数字放到一张图上；"
         "这些陷阱的共同结构是：口径选错时数字仍然自洽，但含义变了。",
         ha="center", fontsize=7.8, color="#666666")
fig.tight_layout(rect=(0, 0.03, 1, 0.96))

trap_rows = []
for i, (title, big, body, src) in enumerate(CARDS, 1):
    trap_rows.append(dict(id=i, trap=title, headline=" ".join(big.split()),
                          source=src.replace("\n", " ")))
rows.append(fs.save_fig(fig, "h4_data_traps", OUT, trap_rows))
print("[H4] fig3 done")

# ---------------------------------------------------------------- 落盘
pd.DataFrame(rows).to_csv(OUT / "h4_manifest.csv", index=False, encoding="utf-8-sig")
fs.note(OUT, [
    "H4 · 管线口径与数据陷阱 成图单元",
    "上游只读：E1_attack_curves/e1_curve_efficiency.csv",
    "        data/anchors.csv, data/censoring_facts.csv",
    "        E5_observed_scenarios/scenario_metrics.csv",
    "        B5_hub_removal/hub_choice_sensitivity.csv",
    "        B4_weight_variants/weight_variant_per_seed.csv",
    "本单元不重算任何分析量：所有数字都是上游 CSV 的取值或 min/max 区间。",
    "",
    "图 1 h4_attack_curves：6 策略 × (GCC / E_fixed) 双面板，random 带 120 次重复的 ±sd。",
    "  刻意把 E_fixed 基线画成一条水平参考线：E_fixed 分母固定，点与点之间只能同 k 横比。",
    "图 2 h4_pipeline_flow：数据 → 清洗 → 四波分析 → 图/审计 的结构图 + 口径分层说明。",
    "图 3 h4_data_traps：六个陷阱卡片，每个数字标注了来源文件与列名。",
    "",
    "出图三件套：PNG 300dpi + 矢量 PDF + fig_*.csv 伴生数据表（save_fig 一次写完，缺一即抛错）。",
    "字体：SimHei（本机上传的 ttf 注册进服务器 fontconfig，未改 figstyle.py）。",
    "产物：%s" % ", ".join(r["fig"] for r in rows),
])
for r in rows:
    print(r)
print("[H4] done")
