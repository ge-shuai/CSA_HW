# -*- coding: utf-8 -*-
"""H3 第三张图：z–P 功能角色地图（数据源 = C5_functional_roles 的实测产出）。

背景：MANIFEST 的 H3 组要求「风险矩阵 / 中心性排序相关热图 / z–P 角色地图」三张图。
前两张由 run_h3.py 出；第三张当时因 C5 零产出而主动缺席（见 H3_figs_risk/RUN.txt）。
本脚本只做这一张图，读 C5 的 CSV，不重算任何角色，也不改 H3 已有的任何文件
（唯一例外是向 H3_figs_risk/RUN.txt **追加**一段说明，不覆盖）。

口径（全部来自 C5，本脚本不改变）：
  - 社区划分 = 加权 Gu 上的 Louvain，canonical seed=42（25 社区 / Q=0.6904 / 最大 87）；
    另有 11 个 seed 的划分用于误差棒（合计 12 个划分）。
  - z、P 在**二值无向** Gb 上算（GA05 原文口径），阈值 z*=2.5、P*=0.62、P**=0.80；
    network_hub 另需模块产出 Y>平均。
  - 节点集 = 671 个有属性节点。

出图三件套：h3_zp_role_map.png / .pdf / fig_h3_zp_role_map.csv（走 figstyle.save_fig）。

运行：cd pipeline && PYTHONHASHSEED=0 python out/C5_functional_roles/run_h3_zp_fig.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parents[1]
sys.path.insert(0, str(PIPELINE))
import figstyle as fs  # noqa: E402

fs.setup()
import matplotlib.pyplot as plt  # noqa: E402

C5 = "C5_functional_roles"
H3_DIR = PIPELINE / "out" / "H3_figs_risk"

Z_STAR, P_STAR, P_DOUBLE_STAR = 2.5, 0.62, 0.80
ROLE_ORDER = ["peripheral", "non-hub_connector", "non-hub_kinless", "provincial_hub",
              "connector_hub", "kinless_hub", "network_hub"]
ROLE_CN = {
    "peripheral": "外围（peripheral）",
    "non-hub_connector": "非枢纽连接者（non-hub connector）",
    "non-hub_kinless": "非枢纽无亲属（non-hub kinless）",
    "provincial_hub": "行省枢纽（provincial hub）",
    "connector_hub": "连接枢纽（connector hub）",
    "kinless_hub": "无亲属枢纽（kinless hub）",
    "network_hub": "网络枢纽（network hub）",
}

zp = fs.read_csv(C5, "zp_roles_canonical.csv")
stab = fs.read_csv(C5, "role_stability_by_node.csv")
rc = fs.read_csv(C5, "role_counts_by_seed.csv")
cs = fs.read_csv(C5, "role_class_stability.csv")
reps = fs.read_csv(C5, "representatives.csv")

zp = zp.merge(stab[["node", "canonical_kept_frac", "n_distinct_roles_across_seeds"]],
              on="node", how="left")
assert len(zp) == 671, "z–P 表应有 671 个节点，实得 %d" % len(zp)
assert zp["canonical_kept_frac"].notna().all()

# ---------------------------------------------------------------- 图
colors = {r: fs.PALETTE_8[i] for i, r in enumerate(ROLE_ORDER)}
fig, (ax, axb) = plt.subplots(1, 2, figsize=(15.2, 6.4),
                              gridspec_kw={"width_ratios": [1.42, 1.0]})

# --- 面板 A：z–P 平面
for r in ROLE_ORDER:
    s = zp[zp["role"] == r]
    ax.scatter(s["P"], s["z"], s=26 if r not in ("peripheral",) else 14,
               c=[colors[r]], alpha=0.55 if r == "peripheral" else 0.9,
               edgecolors="none", label="%s  %d 个" % (ROLE_CN[r], len(s)), zorder=2)
ax.axhline(Z_STAR, color="#444444", lw=1.0, ls="--", zorder=1)
ax.axvline(P_STAR, color="#444444", lw=1.0, ls="--", zorder=1)
ax.axvline(P_DOUBLE_STAR, color="#444444", lw=1.0, ls=":", zorder=1)
ax.text(0.012, Z_STAR + 0.12, "z* = 2.5（枢纽门槛）", fontsize=8.0, color="#333333")
ax.text(P_STAR - 0.012, 6.55, "P* = 0.62", fontsize=8.0, color="#333333", rotation=90, va="top")
ax.text(P_DOUBLE_STAR - 0.012, 6.55, "P** = 0.80", fontsize=8.0, color="#333333",
        rotation=90, va="top")

# 代表节点：每类取“跨 seed 最稳”的 1 个；标签用去掉 .ustc.edu.cn 的完整子域，
# 免得 en.cfmc 被截成 "en" 后与导航枢纽 en.ustc.edu.cn 混淆。
lab = []
for r in ROLE_ORDER:
    sub = reps[reps["role"] == r].sort_values("canonical_kept_frac", ascending=False).head(1)
    for _, x in sub.iterrows():
        nm = str(x["node"]).replace(".ustc.edu.cn", "")
        lab.append((nm, float(x["P_canonical"]), float(x["z_canonical"]), r))
for nm, px, zy, r in lab:
    ax.text(px, zy + 0.14, nm, fontsize=7.9, color="#111111", ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="#cccccc", lw=0.4, alpha=0.80),
            zorder=5, clip_on=False)

ax.set_xlabel("参与系数 P（连边分布在几个模块上；0 = 全部连边都在本模块内）")
ax.set_ylabel("模块内度 z-score（模块内的二值无向连边数，按所属模块标准化）")
ax.set_xlim(-0.03, 1.02)
ax.set_ylim(min(-1.6, zp["z"].min() - 0.3), max(6.9, zp["z"].max() + 0.35))
ax.legend(loc="upper left", fontsize=7.6, ncol=1, labelspacing=0.3)
ax.set_title("(A) 671 个站点在 Guimerà–Amaral z–P 平面的位置\n"
             "z、P 都依赖社区划分：这里是 canonical 划分（seed=42，25 社区 / Q=0.6904）",
             fontsize=9.8)

# --- 面板 B：12 个划分下的角色类规模波动（= 误差棒）
canon_n = {r: int(zp["role"].eq(r).sum()) for r in ROLE_ORDER}
data, ticks, flip_txt = [], [], []
for r in ROLE_ORDER:
    v = rc[r].to_numpy(dtype=float)
    n0 = max(canon_n[r], 1)
    data.append(100.0 * (v / n0 - 1.0))
    ticks.append("%s\n(canon %d)" % (ROLE_CN[r].split("（")[0], canon_n[r]))
    row = cs[cs["role"] == r]
    flip_txt.append(float(row["flip_pct"].iloc[0]) * 100.0 if len(row) else np.nan)

bp = axb.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                 medianprops=dict(color="#111111", lw=1.2))
for patch, r in zip(bp["boxes"], ROLE_ORDER):
    patch.set_facecolor(colors[r])
    patch.set_alpha(0.45)
    patch.set_edgecolor("#333333")
rng = np.random.default_rng(20261113)
for i, (r, vals) in enumerate(zip(ROLE_ORDER, data), start=1):
    axb.scatter(i + rng.uniform(-0.16, 0.16, len(vals)), vals, s=16,
                c=[colors[r]], edgecolors="#222222", linewidths=0.4, zorder=3)
axb.axhline(0.0, color="#444444", lw=1.0, ls="--")
axb.axhspan(-30, 30, color="#2e8b57", alpha=0.06, zorder=0)
axb.set_xticks(range(1, len(ROLE_ORDER) + 1))
axb.set_xticklabels(ticks, fontsize=7.6, rotation=28, ha="right")
axb.set_ylabel("该角色类的节点数相对 canonical 的偏差（%）")
for i, (f, r) in enumerate(zip(flip_txt, ROLE_ORDER), start=1):
    if np.isfinite(f):
        axb.text(i, axb.get_ylim()[1] * 0.97, "翻 %.0f%%" % f, ha="center", va="top",
                 fontsize=7.2, color=(fs.C_DEAD if f > 30 else "#555555"),
                 fontweight=("bold" if f > 30 else "normal"))
axb.set_title("(B) 换社区划分（12 个 seed）后每类规模的波动\n"
              "箱 = 12 个划分；「翻 x%」= 该类 canonical 成员被改判的比例（>30% 则该类名单不可用）",
              fontsize=9.8)

fig.suptitle("H3 · z–P 功能角色地图（数据：C5_functional_roles，671 个有属性节点，"
             "二值无向 Gb 上的 z/P，canonical Louvain seed=42）\n"
             "读法只到「结构位置」：角色描述的是节点在所属模块内外的连边分布，"
             "不是站点功能、重要性或健康状态", fontsize=10.6)
fig.tight_layout(rect=(0, 0, 1, 0.90))

# 伴生数据表：逐节点（图 A 的原始点）+ 每类翻转率/规模区间（图 B 的读法依据）
out = zp[["node", "role", "role_cn", "z", "P", "P_normalized", "k_binary", "k_in_module",
          "module_id", "module_size", "canonical_kept_frac", "n_distinct_roles_across_seeds",
          "in_degree", "ok_ratio_pct", "health_state", "true_fail"]].copy()
out = out.rename(columns={"role": "role_canonical", "z": "z_canonical", "P": "P_canonical"})
role_wide = rc[["seed", "is_canonical"] + ROLE_ORDER].copy()
role_wide = role_wide.rename(columns={r: "n_" + r for r in ROLE_ORDER})
summary = pd.DataFrame({
    "role": ROLE_ORDER,
    "role_cn": [ROLE_CN[r] for r in ROLE_ORDER],
    "n_canonical": [canon_n[r] for r in ROLE_ORDER],
    "n_seed_min": [int(rc[r].min()) for r in ROLE_ORDER],
    "n_seed_max": [int(rc[r].max()) for r in ROLE_ORDER],
    "flip_pct_canonical_members": flip_txt,
    "usable_as_roster": cs.set_index("role").loc[ROLE_ORDER, "usable_as_roster"].astype(bool).tolist(),
})

# 主表逐节点；把每类规模区间压进附加列，保证 fig_*.csv 同时能复现 A 和 B 的读法
meta = summary.set_index("role")
out["role_n_seed_min"] = out["role_canonical"].map(meta["n_seed_min"])
out["role_n_seed_max"] = out["role_canonical"].map(meta["n_seed_max"])
out["role_flip_pct"] = out["role_canonical"].map(meta["flip_pct_canonical_members"])
res = fs.save_fig(fig, "h3_zp_role_map", H3_DIR, out.round(6))

# ---------------------------------------------------------------- RUN.txt 追加（幂等）
# 原始 RUN.txt 的字节原样保留，只在后面接一段；重复运行不会叠出第二段。
run_p = H3_DIR / "RUN.txt"
raw = run_p.read_bytes()
eol = b"\r\n" if b"\r\n" in raw else b"\n"
marker = b"\n" + b"=" * 78
cut = raw.find(marker)
base = raw[:cut] if cut > 0 else raw
base = base.rstrip(b"\r\n")
add = [
    "",
    "=" * 78,
    "【补图 · 2026-11-23】MANIFEST 要求的第三张图 z–P 角色地图现已产出（上面那条“主动缺席”的记录作废）。",
    "数据来源：out/C5_functional_roles/ 的实测产出，不是示意图：",
    "  zp_roles_canonical.csv（671 节点 z/P/角色，canonical Louvain seed=42）",
    "  role_counts_by_seed.csv（12 个划分的角色类计数，用于面板 B 的误差棒）",
    "  role_class_stability.csv（每类成员在 12 个划分间的翻转率）",
    "  representatives.csv（各类代表节点标签）",
    "C5 运行：服务器 <REMOTE_WORKDIR>，python 3.12.15 / networkx 3.6.1，",
    "  SEED=20261113，canonical seed=42 → 25 社区（非单点 25）/ Q=0.69043 / 最大社区 87，",
    "  12 个划分的社区数 23–26、最大社区 83–116；assert_anchors 31/31 PASS。",
    "新图三件套：%s（PNG 300dpi + 矢量 PDF + fig_h3_zp_role_map.csv）" % res["fig"],
    "本图不重算角色，只重绘 C5 的数；H3 原有 3 张图与 h3_manifest.csv 一个字节都没改。",
    "角色定义 = 节点在「模块内度 z」与「参与系数 P」上的位置；它不描述站点功能、",
    "  重要性或健康状态，也不支持「某类角色更易失效」这类因果读法（CONVENTIONS §7）。",
]
new = eol.join(s.encode("utf-8") for s in add)
run_p.write_bytes(base + eol + new + eol)

print("[H3-zp] wrote", res)
print("[H3-zp] role counts:", {r: canon_n[r] for r in ROLE_ORDER})
print("[H3-zp] seed ranges:", summary[["role", "n_seed_min", "n_seed_max",
                                      "flip_pct_canonical_members"]].to_dict(orient="records"))
print("[H3-zp] RUN.txt appended (original lines kept)")
