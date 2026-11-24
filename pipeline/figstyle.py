# -*- coding: utf-8 -*-
"""H 组共享图形样式：中文、统一色表、PNG(300dpi)+矢量 PDF+数据表三件套。

约定（`CONVENTIONS.md` §9 / `MANIFEST.md` H 组）：
- 每个 H 单元每张图必须同时产出 `<name>.png`、`<name>.pdf`、`fig_<name>.csv`；
  `save_fig()` 一次调用把三样都写完，缺一个就抛错，避免"只留了 PNG"。
- 色表在模块级定义、全项目共用一份；同一语义（如真失效/假失效）在所有图里同色。
- 字体只用系统里真实存在的 SimHei；`setup()` 会打印实际解析到的字体文件，
  缺字体时直接抛错而不是静默变成方框。
- 禁止 emoji、禁止在图里写因果措辞（§7）。

本文件不建图、不读数据集，只提供样式与 I/O；所有数值来自 A–G 组已产出的 CSV。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境；必须在 import pyplot 之前设置

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

PIPELINE_DIR = Path(__file__).resolve().parent
OUT_DIR = PIPELINE_DIR / "out"

# ---------------------------------------------------------------- 色表
# 语义色：一个概念在全项目只有一个颜色。改名前先把所有图重跑一遍。
C_OBS = "#1f3b73"          # 观测量 / 真实数据
C_MODEL = "#c0392b"        # 拟合 / 模型曲线
C_NULL = "#7f7f7f"         # 零模型 / 随机基线
C_DEAD = "#b3202c"         # 真失效（A/D/F 类）
C_FALSE_FAIL = "#e0a43c"   # 假失效（B/C 类：认证跳转等）
C_AMBIG = "#9b59b6"        # 口径不明（G 类）
C_PARTIAL = "#f0c419"      # 部分失效
C_ALIVE = "#2e8b57"        # 存活
C_HUB = "#8e44ad"          # 导航枢纽
C_EDGE1 = "#4878a8"        # 附加曲线 1
C_EDGE2 = "#d98a4a"        # 附加曲线 2

MODEL_COLORS = {
    "ER": "#c0392b",
    "WS": "#2e8b57",
    "BA": "#4878a8",
    "configuration": "#d98a4a",
    "config_model": "#d98a4a",
    "ER_gnp": "#c0392b",
}

# 分类色板（用于按类别/按模型着色的散点、条形）
PALETTE_8 = ["#1f3b73", "#c0392b", "#2e8b57", "#d98a4a",
             "#8e44ad", "#4878a8", "#b3202c", "#7f7f7f"]

# 顺序色板：矩阵热图、强度图。viridis 系对色盲友好，且灰度打印仍单调。
HEATMAP_CMAP = "viridis"
DIVERGING_CMAP = "RdBu_r"

SERIES_LABELS = {
    "degree_u_671": "无权度 k（671 节点）",
    "degree_u_676": "无权度 k（676 节点）",
    "strength_u_671": "加权强度 s（671 节点）",
    "in_degree_671": "入度 k_in",
    "out_degree_671": "出度 k_out",
    "out_degree_alive": "出度 k_out（仅活站）",
    "listed_urls": "列出的 URL 数",
}

DIST_LABELS = {
    "powerlaw": "幂律",
    "trunc_powerlaw": "截断幂律",
    "lognormal": "对数正态",
    "exponential": "指数",
    "nbinomial": "负二项",
}


# ---------------------------------------------------------------- setup
def setup(verbose: bool = False) -> dict:
    """设置全局 rcParams，返回实际解析到的中文字体。缺 SimHei 直接抛错。"""
    wanted = ["SimHei", "Microsoft YaHei", "SimSun", "KaiTi", "FangSong"]
    have = {f.name for f in font_manager.fontManager.ttflist}
    cjk = [w for w in wanted if w in have]
    if not cjk:
        raise RuntimeError(
            "没找到任何中文字体（试过 %s）。图里的中文会变成方框，禁止出图。" % wanted
        )
    plt.rcParams.update({
        "font.sans-serif": cjk + ["DejaVu Sans"],
        "font.family": "sans-serif",
        "axes.unicode_minus": False,   # 否则负号在 SimHei 下渲染成方框
        "mathtext.fontset": "dejavusans",  # 对数轴刻度走 mathtext，用 default 会缺 U+2212
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,           # PDF 里嵌 TrueType，文字可选可搜
        "ps.fonttype": 42,
    })
    if verbose:
        paths = {f.name: f.fname for f in font_manager.fontManager.ttflist
                 if f.name in cjk}
        print("[figstyle] CJK chain: %s" % ", ".join(cjk))
        for n in cjk:
            print("[figstyle]   %-18s %s" % (n, paths.get(n)))
    return {"cjk_chain": cjk, "font_file": font_manager.findfont("SimHei")}


# ---------------------------------------------------------------- I/O
def read_csv(module: str, filename: str, **kw) -> pd.DataFrame:
    """读某个 A–G 单元已经产出的 CSV（BOM 自动去掉）。缺文件直接抛错。"""
    p = OUT_DIR / module / filename
    if not p.exists():
        raise FileNotFoundError("缺少上游产出：%s" % p)
    df = pd.read_csv(p, encoding="utf-8-sig", **kw)
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def read_json(module: str, filename: str) -> dict:
    p = OUT_DIR / module / filename
    if not p.exists():
        raise FileNotFoundError("缺少上游产出：%s" % p)
    return json.loads(p.read_text(encoding="utf-8"))


def data_table(path: Path, rows) -> None:
    """把图背后的数据写成 fig_*.csv。rows 允许 DataFrame 或 list[dict]。"""
    if isinstance(rows, pd.DataFrame):
        df = rows
    elif isinstance(rows, (list, tuple)):
        df = pd.DataFrame(list(rows))
    else:
        raise TypeError("fig_*.csv 需要 DataFrame 或 list[dict]，收到 %r" % type(rows))
    if df.empty:
        raise ValueError("图数据表为空：%s" % path.name)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def table_from_columns(cols: dict) -> pd.DataFrame:
    """pd.DataFrame(dict) 的替身：先逐列查长度，长度不一致直接报出是哪一列。

    DataFrame 自己也会查，但它只说 "All arrays must be of the same length"，
    不指出是哪列——图数据表一旦串了长度就是错的图，值得多这一步。
    """
    bad = {k: len(v) for k, v in cols.items() if not hasattr(v, "__len__")}
    if bad:
        raise TypeError("列不是数组（多半是误用了 set / 生成器）：%s" % list(bad))
    n = {k: len(v) for k, v in cols.items()}
    if len(set(n.values())) > 1:
        raise ValueError("图数据表列长不一致：%s" % n)
    return pd.DataFrame(cols)


def save_fig(fig, figname: str, out_dir: Path, data) -> dict:
    """一次产出 <name>.png / .pdf / fig_<name>.csv；三样缺任一抛错。

    fig 是 matplotlib Figure 对象（调用方自己画好并传进来），不是 figure 编号。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / figname
    png, pdf = base.with_suffix(".png"), base.with_suffix(".pdf")
    csvp = out_dir / ("fig_%s.csv" % figname)
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    data_table(csvp, data)
    plt.close(fig)
    missing = [str(p) for p in (png, pdf, csvp) if not p.exists() or p.stat().st_size == 0]
    if missing:
        raise RuntimeError("出图不完整，缺：%s" % missing)
    return {"fig": figname, "png_bytes": png.stat().st_size,
            "pdf_bytes": pdf.stat().st_size,
            "csv_rows": int(pd.read_csv(csvp, encoding="utf-8-sig").shape[0])}


def make_dir(module: str) -> Path:
    p = OUT_DIR / module
    p.mkdir(parents=True, exist_ok=True)
    return p


def note(out_dir: Path, lines) -> None:
    (Path(out_dir) / "RUN.txt").write_text(
        "\n".join(str(s) for s in lines) + "\n", encoding="utf-8")


def cat_colors(k: int):
    """k 个彼此可区分的类别色。

    本项目要着色的类别有 32 个社区，而 matplotlib 3.11 只有 tab10/tab20（没有 tab30），
    8 色循环会大量撞色。这里均匀铺色相，并按 i%3 交替明度与饱和度，
    让相邻类别在色相和明度两个维度上都拉开。
    """
    import colorsys
    out = []
    for i in range(k):
        h = i / max(k, 1)
        l = (0.42, 0.62, 0.52)[i % 3]
        s = (0.75, 0.60, 0.88)[i % 3]
        out.append(colorsys.hls_to_rgb(h, l, s))
    return out


def log_decimal_axis(ax, which: str = "y") -> None:
    """把对数轴的刻度标签换成十进制小数（1 / 0.1 / 0.01）。

    默认的 mathtext 上标负号在 SimHei 链下会渲染成一个说不清的小符号，
    而 CCDF 的纵轴本来就是 1→1/n 的短区间，写小数比写 10^-k 更好读。
    """
    if which not in ("x", "y"):
        raise ValueError("which 只能是 'x' 或 'y'，收到 %r" % which)
    getattr(ax, which + "axis").set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: "%g" % v))


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return ("%." + str(nd) + "f") % x
