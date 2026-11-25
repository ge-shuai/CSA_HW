"""Render the report figures for E3 robustness, E5 controls and missing links."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent.parent / "pipeline"
sys.path.insert(0, str(PIPELINE))

import figstyle as fs
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import PercentFormatter

plt = fs.plt


def save(fig, stem):
    for extension in ("png", "pdf"):
        fig.savefig(HERE / f"{stem}.{extension}")
    plt.close(fig)


def draw_graph(ax, title, hidden):
    nodes = {"A": (0.27, 0.67), "B": (0.73, 0.67),
             "X": (0.27, 0.26), "Y": (0.73, 0.26)}
    ax.set(xlim=(0, 1), ylim=(0, 0.9), aspect="equal")
    ax.set_axis_off()
    ax.set_title(title, pad=10, fontweight="bold")

    def arrow(source, target, hidden_edge=False):
        ax.add_patch(FancyArrowPatch(
            nodes[source], nodes[target], arrowstyle="-|>", shrinkA=13, shrinkB=13,
            mutation_scale=13, linewidth=1.7,
            linestyle="--" if hidden_edge else "-",
            color=fs.C_EDGE2 if hidden_edge else fs.C_OBS,
        ))

    arrow("A", "X")
    arrow("B", "Y")
    if hidden:
        arrow("X", "Y", hidden_edge=True)
    for name, (x, y) in nodes.items():
        fill = fs.C_DEAD if name in {"X", "Y"} else fs.C_OBS
        ax.scatter(x, y, s=510, color=fill, zorder=3)
        ax.text(x, y, name, color="white", ha="center", va="center", zorder=4)


def draw_missing_links():
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.3))
    fig.subplots_adjust(left=0.04, right=0.96, top=0.73, bottom=0.18, wspace=0.12)
    fig.suptitle("失效站点之间的连接可以不同，观测图却相同", y=0.95,
                 fontsize=14, fontweight="bold")
    draw_graph(axes[0], "可能的原图 A", True)
    draw_graph(axes[1], "可能的原图 B", False)
    draw_graph(axes[2], "两者留下的观测图", False)
    fig.text(0.5, 0.07, "A、B 为活站，X、Y 为失效站。实线为观测链接，虚线为可能存在但未采到的出链。",
             ha="center", fontsize=10)
    save(fig, "fig11_missing_outlinks")


def draw_matched_comparison():
    e5 = PIPELINE / "out/E5_observed_scenarios"
    scenarios = ["dead162", "true129"]
    standard = pd.read_csv(e5 / "matched_indegree_summary.csv").set_index("scenario").loc[scenarios]
    sensitivity = pd.read_csv(e5 / "matched_sensitivity_summary.csv")
    joint = sensitivity[sensitivity.control == "indegree_bin_zero_out"].set_index("scenario").loc[scenarios]
    fig, ax = plt.subplots(figsize=(8.6, 3.3))
    fig.subplots_adjust(left=0.20, right=0.97, top=0.92, bottom=0.37)
    ax.axvline(0, color=fs.C_NULL, linewidth=1, linestyle="--")
    positions = np.array([1, 0])
    ax.scatter(standard.observed_minus_uniform, positions + 0.22,
               color=fs.C_NULL, marker="s", s=45, label="同数量均匀随机")
    for offset, mean, low, high, color, marker, label in (
        (0, standard.observed_minus_matched, standard.matched_p025,
         standard.matched_p975, fs.C_EDGE2, "o", "匹配观测入度"),
        (-0.22, joint.observed_minus_random, joint.random_p025,
         joint.random_p975, fs.C_OBS, "D", "同时限定观测出度为零"),
    ):
        y = positions + offset
        ax.hlines(y, standard.observed_E_fixed - high, standard.observed_E_fixed - low,
                  color=color, linewidth=2)
        ax.scatter(mean, y, color=color, marker=marker, s=50, label=label)
    ax.set(yticks=positions, yticklabels=["162 个全挂标记", "129 个判为真失效"],
           ylim=(-0.5, 1.5), xlim=(-0.015, 0.048),
           xlabel="剩余效率差（观测名单减随机对照）")
    ax.yaxis.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.43), ncol=3)
    save(fig, "fig12_matched_efficiency")


def draw_e3_thresholds():
    e3 = PIPELINE / "out/E3_percolation"
    sys.path.insert(0, str(e3))
    from run_e3 import build_sets, gcc_track, ordered, p_c

    data = build_sets()["n674"]
    n0 = data["n0"]
    band = pd.read_csv(e3 / "e3_random_gcc_band.csv")
    steps = np.rint(band.f.to_numpy() * n0).astype(int)
    assert np.array_equal(steps, np.arange(1, len(band) + 1))
    curves = {"random": np.r_[1, band.pct_mean.to_numpy() / 100]}
    for strategy in ("betweenness", "degree"):
        order = ordered(data["G"], data["Gw"], data["Gd"], strategy, "primary")
        curves[strategy] = gcc_track(data["G"], order, len(band)) / n0
    thresholds = pd.read_csv(e3 / "e3_perc_thresholds.csv").query(
        "mode == 'static' and rank_basis == 'primary' and node_set == 'n674'"
    ).set_index(["strategy", "level"])

    fig, ax = plt.subplots(figsize=(8.1, 3.6))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.95, bottom=0.29)
    x = np.arange(len(band) + 1) / n0
    for strategy, label, color in (
        ("random", "随机均值（200 次）", fs.C_NULL),
        ("betweenness", "按介数移除", fs.C_OBS),
        ("degree", "按度移除", fs.C_EDGE2),
    ):
        curve = curves[strategy]
        ax.plot(x, curve, color=color, label=label, linewidth=2)
        for level in (0.5, 0.25):
            _, k = p_c(curve, 1, n0, level, len(band))
            assert k == thresholds.loc[(strategy, level), "k_at_pc"]
            ax.scatter(x[k], curve[k], color=color, s=21, zorder=5)
    for level in (0.5, 0.25):
        ax.axhline(level, color="#899CA7", linestyle="--", linewidth=0.9)
    ax.set(xlim=(0, 0.70), ylim=(0, 1.02), yticks=[0, 0.25, 0.5, 0.75, 1],
           xlabel="移除节点比例", ylabel="最大连通分量占原规模")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)
    save(fig, "fig_e3_threshold_curves")


if __name__ == "__main__":
    fs.setup()
    draw_e3_thresholds()
    draw_missing_links()
    draw_matched_comparison()
