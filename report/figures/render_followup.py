"""Render the two figures added to the report from the E5 follow-up and D4."""

from pathlib import Path
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent.parent / "pipeline"
sys.path.insert(0, str(PIPELINE))

import figstyle as fs
from matplotlib.patches import FancyArrowPatch

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
    path = PIPELINE / "out/E5_observed_scenarios/matched_indegree_summary.csv"
    table = pd.read_csv(path).set_index("scenario")
    table = table.loc[["dead162", "true129"]]
    y = [1, 0]
    fig, ax = plt.subplots(figsize=(8.6, 3.75))
    fig.subplots_adjust(left=0.21, right=0.96, top=0.78, bottom=0.25)
    ax.set_title("入度分层后，观测名单与随机均值的差距缩小", pad=20,
                 fontsize=14, fontweight="bold")
    ax.scatter(table.uniform_mean_E_fixed, y, marker="s", color=fs.C_NULL,
               s=52, label="同数量均匀随机均值")
    ax.hlines(y, table.matched_p025, table.matched_p975, color=fs.C_EDGE2,
              linewidth=2)
    ax.scatter(table.matched_mean_E_fixed, y, color=fs.C_EDGE2, s=57,
               label="入度分层随机均值")
    ax.scatter(table.observed_E_fixed, y, color=fs.C_OBS, marker="D",
               s=57, label="观测名单")
    ax.set(yticks=y, yticklabels=["162 个全挂标记", "129 个判为真失效"],
           ylim=(-0.48, 1.48), xlim=(0.15, 0.24),
           xlabel="移除名单后剩余的固定分母效率")
    ax.yaxis.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.29), ncol=3)
    fig.text(0.5, -0.055, "横线为 300 次分层抽样结果的第 2.5 和 97.5 百分位。",
             ha="center", fontsize=9)
    save(fig, "fig12_matched_efficiency")


if __name__ == "__main__":
    fs.setup()
    draw_missing_links()
    draw_matched_comparison()
