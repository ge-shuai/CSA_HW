# G2_single_point_failure · result.md

`F:` = `out/G2_single_point_failure/` 下文件与列；`c:` = `run_g2.py` 行号（脚本用 `LNO()`/`cite()` 自带上百条行号锚点，`c:67-77`）。
口径：基准图 676 节点 / 2498 无向边，主分量 674（另有一分量 `bm/l.bm` 2 点）；**逐节点静态删除 676 次**（671 有属性 + 5 个悬空引用），`E_fixed` 分母恒为 453602。基准：`E_full`=0.329513、`E_gcc`=`E_fixed`=0.331469（`F:G2_summary.json:baseline_efficiency`；`E_full` 与另外两个不等：其分母含 2 点小分量的点对，这正是三口径必须分列的原因）。

## 1. 结论表

| 指标 | 值 | 来源 |
|---|---|---|
| 删除评估次数 | 676（全图逐点） | `F:G2_summary.json:timing.n_removals_computed` |
| 单点删除后图断连的节点（割点） | **96** | `F:G2_summary.json:counts.n_ap` |
| 有实质影响的节点（丢点对 或 中继项 < −1e-4） | **130**（=130/676，19.2%） | `F:G2_summary.json:counts.n_high_impact`；阈值 `c:604` |
| 删除后完全无变化的节点 | **401**（59.3%） | `F:G2_summary.json:counts.n_zero_all` |
| 度=1 的叶子 | 186（入度=1 的 261） | `F:G2_summary.json:counts.n_leaf,n_indeg1` |
| 叶子删除的最大断连点对数 / 最大 ΔE_fixed | 0 / **−0.00102**（= −0.31% of E_fixed） | `F:G2_summary.json:counts.leaf_pairs_max,leaf_dE_fixed_worst` |
| 影响力集中度（中继项） | Top-1 = **6.11%**、Top-10 = 36.16%、Top-20 = 48.64%、Top-50 = 72.35%、Top-100 = 94.47% | `F:g2_concentration.csv:share_relay_pct` |
| Gini（断连点对 / 中继项 / ΔE_fixed 总量 / 端点机械项） | 0.9125 / 0.9020 / 0.2758 / 0.0931 | `F:G2_summary.json:concentration`；`c:473-476` |
| 130 个高影响节点的健康状态 | healthy 86 / partial 43 / dead 1 | `F:g2_high_impact_by_health.csv:n` |
| 端点机械项占 ΔE_fixed 的比重（全体） | 66.29%（单节点范围 0.105–1.000） | `F:G2_summary.json:counts.total_endpoint_mech,audit.mech_endpoint_share_range` |
| 健全性检查 | `ΔE_fixed>0` 的节点 **0** 个（朴素累加口径会误出 2 个、幅度 2.3e-13）；`ΔE_gcc>0` 的 **405** 个；`ΔE_full>0` 的 **351** 个 | `F:G2_summary.json:audit` |
| 内核校验 | 12 个节点与 `C.eff_three` 逐值比对，max\|diff\| = **2.6e-13**，连通性不一致 0 个 | `F:g2_kernel_validation.csv`；`c:572-573` |

## 2. 必须回答：哪些站点一旦坏了后果最大？

**答：Top-12 全部是 healthy、且全部是割点；名单高度集中在少数"跨院系/跨平台"入口站。**

`F:g2_top_true_spof.csv`（按基线点对损失排序；`dE_fixed` 为负 = 效率损失）：

| # | 节点 | 断连基线点对 | 占基线点对 | ΔE_fixed | 中继项 | 删后分量数 | 健康 |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `employment.ustc.edu.cn` | 14672 | 3.23% | −0.013651（−4.12%） | −0.012220 | 12 | healthy |
| 2 | `lib.ustc.edu.cn` | 13350 | 2.94% | −0.012367 | −0.010962 | 12 | healthy |
| 3 | `en.ustc.edu.cn` | 12024 | 2.65% | −0.010689 | −0.009276 | 11 | healthy |
| 4 | `cec.ustc.edu.cn` | 12024 | 2.65% | −0.008133 | −0.007027 | 11 | healthy |
| 5 | `enbiomed.ustc.edu.cn` | 9366 | 2.06% | −0.007515 | −0.006305 | 9 | healthy |
| 6 | `teach.ustc.edu.cn` | 6700 | 1.48% | **−0.011306** | −0.009676 | 7 | healthy |
| 7 | `sz.ustc.edu.cn` | 6700 | 1.48% | −0.005700 | −0.004445 | 7 | healthy |
| 8 | `lug.ustc.edu.cn` | 6700 | 1.48% | −0.005195 | −0.004136 | 7 | healthy |
| 9 | `fusep.ustc.edu.cn` | 5364 | 1.18% | −0.006990 | −0.005567 | 6 | healthy |
| 10 | `virlab.ustc.edu.cn` | 5364 | 1.18% | −0.003664 | −0.002744 | 6 | healthy |
| 11 | `gradschool.ustc.edu.cn` | 4026 | 0.89% | −0.005458 | −0.004020 | 5 | healthy |
| 12 | `iat.ustc.edu.cn` | 4026 | 0.89% | −0.004117 | −0.002956 | 5 | healthy |

要点：
1. **单点最坏也只损失 4.12% 的 `E_fixed`**（不是数量级崩塌）。"单点故障"在本数据里是真实的但温和的——圈秩 1824（E4 基线）提供了冗余。
2. **排序不能只看断连点对数**：`teach` 断连 6700 对（第 6）但 ΔE_fixed=−0.0113（第 3 序列），它断开的是长距离对；`cec` 断连 12024 对但 ΔE_fixed 只 −0.0081。所以"哪家最疼"必须用 `dE_fixed` 排，`pairs_lost` 只能作辅助（两者 Pearson −0.931、Spearman −0.598，`F:G2_summary.json:concentration`）。
3. **高影响节点几乎全是健康站**：130 个里 86 个 healthy、43 个 partial、仅 1 个 dead（且那个 dead 的 ΔE_fixed 全靠机械项）。即"关键的结构位置"与"当前已经坏了"**不重合**——这正是 G1 风险矩阵要的输入。
4. **集中度极高**：Top-10 承担全部中继损失的 36%，Top-50 承担 72%；Gini(断连点对)=0.912。**必须报分布**：580/676（85.80%）的节点删除后一个点对都不丢（`F:g2_impact_histogram.csv` 第一段），只有 4 个节点丢 >9999 对。

## 3. 关键坑的显式回应

**（坑 1）"度=1 的叶子删了几乎无感"——必须报分布而不是只报 Top。**
- 186 个度=1 节点：断连点对 **0**（`n_leaf_disconnect=0`），最坏 ΔE_fixed = −0.00102（−0.31%）。
- 更大的分布事实：401/676（59.3%）节点删除后无任何影响，其中 261 个是入度=1（死站/叶子类）。**所以"Top-N 单点故障"只有 N≈130 以内的意义，剩下的 80% 节点删除等价于无操作。**
- 若只看 Top-10 会得到"网络极度中心化"的错觉；`g2_concentration.csv` 与 `g2_impact_histogram.csv` 是必须同时引用的两张表。

**（坑 2）ΔE 的机械项必须与中继项分开。**
- 删掉任何节点，所有以它为端点的点对都会记 0，这部分是**机械损失**，不是"网络被切断"。全体节点里端点机械项占 ΔE_fixed 的 **66.29%**，单节点范围 0.105–1.000（`F:G2_summary.json:audit.mech_endpoint_share_range`）。
- 故本单元同时给 `dE_fixed_endpoint_mech` 与 `dE_fixed_relay`，排序以 `relay` 与 `dE_fixed` 双列为准；`g2_top_relay_only.csv` 是按纯中继排序的对照表。

**（坑 3）只有 `E_fixed` 能跨节点比较；`E_full`/`E_gcc` 会给出"删点提升效率"的假象。**
- 硬证据：`ΔE_gcc > 0` 的节点 **405 个**、`ΔE_full > 0` 的 **351 个**，而 `ΔE_fixed > 0` 的 **0 个**（`F:G2_summary.json:audit.n_dE_gcc_positive,n_dE_full_positive,n_dE_fixed_positive`）。连"会断连的节点"里也有 50 个 `ΔE_gcc>0`。**任何"删掉某站后效率上升"的表述都只能来自 E_full/E_gcc 的分母收缩，禁止写成结论。**

**（坑 4）主分析节点集是 671，676 只作附录。**
- 5 个悬空引用（`personnel`、`www2.lib`、`recapi`、`985-211`、`rmd.snst`）的删除结果单列在 `F:g2_appendix_dangling676.csv`：它们都只丢 0 个基线点对（ΔE_fixed 仅 −0.00078~−0.00094，全是端点机械项，中继项 0.0），且不参与主表排序。**"676 口径"不得进入主表。**

**（坑 5）本单元是静态删单点，不是失效预测。**
- 结论措辞："情景模拟显示，删除 X 会使 `E_fixed` 从 0.3315 降到 Y"；不得写"X 坏了会导致…"。健康状态只作对照列（`health_state`/`ok_ratio_pct`），不参与因果解释。

## 4. 可复现与性能（脚本自带内核校验）

- 内核：一趟全源 BFS 同时出 `E_full`/`E_gcc`/`E_fixed` 与分量信息，定义与 `csa_common.eff_three` 一致；抽查 12 个节点，max\|diff\| = **2.6e-13**（断言 <1e-9，`c:572-573`），连通性不一致 0 个。
- 速度：全扫 676 次 = **702.67 s**（1.0394 s/节点），连通用割点捷径（96 个割点重算 + 其余推断）比朴素快 **4.44×**，效率比朴素外推（8206.9 s）快 **11.68×**；总墙钟 **833.83 s**，`seed=20261113`（只用于抽取 12 个校验节点）。
- 版本：python 3.12.13 / networkx 3.6.1 / pandas 3.0.2 / numpy 2.4.4（`F:G2_summary.json:versions`）。
- **注意**：本目录原缺 `RUN.txt` 与 `result.md`（`run_g2.py` 在 `c:894` 才写 RUN.txt、`c:893` 写 result.md，即脚本在生成报告段落之后被终止）。`RUN.txt` 已按 lead 要求依既有 `G2_summary.json` 与脚本常量重建，文件头已注明；本 `result.md` 的数字全部取自既有 CSV/JSON。

## 5. 交付物对照

`g2_single_removal_full.csv`（676 行逐点全量）、`g2_top_true_spof.csv`(30)、`g2_top_relay_only.csv`、`g2_top_by_total_dE_fixed.csv`、`g2_high_impact_list.csv`(130)、`g2_high_impact_by_health.csv`、`g2_all_by_health.csv`、`g2_impact_histogram.csv`(24 bins)、`g2_concentration.csv`、`g2_kernel_validation.csv`、`g2_appendix_dangling676.csv`、`G2_summary.json`、`g2_baseline_distance_profile.npy`（基线全源距离画像，供复核）。
