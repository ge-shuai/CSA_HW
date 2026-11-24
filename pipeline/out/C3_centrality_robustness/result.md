# C3_centrality_robustness · result.md

`F:` = `out/C3_centrality_robustness/` 下文件与列；`c:` = `run_c3.py` 行号。
候选池全程固定为 671 个有属性节点，`TOP_K=20`（`c:49`, RUN.txt `candidate pool identical across variants`）。

## 1. 结论表

| 指标 | 值 | 来源 |
|---|---|---|
| 图口径 a_full | 676 节点 / 2498 无向边 / 2788 弧 | `F:graph_variant_profile.csv:nodes,uedges,darcs` |
| 图口径 b_dehub（删 6 枢纽全部关联边） | 1993 边，删 505 条（20.22% 边、15.68% 权重），25 个有属性点孤立 | 同上 |
| 图口径 c_bb10（只留无向 w≥10 强边） | 675 边，删 1823 条（72.98% 边、仅 6.92% 权重），303 个有属性点孤立 | 同上 |
| 判定阈值 | `J_PASS=0.50`（=两榜至少共有 14/20 人） | `c:54`, `c:20-22` |
| 随机两张 Top-20 的 J 期望 / p99 | 0.01536 / 0.08108 | `c:253`；`F:c3_summary.json:threshold_justification` |
| 阈值 / 随机期望 倍数 | 32.5× | 同上 |
| 12 个指标中"口径门"通过 | **4/12**：`hits_hub`(0.8182)、`hits_auth`(0.6667)、`out_strength`(0.6000)、`in_strength`(0.5385) | `F:decision_table.csv:VERDICT,j_min_worst_over_tie_perms`；`c:318,324` |
| 12 个指标中"口径门"失败 | **8/12**：`pr_uu`(0.4286)、`hidx_in_in`(0.4286)、`in_degree`(0.3793)、`bet_u`(0.3333)、`deg_u`(0.2500)、`pr_dw`(0.2121)、`out_degree`(0.1765)、`kcore`(0.0811) | 同上 |
| 口径+抽样双门通过（无单点脆弱） | **2/12**：`in_strength`、`out_strength` | `F:c3_summary.json:strict_pass_measures`；`c:354` |
| 口径过但"单点脆弱" | `hits_hub`、`hits_auth`（删 10% 节点后 J 最小 0.0256 / 0.0811，p10 0.0811 / 0.1765） | `F:decision_table.csv:boot_drop10_min_j,boot_drop10_p10_j,single_point_fragile`；`c:321` |
| 三口径共同 Top-10 人数（最稳者） | `hits_auth` 10/10、`hits_hub` 9/10；`bet_u` 仅 1/10、`deg_u` 2/10 | `F:decision_table.csv:top10_shared_all3_variants`；`c:322-323` |
| 抽样稳定性（删 10% 节点 ×50 次）相对"上限" | 多数指标 0.90–0.98，`kcore` 0.2458 为孤例 | `F:bootstrap_stability.csv:j_mean_relative_to_ceiling` |
| 并列边界（第 20 名并列节点数，全图口径） | `kcore` 45 个并列、`deg_u`/`out_degree` 3 个、`in_degree` 2 个 | `F:tie_boundary.csv:n_at_value`；`c:348-349` |

主要两两口径 Jaccard（确定性排序；`F:jaccard_matrix_deterministic.csv`，行=指标，列=`a_full|b_dehub` / `a_full|c_bb10` / `b_dehub|c_bb10`）：

| 指标 | a–b | a–c | b–c |
|---|---:|---:|---:|
| deg_u | 0.5385 | 0.4286 | 0.2903 |
| in_degree | 0.6667 | 0.5385 | 0.3793 |
| out_degree | 0.6000 | 0.3793 | 0.2121 |
| in_strength | 0.5385 | 1.0000 | 0.5385 |
| out_strength | 0.6000 | 0.9048 | 0.6000 |
| bet_u | 0.4815 | 0.3793 | 0.3333 |
| pr_dw | 0.3793 | 0.5385 | 0.2121 |
| pr_uu | 0.5385 | 0.7391 | 0.4286 |
| hits_hub | 0.9048 | 0.9048 | 0.8182 |
| hits_auth | 0.8182 | 0.6667 | 0.6667 |
| kcore | 0.5385 | 0.2121 | 0.2903 |
| hidx_in_in | 0.6667 | 0.5385 | 0.4815 |

## 2. 必须回答：榜单稳不稳？

**答：整体不稳。12 个指标里 8 个不能当结论，只有强度类（`in_strength`/`out_strength`）双门全过。**

1. **禁止把 Top-10 当结论**（`c:25` 明文规定，`c:324-327` 执行）：`deg_u`、`in_degree`、`out_degree`、`bet_u`、`pr_dw`、`pr_uu`、`kcore`、`hidx_in_in` 这 8 个指标的 Top-10/Top-20 只允许作"参考排序"，正文不得据此断言"哪个站点更重要"。它们的 `j_min_worst_over_tie_perms` 全部 < 0.50，最低 `kcore` 0.0811（三对口径两两之间只有 1–2 个共同成员）。
2. **"只跟基准比"会翻案 3 个指标**：`pr_uu`、`hidx_in_in`、`in_degree` 与 a_full 比时 J=0.5385 过线，但 a_full 与 c_bb10 或 b–c 之间掉到 0.21–0.48。只报"去枢纽后还稳"是选择性口径（`F:c3_summary.json:flip_if_compared_only_to_reference`，`c:356`）。
3. **能当结论的只有强度类**：`in_strength`/`out_strength` 在三种图口径下 J_min 分别 0.5385/0.6000、删 10% 节点后 J 均值 0.7455/0.7343 且 min 0.5385（`F:bootstrap_stability.csv`）。即"按观察到的链接次数排序"比"按邻居个数排序"更耐口径变更。
4. **HITS 两类要分开读**：口径门通过（0.8182/0.6667）且三口径 Top-10 几乎完全一致（9/10、10/10），但抽样极脆（删 10% 节点后 J 可掉到 0.0256/0.0811）。结论只能写成"在给定图上 hub/authority 榜高度自洽，但对节点抽样极不稳定"，不能写成"榜单稳定"。
5. **k-core 特别不能报**：45 个节点在第 20 名并列（`F:tie_boundary.csv`），任何单一打破顺序都任意；删 10% 节点后 J=0.2085（相对上限 0.2458）。

## 3. 关键坑的显式回应

- **节点集必须同池可比**：三种口径都在 671 个有属性节点上取排名，被删边的孤立点不剔除（否则"变稳"是换池的假象）。c_bb10 下 303 个点孤立（`F:graph_variant_profile.csv:n_attributed_isolated_in_u`），它们的中心性为 0/NaN 排序末尾，不参与 Top-20，但仍在池内。
- **并列必须打散取最坏值**：所有 Jaccard 都给了 100 次随机节点序的 min（`c:50`），主判据用最坏值 `j_min_worst_over_tie_perms`，不用确定性排序值 `j_min_deterministic`。两者在 `deg_u`（0.2903→0.2500）、`out_degree`（0.2121→0.1765）、`kcore`（0.2121→0.0811）上确有差别（`F:jaccard_matrix_deterministic.csv` vs `jaccard_matrix_worstcase.csv`）。
- **有向指标在 b/c 口径下需要弧映射，这是 CONVENTIONS 未定义的扩展口径**（`c:15-17`）。本单元用的规则是"无序点对合并权重≥10 ⇒ 保留两条弧"，并给了"按弧自身权重≥10"的诊断对照：`pr_dw` 在两个映射之间 J=0.4815、`hidx_in_in` 0.4815、`hits_hub` 0.8182，其余多为 0.90–1.00（`F:c3_summary.json:diag_arcs_by_own_weight`）。**读法**：`pr_dw` 的 FAIL 里有一部分是映射规则带来的，不只是图口径本身；这条不确定性已写在 `c:15-17`，但 `run_c3.py` docstring 说"另在 DISSENT.md 标注"，而 `out/C3_centrality_robustness/` 下**并无 DISSENT.md**——这是本单元的交付缺口（见 §5）。
- **随机对照不是显著性检验**：`null_random_j_mean=0.01536` 只是"两张随机 20 人榜"的期望重叠，用来给 0.50 阈值定位（32.5×），不是 p 值。

## 4. 与其它模块的接口

- 本单元结论会约束 C1（中央性矩阵）、C4、C6、G1、H3 里"排序"的可引用性：**任何引用 `deg_u`/`in_degree`/`out_degree`/`bet_u`/`pr_dw`/`pr_uu`/`kcore`/`hidx_in_in` Top-10 的句子，都必须标注"C3 判定口径不稳（J_min<0.50），仅作参考"**。
- `E1_attack_curves` 的攻击曲线用同一批指标排序，其"策略间差异"应理解为对**给定排序**的敏感性，而不是对"该指标真实重要性"的判定（见 E1 的平手敏感性）。

## 5. 交付缺口（不改数据，只记录）

1. `run_c3.py` docstring（`c:17`）承诺把"b/c 口径的弧映射扩展"写进 `DISSENT.md`，实际目录内无该文件。
2. 目录内的 `_chk_c3.txt`、`_repro.txt`、`_view.txt`、`run_stdout.log` 是调试/过程文件，不属于 CONVENTIONS §9 规定的交付物，未在本文件引用其数值。

## 6. 复现

`RUN.txt`：`PYTHONHASHSEED=0 python out/C3_centrality_robustness/run_c3.py`，SEED=20261113，wall 129.25s（bootstrap 100.36s 占主要），anchors 31/31 PASS，networkx 3.6.1 / pandas 3.0.2 / numpy 2.4.4。
