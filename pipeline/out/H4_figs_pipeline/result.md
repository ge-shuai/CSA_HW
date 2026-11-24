# H4_figs_pipeline · 结论表

只读上游 CSV 成图，本单元不产生任何新的分析量。三张图全部走 `figstyle.save_fig()`，
每张配 PNG 300dpi + 矢量 PDF + `fig_*.csv` 数据表。

## 结论 1：移除顺序决定连通性崩塌的速度，但没有任何一种顺序"均匀地"伤网络

| 策略（都是静态 top-k、平手按 FQDN 升序） | 主分量掉到 10% 以下的 f | 该点 GCC |
|---|---:|---:|
| 度（无向无权） | **0.1795** | 4.6% |
| 介数（无向无权） | 0.2196 | 6.1% |
| k-core | 0.2997 | 6.2% |

- 来源：`E1_attack_curves/e1_curve_efficiency.csv`，按 `gcc_pct < 10` 取最小的 `f_actual`。
- 三种"结构中心性"排序在删掉 18%–30% 的节点后就已把主分量打散，而随机删 60% 仍剩 27%。
- 这不等于"这些站点更容易坏"：曲线说的是**移除顺序与连通性之间的对应**，不是真实失效过程。

## 结论 2：有向加权 PageRank 是确定策略里最温和的一条，但仍始终低于随机

| 同 f 下的 GCC 落差（pagerank_dw − random） | 值 |
|---|---:|
| f=0.10 | **−10.5 pp** |
| f=0.20 | **−17.1 pp** |
| f=0.35 | **−32.3 pp** |

- 来源：同表按 `step` 对齐后作差（`step` 两序列完全一致，32 档）。
- 结论只能在 CONVENTIONS §5 的**有向加权主口径**下说。`pagerank_uu`（无权无向）与度曲线几乎重合，
  见 `E1_attack_curves/DISSENT.md`（ρ=0.977）。
- 本图不拿"PageRank 攻击温和"当定性结论使用，只标出它与随机基线的距离。

## 结论 3：效率只能读 E_fixed，且只能同 k 横比

- `E_fixed` 基线 = **0.331469**（分母恒为 453602 个有序对）；图里画成水平参考线而不是起点。
- 反例（`E5_observed_scenarios/scenario_metrics.csv`，`dead162|as_observed`）：
  删 162 个全挂站时 `E_full` = **0.3536**，**高于**基线；同一情景 `E_fixed` = **0.2039**，
  同 k 随机基线 `E_fixed` = **0.1610** → 该情景的连通效率损失比随机**小**。
  `E_full` 上升完全是分母收缩，不能当结论。

## 结论 4：六个数据陷阱都能把结论翻面

| # | 陷阱 | 关键数字 | 来源 |
|---|---|---|---|
| ① | 出度删失 | 162/162 死站出度为 0；以死站为起点的边 = 0 条 | `data/anchors.csv`, `data/censoring_facts.csv` |
| ② | 假失效 | 真 129 + 假 32 + 待定 1 = 162 | `data/anchors.csv` |
| ③ | 禁用列 | `degree` 列最大 146 vs 重算 119；`color` 与 `ok_ratio_pct` 冲突 32 处 | 同上 |
| ④ | 枢纽匹配 | 全名精确 505 条边 vs 前缀匹配 678 条（解析出 36 个"枢纽"） | `B5_hub_removal/hub_choice_sensitivity.csv` |
| ⑤ | 社区含孤立点 | backbone_w10 上 336–340 个社区，其中仅 29–33 个非孤立 | `B4_weight_variants/weight_variant_per_seed.csv` |
| ⑥ | 分母收缩 | `E_full` 0.3536 ↑ 而 `E_fixed` 0.2039 | `E5_observed_scenarios/scenario_metrics.csv` |

## 图的读法与此单元的边界

- 图 1 `h4_attack_curves`：左右面板纵轴口径不同（GCC% / E_fixed），**不可直接比较**。
  随机阴影是 120 次独立重复的 ±1sd；落差小于阴影宽度时不构成差异。
- 图 2 `h4_pipeline_flow`：纯结构图，不含实测数值；箭头只表示"谁读谁"，不代表执行顺序约束。
- 图 3 `h4_data_traps`：每个数字都标注了来源文件与列名，可逐条复核。
- 已知瑕疵：服务器上 SimHei 没有 bold 字重，matplotlib 回退到 400 权重（控制台有 `findfont` 警告），
  卡片标题的粗体实际是常规体。不影响可读性与数字。
- 本单元不重算：图 1 的 3 个数、图 3 的 6 个区间全部是上游 CSV 的取值或 min/max。
