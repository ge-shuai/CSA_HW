# D3_failure_predictors · result.md

`F:` = `out/D3_failure_predictors/` 下文件与列；`c:` = `run_d3.py` 行号。
结局 `true_fail`（A_conn_fail + D_not_found + F_server_error = 129），事件率见下。全部随机过程 seed=20261113。

## 1. 结论表

| 指标 | 值 | 来源 |
|---|---|---|
| 样本（校正前 / 校正后） | 671 / 638（638 = 671 − 32 假失效 − 1 `G_other`） | `c:74-79`；`F:d3_summary.json:n_uncorr,n_corr` |
| 事件数 / 事件率 | 129 / 129；0.1923 / 0.2022 | `F:d3_feature_summary.csv:n_events,event_rate` |
| 入度中位数（IQR） | 2（IQR=3），两样本同 | `F:d3_feature_summary.csv:in_degree_median,in_degree_iqr` |
| 入度均值：事件组 vs 非事件组 | 1.6047 vs 4.7528（校正后 4.8861） | `F:d3_feature_summary.csv:in_degree_mean_events,in_degree_mean_non_events` |
| Spearman(入度, 事件) | −0.2959（校正后 −0.3103） | `F:d3_feature_summary.csv:spearman_in_degree_vs_event` |
| Spearman(log(1+url_count), 事件) | **−0.5447**（校正后 −0.5803） | `F:d3_feature_summary.csv:spearman_log1p_url_vs_event` |
| **主口径模型** | `N2_url_indegree_PRIMARY` = `log1p_url + in_degree`（父模型 `M1_url_only`） | `c:305-306`, `c:311` |
| **主口径结果：入度系数** | −0.0484（se 0.0674，**p=0.4727**）；校正后 −0.0381（se 0.0620，**p=0.5388**） | `F:d3_coefficients.csv:coef,se,p`；`F:d3_summary.json:key_estimates` |
| 入度 per IQR(=3) 的 OR | 0.8649（校正后 0.8919），CI 跨 1（p 不显著） | `F:d3_coefficients.csv:or_at_effect_scale` |
| 单变量（只放入度）`N1` | 系数 −0.4783，se 0.0867，p=3.41e-08（校正后 −0.4900，p=1.86e-08）；cv AUC 0.6969 / 0.7141 | `F:d3_coefficients.csv`；`F:d3_cv_metrics.csv:cv_auc` |
| 只放规模 `M1` | log1p_url 系数 −1.8218，se 0.2221，p=2.34e-16（校正后 −1.9217）；cv AUC 0.8912 / 0.9063 | 同上 |
| **ΔAUC（N2 − M1）** | **+0.0068**，95% CI [0.0025, 0.0113]（校正后 +0.0059，CI [0.0021, 0.0137]），20 次 bootstrap | `F:d3_delta_auc_bootstrap.csv:delta_auc,delta_ci_lo,delta_ci_hi,n_boot_valid`；`c:478-486` |
| 嵌套 LRT：N2 vs M1 | χ²=0.6493，df=1，p=0.4204（校正后 χ²=0.4438，p=0.5053） | `F:d3_model_selection.csv:lrt_vs_parent_chi2,lrt_vs_parent_p` |
| 5 折 CV AUC（主口径 N2） | 0.8980（校正后 0.9122）；10 折 0.8949 / 0.9120 | `F:d3_cv_metrics.csv:cv_auc,k` |
| 置换零模型（20 次打乱标签） | 观测 0.8980 vs null mean 0.4982（sd 0.0243），z=16.48，经验 p=0.04762 | `F:d3_auc_permutation_null.csv`；`c:556` |
| 类别变量：可估计性 | **4/5 个类别水平零事件**，129 个事件全部落在"其他"（301 点，事件率 0.4286）；M2/M3/M4 MLE 不收敛（n_iter=300）、判定分离 | `F:d3_covariate_leakage.csv`；`F:d3_model_selection.csv:separated,diverged_terms`；`c:300-303` |
| 认证站点标记 | 18 个 `B_auth_or_skip`，真失效 0 个；进 638 样本后为 0 → 该项被跳过，M4 在 638 上等价于 M3 | `F:d3_auth_flag_evidence.csv`；`F:d3_skipped_terms.csv`；`c:320-324` |
| 活站内部（`is_partial` 结局，509 站 149 事件） | 入度系数 −0.0073，p=0.587（per IQR=4 的 OR 0.971）；`partial_clean`（80 事件）入度为 +0.0061，p=0.672 → 同样无效应 | `F:d3_alive_only_secondary_models.csv:coef,p,or_at_effect_scale` |

嵌套模型 AIC（校正前样本，`F:d3_model_selection.csv:aic`）：M0 658.87 / M1 388.12 / M2 300.77（分离，无效） / M3 302.37（无效） / M4 272.19（分离，无效） / N1 591.92 / **N2 389.47**。

## 2. 必须回答：控制规模后，入度是否仍有独立效应？

**答：没有。** 三条互补证据：

1. **系数**：主口径 `N2` 里入度系数 −0.0484，p=0.4727（校正后 −0.0381，p=0.5388）。方向为负但完全在噪声内；per IQR 的 OR=0.8649，95% CI 覆盖 1（`F:d3_coefficients.csv`）。
2. **增量判别力**：ΔAUC(N2 − M1) = **+0.0068**（0.68 个百分点，CI [0.0025, 0.0113]）。CI 不含 0 但量级远小于规模项带来的 ΔAUC(N2 − M0) = +0.4023。等价地，AUC 从 0.8912（只有规模）到 0.8980（加进度）。
3. **单变量到多变量的坍缩**：只放入度时 p=3.41e-08（cv AUC 0.6969），一旦同时放 log(url_count)，p 掉到 0.47。说明单变量上的负相关由 `url_count` 承载：`ρ_spearman` 是 −0.2959（入度）vs −0.5447（log url）。这与 D2 的删失事实一致——入度本身也是"被抓到多少页"的产物。
4. **活站内部再检验**：在 509 个活站上把结局换成"部分失效"，入度同样无效应（p=0.587 / 0.672，`F:d3_alive_only_secondary_models.csv`）。所以"结构位置可预测健康"这条在控制规模后不成立，只在**不控制规模**时成立。

## 3. 关键坑的显式回应

**（坑）用交叉验证，不许吹 AUC。**
- 全部判据都来自 5 折分层 CV 的 OOF 预测（`c:399`，`k=5` 主、`k=10` 备，`F:d3_cv_metrics.csv:k`）；5 折与 10 折的 AUC 差 ≤0.003（如 N2 0.8980 vs 0.8949），说明 CV 划分不敏感。
- **0.898 不能当作"能预测失效"的证据**：其中 0.8912 来自 `url_count` 一个变量（M1），类别变量（若可估计）会再加 0.0434（M2−M1）。把 AUC 归给"网络位置"是错的。
- 置换零模型 p=0.04762 是 **20 次置换能给出最小值 1/21**（`F:d3_auc_permutation_null.csv:n_perm=20`），它只说明"比随机标签好"，分辨率不够做显著性判决；不得引用为"p<0.05 显著"。
- 阈值：主判正阈值取 0.5（`c:62`），另报"阈值=先验"与"OOF 上 F1 最优"两档（`F:d3_threshold_metrics.csv`）。F1 最优阈值是在同一份 OOF 上挑的，**乐观偏倚**，`F:d3_f1_threshold_choice.csv:caveat` 已记录（N2 校正前 thr=0.3388→F1=0.6534；校正后 thr=0.4097→F1=0.7099）。

**（坑）类别不平衡 + 类别变量是"结局的孪生项"。**
- 129/671 = 19.2% 事件，EPV 对 N2 为 64.5，对 M4 为 18.43（`F:d3_model_selection.csv:epv`）。
- 更严重的是**定义性泄漏**：B6 的 `title_kind` 里 `fail_placeholder`（标题就是"[连接失败]"）173 点里有 129 个真失效（74.57%），`informative` 449 点里 0 个（`F:d3_covariate_leakage.csv`）。所以"类别"不是预测变量而是结局的另一种写法；M2/M3/M4 的分离（4 个水平零事件）与不收敛正是这一点的机械后果，它们的高 AUC（0.936–0.948）**不可解读**。
- 主口径因此只在无泄漏项上做推断（`log1p_url` + `in_degree`），`c:304-306` 的模型命名（`无泄漏`/`含标题语义（与结局同渠道）`）已把这一点写进代码。

**（坑）多重共线与删失。**
- 入度与 `url_count` 同向（站点被抓到的页数越多，被指向的机会越多），所以"控制规模"会吃掉入度的大部分方差；这不是"入度无用"的因果表述，正确说法是"在本数据里，入度对真失效的**独立**判别贡献 ≤0.7 个百分点"。
- 禁用 `nodes_full.degree`、`color`、以及 `out_degree`（死站出度 100% 删失，§2.1）：`F:d3_feature_summary.csv:degree_u_in_model,out_degree_in_model` 均为 0，`F:d3_summary.json:banned_columns_used` 为空。

## 4. 可复现

`RUN.txt`：`PYTHONHASHSEED=0 python out/D3_failure_predictors/run_d3.py`，SEED=20261113，wall 4.3s，anchors 31/31 PASS（`F:d3_summary.json:anchor_line`），sklearn 1.8.0 / scipy 1.17.1。类别标签由 B6 的 `classify_series` 直接导入（`c:82-103`），与 `title_labels.csv` 一致率 1.0（`F:d3_summary.json:category_agreement_with_title_labels`）。

## 5. 局限

- 横截面单次观测，无时间维度；"预测"仅指同批数据的统计关联，不是对未来失效的预测能力。
- 主口径的 ΔAUC 区间虽不含 0，但 20 次 bootstrap 的 CI 宽度（0.0088）与效应量（0.0068）同量级，不宜当作实质效应。
- 与 CONVENTIONS §2.1 给的 ρ(入度, 可用率)=0.216 不冲突：本单元的 −0.2959 是 ρ_spearman(入度, **二值真失效**)，目标变量不同（连续可用率 vs 真失效指示），正文引用时必须写明是哪一个目标，也不得把总度版本的 0.376 混进来。
