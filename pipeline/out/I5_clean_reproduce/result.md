# I5_clean_reproduce · 可复现抽查

## 做了什么

把整条流水线按**原布局**拷到 `/tmp/repro/`（`pipeline/` + 同级 `CSA_HW/`，共 34 MB），
在新目录里用同一环境重跑两个单元，再逐文件比 md5：

| 单元 | 重跑墙钟 | 说明 |
|---|---:|---|
| `E3_percolation` | 21.83 s | 静态渗流阈值 + 200 reps 随机基线 |
| `E2_dynamic_attack` | 60.59 s | 动态介数/度/PageRank/k-core，404 次重算 |

## 结果：数据层零漂移

| 判读 | 文件数 |
|---|---:|
| 逐字节一致（CSV + result.md + run_*.py） | **15** |
| 预期不同（`RUN.txt` / `*_summary.json` / `run_stdout.log`） | 5 |
| 数值不一致 | **0** |

- E3 的 7 个 CSV 全部 md5 相同；E2 的 4 个 CSV 全部 md5 相同。
- 差异只出现在 `RUN.txt` 与两份 `*_summary.json`，逐字段核对后确认**只差两样东西**：
  1. 墙钟（E3 total 21.5→21.2 s；E2 total 60.6→60.0 s）；
  2. 绝对路径（`cross_check_vs_E1/file`、`static_vs_e1_crosscheck/file` 指向 `/tmp/repro/...`）。
  **没有任何一个统计量变化**。
- 明细见 `I5_reproduce_check.csv`（20 行，含两侧 md5）。

## 数据层与锚点

- 服务器上 `00_build_data.py` 重跑后 **31/31 锚点 PASS**，与 11-22 本机那次一致。
- `pipeline/data/` 的 8 个 CSV 与本机 11-22 产物**除换行符外逐字节一致**（CRLF→LF 之差：
  `anchors.csv` 本地 980 B / 服务器 948 B，归一化后哈希相等）。也就是说跨机器的漂移检查通过了。

## 已知不可复现 / 未复现的部分（必须诚实列出）

| 项 | 状态 |
|---|---|
| `E4_edge_attacks` | 原始运行在 3990 次评估处被外部打断；本轮修 `q_removed` KeyError 后重跑成功（1073 s，rc=0），**当前产出以本轮为准** |
| `F3_template_verdict` | 旧 `power_curve.csv` 与 `run_f3.py` 的 β 网格不一致（旧档 0/0.25/…，脚本 11 档）。本轮用当前脚本重跑覆盖，脚本↔结果已自洽；写 `null_model_audit.csv` 时曾因 `.iloc[0]` 越界中断，已打补丁 |
| `B1_louvain_stability` 等 20 个早期单元 | **未重跑**（成本高、且 11-22 已在本机跑过）。它们的 `RUN.txt` 记的是本机 11-22 的环境 |
| `E5` 的 3 张附表 | 原始运行未产出（`damage_vs_selection_strength` / `contract_crosscheck` / `attack_reference_curve`），主结论 4 张表齐 |

## 局限

- 抽查只覆盖 2 个单元（都在 1 分钟量级）。E1（1823 s）、E6（369 s）、C5、G3（126 s）未重跑，
  因为重跑它们对"是否能复现"的边际信息小于成本。
- 重跑目录 `/tmp/repro/` 与服务器工作目录收工即删；本报告只保留 md5 与墙钟证据。
