# CSA_HW —— USTC 子域网络结构分析（CSA 大作业）

> 交付形态：**可复现分析管线 + 全部图表数据**。不含报告正文。

## 从哪读起

| 文件 | 作用 |
|---|---|
| `pipeline/PROGRESS.md` | **当前状态的唯一权威**：45 个单元的状态、修正过的事实错误、已知缺口、复现方式 |
| `pipeline/NEXT_STEPS.md` | **后续工作指引**：还差什么、逐条怎么改、报告怎么写、答辩 10 问答 |
| `pipeline/CONVENTIONS.md` | 统一口径（建图、三套效率口径、社区计数、因果措辞）——改动它等于改契约 |
| `pipeline/MANIFEST.md` | 45 个单元的任务定义与"关键坑" |
| `pipeline/csa_common.py` | 共享地基：建图、锚点、效率、零模型、社区稳健性。**所有单元只 import，不重写** |

## 快速开始

```bash
conda create -y -n csa python=3.12
pip install -r pipeline/requirements.txt
# 中文字体（否则出图直接抛错）
mkdir -p ~/.fonts && cp /path/to/simhei.ttf ~/.fonts/ && fc-cache -f

cd pipeline
python 00_build_data.py      # 必须 31/31 锚点 PASS，否则停
python out/<单元>/run_<xx>.py
```

**原料位置自动定位**：`csa_common.py` 先读环境变量 `CSA_DATA_DIR`，否则在仓库根找 `USTC网络拓扑分析/`；两处都找不到时抛 `FileNotFoundError`，不做静默回退。

## 目录

- `USTC网络拓扑分析/` 原始爬虫数据（`nodes_full.csv`、`links.csv`、死站清单等）
- `选题分析_USTC网络拓扑/` 前期预实验与选题材料
- `pipeline/data/` 清洗产物与锚点表（由 `00_build_data.py` 生成）
- `pipeline/out/<单元>/` 每个单元的 `result.md` + CSV + `run_*.py` + `RUN.txt`
- `verify/` 独立校验脚本

## 规模

45 个单元中 44 个已交付（仅 C6 未做，理由见 `NEXT_STEPS.md §3.1`）：
44 份 `result.md`、45 份 `RUN.txt`、342 个 CSV、13 张图（PNG 300dpi + 矢量 PDF + 伴生数据表），
外加 I1–I5 五个审计单元（硬伤 0 条）。
