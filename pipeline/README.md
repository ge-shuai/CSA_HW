# CSA 大作业 · 可复现分析流水线

USTC 网站群超链接网络（671 子域 / 2788 有向链接 / 8115 条 URL 健康记录）。
本目录是**分析与图表**的产出；报告层（新题目、结论台账、报告提纲/答辩）在 `../report/`。
仓库根的 `USTC网络拓扑分析/` 是同学的原始数据，**只读**。

## 环境

```bash
pip install -r requirements.txt
```

实测通过版本：Python 3.12.13 · networkx 3.6.1 · pandas 3.0.2 · numpy 2.4.4 ·
scipy 1.17.1 · scikit-learn 1.8.0 · matplotlib 3.11.2。
（同学原环境是 Python 3.13 / networkx 3.4.2；本流水线的锚点已在新环境全部重算。）

## 跑法

```bash
cd pipeline
python 00_build_data.py          # 生成 data/ 下的清洗产物 + 31 项锚点校验
python <模块>/run_*.py           # 各分析单元，彼此独立，可任意顺序
bash run_all.sh                  # 可选：一键全跑（A–H + verify；耗时可能数小时）
bash run_all.sh --only E5,F3     # 只跑匹配的单元
```

`00_build_data.py` 是唯一的前置步骤。它会把 `data/anchors.csv` 写成 31 行 PASS；
**任何一个下游模块开跑前都会先 `csa_common.assert_anchors()`**，对不上直接报错停止。

## 目录

| 路径 | 内容 |
|---|---|
| `CONVENTIONS.md` | **统一口径（先读这个）**：建图规则、三套度/失效/效率口径、零模型规范、禁用措辞 |
| `MANIFEST.md` | 45 个分析单元的分工、依赖与已知坑 |
| `csa_common.py` | 全项目唯一的建图 / 清洗 / 效率 / 零模型 / Louvain 实现 |
| `00_build_data.py` | 清洗产物生成器 + 锚点自检 |
| `data/` | `clean_nodes.csv` `clean_edges.csv` `dead_categories.csv` `anchors.csv` `censoring_facts.csv` `DATA_DICTIONARY.md` |
| `out/<模块>/` | 每个单元自己的 `result.md` + CSV + `run_*.py` + `RUN.txt`（+ 异议写 `DISSENT.md`） |

## 三条最容易踩的坑（详见 CONVENTIONS）

1. **162 个整站不可用站点的出度全为 0**，且以它们为起点的边数为 0。爬虫只能从抓到的页面提链接，
   所以"死站之间是否相邻"在这份数据里是**构造性不可识别**的，不是"测出来不聚集"。
   失效相关分析的主口径必须是**入度**；"死站平均度 1.83 vs 健康站 9.49"这个 5.2 倍是跨口径误比，
   同口径（入度）是 1.83 vs 4.94 = **2.7 倍**。
2. **删点后 `global_efficiency` 上升是分母收缩假象。** 唯一合法的比较是**同一删除数量下跨情景比**。
3. **加权 Louvain 在本图上对 seed 和建图顺序都敏感**（社区数 23–26，最大社区 80–113）。
   所以边序固定为 `links.csv` 行序（`edge_order`），社区一律报分布不报单值。
