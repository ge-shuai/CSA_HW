# 外部审阅简报（REVIEW_BRIEF）

> 给你另一个 agent / 另一位同事：用最少的时间判断"这套东西可不可信、哪里还没做完"。
> 不需要读 595 个文件。

## 项目地址

```
C:\Users\bfy32\Documents\Qoder\2026-11-13\f93e4d48
```

git 仓库已初始化，两个提交，**无远端**（尚未 push）。可搬运副本：`C:\Users\bfy32\Desktop\csa-pipeline.bundle`
（`git clone csa-pipeline.bundle` 即可得到完整历史）。

```
f93e4d48\
├─ README.md               入口：从哪读起、快速开始
├─ pipeline\               ← 全部交付物在这里
│   ├─ PROGRESS.md         当前状态的唯一权威（45 单元状态、修正过的事实错误、已知缺口）
│   ├─ NEXT_STEPS.md       后续工作指引（P0/P1/P2、C6 补法、报告写作、答辩问答）
│   ├─ CONVENTIONS.md      统一口径（建图、E_full/E_gcc/E_fixed、社区计数、因果措辞）
│   ├─ MANIFEST.md         45 个单元的任务定义与"关键坑"
│   ├─ csa_common.py       共享地基（建图/锚点/效率/零模型/社区稳健性）——只 import，不许重写
│   ├─ data\               清洗产物 + 31 项锚点表
│   └─ out\<单元>\         每单元 result.md + CSV + run_*.py + RUN.txt
├─ verify\                 独立校验脚本（check_claims.py 等）
└─ CSA_HW\                 课程/同学的原料仓库（只读，已 gitignore，不属于交付物）
```

## 30 分钟阅读路线（按顺序）

1. `README.md`（2 分钟）
2. `pipeline/PROGRESS.md` §0–§2（5 分钟）——先知道做完什么、改过什么错
3. `pipeline/CONVENTIONS.md` §1/§2/§4/§7（8 分钟）——这是"契约"，所有结论都受它约束
4. `pipeline/out/I1_number_consistency/result.md` + `I2_spec_conflicts/result.md`（5 分钟）
   ——这两份是**审计**，直接告诉你哪里有毛病，不用自己找
5. `pipeline/NEXT_STEPS.md` §3/§4（8 分钟）——没做完的与已知缺陷，逐条带修法

## 你可以直接验的四个硬指标

```bash
cd pipeline
python 00_build_data.py            # 必须 31/31 锚点 PASS
ls data/anchors.csv                # 逐项 expected vs got
python ../verify/check_claims.py   # 声明与数据对账
grep -rn --include='result.md' -E '导致|因为|使得|引发|决定了|传染' .   # 应只剩 7 处禁令原文
```

## 这套东西的可信度靠三件事撑着（要查就查这三条）

1. **每个数字都能指到代码行**：`result.md` 里的值后面都跟着 `run_xx.py:L123` 或 `文件:列`。
   抽查任意 5 个数字，都能在对应 CSV 里找到。
2. **口径冲突被写下来而不是被抹平**：`out/*/DISSENT.md` 有三份（E1 的 PageRank 契约异议最典型：
   契约 66.2/41.5/12.5 复现不出，实测 55.19/3.56/0.59）。
3. **跨机复现过了**：在干净的 Ubuntu 22.04 上重跑，`data/` 产物除换行符外逐字节一致；
   E3/E2 在独立拷贝里重跑的 CSV 逐字节相同（`I5_clean_reproduce`）。

## 已知没做完 / 别去夸大的地方（审阅重点）

| 项 | 状态 |
|---|---|
| C6_centrality_vs_health | **没做**。理由与补法见 `NEXT_STEPS §3.1`（与 D3/D1/G1 三重重复） |
| E5 的 3 张附表 | 缺（主结论 4 张表齐） |
| F3 的 result.md 主体 | 仍指旧版脚本口径，靠文末「重跑修订」段兜底；`NEXT_STEPS §4.10` 要求重写 |
| "149 个部分失效" | D6 实测：剔掉 -1/-2/403 后 63 站实为健康。正文引用必须带限定语（`CONVENTIONS §2.2b`） |
| 6 个 `RUN.txt` | 写死了已删除的服务器路径（死引用，不影响结论） |
| C3 的 `DISSENT.md` | docstring 承诺过但没写 |

## 别动的文件（动了就没法对账）

- `CSA_HW/` —— 只读原料，`git status` 里应始终为空
- `pipeline/csa_common.py`、`pipeline/data/` —— 改它们等于改了全项目的口径基线
- 别人的 `out/<单元>/` 目录 —— 每个单元只写自己的目录

## 想判断"这活干得好不好"，就问三个问题

1. 随便挑 3 个 `result.md` 的数字，能在 CSV 里复现吗？
2. 报告里如果写"删死站让效率上升"，是错的——产出里有没有明确禁止这句？
3. 社区数写"25 个"算不算错？答案：算，必须给 23–26 的区间。

三个都过，这套东西就基本可信；剩下的都是"补完整性"层面的活。
