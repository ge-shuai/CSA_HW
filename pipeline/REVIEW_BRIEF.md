# 外部审阅简报（REVIEW_BRIEF）

> 给你另一个 agent / 另一位同事：用最少的时间判断"这套东西可不可信、哪里还没做完"。
> 不需要读 595 个文件。

## 项目地址

```
C:\Users\bfy32\Documents\Qoder\2026-11-13\f93e4d48
```

git 历史已并入同学的仓库：本地 `main` 同时包含他的历史（`a065a06`）与我们的 5 个交付提交，
由合并提交 `2283116` 接起来；合并前我们的独立历史保留在标签 `premerge-3f88640`。

**已推送（2026-11-24）**：远端新分支 `csa-pipeline` = `a8c4369`，他的 `main` 仍是 `a065a06` 未动。
PR 入口：`https://github.com/ge-shuai/CSA_HW/pull/new/csa-pipeline`。
本地已配 `origin` 并跟踪 `origin/main`，当前 `main` 领先 7 个提交、落后 0（可直接 fast-forward）。
以后再推同一分支：

```bash
git push -u origin main:csa-pipeline
```

旧的离线副本 `C:\Users\bfy32\Desktop\csa-pipeline.bundle` 是**修正前**的版本，不要拿它当审阅入口。

```
CSA_HW\                    仓库根 = 同学的公开仓库（原料与交付物同仓）
├─ README.md               入口：从哪读起、快速开始
├─ USTC网络拓扑分析\        同学的原始数据（nodes_full/links/死站清单等，只读）
├─ archive\00_选题分析_预实验\  前期预实验与选题材料（已废弃，只读）
├─ pipeline\               ← 我们的交付物在这里
│   ├─ REVIEW_BRIEF.md     本文件
│   ├─ WALKTHROUGH.md      从头到尾的解说（接手/创新的人先读这份）
│   ├─ PROGRESS.md         当前状态的唯一权威（45 单元状态、修正过的事实错误、已知缺口）
│   ├─ NEXT_STEPS.md       后续工作指引（P0/P1/P2、C6 补法、报告写作、答辩问答）
│   ├─ CONVENTIONS.md      统一口径（建图、E_full/E_gcc/E_fixed、社区计数、因果措辞）
│   ├─ MANIFEST.md         45 个单元的任务定义与"关键坑"
│   ├─ csa_common.py       共享地基（建图/锚点/效率/零模型/社区稳健性）——只 import，不许重写
│   ├─ data\               清洗产物 + 31 项锚点表
│   └─ out\<单元>\         每单元 result.md + CSV + run_*.py + RUN.txt
└─ verify\                 独立校验脚本（check_claims.py 等）
```


## 30 分钟阅读路线（按顺序）

> 只想懂全貌、不准备逐项核实用的人：改读 `pipeline/WALKTHROUGH.md`（约 8 分钟），
> 它从头讲原料、四处天生缺陷、每组单元在回答什么、以及结论能带到哪一步为止。下面五步是给逐项核实用的。

1. `README.md`（2 分钟）
2. `pipeline/PROGRESS.md` §0–§2（5 分钟）——先知道做完什么、改过什么错
3. `pipeline/CONVENTIONS.md` §1/§2/§4/§7（8 分钟）——这是"契约"，所有结论都受它约束
4. `pipeline/out/I1_number_consistency/result.md` + `I2_spec_conflicts/result.md`（5 分钟）
   ——这两份是**审计**，直接告诉你哪里有毛病，不用自己找
5. `pipeline/NEXT_STEPS.md` §3/§4（8 分钟）——没做完的与已知缺陷，逐条带修法

## 你可以直接验的四个硬指标

```bash
cd pipeline
python 00_build_data.py            # 必须 raw 与 clean reload 各 31/31 PASS
ls data/anchors.csv                # 逐项 expected vs got
python ../verify/check_claims.py   # 现在带断言：契约项违反会退出码 1
# 因果词。注意裸 grep 的命中数不是 7，见 out/I4_causal_language/result.md 的
# 「裸 grep 的命中数怎么对上」一节：要先排掉 I4 自己的前后对照表。
grep -rn --include='result.md' -E '导致|因为|使得|引发|决定了|传染' . \
  | grep -v I4_causal_language     # 剩下的应全部是"不得写成…"这类禁令原文
```

> 2026-11-24 外部审阅的实测结果：`00_build_data.py` 31/31 通过，且重跑后 `git status` 仍干净
> （说明 `data/` 逐字节可复现）；`check_claims.py` 17 项契约断言全过、退出码 0；
> 抽查 5 个数字（A1 幂律 α=1.9759/x_min=4、D4 的 10.636004%=296/2783、E5 的 dE_fixed=−38.473358、
> E1 random E_fixed=0.159113@120reps、G1 的 Q1 名单 94–142）全部在 CSV 里复现；
> 618 处 `run_xx.py:Lnnn` 引用全部命中真实文件且行号不越界。

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
| E5 的 3 张附表 | 已补。另有入度分层随机对照（E5 §8） |
| F3 的 result.md 主体 | 已重写并对照等价区间表区分首段词元与路径指纹 |
| "149 个部分失效" | D6 实测：剔掉 -1/-2/403 后 63 站不再归为部分失效。正文引用必须带限定语（`CONVENTIONS §2.2b`） |
| `RUN.txt` 里的服务器路径 | 原记"6 个 RUN.txt 死引用"，实测是 **15 个 git 跟踪文件**含服务器地址与账号，且不止日志——`run_c5.py`/`run_e6.py`/`run_h3_zp_fig.py` 等**源码**、`E2/E3_summary.json`、`NEXT_STEPS.md`、`PROGRESS.md`、`I2_violations.csv` 都有。2026-11-24 已全部替换为 `<SERVER_HOST>`/`<REMOTE_WORKDIR>` 占位符。密码从未落盘（全库检索 0 命中）。**若将来 push 到任何远端，先确认这批占位符没被新日志重新写入** |
| C3 的 `DISSENT.md` | docstring 承诺过但没写 |

## 别动的文件（动了就没法对账）

- `USTC网络拓扑分析/`、`archive/00_选题分析_预实验/` —— 同学的原始数据与前期材料，只读；改了就和他在仓库里的版本对不上账
- `pipeline/csa_common.py`、`pipeline/data/` —— 改它们等于改了全项目的口径基线
- 别人的 `out/<单元>/` 目录 —— 每个单元只写自己的目录

## 想判断"这活干得好不好"，就问三个问题

1. 随便挑 3 个 `result.md` 的数字，能在 CSV 里复现吗？
2. 报告里如果写"删死站让效率上升"，是错的——产出里有没有明确禁止这句？
3. 社区数写"25 个"算不算错？答案：算，必须给 23–26 的区间。

三个都过，这套东西就基本可信；剩下的都是"补完整性"层面的活。

## 2026-11-24 外部审阅后改了什么（以及两处撤回）

改的四处：

1. **`verify/check_claims.py` 原来不是"对账"**——169 行里零个 assert，只 print，claim 与自己重算值的
   偏差要靠人眼发现；更严重的是它算效率用 `nx.global_efficiency(剩余最大连通块)` = **E_gcc**，
   而 CLAIM 标题直接写着 `E 0.3508 (rises)`，等于全项目唯一的"验收工具"自己在违反 CONVENTIONS §2.3。
   现在：补了 `eff_fixed`（分母恒 n0(n0-1)=453602），把 claim（同学原文，只报偏差）与
   contract（本项目口径，违反退出码 1）分开，加 17 项断言。
   实测 `E_fixed(删162死站)=0.203942`、降幅 38.47%，与 `E5/scenario_metrics.csv` 的
   `dE_fixed_pct=−38.473358` 逐位吻合——**独立重实现与主流水线对上了**。
2. 洗掉 15 个文件里的服务器地址与账号。
3. `I5_clean_reproduce:45` 的「因为」改成冒号；`I4_causal_language` 补记这次漏检及其成因
   （grep 把审计文件自己的对照表算进去，导致兄弟单元的真违规被当成"引用"放过）。
4. `F3_template_verdict/result.md` 顶部加【引用前必读】横幅。

撤回的两条审阅意见（**写在这里是因为它们本身是错误结论，别被下一轮审阅再提一遍**）：

- ~~"34 个 scratch 文件该删"~~：`_chk*.txt`、`run_stdout.log`、`_view.txt` 等**是 I5 复现审计的引用证据**
  （`I5/RUN.txt` 引 75 处、`I5/result.md` 引 49 处），删了会打断引用链。不删。
- ~~"两份 PROGRESS.md 互相冲突"~~：`archive/PROGRESS_2026-11-22.md` 在新版开头已明确标注为归档件，
  不是竞争权威。不合并。

仍可评估 C6 是否值得做。历史审计范围方面，
`I1` 的"315 CSV / 45 result.md"范围声明已过期（实测 342 / 44），它的"真冲突 0 条"只覆盖当时看到的文件。
