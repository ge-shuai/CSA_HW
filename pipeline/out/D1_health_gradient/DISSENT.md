# D1 · DISSENT

对 CONVENTIONS.md 本身**没有异议**；本文件记录的是**任务书 D1 条目里的一处事实性指认错误**，
按 §"若认为本文件有误，把证据写进 DISSENT.md，不要自行改口径"的处置方式留档。

## 议题：题面所说"352 个站点正好 20 条 URL（爬虫上限）"落在哪一列

任务书 D1 原文：
> Special column requirement: 352 sites sit at exactly 20 URLs (a crawler cap), so **url_count is heavily tied at the cap**
> — report url_count's distribution and treat the =20 bin separately rather than letting qcut silently split ties.

实测（`run_d1.py:L351` 产出 `d1_exposure_distribution.csv`；`L333`/`L341` 产出两张 cap_bins 表）：

| 列 | 定义（`pipeline/data/DATA_DICTIONARY.md`） | max | =20 的站点数 | 众数 |
|---|---|---:|---:|---|
| `url_count` | 抓取器报告的该子域 URL 总数 | **302** | **33**（4.92%） | =1（173 个） |
| `listed_urls` | 实际落库的明细条数，**上限 20**（352 个站点正好 20） | **20** | **352**（52.46%） | =20（352 个） |

即 352 这个数属于 **`listed_urls`**，不属于 `url_count`；`url_count` 从未被 20 截断（`sum(url_count)=28346` vs
`sum(listed_urls)=8115`，319 个站点 `url_count>listed_urls`，见 `d1_page_truncation_strata.csv`）。
MANIFEST 的 D1 行写的是"352 个站点正好 20 条 URL"，未指定列名，与 `listed_urls` 一致；任务书把它写成 `url_count` 才是漂移点。
A4 单元已就同一事实留档（`out/A4_directed_structure/truncation_estimate.csv`：`cap_on_listed_urls=20`、
`n_sites_listed_at_cap=352`、`max_url_count=302`）。

## 处置（未改任何口径，只做了两件事）

1. **两列都报**：`url_count` 与 `listed_urls` 同时进入暴露集（`run_d1.py:L67`–`L72`），各出一张十分位表；
2. **堆积值单列**：题面点名的 `=20` 层照样独立成行（`d1_url_count_cap_bins.csv` 第 3 行，n=33），
   另加 `listed_urls=20` 上限层（n=352）与两个更关键的堆积层（`url_count=1` n=173、`99–105` n=176）。
   结论方向不受影响：`url_count=20` 层全挂率 0.0000、`listed_urls=20` 层 0.0057，都不是上限造成的堆积。

补充发现（题面未要求，但与"不许 qcut 静默拆并列"直接相关）：`url_count` 真正的堆积在
**=1（173 个）和 ≈100（99–105 区间 176 个，其中 =100 有 123 个）**，最大值 302；
这说明抓取器每站的页面预算约在 100 而非 20，20 只是落库明细的上限。
