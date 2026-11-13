# USTC 网络拓扑图 —— 数据提取与结构分析

**数据来源**：https://aaaaalbert1958.github.io/USTC-Topology/（纯前端静态页，数据以 `RAW` 常量内嵌在 HTML 中）
**分析对象**：671 个 `*.ustc.edu.cn` 子域、2788 条链接关系、8115 条已收录 URL 明细

---

## 1. 页面机制

页面用 ECharts 力导向图渲染一个「子域 = 节点、子域之间的超链接 = 边」的网络：

| 元素 | 含义 |
|---|---|
| `id` | 子域名，如 `teach.ustc.edu.cn` |
| `label_title` | 该子域首页的网页标题 |
| `url_count` | 抓取器在该子域下发现/尝试的 URL 总数（可为数百） |
| `url_lists` | 每个子域最多 **20 条** 代表 URL，含 HTTP 状态码与标题，点击节点时在右侧面板展示 |
| `links` | `source → target`，`weight` 为同一对子域间被观察到的链接次数 |
| `status_class` | `blue / yellow / red` 三色，对应顶部「全错 / 全正常 / 混合」筛选按钮 |

页面交互：按标题/子域名命名、简化标题、缩放与字号滑块、三种健康度筛选、域名搜索高亮、节点拖动、点击看 URL 列表。

## 2. 规模与健康度

| 指标 | 数值 |
|---|---|
| 子域节点 | 671 |
| 连线 | 2788（平均度 8.3） |
| 已收录 URL（明细可查） | 8115 |
| 其中 HTTP 200 | 6985（86.1%） |
| 其中异常 | 1130（13.9%） |
| 页面统计的 URL 总数 | 28346 |

异常构成（8115 条明细中）：

| 类型 | 条数 | 说明 |
|---|---|---|
| status 0 | 421 | 连接失败（DNS/TCP/TLS 不通、超时） |
| status −1 | 362 | 需要身份认证，或该次运行主动跳过 |
| status −2 | 65 | robots.txt 禁止抓取 |
| 404 | 127 | 页面不存在 |
| 403 | 77 | 访问被拒 |
| 410 | 47 | 已永久删除 |
| 5xx | 28 | 502/500/504/503 |
| 其他 | 3 | 400×2、201×1 |

**按子域看**：162 个子域一条可用页面都没有，149 个部分失效，360 个全部正常。

需要注意，162 个「全挂」里有相当一部分是**假失效**，并非站点真的关停：

| 失效类型 | 子域数 | 说明 |
|---|---|---|
| 连接失败/超时 | 111 | 真故障（DNS 或 TLS 层） |
| 需认证 / 被跳过 | 18 | 如 `passport`、`id`、`manage.faculty` 要先登录 |
| robots 禁止 | 14 | 如 `ftp.lug`、`netfee`，抓取器被拒而非站点失效 |
| 404/410 | 12 | 页面已删但子域还在 |
| 5xx | 6 | 如 `wcm`（502 + 403） |
| 其他 | 1 | 400 |

真正值得追的是那 111 个连接失败与 12 个 404 子域（明细见 `dead_sites_category.csv`、`bad_urls.csv`）。

## 3. 拓扑结构

### 3.1 连通性

676 个图节点（671 个子域 + 5 个只出现在边里的悬空引用：`personnel`、`www2.lib`、`recapi`、`985-211`、`rmd.snst`）几乎全部连通：

- **连通分量 2 个**：主分量 674 个节点，另一个只有 2 个节点 —— `bm.ustc.edu.cn` 与 `l.bm.ustc.edu.cn` 互链成一个封闭小岛，不与校内任何其它站点互联。
- 没有孤立节点，也没有「断层」；整个校级站点网络高度耦合，任一枢纽站点出问题都会在链接层面波及大量邻接页。

### 3.2 枢纽（度数 Top 25）

最高度节点几乎全是**导航型公共站点**：

| 子域 | 度数 | 异常/已收录 |
|---|---|---|
| teach | 146 | 0 / 20 |
| staff | 112 | 2 / 20 |
| wpsearch | 90 | 0 / 20 |
| news | 79 | 0 / 20 |
| en | 77 | 0 / 20 |
| fusep | 75 | 0 / 20 |
| employment / gradschool / physics / yz / faculty | 65–74 | 多为 0–3 |

`teach`（教务）、`staff`（教师主页聚合）、`wpsearch`（站内搜索）、`news`、`en` 构成了事实上的「入口层」，它们是全网链接分布的骨架。

### 3.3 社区结构（标签传播，11 个社区）

| 社区 | 规模 | 代表成员 |
|---|---|---|
| C239 | 557（82.4%） | teach staff wpsearch news en fusep |
| C204 | 44 | biox biomed enbiomed jcyx ioi biotraining |
| C178 | 16 | lab pic gxpt engilab scc lifetech |
| C389 | 14 | vlab cslab soc rec vim fpgaol |
| C291 | 11 | lug 101.lug linux mirrors ftp.lug hack.lug |
| C450 | 10 | cec vcec polychemlab oldcec cmcec |
| C110 | 5 | kdfz jcjy kdfzgxzx kfsy kid |
| C628 | 4 | ai aiethics roboticsbase rccnc |
| C488 | 4 | jjcxmjl jjc xsfw ucas |
| C454 | 4 | faculty-en institution-en team-en fanglab |
| C514 | 2 | l.bm bm |

值得注意的几点：

1. **82% 的站点被吸进同一个社区**。说明校级 CMS / 导航模板高度统一，页面互链模式同质化，链接结构本身几乎不携带语义；「子域 = 一个单位」的直觉在链接层面并不成立。
2. 真正独立成簇的是少数自建系统：**生物医学集群**（biox/biomed/enbiomed/jcyx/ioi）、**实验室集群**（lab/pic/gxpt/engilab）、**硬件与系统类实验平台**（vlab/cslab/rec/vim/fpgaol）、**化学与化工**（cec/vcec/polychemlab）、**LUG 与镜像站**（lug/mirrors/hack.lug/ftp.lug）、**AI**（ai/aiethics/roboticsbase）。
3. `kdfz` 一簇（科大附中相关）与 `bm`/`l.bm` 小岛，是与主站群交互最少的两个系统。

## 4. 方法与局限

- **数据是二次加工的结果**：节点数、URL 数取决于原抓取器的范围、并发与超时设置；`url_count` 与本页可见明细上限（20 条）不一致，明细只能代表抽样。
- **状态码语义**：`0` = 连接失败；`-1` = 需认证或该轮跳过；`-2` = robots 禁止。这些是抓取侧约定，不是 HTTP 语义，故「失效」需按上表拆分看。
- **三色含义**：由数据反推 —— `red` 的 130 个子域根路径 100% 不可用；`yellow` 有 0 个全挂、49 个全好，属「混合」；`blue` 388 个中 32 个在本页 20 条明细里全挂（应是全站抓取里仍有可用页）。页面源码里没有三色与「全错/全正常/混合」按钮的映射表，此处为**推断**，置信度较高但不等于原文。
- **社区划分用标签传播**（确定性顺序、最多 30 轮，6 轮收敛），结果对小图稳定，但 557 节点的巨型社区本质上是「结构同质化」的产物，不宜过度解读为院系归属。

## 5. 文件清单

| 文件 | 内容 |
|---|---|
| `USTC拓扑概览.xlsx` | 4 个工作表：概览、失效分类、枢纽节点、社区结构 |
| `nodes_full.csv` | 671 个子域全字段（标题、URL 数、明细可用/异常、可用率、状态码、颜色、度数、根地址） |
| `links.csv` | 2788 条边（source, target, weight） |
| `urls.csv` | 8115 条 URL 明细（子域、状态码、URL、标题） |
| `bad_urls.csv` | 1130 条异常 URL |
| `dead_sites.csv` | 162 个全挂子域 |
| `dead_sites_category.csv` | 全挂子域按失效类型归类 |
| `partial_fail_sites.csv` | 149 个部分失效子域 |
| `broken_sites.csv` | 311 个含异常 URL 的子域 |
| `hubs.csv` / `communities.csv` / `node_communities.csv` / `node_components.csv` | 拓扑分析结果 |
| `summary.json` | 汇总指标（供程序读取） |

CSV 均为 UTF-8 with BOM，Excel 直接双击不乱码。
