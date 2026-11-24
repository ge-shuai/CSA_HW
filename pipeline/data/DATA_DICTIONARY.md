# 数据字典（由 00_build_data.py 自动生成，勿手改）

版本：{'python': '3.12.13', 'networkx': '3.6.1', 'pandas': '3.0.2', 'numpy': '2.4.4', 'SEED': 20261113, 'PYTHONHASHSEED': '0'}

| file | column | dtype | n_na | example | meaning |
|---|---|---|---|---|---|
| clean_nodes.csv | subdomain | str | 0 | 101.lug.ustc.edu.cn | FQDN，= 图节点主键 node |
| clean_nodes.csv | sub | str | 0 | 101.lug | 短名，= dead_sites_category.csv 的连接键 |
| clean_nodes.csv | title | str | 0 | 欢迎 - Linux 101 |  |
| clean_nodes.csv | url_count | int64 | 0 | 33 | 抓取器报告的该子域 URL 总数 |
| clean_nodes.csv | listed_urls | int64 | 0 | 20 | 实际落库的明细条数，**上限 20**（352 个站点正好 20） |
| clean_nodes.csv | ok | int64 | 0 | 20 |  |
| clean_nodes.csv | bad | int64 | 0 | 0 |  |
| clean_nodes.csv | ok_ratio_pct | float64 | 0 | 100.0 | ok/listed_urls 百分数；健康分层唯一依据（100=healthy, 0=dead, 中间=partial） |
| clean_nodes.csv | status_codes | str | 0 | 200:20 | '码:次数' 串；0=连接失败 -1=需认证/跳过 -2=robots 禁止 |
| clean_nodes.csv | color | str | 0 | blue | **已弃用**：与 ok_ratio_pct 冲突（32 个可用率 0 的站点标 blue），仅存档用 |
| clean_nodes.csv | root_url | str | 0 | https://101.lug.ustc.edu.cn/ |  |
| clean_nodes.csv | node | str | 0 | 101.lug.ustc.edu.cn |  |
| clean_nodes.csv | in_degree | int64 | 0 | 2 |  |
| clean_nodes.csv | out_degree | int64 | 0 | 4 |  |
| clean_nodes.csv | degree_u | int64 | 0 | 5 |  |
| clean_nodes.csv | strength_u | int64 | 0 | 57 |  |
| clean_nodes.csv | in_strength | int64 | 0 | 9 |  |
| clean_nodes.csv | out_strength | int64 | 0 | 48 |  |
| clean_nodes.csv | is_dead | bool | 0 | False |  |
| clean_nodes.csv | is_partial | bool | 0 | False |  |
| clean_nodes.csv | is_healthy | bool | 0 | True |  |
| clean_nodes.csv | health_state | str | 0 | healthy |  |
| clean_nodes.csv | dead_cat | str | 509 | A_conn_fail | 仅整站不可用节点有值；A/D/F=真失效，B/C=假失效，G=待定 |
| clean_nodes.csv | true_fail | bool | 0 | False |  |
| clean_nodes.csv | false_fail | bool | 0 | False |  |
| clean_nodes.csv | ambiguous_fail | bool | 0 | False |  |
| clean_nodes.csv | partial_has_auth_like | bool | 0 | False |  |
| clean_nodes.csv | partial_clean | bool | 0 | False |  |
| clean_nodes.csv | in_gcc | bool | 0 | True | 是否属于无向图最大连通分量(674) |
| clean_nodes.csv | is_dangling_ref | bool | 0 | False |  |
| clean_edges.csv | edge_order | int64 | 0 | 0 |  |
| clean_edges.csv | source | str | 0 | zsb.ustc.edu.cn |  |
| clean_edges.csv | target | str | 0 | zsfw.ustc.edu.cn |  |
| clean_edges.csv | weight | int64 | 0 | 300 | 该有向边上观察到的链接次数 |
| clean_edges.csv | undirected_weight | int64 | 0 | 300 | 反向边权重之和（无向加权图用的就是这个） |
| clean_edges.csv | is_reciprocal | bool | 0 | False | 是否存在反向边 |
| clean_edges.csv | source_is_dead | bool | 0 | False | **恒为 False**：死站的出边不可观测，这是删失证据不是结论 |
| clean_edges.csv | target_is_dead | bool | 0 | False | 有健康站指向它，入边可观测 |
| dead_categories.csv | sub | str | 0 | 19da |  |
| dead_categories.csv | node | str | 0 | 19da.ustc.edu.cn |  |
| dead_categories.csv | cat | str | 0 | A_conn_fail | 失败原因分类 |
| dead_categories.csv | codes | str | 0 | 0:1 | 触发该分类的状态码串 |
| dead_categories.csv | url_count | int64 | 0 | 1 |  |
| dead_categories.csv | is_true_fail | bool | 0 | True | A_conn_fail / D_not_found / F_server_error |
| dead_categories.csv | is_false_fail | bool | 0 | False | B_auth_or_skip / C_robots_blocked |
| dead_categories.csv | is_ambiguous | bool | 0 | False |  |
