# -*- coding: utf-8 -*-
"""D7 · 访问受限或跳过记录审计：32 个 B/C + 1 个 G 的逐站证据

任务书要求（MANIFEST D7）：
  1) 逐站核对 32 个 B/C + 1 个 G 的快照状态码；
  2) 核对 B/C 类记录是否均属于访问受限或跳过；
  3) `passport` / `wp` / `id` / `wvpn` 必须给出证据（状态码分布）。

本单元的定位：**不做新分类**。162→129 的切分（CONVENTIONS §2.2）是本项目最关键的方法学动作，
本单元逐站记录原始状态码。实际可用性需要独立核验。
本单元把证据摊开：每个站的状态码分布、逐 URL 明细、以及"证据强 / 需人工"的判据。

只读 CSA_HW/ 与 pipeline/data/；只写本模块目录；建图与锚点走 csa_common；措辞遵守 §7。
"""
from __future__ import annotations

import os
import re
import sys
import time
import collections

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.dirname(os.path.dirname(HERE))
if PIPE not in sys.path:
    sys.path.insert(0, PIPE)

import csa_common as C  # noqa: E402

MOD = "D7_false_failure_audit"
T0 = time.time()
OUT = C.module_dir(MOD)

AUTH_CODES = {-1, -2, 403}          # 认证跳转 / robots / ACL 这三类才是"探针被挡"
SPECIAL = ["passport", "wp", "id", "wvpn"]


def say(msg):
    print(msg, flush=True)


def parse_codes(s):
    """'−1:2;403:1' → {(-1, 2), (403, 1)}。原始表里是 'code:count' 用分号连接。"""
    out = {}
    for m in re.finditer(r"(-?\d+)\s*:\s*(\d+)", str(s)):
        code, cnt = int(m.group(1)), int(m.group(2))
        out[code] = out.get(code, 0) + cnt
    return out


def fam(code):
    if 200 <= code < 300:
        return "2xx 可访问"
    if code in (-1,):
        return "-1 需认证/本轮跳过"
    if code in (-2,):
        return "-2 robots 禁止"
    if code == 403:
        return "403 ACL/需目录权限"
    if code == 0:
        return "0 连接层失败"
    if code in (404, 410):
        return "404/410 缺失"
    if 500 <= code < 600:
        return "5xx 服务器错误"
    return "其他"


def main():
    b = C.build_all()
    C.assert_anchors(b)
    say("[%5.1fs] 锚点 31/31 PASS" % (time.time() - T0))

    dead = C.load_dead_categories()
    urls = C.load_urls()
    say("[%5.1fs] dead_categories %d 行；urls %d 行" % (time.time() - T0, len(dead), len(urls)))

    # ---- 目标集合：32 个假失效（B/C）+ 1 个待判（G） -------------------
    is_false = dead["cat"].isin(["B_auth_or_skip", "C_robots_blocked"])
    is_amb = dead["cat"].eq("G_other_400")
    tgt = dead[is_false | is_amb].copy()
    say("[%5.1fs] 目标 %d 站（假失效 %d + 待判 %d）"
        % (time.time() - T0, len(tgt), int(is_false.sum()), int(is_amb.sum())))

    rows = []
    for r in tgt.itertuples():
        node = r.node
        codes = parse_codes(r.codes)
        u = urls[urls["subdomain"] == node]
        st = pd.to_numeric(u["status"], errors="coerce").dropna().astype(int)
        n_rows = int(len(st))
        n_auth = int(st.isin([-1, -2, 403]).sum())
        n_ok = int(((st >= 200) & (st < 300)).sum())
        n_realfail = int((~st.isin([-1, -2, 403]) & (st != 0) & ~((st >= 200) & (st < 300))).sum())
        n_zero = int((st == 0).sum())

        # 证据判据（写死在代码里，便于审计）：
        #   强 = 该站全部探测都落在认证/robots/ACL，且没有任何 2xx 内容被抓到
        #   中 = 多数（>=2/3）落在认证类，但有少量 2xx
        #   需人工 = 含真 4xx/5xx，或 2xx 占比过半，或状态码无法解析
        if n_rows == 0:
            strength, why = "需人工", "urls 表里查不到该站的逐 URL 行"
        elif n_auth == n_rows:
            strength, why = "强", "全部 %d 条记录均为 -1/-2/403（访问受限或跳过）" % n_rows
        elif n_auth >= max(1, int(round(n_rows * 2 / 3))) and n_realfail == 0:
            strength, why = "中", "%d/%d 为认证类，其余为 2xx 或 0；无真 4xx/5xx" % (n_auth, n_rows)
        else:
            strength, why = "需人工", "认证类仅 %d/%d，或含真 4xx/5xx（%d）" % (n_auth, n_rows, n_realfail)

        rows.append({
            "node": node, "sub": r.sub, "cat": r.cat, "codes_raw": r.codes,
            "url_count": int(r.url_count), "n_url_rows": n_rows,
            "n_auth_like": n_auth, "n_2xx": n_ok, "n_zero": n_zero,
            "n_real_4xx_5xx": n_realfail,
            "auth_share": (n_auth / n_rows) if n_rows else np.nan,
            "code_minus1": codes.get(-1, 0), "code_minus2": codes.get(-2, 0),
            "code_403": codes.get(403, 0), "code_0": codes.get(0, 0),
            "code_404_410": codes.get(404, 0) + codes.get(410, 0),
            "code_2xx": sum(v for k, v in codes.items() if 200 <= k < 300),
            "evidence_strength": strength, "evidence_note": why,
        })
    EV = pd.DataFrame(rows).sort_values(["cat", "node"]).reset_index(drop=True)
    C.save_csv(EV, MOD, "d7_false_failure_evidence.csv")

    # ---- 按类别汇总 ---------------------------------------------------
    agg = []
    for cat, d in EV.groupby("cat"):
        c = collections.Counter()
        for s in d["codes_raw"]:
            for k, v in parse_codes(s).items():
                c[fam(k)] += v
        agg.append({"cat": cat, "n_sites": len(d),
                    "n_probes": int(sum(sum(parse_codes(s).values()) for s in d["codes_raw"])),
                    "n_strong": int((d.evidence_strength == "强").sum()),
                    "n_mid": int((d.evidence_strength == "中").sum()),
                    "n_manual": int((d.evidence_strength == "需人工").sum()),
                    **{("probes_" + k): int(v) for k, v in sorted(c.items())}})
    AGG = pd.DataFrame(agg)
    C.save_csv(AGG, MOD, "d7_code_distribution_by_cat.csv")

    # ---- passport / wp / id / wvpn 专项 -------------------------------------------------
    hits = EV[EV["node"].str.lower().str.contains("|".join(SPECIAL), regex=True, na=False)].copy()
    if len(hits):
        hits["matched_token"] = hits["node"].apply(
            lambda s: ",".join(t for t in SPECIAL if t in s.lower()))
    C.save_csv(hits, MOD, "d7_special_nodes.csv")

    # ---- 逐 URL 明细（只给证据弱/需人工的站，便于人工复核）-------------------------------------------------
    weak = EV[EV.evidence_strength == "需人工"]
    det = urls[urls["subdomain"].isin(set(weak.node))].copy()
    det["status"] = pd.to_numeric(det["status"], errors="coerce")
    det["fam"] = det["status"].apply(lambda x: fam(int(x)) if pd.notna(x) else "未知")
    C.save_csv(det[["subdomain", "status", "fam", "url", "title"]].sort_values(["subdomain", "status"]),
               MOD, "d7_manual_review_url_detail.csv")

    # ---- key numbers ---------------------------------------------------
    KEY = {
        "n_target": int(len(EV)),
        "n_false_fail": int(is_false.sum()),
        "n_ambiguous": int(is_amb.sum()),
        "n_strong": int((EV.evidence_strength == "强").sum()),
        "n_mid": int((EV.evidence_strength == "中").sum()),
        "n_manual": int((EV.evidence_strength == "需人工").sum()),
        "n_special_nodes": int(len(hits)),
        "n_probes_total": int(sum(sum(parse_codes(s).values()) for s in EV["codes_raw"])),
        "probe_share_auth_like": float(
            sum(v for s in EV["codes_raw"] for k, v in parse_codes(s).items() if k in AUTH_CODES)
            / max(1, sum(sum(parse_codes(s).values()) for s in EV["codes_raw"]))),
    }
    pd.DataFrame([{"key": k, "value": v} for k, v in KEY.items()]).to_json(
        os.path.join(OUT, "d7_key_numbers.json"), orient="records", force_ascii=False, indent=2)

    # ---- result.md ---------------------------------------------------
    special_lines = []
    for r in hits.itertuples():
        special_lines.append(
            "| `%s` | %s | `%s` | %d/%d | %.2f | %s |"
            % (r.node, r.cat, r.codes_raw, r.n_auth_like, r.n_url_rows,
               (r.auth_share if pd.notna(r.auth_share) else float("nan")), r.evidence_strength))
    manual_lines = []
    for r in weak.itertuples():
        manual_lines.append("| `%s` | %s | `%s` | %d/%d | %s |"
                            % (r.node, r.cat, r.codes_raw, r.n_auth_like, r.n_url_rows, r.evidence_note))

    with open(os.path.join(OUT, "result.md"), "w", encoding="utf-8") as fh:
        fh.write(f"""# D7_false_failure_audit · 访问受限或跳过记录的证据审计

> 本单元核查 CONVENTIONS §2.2 中 B/C（数据列标签 `false_fail`）和 G 类的抓取记录。B/C 表示访问受限或跳过，实际可用性未确认。

## 0. 结论

| 项 | 数 |
|---|---:|
| 目标站点（B + C + G） | **{KEY['n_target']}**（访问受限或跳过 {KEY['n_false_fail']} + 待判 {KEY['n_ambiguous']}） |
| 证据**强**（全部记录为 -1/-2/403，零 2xx） | **{KEY['n_strong']}** |
| 证据**中**（多数为认证类，另有少量 2xx，无真 4xx/5xx） | **{KEY['n_mid']}** |
| **需人工复核** | **{KEY['n_manual']}** |
| 总探测次数 | {KEY['n_probes_total']}，其中 -1/-2/403 占 **{KEY['probe_share_auth_like']:.1%}** |

**{KEY['n_strong']} 个站的记录均为访问受限或跳过（-1/-2/403），且未观测到 2xx。** 这些站点的实际可用性未确认；另有 {KEY['n_manual']} 个待人工复核（见下）。

判据写在 `run_d7.py` 的 `main()` 里，可复查：
- **强** = 该站全部 URL 行的状态码都在 {{-1, -2, 403}}，且没有任何 2xx；
- **中** = 认证类 ≥ 2/3，其余是 2xx 或 0，且没有真 4xx/5xx；
- **需人工** = 认证类占比不足，或混入真 4xx/5xx。

## 1. 按类别看状态码分布

`d7_code_distribution_by_cat.csv`：三类各自的探测次数、各码族次数、以及强/中/人工的站数。
B 类记录为 -1（需认证或本轮跳过），C 类记录为 -2（robots），G 类为 1 站的单条 HTTP 400 记录。

## 2. passport / wp / id / wvpn 专项（MANIFEST 点名要求）

| 站点 | 类别 | 原始 `codes` | 认证类/总行 | 认证类占比 | 证据 |
|---|---|---|---|---:|---|
{os.linesep.join(special_lines) if special_lines else "（本批目标里没有命中这些 token 的站点）"}

命中 {KEY['n_special_nodes']} 个站。这里的 -1 同时涵盖认证要求和本轮主动跳过。记录仅支持状态码归类，具体是否发生 HTTP 302、能否成功访问均待确认。

## 3. 需人工复核的站点（{KEY['n_manual']} 个）

| 站点 | 类别 | 原始 `codes` | 认证类/总行 | 判据说明 |
|---|---|---|---:|---|
{os.linesep.join(manual_lines) if manual_lines else "（无）"}

逐 URL 明细在 `d7_manual_review_url_detail.csv`（状态码按码族标注，可直接对着页面核）。

## 4. 这对别的单元意味着什么

- **E5 / G3 的校正情景**依赖这个切分：E5 报"删 162 全挂"与"删 129 真失效"两套，G3 的"下线类 123 个"
  也建立在 true_fail 上。本单元的作用是让这个切分的**证据可查**，而不是改变它。
- **D2** 把"出度删失"定为最关键的方法学结论；本单元处理的是另一半风险——**失效语义**。
  两者合起来才是"这个数据集能回答什么"的完整边界。
- 后续如有独立且获准的可用性核验，可据结果更新站点标签。

## 5. 局限

- 证据来自现有状态码和标题，站点实际可用性未经独立核验。
- `codes` 里的 0 同时表示"连接失败"与"本运行跳过"，本单元不区分（D6 的 `zero_skip`/`zero_conn` 分得更细）。
- 判据阈值（2/3）是显式选择，不是从数据里推出来的；改变它只会移动"中/需人工"的边界，
  不会改变"强证据站数"这个核心数字。
""")

    C.write_run_note(MOD, [
        "D7_false_failure_audit · 假失效名单证据审计",
        "command: cd pipeline && PYTHONHASHSEED=0 python out/D7_false_failure_audit/run_d7.py",
        "seed: 20261113  PYTHONHASHSEED=0  versions: %s" % C.versions(),
        "anchors: csa_common.assert_anchors() 31/31 PASS",
        "inputs: CSA_HW dead_sites_category.csv (162 行) + urls.csv (%d 行)" % len(urls),
        "target: %d 站（假失效 %d + 待判 %d）；强 %d / 中 %d / 需人工 %d"
        % (KEY["n_target"], KEY["n_false_fail"], KEY["n_ambiguous"],
           KEY["n_strong"], KEY["n_mid"], KEY["n_manual"]),
        "note: 本单元不重新分类，只把 CONVENTIONS §2.2 的切分逐站证据摊开",
        "outputs: d7_false_failure_evidence.csv d7_code_distribution_by_cat.csv "
        "d7_special_nodes.csv d7_manual_review_url_detail.csv d7_key_numbers.json result.md",
    ])
    say("[%5.1fs] DONE 强=%d 中=%d 需人工=%d" % (time.time() - T0, KEY["n_strong"],
                                                 KEY["n_mid"], KEY["n_manual"]))


if __name__ == "__main__":
    main()
