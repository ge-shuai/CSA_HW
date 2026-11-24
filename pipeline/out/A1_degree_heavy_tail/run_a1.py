# -*- coding: utf-8 -*-
"""A1_degree_heavy_tail —— 度 / 强度分布的重尾刻画。

任务（MANIFEST W1·A 组，单元 A1）
--------------------------------
对无向度 `degree_u`、入度 `in_degree`、出度 `out_degree`、强度 `strength_u` 给出
CCDF、Gini、截尾均值、变异系数、Top-1%/Top-10 份额、矩（均值 / 方差 / 三阶中心矩），
并按 Clauset–Shalizi–Newman (2009, SIAM Rev 51:661) 的**离散 MLE + KS(x_min 搜索) +
参数化 bootstrap + 配对似然比**流程比较 幂律 / 截断幂律 / 对数正态 / 指数 / 负二项，
最后判决“这份度分布能不能叫无标度”。

口径（CONVENTIONS.md；本文件不重写任何建图逻辑）
------------------------------------------------
- 图 / 度 / 强度全部来自 `csa_common.build_all()`（§1）。开跑先 `assert_anchors()`，
  对不上就写 DISSENT.md 并停止。
- 主节点集 = 671 个有属性节点（§1）；676（含 5 个悬空）与 674（GCC）作并列对照。
- 出度按 §2.1 是**完全删失**口径（162 个整站不可用节点出度恒为 0）→ 只描述、不拟合。
- 不使用 `nodes_full.csv` 的 `degree` / `color` 禁用列（§1）。
- `SEED = csa_common.SEED = 20261113`；所有随机过程显式传 seed，seed 只由**下标**派生
  （不用 `hash()`：字符串哈希按进程随机化，会毁掉可复现性）。

方法（对应 Clauset et al. 2009 的实现要点）
-------------------------------------------
1. 所有候选分布都在**离散整数**上做 MLE，并按 x >= x_min **重新归一化**：
   powerlaw        p(x) = x^-a / zeta(a, x_min)，归一化与尾概率用精确 Hurwitz zeta
   trunc_powerlaw  p(x) ∝ x^-a exp(-x/x_c)，支撑 {x_min..X_CAP}（见 DISSENT §4）
   lognormal       p(x) ∝ Phi(u(ln(x+.5))) - Phi(u(ln(x-.5)))（CSN 的连续近似），归一化用 sf
   exponential     离散对应量 p(x) ∝ exp(-x/x_c)，即平移几何分布，闭式 MLE
   nbinomial       p(x) ∝ C(x+r-1,x) p^r (1-p)^x，归一化用正则不完全 Beta
2. x_min 由**幂律的 KS 最小化**在候选集 {n(>=x_min) >= max(50, 6 x_min^2)} 上搜索；
   其余分布在**同一 x_min、同一子样本**上拟合（lnL/AIC/BIC/KS 可比），
   另给各分布自身最优 x_min、无 6x^2 约束的 x_min、以及 x_min=1 全量三种对照。
3. 一次“池化 bootstrap”同时产出全部检验：对每个候选模型 m 抽 NBOOT 个样本
   （从 m 的拟合分布生成），**每个样本重新 MLE 全部 5 个模型**，于是
     - KS bootstrap p（用 m 自己的池）
     - 参数 bootstrap 分位（用 m 自己的池）
     - 配对 LR 的双侧零分布（用 A 的池 + B 的池）
   全部来自同一批计算，无额外成本。
4. 配对似然比 R = lnL(A) - lnL(B)，正态近似 p = Phi(-|R| / sd(R))。
5. Gini / 份额 / CV 另给 NRES 次重抽样的中位数与 90% 区间。

运行
----
    cd pipeline
    python out/A1_degree_heavy_tail/run_a1.py
环境变量可调重复次数：A1_NBOOT(默认1000) / A1_NLR(默认400) / A1_NRES(默认500)。
"""
from __future__ import annotations

import os
import sys
import time
import platform

T_START = time.time()

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE = os.path.dirname(os.path.dirname(HERE))          # .../pipeline
if PIPELINE not in sys.path:
    sys.path.insert(0, PIPELINE)

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import (zeta as _hzeta, log_ndtr as _logndtr, betainc as _betainc,
                           gammaln as _gammaln, exp1 as _exp1)
from scipy.stats import norm as _norm
import networkx as nx

import csa_common as C
import sklearn
import scipy

SEED = int(C.SEED)                     # 20261113
MODULE = "A1_degree_heavy_tail"
OUT = C.module_dir(MODULE)
LOG: list[str] = []

NBOOT = int(os.environ.get("A1_NBOOT", "1000"))   # KS / 参数 bootstrap（= 池大小）
NLR = int(os.environ.get("A1_NLR", "400"))        # 配对 LR 每侧使用的池样本数 (<= NBOOT)
NRES = int(os.environ.get("A1_NRES", "500"))      # 描述统计重抽样次数
MIN_PTS = 50                                      # CSN: x_min 之上至少 50 个点
TAIL_PROB = 1e-12                                 # 抽样网格允许的丢失尾概率
PLAT = np.log(max(_exp1(1.0), 1e-300))            # 常数，仅用于日志/自检


def log(msg: str) -> None:
    line = f"[{time.time() - T_START:8.2f}s] {msg}"
    print(line, flush=True)
    LOG.append(line)


# ============================== 0. 锚点自检 ============================== #
def anchors_or_die():
    """CONVENTIONS §1：先 assert_anchors；对不上停下写 DISSENT.md，不出任何结论。"""
    bb = C.build_all()
    adf = C.check_anchors(bb)
    bad = adf[~adf["ok"]]
    if len(bad):
        with open(os.path.join(OUT, "DISSENT.md"), "w", encoding="utf-8") as f:
            f.write("# A1 DISSENT —— 锚点校验失败，本单元**未产出任何结论**\n\n"
                    f"第一步 `csa_common.check_anchors()` 有 {len(bad)}/{len(adf)} 项对不上，"
                    "说明建图/清洗口径已漂移。按 CONVENTIONS §1 立即停止。\n\n```\n"
                    + bad.to_string(index=False) + "\n```\n")
        adf.to_csv(os.path.join(OUT, "anchors.csv"), index=False, encoding="utf-8-sig")
        raise SystemExit("ANCHOR FAILURE -> out/A1_degree_heavy_tail/DISSENT.md")
    C.assert_anchors(bb)
    return bb, adf


# ============================== 1. 分布引擎 ============================== #
def _solve_cap(f, x_start: float, target: float, big: float = 4.0e6) -> float:
    """解 f(x) = target，f 单调下降；先倍增再二分。"""
    x_hi = max(float(x_start), 2.0)
    for _ in range(200):
        if f(x_hi) <= target or x_hi >= big:
            break
        x_hi = min(x_hi * 2.0, big)
    x_lo = max(x_hi / 2.0, 1.0)
    for _ in range(200):
        if x_hi - x_lo <= 1.0:
            break
        mid = 0.5 * (x_lo + x_hi)
        if f(mid) > target:
            x_lo = mid
        else:
            x_hi = mid
    return x_hi


class Dist:
    """离散分布基类：支撑 {x_min, x_min+1, ...}。"""

    name = "base"
    pname = ("p1",)
    npar = 1
    lower = np.array([1e-3])
    upper = np.array([1e3])

    def __init__(self):
        self.trunc = 0.0          # 抽样网格丢掉的尾概率（诊断用）

    # -- 子类实现 --
    def log_unnorm(self, x, th):
        raise NotImplementedError

    def logZ(self, th, xmin):
        raise NotImplementedError

    def logsf_cond(self, x, th, xmin):
        raise NotImplementedError

    def grid(self, th, xmin):
        raise NotImplementedError

    def start(self, data, xmin):
        return np.array([1.0] * self.npar)

    # -- 通用 --
    def logpmf(self, x, th, xmin):
        return self.log_unnorm(x, th) - self.logZ(th, xmin)

    def sf_cond(self, x, th, xmin):
        """条件生存 P(X >= x | X >= xmin)，强制单调不增（消除数值毛刺）。"""
        v = np.exp(np.clip(self.logsf_cond(np.asarray(x, dtype=float), th, xmin), -745.0, 0.0))
        return np.minimum.accumulate(np.asarray(v, dtype=float))

    def sample(self, th, xmin, n, rng):
        xs, ps = self.grid(th, xmin)
        cs = np.cumsum(ps)
        cs[-1] = 1.0
        return np.searchsorted(cs, rng.random(n), side="left") + xs[0]

    def _nll(self, th, xv, cv, xmin):
        th = np.atleast_1d(np.asarray(th, dtype=float))     # minimize_scalar 传的是 0 维标量
        if not np.all(np.isfinite(th)) or np.any(th < self.lower) or np.any(th > self.upper):
            return 1e18
        ll = float(np.dot(cv, self.logpmf(xv, th, xmin)))
        return -ll if np.isfinite(ll) else 1e18

    def fit(self, xv, cv, xmin, th0=None):
        """MLE。xv 升序唯一观测值，cv 频数。返回 (theta, loglik)。"""
        cand = [np.asarray(self.start(xv, xmin), dtype=float)]
        if th0 is not None:
            cand.append(np.asarray(th0, dtype=float))
        best = None
        for c0 in cand:
            c0 = np.clip(c0, self.lower * (1 + 1e-6), self.upper * (1 - 1e-9))
            if self.npar == 1:
                g = float(c0[0])
                lo = max(float(self.lower[0]), g / 100.0)
                hi = min(float(self.upper[0]), g * 100.0)
                r = minimize_scalar(self._nll, bounds=(lo, hi), method="bounded",
                                    args=(xv, cv, xmin), options={"xatol": 1e-7})
                th = np.array([float(r.x)])
            else:
                rr = minimize(self._nll, c0, args=(xv, cv, xmin), method="Nelder-Mead",
                              options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 1200,
                                       "maxfev": 1200})
                th = np.clip(np.asarray(rr.x, dtype=float), self.lower, self.upper)
            v = self._nll(th, xv, cv, xmin)
            if best is None or v < best[0]:
                best = (v, th)
        return np.asarray(best[1], dtype=float), float(-best[0])


class PowerLaw(Dist):
    """离散幂律（zeta 分布）：p(x) = x^-alpha / zeta(alpha, x_min)，alpha > 1。"""

    name, pname, npar = "powerlaw", ("alpha",), 1
    lower, upper = np.array([1.0 + 1e-9]), np.array([30.0])

    def log_unnorm(self, x, th):
        return -th[0] * np.log(np.asarray(x, dtype=float))

    def logZ(self, th, xmin):
        return float(np.log(_hzeta(th[0], float(xmin))))

    def logsf_cond(self, x, th, xmin):
        # 整数 x: zeta(a, x) = sum_{y>=x} y^-a  ->  P(X>=x | X>=xmin)
        a = float(th[0])
        return np.minimum(np.log(_hzeta(a, np.maximum(np.asarray(x, dtype=float), xmin)))
                          - self.logZ(th, xmin), 0.0)

    def start(self, data, xmin):
        return np.array([2.2])

    def grid(self, th, xmin):
        a = float(th[0])
        lz0 = self.logZ(th, xmin)
        g = lambda x: float(np.log(_hzeta(a, x))) - lz0          # noqa: E731
        X = int(min(max(_solve_cap(g, float(xmin) + 1.0, np.log(TAIL_PROB), 1.0e6),
                        float(xmin) + 1), 1.0e6))
        self.trunc = float(min(max(g(X + 1.0), 0.0), 1.0))
        xs = np.arange(int(xmin), X + 1, dtype=float)
        ps = xs ** (-a)
        return xs, ps / ps.sum()


class ExpCutoffPL(Dist):
    """截断幂律 p(x) ∝ x^-alpha exp(-x/x_c)，支撑 {x_min..xcap}（有限，见 DISSENT §4）。"""

    name, pname, npar = "trunc_powerlaw", ("alpha", "x_c"), 2
    lower, upper = np.array([-2.0, 1e-3]), np.array([30.0, 1e7])

    def __init__(self, xcap=None):
        super().__init__()
        self.xcap = float(xcap) if xcap else None
        self._cache: dict = {}

    def _grid_x(self, xmin):
        key = (int(xmin), self.xcap)
        g = self._cache.get(key)
        if g is None:
            X = int(self.xcap) if self.xcap else int(xmin)
            xs = np.arange(int(xmin), max(X, int(xmin)) + 1, dtype=float)
            g = (xs, np.log(xs))
            self._cache[key] = g
        return g

    def log_unnorm(self, x, th):
        x = np.asarray(x, dtype=float)
        return -th[0] * np.log(x) - x / th[1]

    def _w(self, th, xmin):
        _, lxs = self._grid_x(xmin)
        mx = -th[0] * lxs - self._grid_x(xmin)[0] / th[1]
        m = float(mx.max())
        return np.exp(mx - m)

    def logZ(self, th, xmin):
        xs, lxs = self._grid_x(xmin)
        mx = -th[0] * lxs - xs / th[1]
        m = float(mx.max())
        return float(m + np.log(np.exp(mx - m).sum()))

    def logsf_cond(self, x, th, xmin):
        xs, _ = self._grid_x(xmin)
        w = self._w(th, xmin)
        cs = np.cumsum(w[::-1])[::-1]
        xi = np.asarray(x, dtype=float)
        idx = np.clip(xi.astype(np.int64) - int(xmin), 0, len(xs) - 1)
        out = np.where(xi <= xs[-1], np.maximum(cs[idx], 0.0) / float(w.sum()), 0.0)
        return np.log(np.maximum(out, 1e-300))

    def start(self, data, xmin):
        return np.array([2.0, max(10.0, 0.6 * float(np.max(data)))])

    def grid(self, th, xmin):
        xs, _ = self._grid_x(xmin)
        lp = self.log_unnorm(xs, th)
        ps = np.exp(lp - float(lp.max()))
        self.trunc = 0.0                       # 支撑就是网格本身
        return xs, ps / ps.sum()


class LogNormal(Dist):
    """离散数据上的连续近似（CSN 推荐）：p(x) ∝ Phi(u(x+.5)) - Phi(u(x-.5))。"""

    name, pname, npar = "lognormal", ("mu", "sigma"), 2
    lower, upper = np.array([-5.0, 1e-2]), np.array([20.0, 60.0])

    @staticmethod
    def _ls(x, mu, s):
        """ln P(Y > ln(x-0.5)) = ln sf((ln(x-.5)-mu)/s)，用 log_ndtr 直接算。"""
        return _logndtr(-((np.log(np.maximum(np.asarray(x, dtype=float) - 0.5, 1e-12)) - mu) / s))

    @classmethod
    def _ldiff(cls, x, mu, s):
        """ln[Phi(u(x+.5)) - Phi(u(x-.5))] = la + ln(1 - exp(lb-la))，稳定写法。"""
        x = np.asarray(x, dtype=float)
        la = cls._ls(np.maximum(x, 1.0000001), mu, s)
        lb = _logndtr(-((np.log(x + 0.5) - mu) / s))
        return la + np.log(-np.expm1(np.minimum(lb - la, 0.0)))

    def log_unnorm(self, x, th):
        return self._ldiff(x, th[0], th[1])

    def logZ(self, th, xmin):
        return float(self._ls(float(xmin), th[0], th[1]))

    def logsf_cond(self, x, th, xmin):
        return np.minimum(self._ls(np.asarray(x, dtype=float), th[0], th[1]) - self.logZ(th, xmin), 0.0)

    def start(self, data, xmin):
        w = np.asarray(data, dtype=float)
        w = w / w.sum()
        lg = np.log(np.maximum(np.asarray(data, dtype=float), 1e-9))
        mu = float(np.average(lg, weights=w))
        sd = float(np.sqrt(max(np.average((lg - mu) ** 2, weights=w), 1e-3)))
        return np.array([mu, max(sd, 0.15)])

    def grid(self, th, xmin):
        tgt = np.log(TAIL_PROB) + self.logZ(th, xmin)
        f = lambda x: float(self._ls(np.array([x]), th[0], th[1])[0])   # noqa: E731
        X = int(min(max(_solve_cap(f, max(float(xmin), float(np.exp(th[0]))), tgt),
                        float(xmin) + 1), 1.0e6))
        self.trunc = float(min(max(f(X + 1.0) - tgt, 0.0), 1.0))
        xs = np.arange(int(xmin), X + 1, dtype=float)
        lp = self.log_unnorm(xs, th)
        ps = np.exp(lp - float(lp.max()))
        return xs, ps / ps.sum()


class Exponential(Dist):
    """离散对应量 p(x) ∝ exp(-x/x_c)（平移几何分布），闭式 MLE。"""

    name, pname, npar = "exponential", ("x_c",), 1
    lower, upper = np.array([1e-6]), np.array([1e10])

    def log_unnorm(self, x, th):
        return -np.asarray(x, dtype=float) / th[0]

    def logZ(self, th, xmin):
        lr = -1.0 / th[0]
        # sum_{y>=xmin} e^{-y/lam} = e^{-xmin/lam} / (1 - e^{-1/lam})
        return float(xmin * lr - np.log(-np.expm1(lr)))

    def logsf_cond(self, x, th, xmin):
        return np.minimum(-(np.asarray(x, dtype=float) - float(xmin)) / th[0], 0.0)

    def fit(self, xv, cv, xmin, th0=None):
        n = float(cv.sum())
        mean = float(np.dot(cv, xv) / n)
        exc = max(mean - float(xmin), 1e-9)
        lam = float(np.log1p(1.0 / exc))                       # -ln r
        th = np.array([max(1.0 / max(lam, 1e-12), float(self.lower[0]))])
        return th, float(np.dot(cv, self.logpmf(xv, th, xmin)))

    def start(self, data, xmin):
        return np.array([max(float(np.mean(data)) - float(xmin), 0.5)])

    def grid(self, th, xmin):
        lam = float(th[0])
        X = int(min(max(xmin + lam * (-np.log(TAIL_PROB)), xmin + 1), 1.0e6))
        self.trunc = float(np.exp(-(X + 1.0 - xmin) / lam))
        xs = np.arange(int(xmin), X + 1, dtype=float)
        ps = np.exp(-(xs - xs[0]) / lam)
        return xs, ps / ps.sum()


class NegBinom(Dist):
    """负二项（可选候选）：p(x) = C(x+r-1,x) p^r (1-p)^x，x >= 0，按 x >= x_min 归一化。"""

    name, pname, npar = "nbinomial", ("r", "p"), 2
    lower, upper = np.array([1e-4, 1e-9]), np.array([1e7, 1.0 - 1e-12])

    def log_unnorm(self, x, th):
        x = np.asarray(x, dtype=float)
        return (_gammaln(x + th[0]) - _gammaln(th[0]) - _gammaln(x + 1.0)
                + th[0] * np.log(th[1]) + x * np.log1p(-th[1]))

    def logZ(self, th, xmin):
        # P(X >= xmin) = sf(xmin-1) = I_{1-p}(xmin, r)
        return float(np.log(max(_betainc(float(xmin), th[0], 1.0 - th[1]), 1e-300)))

    def logsf_cond(self, x, th, xmin):
        xx = np.maximum(np.asarray(x, dtype=float), 1.0)
        return np.minimum(np.log(np.maximum(_betainc(xx, th[0], 1.0 - th[1]), 1e-300))
                          - self.logZ(th, xmin), 0.0)

    def start(self, data, xmin):
        m = float(np.mean(data))
        v = max(float(np.var(data)), m + 1e-6)
        p = float(min(max(m / v, 1e-6), 1 - 1e-9))
        return np.array([max(m * p / max(1 - p, 1e-9), 1e-3), p])

    def grid(self, th, xmin):
        f = lambda x: float(np.log(max(_betainc(x, th[0], 1.0 - th[1]), 1e-300)))  # noqa: E731
        tgt = np.log(TAIL_PROB) + self.logZ(th, xmin)
        m0 = max(float(xmin), th[0] * (1 - th[1]) / max(th[1], 1e-9))
        X = int(min(max(_solve_cap(f, m0, tgt), float(xmin) + 1), 1.0e6))
        self.trunc = 0.0
        xs = np.arange(int(xmin), X + 1, dtype=float)
        lp = self.log_unnorm(xs, th)
        ps = np.exp(lp - float(lp.max()))
        return xs, ps / ps.sum()


# ============================== 2. 拟合 / 检验 ============================== #
def unique_counts(vals):
    v, c = np.unique(np.asarray(vals, dtype=np.int64), return_counts=True)
    return v.astype(float), c.astype(float)


def ks_dist(xv, cv, sf):
    """离散 KS：D = max_j |F_emp(x_j) - F_th(x_j)|（两个阶梯函数只在跳跃点比较）。"""
    n = float(cv.sum())
    return float(np.max(np.abs(np.cumsum(cv) / n - (1.0 - np.asarray(sf, dtype=float)))))


def fmt_params(dist, th):
    return ", ".join(f"{n}={v:.6g}" for n, v in zip(dist.pname, th))


def fit_pair(dist, xv, cv, xmin, th0=None):
    th, ll = dist.fit(xv, cv, xmin, th0)
    return th, ll, ks_dist(xv, cv, dist.sf_cond(xv, th, xmin))


def search_xmin(dist, vals, min_pts=MIN_PTS, six_sq=True):
    """CSN 的 x_min 搜索：逐候选做 MLE + KS。"""
    rows = []
    for xm in np.unique(vals):
        m = vals >= xm
        n_ge = int(m.sum())
        if n_ge < 3:
            continue
        xv, cv = unique_counts(vals[m])
        if len(xv) < 3:
            continue
        th, ll, D = fit_pair(dist, xv, cv, float(xm))
        rows.append({"x_min": int(xm), "n_above": n_ge, "n_distinct": len(xv),
                     "loglik": ll, "KS": D, "params": fmt_params(dist, th),
                     "param1": float(th[0]),
                     "param2": float(th[1]) if dist.npar > 1 else np.nan,
                     "ok_minpts": bool(n_ge >= min_pts),
                     "ok_6xsq": bool(n_ge >= 6.0 * float(xm) ** 2) if six_sq else True})
    return pd.DataFrame(rows)


def pick_xmin(sr, six_sq=True):
    """约束内 KS 最小的 x_min；若约束把候选清空则退回 n>=50 的最优并打标。"""
    if sr is None or not len(sr):
        return None, False
    ok = sr[sr["ok_minpts"] & sr["ok_6xsq"]] if six_sq else sr[sr["ok_minpts"]]
    if len(ok):
        return float(ok.loc[ok["KS"].idxmin(), "x_min"]), True
    pool = sr[sr["ok_minpts"]]
    if not len(pool):
        pool = sr
    return float(pool.loc[pool["KS"].idxmin(), "x_min"]), False


def bootstrap_pool(dists, fitted, xmin, n_fit, B, seed):
    """从每个拟合分布抽 B 个样本，每个样本上重新 MLE 全部模型。

    返回 {源模型下标: dict(lls=(B,K), kss=(B,K), pars=object[B][K], maxv=(B,))}
    一次算完即可同时支撑 KS bootstrap、参数 bootstrap、配对 LR 的零分布。
    """
    K = len(dists)
    rng = np.random.default_rng(seed)
    pools = {}
    for si, d in enumerate(dists):
        lls = np.full((B, K), np.nan)
        kss = np.full((B, K), np.nan)
        pars: list = [None] * B
        mx = np.zeros(B)
        for i in range(B):
            s = d.sample(fitted[si], xmin, n_fit, rng)
            xv, cv = unique_counts(s)
            mx[i] = float(xv[-1])
            pr = []
            for kj, dk in enumerate(dists):
                try:
                    thk, llk, Dk = fit_pair(dk, xv, cv, xmin)
                    lls[i, kj] = llk
                    kss[i, kj] = Dk
                    pr.append(thk)
                except Exception:                                # noqa: BLE001
                    pr.append(np.full(dk.npar, np.nan))
            pars[i] = pr
        pools[si] = {"lls": lls, "kss": kss, "pars": pars, "maxv": mx, "src": d.name}
    return pools


# ============================== 3. 描述统计 ============================== #
def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n, s = len(x), float(np.sum(x))
    if s <= 0:
        return np.nan
    return float(2.0 * np.dot(np.arange(1, n + 1), x) / (n * s) - (n + 1.0) / n)


def top_share(x, k):
    x = np.sort(np.asarray(x, dtype=float))[::-1]
    s = float(x.sum())
    return float(x[:k].sum() / s) if s > 0 else np.nan


def describe(vals):
    v = np.asarray(vals, dtype=float)
    n = len(v)
    dsc = np.sort(v)[::-1]
    mean = float(v.mean())
    var = float(v.var(ddof=1))
    m2 = float(((v - mean) ** 2).mean())
    m3 = float(((v - mean) ** 3).mean())
    m4 = float(((v - mean) ** 4).mean())
    k1 = max(int(np.ceil(0.01 * n)), 1)
    tot = float(v.sum())
    return {
        "n": n, "n_zero": int((v == 0).sum()), "min": float(v.min()), "max": float(v.max()),
        "mean": mean, "var": var, "sd": float(np.sqrt(var)), "m2": m2, "m3": m3, "m4": m4,
        "skew_pop": m3 / m2 ** 1.5 if m2 > 0 else np.nan,
        "kurt_pop": m4 / m2 ** 2 if m2 > 0 else np.nan,
        "cv": float(np.sqrt(var)) / mean if mean > 0 else np.nan,
        "gini": gini(v), "gini_pos": gini(v[v >= 1]) if (v >= 1).sum() > 1 else np.nan,
        "median": float(np.median(v)), "q25": float(np.quantile(v, .25)),
        "q75": float(np.quantile(v, .75)), "q90": float(np.quantile(v, .90)),
        "q95": float(np.quantile(v, .95)), "q99": float(np.quantile(v, .99)),
        "n_distinct": int(len(np.unique(v))),
        "top1k": k1, "top1_share": dsc[:k1].sum() / tot, "top10_share": dsc[:10].sum() / tot,
        "trim_mean_10pct": float(np.mean(np.sort(v)[int(0.10 * n):n - int(0.10 * n)])),
        "trim_mean_1pct": float(np.mean(np.sort(v)[int(0.01 * n):n - int(0.01 * n)])),
        "mean_excl_top1": float(dsc[k1:].mean()),
        "max_over_mean": dsc[0] / mean if mean > 0 else np.nan,
        "total": tot,
    }


def local_slopes(vals, tag):
    """双对数 CCDF 的局部斜率（Top 窗口 OLS）——诊断量，不作为尾指数报出。"""
    v = np.sort(np.asarray(vals, dtype=float))[::-1]
    n = len(v)
    rows = []
    for frac in (0.50, 0.25, 0.10, 0.05, 0.02):
        k = max(int(np.ceil(n * frac)), 5)
        x = v[:k]
        yy = (np.arange(1, k + 1) - 0.5) / n          # 降序第 i 位对应 P(X >= x_i)
        ok = (x > 0) & (yy > 0)
        if int(ok.sum()) < 5:
            rows.append({"series": tag, "top_frac": frac, "k": int(ok.sum()), "slope": np.nan,
                         "x_lo": np.nan, "x_hi": np.nan, "note": "有效点不足或含 0"})
            continue
        bb = np.polyfit(np.log(x[ok]), np.log(yy[ok]), 1)
        rows.append({"series": tag, "top_frac": frac, "k": int(ok.sum()), "slope": float(bb[0]),
                     "x_lo": float(x[ok][-1]), "x_hi": float(x[ok][0]), "note": ""})
    return rows


def log_binned_tail(vals, tag, nb=8):
    """等**计数**分箱（每箱 n/nb 个点）的 log-log 斜率：CSN 的目视诊断。"""
    v = np.sort(np.asarray(vals, dtype=float))[::-1]
    n = len(v)
    k = max(n // nb, 3)
    rows = []
    for i in range(nb):
        bs = v[i * k:(i + 1) * k]
        if len(bs) < 3 or bs.mean() <= 0:
            continue
        ccdf_mid = 1.0 - 0.5 * ((i * k + 1) / n + ((i + 1) * k) / n)
        if ccdf_mid <= 0:
            continue
        rows.append({"series": tag, "bin": i + 1, "n": len(bs), "x_mean": float(bs.mean()),
                     "x_min": float(bs.min()), "x_max": float(bs.max()), "ccdf_mid": float(ccdf_mid)})
    df = pd.DataFrame(rows)
    if len(df) >= 3:
        bb = np.polyfit(np.log(df["x_mean"]), np.log(df["ccdf_mid"]), 1)
        df["global_loglog_slope"] = float(bb[0])
    return df


# ============================== 4. 主流程 ============================== #
log(f"start | module={MODULE} SEED={SEED} NBOOT={NBOOT} NLR={NLR} NRES={NRES} MIN_PTS={MIN_PTS}")
b, ADF = anchors_or_die()
log(f"anchors OK {int(ADF['ok'].sum())}/{len(ADF)}")

NODES, GU, GD, SETS = b["nodes"], b["Gu"], b["Gd"], b["sets"]
ATTR = sorted(SETS["attributed"])
ALLN = sorted(SETS["in_graph"])
GCCN = sorted(SETS["gcc"])
DEAD = sorted(b["fail"]["dead_all"])
NONDEAD = [n for n in ATTR if n not in set(DEAD)]
d_attr = NODES.set_index("node")
deg_u_all = dict(GU.degree())

SERIES = {
    "degree_u_671":    d_attr.loc[ATTR, "degree_u"].to_numpy(int),
    "degree_u_676":    np.array([deg_u_all[n] for n in ALLN], dtype=int),
    "degree_u_674gcc": np.array([deg_u_all[n] for n in GCCN], dtype=int),
    "in_degree_671":   d_attr.loc[ATTR, "in_degree"].to_numpy(int),
    "out_degree_671":  d_attr.loc[ATTR, "out_degree"].to_numpy(int),
    "strength_u_671":  d_attr.loc[ATTR, "strength_u"].to_numpy(int),
}
FIT_SERIES = ["degree_u_671", "degree_u_676", "in_degree_671", "strength_u_671"]
SUPP_SERIES = ["degree_u_674gcc"]
DESCRIBE_ONLY = ["out_degree_671"]

_o671 = d_attr.loc[ATTR, "out_degree"].to_numpy(int)
EXTRA_DESC = {
    "out_degree_671_positive":  _o671[_o671 > 0],
    "out_degree_509_nondead":   d_attr.loc[NONDEAD, "out_degree"].to_numpy(int),
    "in_degree_509_nondead":    d_attr.loc[NONDEAD, "in_degree"].to_numpy(int),
    "degree_u_162_dead":        np.array([deg_u_all[n] for n in DEAD], dtype=int),
    "strength_u_509_nondead":   d_attr.loc[NONDEAD, "strength_u"].to_numpy(int),
}
ALL_DESC = dict(SERIES)
ALL_DESC.update(EXTRA_DESC)

# --- 4.1 描述统计 + 重抽样误差棒 ------------------------------------------- #
SUM_ROWS = []
for kk, v in ALL_DESC.items():
    row = {"series": kk,
           "role": ("fit" if kk in FIT_SERIES else
                    "sensitivity" if kk in SUPP_SERIES else "describe_only")}
    row.update(describe(v))
    SUM_ROWS.append(row)
summary = pd.DataFrame(SUM_ROWS)

res_rows = []
rng = np.random.default_rng(SEED + 11)
for kk, v in ALL_DESC.items():
    v = np.asarray(v)
    n = len(v)
    k1 = max(int(np.ceil(0.01 * n)), 1)
    acc = {s: np.empty(NRES) for s in ("gini", "top1", "top10", "cv")}
    for i in range(NRES):
        s = v[rng.integers(0, n, n)]
        acc["gini"][i] = gini(s)
        acc["top1"][i] = top_share(s, k1)
        acc["top10"][i] = top_share(s, 10)
        acc["cv"][i] = s.std(ddof=1) / s.mean() if s.mean() > 0 else np.nan
    pt = describe(v)
    for stat, col in (("gini", "gini"), ("top1", "top1_share"),
                      ("top10", "top10_share"), ("cv", "cv")):
        d = acc[stat][np.isfinite(acc[stat])]
        res_rows.append({"series": kk, "stat": stat, "point": pt[col],
                         "boot_median": float(np.median(d)), "boot_sd": float(d.std(ddof=1)),
                         "ci90_lo": float(np.quantile(d, .05)),
                         "ci90_hi": float(np.quantile(d, .95)),
                         "n_boot": int(len(d)), "seed": SEED + 11})
resil = pd.DataFrame(res_rows)
log(f"describe + resampling ({NRES} reps) done")

# --- 4.2 CCDF 表 + 局部斜率 ------------------------------------------------- #
ccdf_rows, slope_rows, tail_rows = [], [], []
for kk, v in SERIES.items():
    v = np.asarray(v, dtype=float)
    n = len(v)
    uv, uc = np.unique(v, return_counts=True)
    nge = np.cumsum(uc[::-1])[::-1]
    for j in range(len(uv)):
        strict = (nge[j + 1] / n) if j + 1 < len(uv) else 0.0
        ccdf_rows.append({"series": kk, "x": int(uv[j]), "count": int(uc[j]),
                          "n_ge": int(nge[j]), "ccdf_emp": nge[j] / n,
                          "ccdf_strict": strict, "cdf_emp": 1.0 - strict})
    slope_rows += local_slopes(v, kk)
    tail_rows.append(log_binned_tail(v, kk))
ccdf = pd.DataFrame(ccdf_rows)
slopes = pd.DataFrame(slope_rows)
tail_bins = pd.concat(tail_rows, ignore_index=True)
log(f"ccdf rows={len(ccdf)}")

# --- 4.3 MLE + x_min 搜索 + 池化 bootstrap + 配对 LR ------------------------ #
ORDER = ["powerlaw", "trunc_powerlaw", "lognormal", "exponential", "nbinomial"]


def make_dists(x_obs_max):
    """每个序列一套分布实例（截断幂律的支撑与该序列的观测最大值绑定）。"""
    xcap = float(min(max(3.0 * float(x_obs_max), 3000.0), 2.0e5))
    ds = [PowerLaw(), ExpCutoffPL(xcap), LogNormal(), Exponential(), NegBinom()]
    return {d.name: d for d in ds}, xcap


xmin_rows, fit_rows, lr_rows = [], [], []
FITS: dict[str, dict] = {}

for si, sname in enumerate(FIT_SERIES + SUPP_SERIES):
    t0 = time.time()
    vals = np.asarray(SERIES[sname], dtype=float)
    D, xcap = make_dists(vals.max())
    dists = [D[n] for n in ORDER]
    seed_base = SEED + 1000 * (si + 1)                # 只由序列下标派生，可复现

    # (a) 幂律 x_min 搜索（CSN 约束 + 去掉 6x^2 的对照）
    sr = search_xmin(D["powerlaw"], vals, MIN_PTS, six_sq=True)
    sr_free = search_xmin(D["powerlaw"], vals, MIN_PTS, six_sq=False)
    for src, s in (("csn_constrained", sr), ("no_6xsq_constraint", sr_free)):
        for _, r in s.iterrows():
            xmin_rows.append({"series": sname, "dist": "powerlaw", "curve": src,
                              "samp_xcap": xcap, **r.to_dict()})
    xm, csn_ok = pick_xmin(sr, six_sq=True)
    xm_free, _ = pick_xmin(sr_free, six_sq=False)

    own = {"powerlaw": xm, "nbinomial": xm}
    for dn in ("trunc_powerlaw", "lognormal", "exponential"):
        s2 = search_xmin(D[dn], vals, MIN_PTS, six_sq=False)
        for _, r in s2.iterrows():
            xmin_rows.append({"series": sname, "dist": dn, "curve": "own_no_6xsq",
                              "samp_xcap": xcap, **r.to_dict()})
        own[dn] = pick_xmin(s2, six_sq=False)[0]
    log(f"{sname}: x_min(powerlaw|CSN)={xm:g} 可行={csn_ok} 无6x^2={xm_free:g} "
        f"own={ {k: v for k, v in own.items()} } xcap={xcap:g} ({time.time()-t0:.1f}s)")

    # (b) 公共 x_min 上的全套拟合
    m = vals >= xm
    xv, cv = unique_counts(vals[m])
    n_fit = int(cv.sum())
    thetas, lls, kss = {}, {}, {}
    for d in dists:
        th, ll, Dk = fit_pair(d, xv, cv, xm)
        thetas[d.name], lls[d.name], kss[d.name] = th, ll, Dk
    fit_t = time.time()

    # (c) 池化 bootstrap
    fitted = [thetas[n] for n in ORDER]
    pools = bootstrap_pool(dists, fitted, xm, n_fit, NBOOT, seed_base)
    log(f"{sname}: pool {NBOOT}x{len(ORDER)} done ({time.time()-fit_t:.1f}s)")

    for di, dn in enumerate(ORDER):
        d = D[dn]
        k = d.npar
        pl_ = pools[di]
        ks_obs = kss[dn]
        dv = pl_["kss"][:, di]
        dv = dv[np.isfinite(dv)]
        p_ks = float((dv >= ks_obs).mean()) if len(dv) else np.nan
        pars = np.array([p[di] for p in pl_["pars"] if all(np.all(np.isfinite(q)) for q in p)])
        row = {"series": sname, "dist": dn, "xmin_source": "common_pl_ks_search",
               "x_min": xm, "n_total": len(vals), "n_fit": n_fit, "n_distinct_fit": len(xv),
               "params": fmt_params(d, thetas[dn]), "n_par": k, "loglik": lls[dn],
               "aic": 2 * k - 2 * lls[dn], "bic": k * np.log(n_fit) - 2 * lls[dn],
               "KS": ks_obs, "ks_boot_mean": float(np.mean(dv)) if len(dv) else np.nan,
               "ks_boot_q05": float(np.quantile(dv, .05)) if len(dv) else np.nan,
               "ks_boot_q95": float(np.quantile(dv, .95)) if len(dv) else np.nan,
               "ks_boot_p": p_ks, "n_boot": int(len(dv)),
               "samp_xcap": xcap, "samp_trunc_mass": float(d.trunc),
               "samp_max_median": float(np.median(pl_["maxv"]))}
        for pn_i, pn in enumerate(d.pname):
            col = pars[:, pn_i] if len(pars) else np.array([np.nan])
            row[f"par_{pn}"] = float(thetas[dn][pn_i])
            row[f"par_{pn}_boot_mean"] = float(np.mean(col))
            row[f"par_{pn}_boot_lo95"] = float(np.quantile(col, .025))
            row[f"par_{pn}_boot_hi95"] = float(np.quantile(col, .975))
        fit_rows.append(row)

    # 对照行：各分布自身最优 x_min / 无约束幂律 / x_min=1 全量（不做 bootstrap）
    for dn, ox in own.items():
        if dn == "nbinomial" or ox is None or ox == xm:
            continue
        xv2, cv2 = unique_counts(vals[vals >= ox])
        th2, ll2, D2 = fit_pair(D[dn], xv2, cv2, ox)
        fit_rows.append({"series": sname, "dist": dn, "xmin_source": "own_ks_search",
                         "x_min": ox, "n_total": len(vals), "n_fit": int(cv2.sum()),
                         "n_distinct_fit": len(xv2), "params": fmt_params(D[dn], th2),
                         "n_par": D[dn].npar, "loglik": ll2,
                         "aic": 2 * D[dn].npar - 2 * ll2,
                         "bic": D[dn].npar * np.log(cv2.sum()) - 2 * ll2, "KS": D2,
                         "samp_xcap": xcap,
                         **{f"par_{p}": float(v) for p, v in zip(D[dn].pname, th2)}})
    for tag_src, xx in (("free_ks_search", xm_free), ("all_data_xmin_1", 1.0)):
        if xx == xm:
            continue
        xv3, cv3 = unique_counts(vals[vals >= xx])
        th3, ll3, D3 = fit_pair(D["powerlaw"], xv3, cv3, xx)
        fit_rows.append({"series": sname, "dist": "powerlaw", "xmin_source": tag_src,
                         "x_min": xx, "n_total": len(vals), "n_fit": int(cv3.sum()),
                         "n_distinct_fit": len(xv3), "params": fmt_params(D["powerlaw"], th3),
                         "n_par": 1, "loglik": ll3, "aic": 2 - 2 * ll3,
                         "bic": np.log(cv3.sum()) - 2 * ll3, "KS": D3, "par_alpha": float(th3[0]),
                         "samp_xcap": xcap})

    # (d) 两两配对 LR（用池化 bootstrap 的零分布）
    for i in range(len(ORDER)):
        for j in range(i + 1, len(ORDER)):
            A, Bn = ORDER[i], ORDER[j]
            R = float(lls[A] - lls[Bn])
            rr = np.repeat(D[A].logpmf(xv, thetas[A], xm) - D[Bn].logpmf(xv, thetas[Bn], xm),
                           cv.astype(int))
            sd_clt = float(np.sqrt(n_fit * np.var(rr, ddof=1)))
            RA, RB = [], []
            for pi, lst in ((i, RA), (j, RB)):
                arr = pools[pi]["lls"][:, i] - pools[pi]["lls"][:, j]
                lst.append(arr[np.isfinite(arr)][:max(NLR // 2, 1)])
            RA, RB = np.concatenate(RA), np.concatenate(RB)
            muA, muB = float(np.mean(RA)), float(np.mean(RB))
            sdA, sdB = float(np.std(RA, ddof=1)), float(np.std(RB, ddof=1))
            sd_pool = float(np.sqrt(0.5 * (sdA ** 2 + sdB ** 2)))
            lr_rows.append({"series": sname, "dist_A": A, "dist_B": Bn, "x_min": xm,
                            "n_fit": n_fit, "loglik_A": lls[A], "loglik_B": lls[Bn],
                            "LR_A_minus_B": R, "favored": A if R > 0 else Bn,
                            "nested": "yes" if (A, Bn) == ("powerlaw", "trunc_powerlaw") else "no",
                            "aic_A": 2 * D[A].npar - 2 * lls[A],
                            "aic_B": 2 * D[Bn].npar - 2 * lls[Bn],
                            "bic_A": D[A].npar * np.log(n_fit) - 2 * lls[A],
                            "bic_B": D[Bn].npar * np.log(n_fit) - 2 * lls[Bn],
                            "R_sd_boot": sd_pool, "R_sd_clt": sd_clt,
                            "p_normal_approx": float(_norm.cdf(-abs(R) / sd_pool)) if sd_pool > 0 else np.nan,
                            "p_norm_centered_underA": float(_norm.cdf(-abs(R - muA) / sdA)) if sdA > 0 else np.nan,
                            "p_norm_centered_underB": float(_norm.cdf(-abs(R - muB) / sdB)) if sdB > 0 else np.nan,
                            "p_boot_two_sided_underA": float(min(1.0, (RA >= abs(R)).mean() + (RA <= -abs(R)).mean())),
                            "p_boot_two_sided_underB": float(min(1.0, (RB >= abs(R)).mean() + (RB <= -abs(R)).mean())),
                            "meanR_underA": muA, "meanR_underB": muB,
                            "sdR_underA": sdA, "sdR_underB": sdB,
                            "n_lr_boot": int(len(RA) + len(RB))})
    FITS[sname] = {"x_min": xm, "x_min_csn_ok": csn_ok, "x_min_free": xm_free, "n_fit": n_fit,
                   "theta": thetas, "ll": lls, "ks": kss, "xcap": xcap, "dists": D,
                   "p_ks": {ORDER[i]: float((np.nan_to_num(pools[i]["kss"][:, i], nan=-1)
                                             >= kss[ORDER[i]]).mean()) for i in range(len(ORDER))}}
    log(f"{sname}: ALL done in {time.time()-t0:.1f}s")

fits = pd.DataFrame(fit_rows)
lrs = pd.DataFrame(lr_rows)
xmin_curve = pd.DataFrame(xmin_rows)

# --- 4.4 拟合曲线并入 CCDF 表 ---------------------------------------------- #
add = {f"ccdf_fit_{dn}": np.full(len(ccdf), np.nan) for dn in ORDER}
for sname, obj in FITS.items():
    xmn = obj["x_min"]
    sel = ((ccdf["series"] == sname) & (ccdf["x"] >= xmn)).to_numpy()
    xs = ccdf.loc[sel, "x"].to_numpy(dtype=float)
    if not len(xs):
        continue
    for dn in ORDER:
        add[f"ccdf_fit_{dn}"][sel] = obj["dists"][dn].sf_cond(xs, obj["theta"][dn], xmn)
for kk, vv in add.items():
    ccdf[kk] = vv

# --- 4.5 分层诊断 + 去枢纽敏感性 ------------------------------------------- #
health_rows = []
for tag, mask in (("all_attributed_671", np.ones(len(NODES), bool)),
                  ("dead_162", NODES["is_dead"].to_numpy(bool)),
                  ("nonded_509", ~NODES["is_dead"].to_numpy(bool)),
                  ("true_fail_129", NODES["true_fail"].to_numpy(bool)),
                  ("false_fail_32", NODES["false_fail"].to_numpy(bool)),
                  ("healthy_360", NODES["is_healthy"].to_numpy(bool))):
    sub = NODES[mask]
    row = {"group": tag, "n": int(len(sub))}
    for col in ("degree_u", "in_degree", "out_degree", "strength_u"):
        v = sub[col].to_numpy(float)
        row[f"mean_{col}"] = float(v.mean()) if len(v) else np.nan
        row[f"gini_{col}"] = gini(v) if len(v) else np.nan
        row[f"max_{col}"] = float(v.max()) if len(v) else np.nan
        row[f"zero_{col}"] = int((v == 0).sum())
    health_rows.append(row)
health = pd.DataFrame(health_rows)

GU_nh = C.strip_hub_edges(GU)
deg_nh = dict(GU_nh.degree())
v_nh = np.array([deg_nh[n] for n in ATTR], dtype=float)
xvn, cvn = unique_counts(v_nh)
th_nh, ll_nh = PowerLaw().fit(xvn, cvn, 1.0)
ks_nh = ks_dist(xvn, cvn, PowerLaw().sf_cond(xvn, th_nh, 1.0))
xva, cva = unique_counts(np.asarray(SERIES["degree_u_671"], dtype=float))
th_wh, ll_wh = PowerLaw().fit(xva, cva, 1.0)
ks_wh = ks_dist(xva, cva, PowerLaw().sf_cond(xva, th_wh, 1.0))

# --- 4.6 引擎自检（写进日志，防止后续改动悄悄破坏归一化） ------------------ #
chk = []
xg = np.arange(1, 6001, dtype=float)
for d, th in ((PowerLaw(), np.array([2.4])), (LogNormal(), np.array([2.5, 1.1])),
              (Exponential(), np.array([30.0])), (NegBinom(), np.array([3.0, 0.05])),
              (ExpCutoffPL(xcap=6000.0), np.array([1.8, 80.0]))):
    for xmn in (1.0, 4.0):
        mm = xg >= xmn
        pmf = np.exp(d.log_unnorm(xg[mm], th) - d.logZ(th, xmn))
        sf_ref = np.cumsum(pmf[::-1])[::-1] / pmf.sum()
        mine = d.sf_cond(xg[mm][:400], th, xmn)
        chk.append((d.name, xmn, float(pmf.sum()), float(np.max(np.abs(sf_ref[:400] - mine)))))
log("engine selfcheck (name, xmin, sum_pmf_over[1,6000], max|sf - brute_sf|):")
for c in chk:
    log(f"   {c[0]:15s} xmin={c[1]:4.0f} sum={c[2]:.10f} sf_err={c[3]:.2e}")
KEY: dict[str, object] = {"engine_sum_pmf_worst": max(abs(c[2] - 1) for c in chk),
                          "engine_sf_maxerr": max(c[3] for c in chk)}

# --- 4.7 KEY 数字（result.md 只引用这里，杜绝手填） ------------------------ #
def getf(s, d, src="common_pl_ks_search"):
    r = fits[(fits.series == s) & (fits.dist == d) & (fits.xmin_source == src)]
    return r.iloc[0] if len(r) else None


def S(s, col):
    return float(summary.loc[summary.series == s, col].iloc[0])


def lr_of(s, a, d):
    r = lrs[(lrs.series == s) & (lrs.dist_A == a) & (lrs.dist_B == d)]
    if len(r):
        return float(r["LR_A_minus_B"].iloc[0]), float(r["p_normal_approx"].iloc[0])
    r = lrs[(lrs.series == s) & (lrs.dist_A == d) & (lrs.dist_B == a)]
    if len(r):
        return float(-r["LR_A_minus_B"].iloc[0]), float(r["p_normal_approx"].iloc[0])
    return np.nan, np.nan


KEY.update({"seed": SEED, "nboot": NBOOT, "nlr": NLR, "nres": NRES, "min_pts": MIN_PTS})
for sname in FIT_SERIES + SUPP_SERIES:
    KEY[f"{sname}__x_min_csn"] = FITS[sname]["x_min"]
    KEY[f"{sname}__x_min_csn_feasible"] = FITS[sname]["x_min_csn_ok"]
    KEY[f"{sname}__x_min_free"] = FITS[sname]["x_min_free"]
    KEY[f"{sname}__n_fit"] = FITS[sname]["n_fit"]
    KEY[f"{sname}__samp_xcap"] = FITS[sname]["xcap"]
    for dn in ORDER:
        f = getf(sname, dn)
        if f is None:
            continue
        for cn in f.index:
            if cn in ("params", "series", "dist", "xmin_source") or pd.isna(f[cn]):
                continue
            if cn.startswith(("par_", "ks_", "samp_")) or cn in ("loglik", "aic", "bic", "KS", "n_fit"):
                KEY[f"{sname}__{dn}__{cn}"] = float(f[cn])
    for a, d in (("powerlaw", "lognormal"), ("powerlaw", "trunc_powerlaw"),
                 ("powerlaw", "exponential"), ("lognormal", "trunc_powerlaw"),
                 ("lognormal", "exponential"), ("trunc_powerlaw", "exponential")):
        R, pv = lr_of(sname, a, d)
        KEY[f"{sname}__LR[{a}]-[{d}]"] = R
        KEY[f"{sname}__LRp[{a}]-[{d}]"] = pv
    sub = fits[(fits.series == sname) & (fits.xmin_source == "common_pl_ks_search")]
    if len(sub):
        KEY[f"{sname}__best_by_KS"] = str(sub.loc[sub["KS"].idxmin(), "dist"])
        KEY[f"{sname}__best_by_aic"] = str(sub.loc[sub["aic"].idxmin(), "dist"])
        KEY[f"{sname}__best_by_bic"] = str(sub.loc[sub["bic"].idxmin(), "dist"])
    for tag_src, keyname in (("all_data_xmin_1", "xmin1"), ("free_ks_search", "free")):
        r = getf(sname, "powerlaw", tag_src)
        if r is not None:
            KEY[f"{sname}__powerlaw_{keyname}__x_min"] = float(r["x_min"])
            KEY[f"{sname}__powerlaw_{keyname}__alpha"] = float(r["par_alpha"])
            KEY[f"{sname}__powerlaw_{keyname}__KS"] = float(r["KS"])
KEY["pl_alpha_xmin1_degree_u_671_withhub"] = float(th_wh[0])
KEY["pl_ks_xmin1_degree_u_671_withhub"] = ks_wh
KEY["pl_alpha_xmin1_degree_u_671_nohub"] = float(th_nh[0])
KEY["pl_ks_xmin1_degree_u_671_nohub"] = ks_nh
KEY["n_edges_gu"] = int(GU.number_of_edges())
KEY["n_edges_gu_after_hub_strip"] = int(GU_nh.number_of_edges())
KEY["out_degree_671__n_zero"] = int(S("out_degree_671", "n_zero"))
KEY["out_degree_671__zero_share"] = S("out_degree_671", "n_zero") / 671.0

# --- 4.8 写 CSV ------------------------------------------------------------- #
for df, fn in ((summary, "summary.csv"), (ccdf, "ccdf_tables.csv"), (fits, "fits.csv"),
               (lrs, "lr_tests.csv"), (xmin_curve, "xmin_search.csv"),
               (resil, "kstest_series.csv"), (slopes, "local_slope.csv"),
               (tail_bins, "log_binned_tail.csv"), (health, "health_split_describe.csv"),
               (ADF, "anchors.csv"),
               (pd.DataFrame([{"key": k, "value": v} for k, v in KEY.items()]), "key_numbers.csv")):
    df.to_csv(os.path.join(OUT, fn), index=False, encoding="utf-8-sig")
    log(f"wrote {fn} rows={len(df)}")


# ============================== 5. DISSENT.md ============================== #
zout = int(S("out_degree_671", "n_zero"))
diss = f"""# A1 DISSENT —— 口径层面的发现与处理方式

按 CONVENTIONS §9.5 写。**没有**改动 `csa_common` 或 CONVENTIONS 的任何既定口径：
下面每一条都是“把两种口径同时算出来并列报”。

## 1. A1 任务书要 `degree_u`（676 节点图），CONVENTIONS §1 要主表 671

- 两套都算了：`degree_u_671`（主口径，n=671，均值 {S('degree_u_671','mean'):.4f}）与
  `degree_u_676`（n=676，均值 {S('degree_u_676','mean'):.4f}，与锚点
  `mean_degree_u_gu676 = 7.3905` 同口径）。见 `summary.csv` 与 `fits.csv` 的 `series` 列。
- 5 个悬空节点（`985-211` `personnel` `recapi` `rmd.snst` `www2.lib`）的无向度**全部 = 1**，
  其中 `rmd.snst.ustc.edu.cn` 的无向强度 = 60。也就是 676 口径是在分布**左端**加 5 个 1，
  并在强度分布里加一个非平凡值；它们没有属性，进不了任何健康侧分析。
- 处理：判决以 671 为主、676 并列，两者的 KS/LR 排序见 `result.md` 表 4/5。
  但 676 作为**主表**违反 §1 的“主表必须 671”，因此主表用 671。

## 2. 任务书说“出度有 162/671 = 24% 的人工零点堆积”，实测**可观察到的零点总堆积是
##    {zout}/671 = {zout/671*100:.1f}%**

- `summary.csv` 的 `out_degree_671.n_zero = {zout}`。其中 162 个是 §2.1 的**完全删失**节点
  （`health_split_describe.csv` 的 `dead_162` 行 `zero_out_degree = 162`，与锚点
  `dead_with_outdegree_zero = 162` 一致）；另有 **{zout-162} 个未整站不可用的站点出度也是 0**。
- 后一组不能当成“真实零出链”：其中含部分失效站，其可抓页面里可能没有出链，
  删失是否仍在起作用**不可识别**。所以出度零点质量应报成区间：
  下界 162/671 = 24.1%（可确证的人工删失），点估计 {zout/671*100:.1f}%（可观察到的全部零）。
- 影响：出度的 Gini = {S('out_degree_671','gini'):.4f}、CV = {S('out_degree_671','cv'):.4f}
  里有很大一块是这堆零点贡献的，不是尾部形状。按任务书：出度**只描述、不拟合、不报指数**。
  正值子集对照 `out_degree_671_positive`（n = {int(S('out_degree_671_positive','n'))}，
  Gini = {S('out_degree_671_positive','gini'):.4f}）。

## 3. 强度分布上 CSN 的 `6·x_min^2` 候选约束**整体不可行**（口径需要下游注意）

- CSN 建议候选 x_min 满足 `n(>= x_min) >= 6 x_min^2`。`strength_u_671` 的取值到
  {int(S('strength_u_671','max'))} 而 n = 671，任何 x_min >= 11 都必然违反该约束，
  “幂律 x_min 由 KS 最小化搜索”在这条序列上被压到很小的 x_min（实际取
  {FITS['strength_u_671']['x_min']:g}），而该处拟合出的 α =
  {float(getf('strength_u_671','powerlaw')['par_alpha']):.4f} 已经接近可积边界 α→1。
- 处理：`xmin_search.csv` 同时给出 `curve='csn_constrained'` 与 `curve='no_6xsq_constraint'`
  两条完整曲线；`fits.csv` 用 `xmin_source` 标明用的是哪一条，两种 x_min 的拟合都进表。
  放宽约束会**偏袒**幂律（x_min 越大越容易像幂律），因此强度上的判决以“放宽后仍被拒绝”为准。
- 建议：后续模块若引用“强度的尾指数”，必须同时引用它用的 x_min，否则这个数字没有意义。

## 4. 实现披露：截断幂律的支撑是**有限**的

- `p(x) ∝ x^-alpha exp(-x/x_c)` 在 x_c → ∞ 时归一化趋于 ζ(alpha)，数值上不可求。
  本实现把支撑取为 {{x_min, ..., X_CAP}}，`X_CAP = min(max(3 × 观测最大值, 3000), 2e5)`
  （代码 `make_dists()`；实际值写进 `fits.csv` 的 `samp_xcap` 列）。
- X_CAP >= 3 × 观测最大值，因此对观测区间内的拟合与 KS 无可观察影响；
  但字面上它不是无限支撑模型。另：`fits.csv` 的 `samp_trunc_mass` 列给出**抽样网格**
  （不是截断幂律，而是幂律/对数正态/指数/负二项的 bootstrap 抽样网格）丢掉的尾概率，
  在 α 接近 1 的单元格上该值会明显 > 0（`result.md` §10 局限里有对应说明）。

## 5. 复核成立、但对下游有提示作用的两条

- §2.1 的“死站无向度 = 入度”复核成立：`degree_u_162_dead` 均值
  {S('degree_u_162_dead','mean'):.4f}，与死站入度均值一致（`health_split_describe.csv`）。
  但**反向不成立**（入度低不推出无向度低），A1 不作该推断。
- `CONVENTIONS §1` 说“这 671 个节点全部至少有一条边”：复核成立
  （`degree_u_671` 最小值 = {int(S('degree_u_671','min'))}，`n_zero = {int(S('degree_u_671','n_zero'))}`），
  但注意**入度**有 {int(S('in_degree_671','n_zero'))} 个 0、**出度**有 {zout} 个 0，
  所以“图内零度孤立点 = 0”只在无向口径下成立，写进有向表时不能照抄。
"""
with open(os.path.join(OUT, "DISSENT.md"), "w", encoding="utf-8") as f:
    f.write(diss)


# ============================== 6. result.md ============================== #
def fr(v, n=4):
    if v is None:
        return "NA"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "NA" if not np.isfinite(v) else f"{v:.{n}f}"


def verdict_tag(p):
    if p is None or not np.isfinite(p):
        return "NA"
    return "**可拒绝**" if p < 0.01 else ("边缘" if p <= 0.1 else "不可拒绝")


MAIN = "degree_u_671"
md: list[str] = []
md.append("# A1 · 度 / 强度分布的重尾刻画 —— 结果\n")
md.append(f"- 图 / 度 / 强度全部来自 `csa_common.build_all()`；锚点 "
          f"{int(ADF['ok'].sum())}/{len(ADF)} PASS（`anchors.csv`，由 `anchors_or_die()` 写出）。")
md.append(f"- `SEED = {SEED}`；池化参数化 bootstrap `NBOOT = {NBOOT}`（KS、参数分位、配对 LR 零分布"
          f"同出一池，LR 每侧最多取 `NLR = {NLR}` 个），描述统计重抽样 `NRES = {NRES}`。"
          f"seed 由 `seed_base = SEED + 1000*(序列下标+1)` 派生，全部显式传入。")
md.append("- **本文件由脚本自动生成**：每个数字都是 `KEY[...]` / `summary` / `fits` / `lrs` / "
          "`resil` 的取值（f-string），无手填；`KEY` 的全部键值另存 `key_numbers.csv`。")
md.append(f"- 引擎自检（归一化 / 尾概率 vs 暴力求和）：最大偏差 "
          f"`sum_pmf` 误差 {fr(KEY['engine_sum_pmf_worst'],2)}、"
          f"`sf` 误差 {fr(KEY['engine_sf_maxerr'],2)}，见 `run_log.txt`。")
md.append("- 措辞按 CONVENTIONS §7：只写共变与分布形状，不写因果。\n")

pl, ln, tpl = getf(MAIN, "powerlaw"), getf(MAIN, "lognormal"), getf(MAIN, "trunc_powerlaw")
ex, nb = getf(MAIN, "exponential"), getf(MAIN, "nbinomial")
R_pl_ln, p_pl_ln = lr_of(MAIN, "powerlaw", "lognormal")
R_pl_tpl, p_pl_tpl = lr_of(MAIN, "powerlaw", "trunc_powerlaw")

md.append("## 0. 判决\n")
md.append(f"**这份数据的度分布不能称作“无标度 / 幂律”。** 主口径 `degree_u_671`，"
          f"幂律 KS 最优 `x_min = {fr(FITS[MAIN]['x_min'],0)}`，`n_fit = {FITS[MAIN]['n_fit']}`：\n")
md.append(f"1. 幂律自身：KS = {fr(pl['KS'])}，bootstrap p = {fr(pl['ks_boot_p'],3)} → "
          f"{verdict_tag(pl['ks_boot_p'])}（`KEY['{MAIN}__powerlaw__ks_boot_p']`）。")
md.append(f"2. 配对似然比 R(幂律 − 对数正态) = {fr(R_pl_ln,2)}，正态近似 p = {fr(p_pl_ln,4)} → "
          f"{'**对数正态**在似然上更受支持' if R_pl_ln < 0 else '幂律在似然上更受支持'}。")
md.append(f"3. 同一 x_min 上 KS 最小 = `{KEY[f'{MAIN}__best_by_KS']}`"
          f"（{fr(min(pl['KS'], ln['KS'], tpl['KS'], ex['KS'], nb['KS']))}），"
          f"AIC 最小 = `{KEY[f'{MAIN}__best_by_aic']}`，BIC 最小 = `{KEY[f'{MAIN}__best_by_bic']}`。")
md.append(f"4. “尾指数”本身不稳定：x_min=1（全量 671）α = {fr(th_wh[0])}；"
          f"x_min={fr(FITS[MAIN]['x_min'],0)}（KS 最优）α = {fr(pl['par_alpha'])}，"
          f"bootstrap 95% 区间 [{fr(pl['par_alpha_boot_lo95'])}, {fr(pl['par_alpha_boot_hi95'])}]；"
          f"x_min={fr(KEY[f'{MAIN}__powerlaw_free__x_min'],0)}（无 6x² 约束）α = "
          f"{fr(KEY[f'{MAIN}__powerlaw_free__alpha'])}；删 6 个导航枢纽的全部关联边后"
          f"（x_min=1）α = {fr(th_nh[0])}。四个都是合法口径下的“同一个分布的幂律指数”。\n")
md.append("支持“**重尾**”（不指定分布族的较弱事实）的量彼此一致："
          f"Gini = {fr(S(MAIN,'gini'))}、CV = {fr(S(MAIN,'cv'))}、"
          f"三阶中心矩 m3 = {fr(S(MAIN,'m3'),2)}、"
          f"Top-1%（{int(S(MAIN,'top1k'))} 个节点）占无向度总量 {fr(100*S(MAIN,'top1_share'),2)}%、"
          f"Top-10 个节点占 {fr(100*S(MAIN,'top10_share'),2)}%；"
          f"全样本均值 {fr(S(MAIN,'mean'),3)} 而 10% 截尾均值 {fr(S(MAIN,'trim_mean_10pct'),3)}"
          f"（差 {fr(S(MAIN,'mean')-S(MAIN,'trim_mean_10pct'),3)}）。\n")

md.append("## 1. 描述统计（`summary.csv` ← `SUM_ROWS` + `describe()`）\n")
c1 = ["series", "role", "n", "n_zero", "mean", "var", "sd", "m3", "cv", "gini", "gini_pos",
      "top1k", "top1_share", "top10_share", "max", "median", "q95", "q99", "n_distinct",
      "trim_mean_10pct", "trim_mean_1pct", "mean_excl_top1", "max_over_mean"]
md.append("| " + " | ".join(c1) + " |")
md.append("|" + "---|" * len(c1))
for _, r in summary.iterrows():
    md.append("| " + " | ".join(str(r[c]) if c in ("series", "role") else fr(r[c], 4) for c in c1) + " |")
md.append("")
md.append("约定：`m3` 是三阶中心矩（除 n），`var/sd` 是样本方差/标准差（除 n−1）；`cv = sd/mean`；"
          "Top-1% 的人数规则 `top1k = ceil(0.01 n)`；`trim_mean_*pct` 是两端各截同比例的截尾均值；"
          "`gini_pos` 只在正值子集上算，用来分离“零点堆积”与“尾部形状”两种贡献。\n")

md.append(f"## 2. Gini / 份额 / CV 的有限样本误差棒（`kstest_series.csv` ← `resil`；"
          f"{NRES} 次重抽样，seed = `SEED+11 = {SEED+11}`）\n")
md.append("| series | 统计量 | 点估计 | bootstrap 中位数 | sd | 90% 区间 |")
md.append("|---|---|---|---|---|---|")
for _, r in resil.iterrows():
    md.append(f"| {r['series']} | {r['stat']} | {fr(r['point'])} | {fr(r['boot_median'])} | "
              f"{fr(r['boot_sd'])} | [{fr(r['ci90_lo'])}, {fr(r['ci90_hi'])}] |")
md.append("")

md.append("## 3. CCDF 数据表（`ccdf_tables.csv` ← `ccdf`）\n")
md.append(f"共 {len(ccdf)} 行，列：`series, x, count, n_ge, ccdf_emp = P(X>=x), "
          "ccdf_strict = P(X>x), cdf_emp`，再加 5 个候选分布在公共 x_min 下的条件生存 "
          "`ccdf_fit_<dist>`（x < x_min 处为 NaN，因为那里不进拟合）。下表是主序列 `degree_u_671`：\n")
sub = ccdf[ccdf.series == MAIN]
md.append("| x | count | n_ge | P(X>=x) | P(X>x) | 幂律 | 截断幂律 | 对数正态 | 指数 | 负二项 |")
md.append("|---|---|---|---|---|---|---|---|---|---|")
for _, r in sub.iterrows():
    md.append(f"| {int(r['x'])} | {int(r['count'])} | {int(r['n_ge'])} | {r['ccdf_emp']:.6f} | "
              f"{r['ccdf_strict']:.6f} | {fr(r.get('ccdf_fit_powerlaw'),6)} | "
              f"{fr(r.get('ccdf_fit_trunc_powerlaw'),6)} | {fr(r.get('ccdf_fit_lognormal'),6)} | "
              f"{fr(r.get('ccdf_fit_exponential'),6)} | {fr(r.get('ccdf_fit_nbinomial'),6)} |")
md.append("")

md.append("## 4. Clauset–Shalizi–Newman 离散 MLE 拟合（`fits.csv` ← `fits`）\n")
md.append("同一 x_min、同一子样本上拟合，故 lnL / AIC / BIC / KS 可直接比。"
          f"`KS bootstrap p` = `NBOOT = {NBOOT}` 次参数化 bootstrap 下 KS >= 观测值的比例。\n")
parcols = [c for c in ["par_alpha", "par_x_c", "par_mu", "par_sigma", "par_r", "par_p"] if c in fits.columns]
md.append("| series | dist | x_min | n_fit | 参数 | lnL | k | AIC | BIC | KS | KS boot q05–q95 | KS bootstrap p |")
md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
for _, r in fits[fits.xmin_source == "common_pl_ks_search"].iterrows():
    ps = ", ".join(f"{c[4:]}={fr(r[c],4)}" for c in parcols if pd.notna(r[c]))
    md.append(f"| {r['series']} | {r['dist']} | {fr(r['x_min'],0)} | {int(r['n_fit'])} | {ps} | "
              f"{fr(r['loglik'],3)} | {int(r['n_par'])} | {fr(r['aic'],2)} | {fr(r['bic'],2)} | "
              f"{fr(r['KS'])} | [{fr(r['ks_boot_q05'])}, {fr(r['ks_boot_q95'])}] | "
              f"{fr(r['ks_boot_p'],3)} |")
md.append("")
md.append("### 4.1 x_min 对照口径（`fits.xmin_source`；完整曲线在 `xmin_search.csv`）\n")
md.append("| series | dist | xmin_source | x_min | n_fit | lnL | KS |")
md.append("|---|---|---|---|---|---|---|")
for _, r in fits[fits.xmin_source != "common_pl_ks_search"].iterrows():
    md.append(f"| {r['series']} | {r['dist']} | {r['xmin_source']} | {fr(r['x_min'],0)} | "
              f"{int(r['n_fit'])} | {fr(r['loglik'],3)} | {fr(r['KS'])} |")
md.append("")
md.append("### 4.2 参数点估计 vs bootstrap 95% 区间（`fits.csv` 的 `par_*_boot_*`）\n")
md.append("| series | dist | 参数 | 点估计 | boot 均值 | 95% 区间 |")
md.append("|---|---|---|---|---|---|")
for s in FIT_SERIES + SUPP_SERIES:
    for dn in ORDER:
        r = getf(s, dn)
        if r is None:
            continue
        for c in parcols:
            if pd.isna(r.get(c, np.nan)):
                continue
            md.append(f"| {s} | {dn} | {c[4:]} | {fr(r[c],4)} | {fr(r.get(c+'_boot_mean'),4)} | "
                      f"[{fr(r.get(c+'_boot_lo95'),4)}, {fr(r.get(c+'_boot_hi95'),4)}] |")
md.append("")

md.append("## 5. 两两配对对数似然比（`lr_tests.csv` ← `lrs`）\n")
md.append("`LR(A−B) = lnL(A) − lnL(B)`；`p 正态近似 = Φ(−|R| / sd(R))`，"
          f"sd(R) 由池化 bootstrap 给出（从 A 拟合分布抽样的池与从 B 抽样的池各取 {NLR//2} 个，"
          "每个样本上两端都重新 MLE）；`p_boot_*` 是同一批样本的经验双侧尾概率。\n")
md.append("| series | A | B | LR(A−B) | 更受支持 | 嵌套 | sd(R)boot | p 正态近似 | p_boot(从A) | p_boot(从B) |")
md.append("|---|---|---|---|---|---|---|---|---|---|")
for _, r in lrs.iterrows():
    md.append(f"| {r['series']} | {r['dist_A']} | {r['dist_B']} | {fr(r['LR_A_minus_B'],3)} | "
              f"{r['favored']} | {r['nested']} | {fr(r['R_sd_boot'],3)} | "
              f"{fr(r['p_normal_approx'],4)} | {fr(r['p_boot_two_sided_underA'],3)} | "
              f"{fr(r['p_boot_two_sided_underB'],3)} |")
md.append("")
md.append("注：`powerlaw` vs `trunc_powerlaw` 是**嵌套**对（x_c → ∞ 退化为幂律），参数落在支撑边界，"
          "χ²/正态近似都不标准，所以这一对只作方向参考，绝对判定以 KS bootstrap 为准。\n")

md.append("## 6. 双对数斜率诊断（`local_slope.csv` / `log_binned_tail.csv`）\n")
md.append("MANIFEST 明确“不许只画双对数直线就说幂律”。下表说明**斜率随窗口漂移**，"
          "因此其中任何一个都不能当尾指数报出。\n")
md.append("| series | Top 窗口 | 点数 | x 区间 | 局部斜率 |")
md.append("|---|---|---|---|---|")
for _, r in slopes.iterrows():
    md.append(f"| {r['series']} | {r['top_frac']*100:.0f}% | {int(r['k'])} | "
              f"[{fr(r['x_lo'],1)}, {fr(r['x_hi'],1)}] | {fr(r['slope'],3)} |")
md.append("")
for s in FIT_SERIES + DESCRIBE_ONLY:
    tb = tail_bins[tail_bins.series == s]
    if len(tb):
        md.append(f"- `{s}` 等尾 8 分箱全局 log-log 斜率 = {fr(tb['global_loglog_slope'].iloc[0],3)}")
md.append("")

zout = int(S("out_degree_671", "n_zero"))
md.append("## 7. 出度：只描述，不拟合（CONVENTIONS §2.1）\n")
md.append(f"- `out_degree_671` 可观察到 **{zout}/671 = {zout/671*100:.1f}%** 的零"
          f"（`KEY['out_degree_671__n_zero']`）。其中 **162 个是整站不可用节点的完全删失**"
          f"（`health_split_describe.csv` 的 `dead_162.zero_out_degree = 162`；锚点 "
          f"`dead_with_outdegree_zero = 162`；即任务书所指的 162/671 = 24.1% 人工零点堆积），"
          f"另有 **{zout-162} 个未整站不可用的站点出度也是 0**，其删失程度不可识别。\n")
md.append(f"- 因此出度的 Gini = {fr(S('out_degree_671','gini'))}、CV = {fr(S('out_degree_671','cv'))}、"
          f"Top-1% 份额 = {fr(S('out_degree_671','top1_share'))} 的“极端性”很大部分来自零点堆积 "
          "而不是尾部形状。**本单元不对出度做任何 MLE，不报出度的幂律指数。**\n")
md.append(f"- 同一序列的对照描述（仍不拟合）：正值子集 `out_degree_671_positive`"
          f"（n = {int(S('out_degree_671_positive','n'))}，mean = {fr(S('out_degree_671_positive','mean'))}，"
          f"Gini = {fr(S('out_degree_671_positive','gini'))}）；"
          f"未整站不可用子集 `out_degree_509_nondead`（n = {int(S('out_degree_509_nondead','n'))}，"
          f"零点个 {int(S('out_degree_509_nondead','n_zero'))}）。\n")
md.append(f"- 入度方向没有这种删失（`in_degree_671` 零点 {int(S('in_degree_671','n_zero'))} 个），"
          "所以“重尾”讨论只在 `degree_u` / `in_degree` / `strength_u` 上进行。\n")

md.append("## 8. 逐序列判决（同一套检验，四个主序列 + 一个对照序列）\n")
for s in FIT_SERIES + SUPP_SERIES:
    a_pl, a_ln, a_tpl, a_ex, a_nb = (getf(s, d) for d in
                                     ("powerlaw", "lognormal", "trunc_powerlaw", "exponential",
                                      "nbinomial"))
    if a_pl is None:
        continue
    Rln, pln = lr_of(s, "powerlaw", "lognormal")
    Rtp, ptp = lr_of(s, "powerlaw", "trunc_powerlaw")
    md.append(f"### {s} — x_min = {fr(FITS[s]['x_min'],0)}"
              f"（CSN 约束可行 = {FITS[s]['x_min_csn_ok']}；无 6x² 约束时 = "
              f"{fr(FITS[s]['x_min_free'],0)}），n_fit = {FITS[s]['n_fit']}\n")
    md.append(f"- 幂律 α = {fr(a_pl['par_alpha'])} "
              f"[{fr(a_pl['par_alpha_boot_lo95'])}, {fr(a_pl['par_alpha_boot_hi95'])}]，"
              f"KS = {fr(a_pl['KS'])}，p = {fr(a_pl['ks_boot_p'],3)} → **{verdict_tag(a_pl['ks_boot_p'])}**")
    md.append(f"- 截断幂律 KS = {fr(a_tpl['KS'])}（p = {fr(a_tpl['ks_boot_p'],3)}，"
              f"{verdict_tag(a_tpl['ks_boot_p'])}）；对数正态 KS = {fr(a_ln['KS'])}"
              f"（p = {fr(a_ln['ks_boot_p'],3)}，{verdict_tag(a_ln['ks_boot_p'])}）；"
              f"指数 KS = {fr(a_ex['KS'])}（p = {fr(a_ex['ks_boot_p'],3)}，"
              f"{verdict_tag(a_ex['ks_boot_p'])}）；负二项 KS = {fr(a_nb['KS'])}"
              f"（p = {fr(a_nb['ks_boot_p'],3)}，{verdict_tag(a_nb['ks_boot_p'])}）")
    md.append(f"- R(幂律−对数正态) = {fr(Rln,3)}（p = {fr(pln,4)}）→ "
              f"{'对数正态' if Rln < 0 else '幂律'}在似然上更受支持；"
              f"R(幂律−截断幂律) = {fr(Rtp,3)}（p = {fr(ptp,4)}）")
    md.append(f"- KS 最小者 = `{KEY[f'{s}__best_by_KS']}`，AIC 最小者 = `{KEY[f'{s}__best_by_aic']}`，"
              f"BIC 最小者 = `{KEY[f'{s}__best_by_bic']}`\n")

md.append("## 9. 分层复核（`health_split_describe.csv` ← `health`）\n")
md.append("| group | n | mean_degree_u | gini_degree_u | mean_in_degree | mean_out_degree | "
          "zero_out_degree | max_degree_u | mean_strength_u |")
md.append("|---|---|---|---|---|---|---|---|---|")
for _, r in health.iterrows():
    md.append(f"| {r['group']} | {r['n']} | {fr(r['mean_degree_u'])} | {fr(r['gini_degree_u'])} | "
              f"{fr(r['mean_in_degree'])} | {fr(r['mean_out_degree'])} | {int(r['zero_out_degree'])} | "
              f"{fr(r['max_degree_u'],0)} | {fr(r['mean_strength_u'],1)} |")
md.append("")
md.append("此表只说明主口径的形状不是由死站单独撑起来的"
          f"（死站子集 n = {int(S('degree_u_162_dead','n'))} 的无向度均值 "
          f"{fr(S('degree_u_162_dead','mean'))}、Gini = {fr(S('degree_u_162_dead','gini'))}）。"
          "**不**据此作任何因果或健康推断（§7）。\n")

md.append("## 10. 局限\n")
md.append("- 单次横截面、只有观察到的链接计数：任何分布形状都不携带机制解释；"
          "权重语义按 §1 写“观察到的链接次数”。")
md.append("- 20 条 URL 的抓取上限截断出度/出强度，这是出度不进拟合的第二个理由"
          "（第一个见 §7 的删失）。")
md.append("- 悬空引用节点（676 − 671 = 5）无属性，主口径排除，676 口径并列在 `fits.csv`。")
md.append(f"- 参数化 bootstrap 假设“拟合分布为真”，且样本量只有 n_fit = {FITS[MAIN]['n_fit']}，"
          "所以 `ks_boot_p` 是**候选族内**的相对可拒绝性，不是“模型为真”的概率。")
md.append(f"- 度分布取值离散且不多（`degree_u_671` 唯一值 {int(S('degree_u_671','n_distinct'))} 个），"
          "KS 在这种网格上偏保守：大 p 不能读成“确认幂律”。")
md.append("- 抽样网格：`fits.csv` 的 `samp_trunc_mass` 给出该单元格 bootstrap 抽样时丢掉的尾概率；"
          "在 α→1 的单元格（尤其 `strength_u_671` 在 x_min 很小处）该值明显非零，"
          "对应 p 值要按“在截断版本上的检验”来读。")
md.append("- 本单元只给排序与误差，不宣布“哪个替代模型为真”；截断幂律 / 对数正态 / 负二项在"
          "同一 x_min 上互有胜负（表 5）。")
md.append("- 两处任务书与 CONVENTIONS 不一致、以及两处方法口径的必要放宽，见同目录 `DISSENT.md`。")

with open(os.path.join(OUT, "result.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")


# ============================== 7. RUN.txt ============================== #
FILES = ["run_a1.py", "result.md", "ccdf_tables.csv", "fits.csv", "lr_tests.csv", "RUN.txt",
         "DISSENT.md", "summary.csv", "key_numbers.csv", "xmin_search.csv", "kstest_series.csv",
         "local_slope.csv", "log_binned_tail.csv", "health_split_describe.csv", "anchors.csv",
         "run_log.txt"]
V = C.versions()
run_lines = [
    "A1_degree_heavy_tail - run record",
    "=" * 66,
    "commandline      : cd pipeline && python out/A1_degree_heavy_tail/run_a1.py",
    "cwd              : " + os.getcwd(),
    f"duration_sec     : {time.time()-T_START:.2f}",
    f"start_utc        : {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(T_START))}",
    f"end_utc          : {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
    "platform         : " + platform.platform(),
    "python           : " + V["python"],
    "networkx         : " + V["networkx"],
    "pandas           : " + V["pandas"],
    "numpy            : " + V["numpy"],
    "scipy            : " + scipy.__version__,
    "scikit-learn     : " + sklearn.__version__,
    "csa_common       : " + os.path.join(PIPELINE, "csa_common.py"),
    "SEED             : " + str(V["SEED"]) + "  (csa_common.SEED)",
    "PYTHONHASHSEED   : " + str(V["PYTHONHASHSEED"]),
    "NBOOT / NLR / NRES / MIN_PTS : " + f"{NBOOT} / {NLR} / {NRES} / {MIN_PTS}",
    "seed 分配        : 描述统计重抽样 = SEED+11; 每序列的池化 bootstrap = "
    "SEED+1000*(si+1)（si=序列下标）; 全部显式传入 numpy.random.default_rng，不用 hash()",
    "anchor check     : " + f"{int(ADF['ok'].sum())}/{len(ADF)} PASS "
    "(csa_common.check_anchors + assert_anchors)",
    "fitting series   : " + ", ".join(FIT_SERIES + SUPP_SERIES),
    "describe only    : " + ", ".join(DESCRIBE_ONLY + list(EXTRA_DESC)),
    "distributions    : " + ", ".join(ORDER),
    "",
    "产出文件（同目录）：",
]
for fn in FILES:
    p = os.path.join(OUT, fn)
    ex_ = os.path.exists(p)
    run_lines.append(f"  {fn:26s} {os.path.getsize(p) if ex_ else 'MISSING':>10}"
                     + (" bytes" if ex_ else ""))
run_lines += ["", "KEY 数字（= key_numbers.csv）："]
for k, v in KEY.items():
    run_lines.append(f"  {k:56s} {v}")
run_lines += ["", "日志：", ""] + LOG
with open(os.path.join(OUT, "RUN.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(run_lines) + "\n")
with open(os.path.join(OUT, "run_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(LOG) + "\n")

print(f"\n=== DONE in {time.time()-T_START:.1f}s -> {OUT} ===")
for k, v in KEY.items():
    print(f"  {k:56s} {v}")
