"""상관관계 + lead-lag 산출 모듈.

score_history.json 의 시계열을 입력으로 받아:
- correlation_matrix.json: 모든 sector 쌍의 60일 Pearson 상관
- lead_lag_matrix.json: 각 leader 에 대해 best lag k (1·3·5·10·15) + hit rate

hit_rate 정의: leader 의 일별 점수 변화 분포에서 상위 25% percentile 초과한 시점 t 들 중,
  follower 가 t+k 에서 자기 분포 상위 25% 초과한 비율.
"""

from __future__ import annotations

import math


def pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2 or len(y) != n:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    dy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def rank(values: list[float]) -> list[float]:
    """Average rank with ties (mid-rank). Spearman 용. 1-indexed.
    예: [3, 1, 4, 1, 5] → [3, 1.5, 4, 1.5, 5] (1 두 개는 1·2 평균 = 1.5).
    """
    n = len(values)
    if n == 0:
        return []
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-indexed mid-rank
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    """순위 기반 Pearson — outlier 와 비선형 단조 관계에 robust."""
    return pearson(rank(x), rank(y))


def two_factor_regress(
    y: list[float], x1: list[float], x2: list[float]
) -> tuple[float, float, float, list[float]]:
    """OLS y = α + β1·x1 + β2·x2 + ε. Returns (α, β1, β2, residuals).
    Normal equation 직접 풀이 (numpy 안 씀).
    """
    n = len(y)
    if n < 3 or len(x1) != n or len(x2) != n:
        return 0.0, 0.0, 0.0, list(y)
    my = sum(y) / n
    m1 = sum(x1) / n
    m2 = sum(x2) / n
    yc = [yi - my for yi in y]
    xc1 = [a - m1 for a in x1]
    xc2 = [a - m2 for a in x2]
    s11 = sum(a * a for a in xc1)
    s22 = sum(a * a for a in xc2)
    s12 = sum(a * b for a, b in zip(xc1, xc2))
    s1y = sum(a * b for a, b in zip(xc1, yc))
    s2y = sum(a * b for a, b in zip(xc2, yc))
    det = s11 * s22 - s12 * s12
    if abs(det) < 1e-10:
        return my, 0.0, 0.0, [yi - my for yi in y]
    beta1 = (s22 * s1y - s12 * s2y) / det
    beta2 = (s11 * s2y - s12 * s1y) / det
    alpha = my - beta1 * m1 - beta2 * m2
    residuals = [yi - alpha - beta1 * a - beta2 * b for yi, a, b in zip(y, x1, x2)]
    return alpha, beta1, beta2, residuals


def vasicek_shrink(beta: float, sample_mean: float = 1.0, weight: float = 0.5) -> float:
    """β 추정 noise 줄이려고 sample mean (default 1.0) 쪽으로 부분 끌어당김.
    weight=1 이면 그대로, 0 이면 완전 mean. 기본 0.5 (절반).
    """
    return weight * beta + (1.0 - weight) * sample_mean


def diff(values: list[float]) -> list[float]:
    """First difference: [v_t - v_{t-1}]. 길이 n-1."""
    return [values[i] - values[i - 1] for i in range(1, len(values))]


def percentile(values: list[float], p: float) -> float:
    """p 분위수 (0~1). values 가 비면 0.0. linear interpolation."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = p * (len(s) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return s[lo]
    frac = rank - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _build_series(history: list[dict], window: int) -> tuple[list[str], dict[str, list[float]]]:
    """history (latest first) 를 window 길이로 잘라 sector_id → [old, ..., latest] 시계열로 변환.
    누락 entry 가 있는 sector 는 drop.
    """
    recent = history[:window]
    if not recent:
        return [], {}
    # history 는 latest first → 시간 순으로 뒤집어 [old → latest] 가 되도록
    chronological = list(reversed(recent))

    series: dict[str, list[float]] = {}
    for entry in chronological:
        scores = entry.get("scores", {})
        for sid, val in scores.items():
            series.setdefault(sid, []).append(float(val))

    n = len(chronological)
    series = {sid: vals for sid, vals in series.items() if len(vals) == n}
    sectors = sorted(series.keys())
    return sectors, series


def _residualize(
    series: dict[str, list[float]],
    market_keys: tuple[str, str] = ("_market_kodex200", "_market_kosdaq150"),
) -> dict[str, list[float]] | None:
    """각 sector 의 차분 시계열을 두 시장 (KOSPI 200, KOSDAQ 150) 차분에 회귀해 잔차 반환.
    Vasicek shrinkage 로 β 안정화. market 시계열이 없으면 None — caller fallback.
    """
    m1_key, m2_key = market_keys
    if m1_key not in series or m2_key not in series:
        return None
    m1 = diff(series[m1_key])
    m2 = diff(series[m2_key])
    if len(m1) < 5 or len(m2) < 5:
        return None

    # 1차 패스 — 모든 sector 의 β 추정 (sample mean β 산출용).
    diffs: dict[str, list[float]] = {}
    raw_betas: dict[str, tuple[float, float]] = {}
    for sid, vals in series.items():
        if sid in (m1_key, m2_key):
            continue
        d = diff(vals)
        if len(d) != len(m1):
            continue
        diffs[sid] = d
        _, b1, b2, _ = two_factor_regress(d, m1, m2)
        raw_betas[sid] = (b1, b2)

    if not raw_betas:
        return None

    # Sample mean β (sector-population). Vasicek 의 prior.
    mean_b1 = sum(b[0] for b in raw_betas.values()) / len(raw_betas)
    mean_b2 = sum(b[1] for b in raw_betas.values()) / len(raw_betas)

    # 2차 패스 — Vasicek shrunken β 로 잔차 재계산.
    residuals: dict[str, list[float]] = {}
    for sid, d in diffs.items():
        b1_raw, b2_raw = raw_betas[sid]
        b1 = vasicek_shrink(b1_raw, mean_b1, 0.5)
        b2 = vasicek_shrink(b2_raw, mean_b2, 0.5)
        # α 도 shrunken β 로 재계산.
        my = sum(d) / len(d)
        mm1 = sum(m1) / len(m1)
        mm2 = sum(m2) / len(m2)
        alpha = my - b1 * mm1 - b2 * mm2
        eps = [yi - alpha - b1 * a - b2 * b for yi, a, b in zip(d, m1, m2)]
        residuals[sid] = eps
    return residuals


def compute_correlation_matrix(history: list[dict], window: int = 60) -> dict:
    """시장 효과 제거된 sector 간 상관 (정확도 pipeline):
    1. 차분 (first difference) — stationary 한 변화량 시계열.
    2. Two-factor 회귀 (KOSPI 200 + KOSDAQ 150 차분) → 잔차.
    3. Vasicek shrinkage on β (sample mean 50%).
    4. Spearman rank correlation on 잔차.

    market 시계열 없으면 fallback: 원시 점수 시계열 + Pearson (옛 동작).
    """
    sectors, series = _build_series(history, window)
    n_obs = len(series[sectors[0]]) if sectors else len(history[:window])

    if n_obs < 30 or not sectors:
        return {"window_days": window, "n_observations": n_obs, "matrix": {}}

    residuals = _residualize(series)
    use_residuals = residuals is not None
    follower_pool = [s for s in sectors if not s.startswith("_market")]

    matrix: dict[str, dict[str, float]] = {}
    for s1 in follower_pool:
        matrix[s1] = {}
        for s2 in follower_pool:
            if s1 == s2:
                matrix[s1][s2] = 1.0
                continue
            if use_residuals and residuals is not None:
                r = spearman(residuals[s1], residuals[s2])
            else:
                r = pearson(series[s1], series[s2])
            matrix[s1][s2] = round(r, 3)

    return {
        "window_days": window,
        "n_observations": n_obs,
        "matrix": matrix,
        "method": "partial_spearman" if use_residuals else "raw_pearson",
    }


def compute_lead_lag_matrix(
    history: list[dict],
    window: int = 60,
    lags: list[int] | None = None,
    hit_threshold_percentile: float = 0.75,
    top_followers: int = 5,
) -> dict:
    """각 (leader, follower) 쌍에 대해 best lag (correlation 절댓값 최대) + hit_rate 산출.

    hit_rate: leader 의 t→t+1 점수 변화 중 상위 25% percentile 초과한 시점 집합 S 추출.
      S 안의 각 t 에 대해 follower 의 (t+k)→(t+k+1) 변화가 follower 자기 25% 초과면 hit.
    """
    if lags is None:
        lags = [1, 3, 5, 10, 15]

    sectors, series = _build_series(history, window)
    n_obs = len(series[sectors[0]]) if sectors else len(history[:window])

    if n_obs < 30 or not sectors:
        return {
            "window_days": window,
            "n_observations": n_obs,
            "lags": lags,
            "leaders": {},
        }

    # 정확도 pipeline: 차분 + two-factor 잔차 + spearman lag.
    # 잔차 시계열 자체가 이미 차분 후 잔차 (length n-1).
    residuals = _residualize(series)
    use_residuals = residuals is not None

    follower_pool = [s for s in sectors if not s.startswith("_market")]

    # leader 의 잔차 변화량 분포로 hit threshold (signal 시점) 결정 — 시장 효과 제거 후의 진짜 강세 시점.
    if use_residuals and residuals is not None:
        thresholds: dict[str, float] = {
            sid: percentile(eps, hit_threshold_percentile) for sid, eps in residuals.items()
        }
    else:
        # fallback — 원시 점수 차분 분포.
        deltas: dict[str, list[float]] = {
            sid: [vals[i] - vals[i - 1] for i in range(1, len(vals))]
            for sid, vals in series.items()
        }
        thresholds = {sid: percentile(d, hit_threshold_percentile) for sid, d in deltas.items()}

    leaders: dict[str, dict] = {}
    for leader in follower_pool:  # leader 도 micro 만 (broad market 제외 — anchor 안 됨)
        leader_eps = residuals[leader] if use_residuals and residuals is not None else None
        if leader_eps is None and not use_residuals:
            leader_eps = [series[leader][i] - series[leader][i - 1] for i in range(1, len(series[leader]))]
        if leader_eps is None:
            continue
        leader_thr = thresholds[leader]

        followers_out: list[dict] = []
        for follower in follower_pool:
            if follower == leader:
                continue
            f_eps = residuals[follower] if use_residuals and residuals is not None else None
            if f_eps is None and not use_residuals:
                f_eps = [series[follower][i] - series[follower][i - 1] for i in range(1, len(series[follower]))]
            if f_eps is None:
                continue
            f_thr = thresholds[follower]

            best = {"lag": lags[0], "correlation": 0.0, "hit_rate": 0.0, "n_signals": 0}
            best_abs = -1.0
            for k in lags:
                if k >= len(leader_eps):
                    continue
                # cross-correlation on residuals (잔차 시계열 lag).
                lx = leader_eps[: len(leader_eps) - k]
                fy = f_eps[k:]
                corr = spearman(lx, fy) if use_residuals else pearson(lx, fy)

                # hit rate: leader 잔차 변화 상위 25% 시점 → follower 도 t+k 시점 상위 25% 비율.
                signal_indices = [t for t, d in enumerate(leader_eps) if d > leader_thr]
                hits = 0
                considered = 0
                for t in signal_indices:
                    fi = t + k
                    if fi >= len(f_eps):
                        continue
                    considered += 1
                    if f_eps[fi] > f_thr:
                        hits += 1
                hit_rate = (hits / considered) if considered > 0 else 0.0

                if abs(corr) > best_abs:
                    best_abs = abs(corr)
                    best = {
                        "lag": k,
                        "correlation": round(corr, 3),
                        "hit_rate": round(hit_rate, 3),
                        "n_signals": considered,
                    }

            if best["n_signals"] > 0 or best["correlation"] != 0.0:
                followers_out.append(
                    {
                        "sector_id": follower,
                        "best_lag": best["lag"],
                        "correlation": best["correlation"],
                        "hit_rate": best["hit_rate"],
                        "n_signals": best["n_signals"],
                    }
                )

        followers_out.sort(key=lambda f: abs(f["correlation"]), reverse=True)
        leaders[leader] = {"followers": followers_out[:top_followers]}

    return {
        "window_days": window,
        "n_observations": n_obs,
        "lags": lags,
        "hit_threshold_percentile": hit_threshold_percentile,
        "leaders": leaders,
        "method": "partial_spearman" if use_residuals else "raw_pearson",
    }


def followers_at_lag(
    lead_lag: dict, leader_id: str, target_lag: int, top: int = 3
) -> list[dict]:
    """lead_lag matrix 에서 특정 leader 의 followers 중 best_lag == target_lag 인 것만 추려 반환.
    correlation 내림차순 top N. next_cycle.json 빌드용 헬퍼.
    """
    leaders = lead_lag.get("leaders", {})
    entry = leaders.get(leader_id)
    if not entry:
        return []
    matched = [f for f in entry.get("followers", []) if f.get("best_lag") == target_lag]
    matched.sort(key=lambda f: f.get("correlation", 0), reverse=True)
    return matched[:top]


def co_movers(
    correlation: dict, sector_id: str, threshold: float = 0.6, top: int = 3
) -> list[dict]:
    """correlation_matrix 에서 sector_id 와 threshold 이상 양의 상관인 top N (자기 자신 제외).
    """
    matrix = correlation.get("matrix", {})
    row = matrix.get(sector_id, {})
    pairs = [
        {"sector_id": s, "correlation": round(c, 3)}
        for s, c in row.items()
        if s != sector_id and not s.startswith("_market") and c >= threshold
    ]
    pairs.sort(key=lambda p: p["correlation"], reverse=True)
    return pairs[:top]
