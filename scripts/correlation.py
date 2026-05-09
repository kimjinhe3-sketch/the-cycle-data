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


def compute_correlation_matrix(history: list[dict], window: int = 60) -> dict:
    sectors, series = _build_series(history, window)
    n_obs = len(series[sectors[0]]) if sectors else len(history[:window])

    if n_obs < 30 or not sectors:
        return {"window_days": window, "n_observations": n_obs, "matrix": {}}

    matrix: dict[str, dict[str, float]] = {}
    for s1 in sectors:
        matrix[s1] = {}
        for s2 in sectors:
            if s1 == s2:
                matrix[s1][s2] = 1.0
            else:
                matrix[s1][s2] = round(pearson(series[s1], series[s2]), 3)

    return {"window_days": window, "n_observations": n_obs, "matrix": matrix}


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

    # 일별 변화 (delta) 시계열 + sector 별 임계값 사전 계산
    deltas: dict[str, list[float]] = {}
    thresholds: dict[str, float] = {}
    for sid, vals in series.items():
        d = [vals[i] - vals[i - 1] for i in range(1, len(vals))]
        deltas[sid] = d
        thresholds[sid] = percentile(d, hit_threshold_percentile)

    # follower 후보: micro sector 만 (broad market `_market_*` 는 leader 로만 쓰고 follower 제외)
    follower_pool = [s for s in sectors if not s.startswith("_market")]

    leaders: dict[str, dict] = {}
    for leader in sectors:
        leader_vals = series[leader]
        leader_deltas = deltas[leader]
        leader_thr = thresholds[leader]

        followers_out: list[dict] = []
        for follower in follower_pool:
            if follower == leader:
                continue
            f_vals = series[follower]
            f_deltas = deltas[follower]
            f_thr = thresholds[follower]

            best = {"lag": lags[0], "correlation": 0.0, "hit_rate": 0.0, "n_signals": 0}
            best_abs = -1.0
            for k in lags:
                if k >= len(leader_vals):
                    continue
                # cross-correlation: leader[0..n-k-1] vs follower[k..n-1]
                lx = leader_vals[: len(leader_vals) - k]
                fy = f_vals[k:]
                corr = pearson(lx, fy)

                # hit rate: leader_deltas index t → leader 의 (t→t+1) 변화. delta length = n-1.
                # 그 시점 t 에 follower 의 (t+k → t+k+1) 변화는 f_deltas[t+k] (가 존재해야).
                signal_indices = [
                    t for t, d in enumerate(leader_deltas) if d > leader_thr
                ]
                hits = 0
                considered = 0
                for t in signal_indices:
                    fi = t + k
                    if fi >= len(f_deltas):
                        continue
                    considered += 1
                    if f_deltas[fi] > f_thr:
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
