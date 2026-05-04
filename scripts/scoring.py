"""섹터/종목 점수 계산 모듈.

원래 mock 구현 (sectorRows.ts) 의 5축 점수 (rs/flow/breadth/fatigue/catalyst)
공식을 진짜 OHLCV 시계열 위에 다시 얹는다. 모든 점수는 0~100 으로 정규화.

- rs (가격강도)   : 종목들의 20일 수익률을 백분위로 환산한 평균
- flow (자금유입) : 5일 평균 거래대금 / 20일 평균 거래대금 비율 → 백분위
- breadth (종목폭): 5일 양봉(종가>20일전 종가) 비중
- fatigue (피로도): 14일 RSI 기반, 70 이상 누적 비중
- catalyst (이슈도): 거래대금 표준편차 spike (Z-score)
- total           : 0.30·rs + 0.25·flow + 0.20·breadth − 0.15·fatigue + 0.10·catalyst (clamp 0~100)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass
class SectorScores:
    rs: float
    flow: float
    breadth: float
    fatigue: float
    catalyst: float
    total: float


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _percentile_rank(value: float, dist: list[float]) -> float:
    """value 가 dist 안에서 차지하는 백분위 (0~100). dist 가 비면 50."""
    if not dist:
        return 50.0
    sorted_d = sorted(dist)
    n = len(sorted_d)
    le = sum(1 for v in sorted_d if v <= value)
    return (le / n) * 100


def _rsi(closes: list[float], period: int = 14) -> float:
    """단일 기간 RSI (마지막 값). 데이터 부족시 50 반환."""
    if len(closes) < period + 1:
        return 50.0
    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        if diff >= 0:
            gains.append(diff)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-diff)
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _safe_pct(curr: float, base: float) -> float:
    if base <= 0:
        return 0.0
    return ((curr - base) / base) * 100


def stock_features(df: pd.DataFrame) -> dict:
    """단일 종목 OHLCV(시가총액 포함) DataFrame 으로부터 1차 지표 추출.
    df 의 인덱스는 날짜, 컬럼은 ['종가', '거래량', '거래대금'] 가정.
    """
    if df.empty:
        return {"ret20": 0.0, "ret5": 0.0, "tv_ratio": 1.0, "rsi": 50.0, "tv_z": 0.0}

    closes = df["종가"].astype(float).tolist()
    tv = df["거래대금"].astype(float).tolist() if "거래대금" in df.columns else []

    last = closes[-1]
    ret20 = _safe_pct(last, closes[-21]) if len(closes) >= 21 else 0.0
    ret5 = _safe_pct(last, closes[-6]) if len(closes) >= 6 else 0.0

    if len(tv) >= 20:
        tv_avg5 = sum(tv[-5:]) / 5
        tv_avg20 = sum(tv[-20:]) / 20
        tv_ratio = tv_avg5 / tv_avg20 if tv_avg20 > 0 else 1.0
    else:
        tv_ratio = 1.0

    rsi = _rsi(closes)

    if len(tv) >= 20:
        mean = sum(tv[-20:]) / 20
        var = sum((v - mean) ** 2 for v in tv[-20:]) / 20
        std = var**0.5
        tv_z = (tv[-1] - mean) / std if std > 0 else 0.0
    else:
        tv_z = 0.0

    return {"ret20": ret20, "ret5": ret5, "tv_ratio": tv_ratio, "rsi": rsi, "tv_z": tv_z}


def aggregate_sector_scores(
    sector_features: list[dict],
    market_ret_distribution: list[float],
    market_tv_ratio_distribution: list[float],
) -> SectorScores:
    """섹터에 속한 종목들의 features 를 받아서 5축 점수로 환산.

    - ret_distribution / tv_ratio_distribution 은 시장 전체 분포 (백분위 산정용)
    """
    if not sector_features:
        return SectorScores(50, 50, 50, 50, 50, 50)

    # rs: 평균 20일 수익률을 시장 분포에서 백분위로 변환
    avg_ret20 = sum(f["ret20"] for f in sector_features) / len(sector_features)
    rs = _percentile_rank(avg_ret20, market_ret_distribution)

    # flow: 평균 거래대금 비율 → 백분위
    avg_tv_ratio = sum(f["tv_ratio"] for f in sector_features) / len(sector_features)
    flow = _percentile_rank(avg_tv_ratio, market_tv_ratio_distribution)

    # breadth: 5일 수익률 양수 비중
    pos_count = sum(1 for f in sector_features if f["ret5"] > 0)
    breadth = (pos_count / len(sector_features)) * 100

    # fatigue: 평균 RSI 가 50 이상이면 그 거리만큼 피로도 증가
    avg_rsi = sum(f["rsi"] for f in sector_features) / len(sector_features)
    # 50→0, 70→50, 90→100 형태로 매핑
    fatigue = _clamp((avg_rsi - 50) * 2.5)

    # catalyst: 거래대금 Z-score 가 양수인 종목 비중 (어떤 종목이 spike 났는지)
    spike_count = sum(1 for f in sector_features if f["tv_z"] > 1.0)
    catalyst = (spike_count / len(sector_features)) * 100

    total = _clamp(0.30 * rs + 0.25 * flow + 0.20 * breadth - 0.15 * fatigue + 0.10 * catalyst)
    return SectorScores(rs, flow, breadth, fatigue, catalyst, total)


def derive_state(scores: SectorScores) -> str:
    if scores.fatigue > 75 and scores.rs > 60:
        return "overheated"
    if scores.rs > 65 and scores.breadth > 55:
        return "expanding"
    if scores.flow > 65 and scores.rs < 55:
        return "early"
    if scores.flow < 35 and scores.rs < 35:
        return "watch"
    return "neutral"
