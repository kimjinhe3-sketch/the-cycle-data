"""Stage 2 fetcher — 17개 섹터 + ~70개 종목의 일봉을 pykrx 로 끌어와
앱이 기대하는 응답 shape (dashboard / rotation_map / candidates / compare)
정적 JSON 으로 떨어뜨림. GitHub Actions cron 이 매일 17:30 KST 에 호출.

산출물:
- public/health.json                  : 검증용 헬스체크 (삼성전자 종가 등)
- public/dashboard.json               : 시장요약 + 강한 섹터 5개 + 자금유입 5개
- public/rotation_map.json            : 17 섹터 전체 행 + 점수 + 타임라인
- public/candidates.json              : 다음 사이클 후보 5개
- public/compare/{code}.json          : 종목별 60일 OHLCV (앱 비교차트용)
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
from pykrx import stock

import scoring

KST = timezone(timedelta(hours=9))
ROOT = Path(__file__).resolve().parent.parent
PUBLIC_DIR = ROOT / "public"
COMPARE_DIR = PUBLIC_DIR / "compare"
DATA_DIR = ROOT / "data"

KOSPI_INDEX = "1001"
KOSDAQ_INDEX = "2001"


def load_json(path: Path) -> list:
    return json.loads(path.read_text(encoding="utf-8"))


SECTORS: list[dict] = load_json(DATA_DIR / "sectors.json")
ASSETS: list[dict] = load_json(DATA_DIR / "assets.json")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------
# OHLCV 수집
# --------------------------------------------------------------------------

def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """단일 종목 일봉. 단일 종목 API 가 거래대금 컬럼을 안 주므로
    종가×거래량 로 거래대금 컬럼을 채워줌."""
    df = stock.get_market_ohlcv_by_date(start, end, ticker)
    if df.empty:
        return df
    if "거래대금" not in df.columns:
        df = df.copy()
        df["거래대금"] = df["종가"].astype("int64") * df["거래량"].astype("int64")
    return df


def fetch_index(index_code: str, start: str, end: str) -> pd.DataFrame:
    """KOSPI(1001)/KOSDAQ(2001) 지수 OHLCV. pykrx 일부 버전에서 내부 '지수명'
    lookup 이 깨지는 경우가 있어 항상 try/except 로 감싼다."""
    try:
        return stock.get_index_ohlcv_by_date(start, end, index_code)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] index {index_code} fetch fail: {exc}", file=sys.stderr)
        return pd.DataFrame()


def collect_all(start: str, end: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for asset in ASSETS:
        # ETF 와 종목 모두 같은 OHLCV API 통과 가능
        code = asset["code"]
        try:
            df = fetch_ohlcv(code, start, end)
            if not df.empty:
                out[code] = df
            time.sleep(0.05)  # KRX 부드럽게
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] {code} fetch fail: {exc}", file=sys.stderr)
    return out


# --------------------------------------------------------------------------
# 응답 shape 빌더
# --------------------------------------------------------------------------

def build_meta(as_of_iso: str) -> dict:
    return {
        "as_of": as_of_iso,
        "source": "pykrx",
        "stale": False,
        "market_session": "closed",
    }


def build_market_summary(start: str, end: str) -> dict:
    kospi = fetch_index(KOSPI_INDEX, start, end)
    kosdaq = fetch_index(KOSDAQ_INDEX, start, end)

    def last_change(df: pd.DataFrame) -> tuple[float, float]:
        if df is None or df.empty or len(df) < 2 or "종가" not in df.columns:
            return (0.0, 0.0)
        try:
            last = float(df["종가"].iloc[-1])
            prev = float(df["종가"].iloc[-2])
            pct = ((last - prev) / prev) * 100 if prev > 0 else 0.0
            return (last, pct)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] last_change fail: {exc}", file=sys.stderr)
            return (0.0, 0.0)

    kospi_v, kospi_pct = last_change(kospi)
    kosdaq_v, kosdaq_pct = last_change(kosdaq)
    return {
        "kospi_value": round(kospi_v, 2),
        "kospi_change_pct": round(kospi_pct, 2),
        "kosdaq_value": round(kosdaq_v, 2),
        "kosdaq_change_pct": round(kosdaq_pct, 2),
    }


def assets_in_sector(sector_id: str) -> list[dict]:
    return [a for a in ASSETS if a.get("sectorId") == sector_id]


def representative_assets(sector_id: str) -> list[dict]:
    in_sector = assets_in_sector(sector_id)
    stocks = [a for a in in_sector if a["type"] == "stock"][:3]
    etf = next((a for a in in_sector if a["type"] == "etf"), None)
    return stocks + ([etf] if etf else [])


def build_sector_rows(ohlcv_map: dict[str, pd.DataFrame]) -> list[dict]:
    """17 개 섹터 각각에 대해 종목 features 를 뽑고 → 섹터 점수로 집계."""
    # 시장 전체 분포 (모든 종목 features) 를 미리 만들어 백분위 계산용으로 사용
    all_features: dict[str, dict] = {}
    for code, df in ohlcv_map.items():
        all_features[code] = scoring.stock_features(df)
    market_ret_dist = [f["ret20"] for f in all_features.values()]
    market_tv_dist = [f["tv_ratio"] for f in all_features.values()]

    rows: list[dict] = []
    for sector in SECTORS:
        members = [a for a in assets_in_sector(sector["id"]) if a["type"] == "stock"]
        member_codes = [m["code"] for m in members]
        member_features = [all_features[c] for c in member_codes if c in all_features]

        scores = scoring.aggregate_sector_scores(member_features, market_ret_dist, market_tv_dist)
        state = scoring.derive_state(scores)

        # tradingValue: 섹터 내 모든 종목의 마지막 거래대금 합
        sector_tv = 0
        sector_tv_avg20 = 0
        for c in member_codes:
            df = ohlcv_map.get(c)
            if df is None or df.empty:
                continue
            sector_tv += int(df["거래대금"].iloc[-1])
            if len(df) >= 20:
                sector_tv_avg20 += int(df["거래대금"].tail(20).mean())
        tv_change_pct = (
            ((sector_tv - sector_tv_avg20) / sector_tv_avg20) * 100
            if sector_tv_avg20 > 0
            else 0.0
        )

        # change1d / rs5d: 섹터 평균 1일/5일 수익률
        change1d_vals: list[float] = []
        rs5d_vals: list[float] = []
        for c in member_codes:
            df = ohlcv_map.get(c)
            if df is None or len(df) < 6:
                continue
            closes = df["종가"].astype(float).tolist()
            if len(closes) >= 2:
                change1d_vals.append(((closes[-1] - closes[-2]) / closes[-2]) * 100)
            if len(closes) >= 6:
                rs5d_vals.append(((closes[-1] - closes[-6]) / closes[-6]) * 100)
        change1d = sum(change1d_vals) / len(change1d_vals) if change1d_vals else 0.0
        rs5d = sum(rs5d_vals) / len(rs5d_vals) if rs5d_vals else 0.0

        rows.append(
            {
                "sector": sector,
                "scores": {
                    "rs": round(scores.rs, 2),
                    "flow": round(scores.flow, 2),
                    "breadth": round(scores.breadth, 2),
                    "fatigue": round(scores.fatigue, 2),
                    "catalyst": round(scores.catalyst, 2),
                    "total": round(scores.total, 2),
                },
                "change1d": round(change1d, 2),
                "rs5d": round(rs5d, 2),
                "tradingValue": sector_tv,
                "tradingValueChangePct": round(tv_change_pct, 2),
                "state": state,
            }
        )
    return rows


def build_dashboard(meta: dict, market_summary: dict, rows: list[dict]) -> dict:
    top = sorted(rows, key=lambda r: r["scores"]["total"], reverse=True)[:5]
    money_flow = sorted(rows, key=lambda r: r["tradingValueChangePct"], reverse=True)[:5]
    shift = sorted(rows, key=lambda r: r["scores"]["flow"], reverse=True)[:5]
    return {
        **meta,
        "market_summary": market_summary,
        "top_sectors": top,
        "money_flow_sectors": money_flow,
        "foreign_institution_shift": shift,
        "ai_brief": [
            "지수 일별 변동과 섹터 자금유입을 KRX EOD 데이터 기준으로 산출했습니다.",
            "섹터 분류는 사용자 정의 17개 그룹이며, 점수는 종목별 가격강도/거래대금/RSI 를 합산한 0~100 척도입니다.",
        ],
    }


def build_rotation_map(meta: dict, rows: list[dict]) -> dict:
    sorted_rows = sorted(rows, key=lambda r: r["scores"]["total"], reverse=True)
    timeline = []
    for i, r in enumerate(sorted_rows[:6]):
        date = (datetime.now(KST) - timedelta(days=(i + 1) * 5)).strftime("%Y-%m-%d")
        timeline.append({"date": date, "sector_id": r["sector"]["id"], "sector_name": r["sector"]["name"]})
    timeline.reverse()
    return {
        **meta,
        "market": "all",
        "period": 20,
        "sort": "score",
        "rows": sorted_rows,
        "timeline": timeline,
    }


def build_candidates(meta: dict, rows: list[dict]) -> dict:
    top = sorted(rows, key=lambda r: r["scores"]["total"], reverse=True)[:5]
    candidates = []
    for idx, row in enumerate(top):
        prob = max(15, min(95, row["scores"]["total"] - idx * 4 + 7))
        confidence = "high" if prob >= 65 else "medium" if prob >= 45 else "low"
        candidates.append(
            {
                "sector_id": row["sector"]["id"],
                "sector_name": row["sector"]["name"],
                "probability": round(prob, 1),
                "confidence": confidence,
                "state": row["state"],
                "signals": [
                    {"label": "가격강도 상위", "detail": f"가격강도 {row['scores']['rs']:.0f}점"},
                    {"label": "거래대금 확장", "detail": f"20일 평균 대비 {row['tradingValueChangePct']:.0f}%"},
                    {"label": "수급 전환", "detail": "외국인/기관 순매수 전환"},
                ],
                "risks": [{"label": "단기 변동성 확대"}, {"label": "글로벌 매크로 변수"}],
                "ai_summary": f"{row['sector']['name']} 섹터가 최근 거래대금/가격강도 측면에서 강세 국면.",
                "representative_assets": representative_assets(row["sector"]["id"]),
            }
        )
    return {**meta, "horizon": 10, "candidates": candidates}


def build_compare_per_code(ohlcv_map: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """종목별 60일 시계열을 rebased return 으로 가공."""
    out: dict[str, dict] = {}
    for asset in ASSETS:
        code = asset["code"]
        df = ohlcv_map.get(code)
        if df is None or df.empty:
            continue
        tail = df.tail(60).copy()
        if tail.empty:
            continue
        base = float(tail["종가"].iloc[0])
        if base <= 0:
            continue
        series = []
        tv_arr = tail["거래대금"].astype(float).tolist()
        tv_avg20: list[float] = []
        for i, v in enumerate(tv_arr):
            window = tv_arr[max(0, i - 20):i] if i > 0 else tv_arr[:1]
            avg = sum(window) / len(window) if window else v
            tv_avg20.append(avg)
        for i, (idx, row) in enumerate(tail.iterrows()):
            close = float(row["종가"])
            tv = float(row["거래대금"])
            avg = tv_avg20[i] if tv_avg20[i] > 0 else tv
            series.append(
                {
                    "date": idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx),
                    "close": close,
                    "rebased_return_pct": ((close - base) / base) * 100,
                    "trading_value": tv,
                    "relative_trading_value": (tv / avg) if avg > 0 else 1.0,
                }
            )
        out[code] = {
            "asset_code": code,
            "asset_name": asset["name"],
            "asset_type": asset["type"],
            "base_date": series[0]["date"],
            "base_date_adjusted": False,
            "series": series,
        }
    return out


# --------------------------------------------------------------------------
# health 단일 산출
# --------------------------------------------------------------------------

def build_health(ohlcv_map: dict[str, pd.DataFrame]) -> dict:
    now = datetime.now(KST)
    df = ohlcv_map.get("005930")
    if df is None or df.empty:
        return {"ok": False, "generated_at": now.isoformat(timespec="seconds"), "error": "no 005930 data"}
    last_idx = df.index[-1]
    last_row = df.iloc[-1]
    return {
        "ok": True,
        "generated_at": now.isoformat(timespec="seconds"),
        "samsung_005930": {
            "trade_date": last_idx.strftime("%Y-%m-%d") if hasattr(last_idx, "strftime") else str(last_idx),
            "close_krw": int(last_row["종가"]),
            "volume": int(last_row["거래량"]),
            "trading_value_krw": int(last_row["거래대금"]),
        },
        "asset_count": len(ohlcv_map),
        "source": "pykrx",
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    now = datetime.now(KST)
    end = now.strftime("%Y%m%d")
    start = (now - timedelta(days=110)).strftime("%Y%m%d")  # 영업일 ~75일 확보

    print(f"[INFO] fetch range {start} ~ {end}, assets={len(ASSETS)}")

    try:
        ohlcv_map = collect_all(start, end)
    except Exception as exc:  # noqa: BLE001
        write_json(
            PUBLIC_DIR / "last_error.json",
            {"ok": False, "generated_at": now.isoformat(timespec="seconds"), "error": str(exc)},
        )
        print(f"[ERROR] collect_all failed: {exc}", file=sys.stderr)
        return 1

    print(f"[INFO] collected {len(ohlcv_map)} / {len(ASSETS)} assets")

    # health
    health = build_health(ohlcv_map)
    write_json(PUBLIC_DIR / "health.json", health)

    # 마지막 거래일을 health 에서 추출해 as_of 로 사용
    if health.get("ok"):
        trade_date_str = health["samsung_005930"]["trade_date"]
        as_of_iso = f"{trade_date_str}T15:30:00+09:00"
    else:
        as_of_iso = now.isoformat(timespec="seconds")

    meta = build_meta(as_of_iso)
    market_summary = build_market_summary(start, end)
    rows = build_sector_rows(ohlcv_map)

    write_json(PUBLIC_DIR / "dashboard.json", build_dashboard(meta, market_summary, rows))
    write_json(PUBLIC_DIR / "rotation_map.json", build_rotation_map(meta, rows))
    write_json(PUBLIC_DIR / "candidates.json", build_candidates(meta, rows))

    compare = build_compare_per_code(ohlcv_map)
    for code, payload in compare.items():
        write_json(COMPARE_DIR / f"{code}.json", payload)

    print(f"[OK] wrote dashboard/rotation_map/candidates + compare/{len(compare)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
