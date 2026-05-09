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

try:
    import FinanceDataReader as fdr
except Exception:  # noqa: BLE001
    fdr = None  # 없어도 fetch 자체는 진행 (KOSPI/KOSDAQ 만 0 으로 떨어짐)

import scoring
import correlation as corr_mod

KST = timezone(timedelta(hours=9))
ROOT = Path(__file__).resolve().parent.parent
PUBLIC_DIR = ROOT / "public"
COMPARE_DIR = PUBLIC_DIR / "compare"
DATA_DIR = ROOT / "data"

KOSPI_INDEX = "1001"
KOSDAQ_INDEX = "2001"

# Broad-market ETF tickers used as leader-only nodes in score_history / lead-lag.
# Phase 1 후행/선행 분리 plan — 시장 전체 vs 섹터 분리 detect 용. sector ETF 는 underlying micro 와
# redundant 라 의도적으로 포함 안 함. 일별 변동률(%) 만 저장. follower 로는 사용 안 함.
BROAD_MARKET_ETFS: dict[str, str] = {
    "_market_kodex200": "069500",   # KODEX 200 — KOSPI 200 추종, 가장 큰 거래량
    "_market_tiger200": "102110",   # TIGER 200 — 보조 reference
}


def load_json(path: Path) -> list:
    return json.loads(path.read_text(encoding="utf-8"))


SECTORS: list[dict] = load_json(DATA_DIR / "sectors.json")
ASSETS: list[dict] = load_json(DATA_DIR / "assets.json")

# 섹터 계층 (대/중/소). Cycle Map drill-down + macro/meso 점수 집계용.
try:
    _hier = json.loads((DATA_DIR / "sector_hierarchy.json").read_text(encoding="utf-8"))
    HIERARCHY: dict = _hier
    MICRO_TO_MESO: dict[str, str] = _hier.get("micro_to_meso", {})
    MESO_TO_MACRO: dict[str, str] = {m["id"]: m.get("macro", "") for m in _hier.get("meso", [])}
    MESO_LIST: list[dict] = _hier.get("meso", [])
    MACRO_LIST: list[dict] = _hier.get("macro", [])
except Exception as _exc:  # noqa: BLE001
    print(f"[WARN] sector_hierarchy.json load fail: {_exc}", file=sys.stderr)
    HIERARCHY, MICRO_TO_MESO, MESO_TO_MACRO, MESO_LIST, MACRO_LIST = {}, {}, {}, [], []

# 사용자 큐레이션 '매의 눈' 픽 — representative_assets 우선 후보.
# 일단 OFF. 다시 켜려면 HAWK_EYE_ENABLED = True.
HAWK_EYE_ENABLED = True
try:
    _hawk_raw = json.loads((DATA_DIR / "hawk_eye.json").read_text(encoding="utf-8"))
    _hawk_picks: dict[str, list[str]] = _hawk_raw.get("picks", {})
except Exception as _exc:  # noqa: BLE001
    print(f"[WARN] hawk_eye.json load fail (continuing with empty): {_exc}", file=sys.stderr)
    _hawk_picks = {}
HAWK_EYE: dict[str, list[str]] = _hawk_picks if HAWK_EYE_ENABLED else {}


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
    """KOSPI(1001)/KOSDAQ(2001) 지수 OHLCV.
    - pykrx 의 get_index_ohlcv_by_date 는 일부 버전에서 '지수명' KeyError 로 죽음.
    - 우선 FinanceDataReader 로 시도 (KS11=KOSPI, KQ11=KOSDAQ), 실패시 pykrx 시도.
    """
    fdr_symbol = {"1001": "KS11", "2001": "KQ11"}.get(index_code)
    if fdr is not None and fdr_symbol:
        try:
            start_iso = f"{start[:4]}-{start[4:6]}-{start[6:]}"
            end_iso = f"{end[:4]}-{end[4:6]}-{end[6:]}"
            df = fdr.DataReader(fdr_symbol, start_iso, end_iso)
            if not df.empty:
                # 우리 코드는 '종가' 컬럼명을 기대하므로 매핑.
                if "Close" in df.columns and "종가" not in df.columns:
                    df = df.rename(columns={"Close": "종가"})
                return df
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] fdr index {fdr_symbol} fail: {exc}", file=sys.stderr)

    try:
        return stock.get_index_ohlcv_by_date(start, end, index_code)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] pykrx index {index_code} fail: {exc}", file=sys.stderr)
        return pd.DataFrame()


def verify_asset_names() -> list[str]:
    """assets.json 에 적힌 우리 이름과 KRX 공식 종목명을 대조해서 mismatch 출력.
    - ETF 는 pykrx 의 get_market_ticker_name 으로 조회 안 되므로 skip.
    - 일부 종목 (신상장 등) 도 KRX 응답이 비정상일 수 있는데 진짜 매핑 오류와 구분
      하기 위해 별도 prefix 로 라벨링."""
    mismatches: list[str] = []
    for asset in ASSETS:
        if asset.get("type") == "etf":
            # ETF 는 pykrx get_market_ticker_name 대상 아님.
            continue
        code = asset["code"]
        our_name = asset["name"]
        try:
            krx_name = stock.get_market_ticker_name(code)
        except Exception as exc:  # noqa: BLE001
            mismatches.append(f"{code} [{our_name}] → KRX lookup 실패: {exc}")
            continue
        if not isinstance(krx_name, str) or not krx_name.strip():
            # 신상장/특수 종목은 KRX 응답이 비어 올 수 있음 — info 레벨로만.
            mismatches.append(f"INFO {code} [{our_name}]: KRX 응답 비정상 (신상장/auth 가능성)")
            continue
        norm_a = our_name.replace(" ", "").replace("(주)", "").lower()
        norm_b = krx_name.replace(" ", "").replace("(주)", "").lower()
        if norm_a != norm_b:
            mismatches.append(f"MISMATCH {code}: 우리={our_name!r} / KRX={krx_name!r}")
    return mismatches


def collect_all(start: str, end: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    total = len(ASSETS)
    for i, asset in enumerate(ASSETS):
        # ETF 와 종목 모두 같은 OHLCV API 통과 가능
        code = asset["code"]
        try:
            df = fetch_ohlcv(code, start, end)
            if not df.empty:
                out[code] = df
            time.sleep(0.02)  # KRX 부드럽게 (extended assets 700+ 고려 sleep 단축)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] {code} fetch fail: {exc}", file=sys.stderr)
        if (i + 1) % 100 == 0:
            print(f"[INFO] collect_all progress {i+1}/{total}", file=sys.stderr)
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


def asset_sector_ids(asset: dict) -> list[str]:
    """멀티태그(sectorIds) 우선, 없으면 단일 sectorId 폴백."""
    if isinstance(asset.get("sectorIds"), list):
        return asset["sectorIds"]
    if asset.get("sectorId"):
        return [asset["sectorId"]]
    return []


def assets_in_sector(sector_id: str) -> list[dict]:
    return [a for a in ASSETS if sector_id in asset_sector_ids(a)]


def representative_assets(sector_id: str) -> list[dict]:
    in_sector = assets_in_sector(sector_id)
    etf = next((a for a in in_sector if a["type"] == "etf"), None)

    # 매의 눈 픽이 있으면 그게 우선 (사용자 큐레이션).
    picks = HAWK_EYE.get(sector_id, [])
    if picks:
        by_code = {a["code"]: a for a in in_sector}
        hawk_assets = [by_code[c] for c in picks if c in by_code]
        if hawk_assets:
            return hawk_assets + ([etf] if etf else [])

    # 폴백: 섹터 내 첫 3개 종목 + ETF.
    stocks = [a for a in in_sector if a["type"] == "stock"][:3]
    return stocks + ([etf] if etf else [])


def build_sector_rows(
    ohlcv_map: dict[str, pd.DataFrame],
    lookback: int = 20,
    market_filter: str = "all",
    code_market: dict[str, str] | None = None,
) -> list[dict]:
    """모든 섹터에 대해 lookback 일 윈도우로 features → 섹터 점수 집계.
    market_filter: 'all' | 'kospi' | 'kosdaq' — 그 시장 종목만으로 점수 계산.
    code_market: code → 'KOSPI'/'KOSDAQ' 매핑 (universe 에서 미리 빌드).
    """
    def _allowed(code: str) -> bool:
        if market_filter == "all":
            return True
        if not code_market:
            return True  # 매핑 없으면 필터 무시.
        m = (code_market.get(code) or "").upper()
        return m == market_filter.upper()

    # 시장 전체 분포 (필터 적용된 종목 features) 를 미리 만들어 백분위 계산용으로 사용
    all_features: dict[str, dict] = {}
    for code, df in ohlcv_map.items():
        if not _allowed(code):
            continue
        all_features[code] = scoring.stock_features(df, lookback=lookback)
    market_ret_dist = [f["ret_main"] for f in all_features.values()]
    market_tv_dist = [f["tv_ratio"] for f in all_features.values()]

    rows: list[dict] = []
    for sector in SECTORS:
        members = [a for a in assets_in_sector(sector["id"]) if a["type"] == "stock"]
        # 시장 필터 적용: 해당 시장 종목만 섹터 점수에 사용.
        member_codes = [m["code"] for m in members if _allowed(m["code"])]
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
    weak = sorted(rows, key=lambda r: r["scores"]["total"])[:5]
    money_flow = sorted(rows, key=lambda r: r["tradingValueChangePct"], reverse=True)[:5]
    shift = sorted(rows, key=lambda r: r["scores"]["flow"], reverse=True)[:5]
    return {
        **meta,
        "market_summary": market_summary,
        "top_sectors": top,
        "weak_sectors": weak,
        "money_flow_sectors": money_flow,
        "foreign_institution_shift": shift,
        "ai_brief": [
            "지수 일별 변동과 섹터 자금유입을 KRX EOD 데이터 기준으로 산출했습니다.",
            "섹터 분류는 사용자 정의 17개 그룹이며, 점수는 종목별 가격강도/거래대금/RSI 를 합산한 0~100 척도입니다.",
        ],
    }


def _truncate_to_date(ohlcv_map: dict[str, pd.DataFrame], end_date) -> dict[str, pd.DataFrame]:
    """OHLCV 를 특정 날짜 (포함) 까지로 자른 새 dict 반환."""
    out: dict[str, pd.DataFrame] = {}
    for code, df in ohlcv_map.items():
        if df is None or df.empty:
            continue
        truncated = df.loc[df.index <= end_date]
        if not truncated.empty:
            out[code] = truncated
    return out


def build_rotation_timeline(ohlcv_map: dict[str, pd.DataFrame], lookback: int = 20,
                            points: int = 6, gap_days: int = 5) -> list[dict]:
    """과거 5일/10일/15일... 시점에 실제 1위/꼴찌 섹터를 계산해 추이로 반환.
    - points: 데이터 포인트 개수 (기본 6 = 약 30일치)
    - gap_days: 포인트 사이 간격 (영업일 기준 5일)
    """
    if not ohlcv_map:
        return []

    # 가장 최근 거래일 인덱스 (OHLCV 의 max date)
    all_dates = sorted({d for df in ohlcv_map.values() for d in df.index})
    if len(all_dates) < lookback + gap_days:
        return []

    timeline: list[dict] = []
    for i in range(points):
        offset_from_end = i * gap_days
        end_idx = len(all_dates) - 1 - offset_from_end
        if end_idx < lookback:
            break
        anchor = all_dates[end_idx]
        truncated = _truncate_to_date(ohlcv_map, anchor)
        rows = build_sector_rows(truncated, lookback=lookback)
        if not rows:
            continue
        top = max(rows, key=lambda r: r["scores"]["total"])
        bottom = min(rows, key=lambda r: r["scores"]["total"])
        timeline.append({
            "date": anchor.strftime("%Y-%m-%d") if hasattr(anchor, "strftime") else str(anchor)[:10],
            "leading": {"sector_id": top["sector"]["id"], "sector_name": top["sector"]["name"]},
            "lagging": {"sector_id": bottom["sector"]["id"], "sector_name": bottom["sector"]["name"]},
        })

    timeline.reverse()  # 오래된 → 최신 순
    return timeline


def aggregate_to_meso(micro_rows: list[dict]) -> list[dict]:
    """micro 점수를 meso 로 weighted average (가중치: tradingValue). MESO_LIST 순서 유지."""
    by_meso: dict[str, list[dict]] = {}
    for r in micro_rows:
        meso = MICRO_TO_MESO.get(r["sector"]["id"])
        if not meso:
            continue
        by_meso.setdefault(meso, []).append(r)

    meso_meta_by_id = {m["id"]: m for m in MESO_LIST}
    out: list[dict] = []
    for meso_id, children in by_meso.items():
        meta = meso_meta_by_id.get(meso_id, {"id": meso_id, "name": meso_id, "shortName": meso_id})
        # 가중치 = tradingValue. 0 이면 동일가중.
        total_w = sum(c.get("tradingValue", 0) for c in children) or len(children)
        def wavg(field: str) -> float:
            if total_w == 0:
                return 0.0
            return sum(c["scores"][field] * (c.get("tradingValue") or 1) for c in children) / total_w
        scores = {f: round(wavg(f), 2) for f in ("rs", "flow", "breadth", "fatigue", "catalyst", "total")}
        sector_tv = sum(c.get("tradingValue", 0) for c in children)
        sector_tv_chg = sum(c.get("tradingValueChangePct", 0) * (c.get("tradingValue") or 1) for c in children) / total_w
        change1d = sum(c.get("change1d", 0) * (c.get("tradingValue") or 1) for c in children) / total_w
        rs5d = sum(c.get("rs5d", 0) * (c.get("tradingValue") or 1) for c in children) / total_w
        # state 는 hottest child 의 state.
        state = max(children, key=lambda c: c["scores"]["total"]).get("state", "neutral")
        out.append({
            "sector": {"id": meta["id"], "name": meta.get("name", meta["id"]), "shortName": meta.get("shortName", meta["id"])},
            "scores": scores,
            "change1d": round(change1d, 2),
            "rs5d": round(rs5d, 2),
            "tradingValue": int(sector_tv),
            "tradingValueChangePct": round(sector_tv_chg, 2),
            "state": state,
            "macro": meta.get("macro"),
            "child_micro_ids": [c["sector"]["id"] for c in children],
        })
    return out


def aggregate_to_macro(meso_rows: list[dict]) -> list[dict]:
    """meso → macro weighted average."""
    by_macro: dict[str, list[dict]] = {}
    for r in meso_rows:
        macro = r.get("macro") or MESO_TO_MACRO.get(r["sector"]["id"])
        if not macro:
            continue
        by_macro.setdefault(macro, []).append(r)

    macro_meta_by_id = {m["id"]: m for m in MACRO_LIST}
    out: list[dict] = []
    for macro_id, children in by_macro.items():
        meta = macro_meta_by_id.get(macro_id, {"id": macro_id, "name": macro_id, "shortName": macro_id})
        total_w = sum(c.get("tradingValue", 0) for c in children) or len(children)
        def wavg(field: str) -> float:
            if total_w == 0:
                return 0.0
            return sum(c["scores"][field] * (c.get("tradingValue") or 1) for c in children) / total_w
        scores = {f: round(wavg(f), 2) for f in ("rs", "flow", "breadth", "fatigue", "catalyst", "total")}
        sector_tv = sum(c.get("tradingValue", 0) for c in children)
        sector_tv_chg = sum(c.get("tradingValueChangePct", 0) * (c.get("tradingValue") or 1) for c in children) / total_w
        change1d = sum(c.get("change1d", 0) * (c.get("tradingValue") or 1) for c in children) / total_w
        rs5d = sum(c.get("rs5d", 0) * (c.get("tradingValue") or 1) for c in children) / total_w
        state = max(children, key=lambda c: c["scores"]["total"]).get("state", "neutral")
        out.append({
            "sector": {"id": meta["id"], "name": meta.get("name", meta["id"]), "shortName": meta.get("shortName", meta["id"])},
            "scores": scores,
            "change1d": round(change1d, 2),
            "rs5d": round(rs5d, 2),
            "tradingValue": int(sector_tv),
            "tradingValueChangePct": round(sector_tv_chg, 2),
            "state": state,
            "child_meso_ids": [c["sector"]["id"] for c in children],
        })
    return out


def build_rotation_map(meta: dict,
                       rows_by_period: dict[int, list[dict]],
                       rows_by_period_kospi: dict[int, list[dict]],
                       rows_by_period_kosdaq: dict[int, list[dict]],
                       timeline: list[dict],
                       meso_rows_by_period: dict[int, list[dict]] | None = None,
                       macro_rows_by_period: dict[int, list[dict]] | None = None) -> dict:
    """기간별 + 시장별 점수 + 진짜 timeline + 계층(meso/macro) 점수. rows 는 default(전체+20일+소분류) 호환성 유지."""
    default_period = 20
    default_rows = sorted(
        rows_by_period.get(default_period, []),
        key=lambda r: r["scores"]["total"],
        reverse=True,
    )

    def _bundle(by_period: dict[int, list[dict]]) -> dict[str, list[dict]]:
        return {
            str(p): sorted(rs, key=lambda r: r["scores"]["total"], reverse=True)
            for p, rs in by_period.items()
        }

    out = {
        **meta,
        "market": "all",
        "period": default_period,
        "sort": "score",
        "rows": default_rows,
        "rows_by_period": _bundle(rows_by_period),
        "rows_by_period_kospi": _bundle(rows_by_period_kospi),
        "rows_by_period_kosdaq": _bundle(rows_by_period_kosdaq),
        "timeline": timeline,
    }
    if meso_rows_by_period:
        out["meso_rows_by_period"] = _bundle(meso_rows_by_period)
    if macro_rows_by_period:
        out["macro_rows_by_period"] = _bundle(macro_rows_by_period)
    if HIERARCHY:
        out["hierarchy"] = HIERARCHY
    return out


def _candidates_for_rows(rows: list[dict], lookback: int) -> list[dict]:
    top = sorted(rows, key=lambda r: r["scores"]["total"], reverse=True)[:5]
    out = []
    for idx, row in enumerate(top):
        prob = max(15, min(95, row["scores"]["total"] - idx * 4 + 7))
        confidence = "high" if prob >= 65 else "medium" if prob >= 45 else "low"
        out.append(
            {
                "sector_id": row["sector"]["id"],
                "sector_name": row["sector"]["name"],
                "probability": round(prob, 1),
                "confidence": confidence,
                "state": row["state"],
                "signals": [
                    {"label": "가격강도 상위", "detail": f"가격강도 {row['scores']['rs']:.0f}점"},
                    {"label": "거래대금 확장", "detail": f"{lookback}일 평균 대비 {row['tradingValueChangePct']:.0f}%"},
                    {"label": "수급 전환", "detail": "외국인/기관 순매수 전환"},
                ],
                "risks": [{"label": "단기 변동성 확대"}, {"label": "글로벌 매크로 변수"}],
                "ai_summary": f"{row['sector']['name']} 섹터가 최근 거래대금/가격강도 측면에서 강세 국면.",
                "representative_assets": representative_assets(row["sector"]["id"]),
            }
        )
    return out


def build_candidates(meta: dict, rows: list[dict],
                     rows_by_period: dict[int, list[dict]] | None = None) -> dict:
    candidates = _candidates_for_rows(rows, lookback=20)
    out: dict = {**meta, "horizon": 10, "candidates": candidates}
    if rows_by_period:
        out["candidates_by_period"] = {
            str(p): _candidates_for_rows(rs, lookback=p)
            for p, rs in rows_by_period.items()
        }
    return out


def build_universe() -> dict:
    """KRX 전종목 (KOSPI + KOSDAQ) 코드/이름/시장 한 번에 빌드.
    FinanceDataReader.StockListing('KRX') 가 단일 호출로 ~2,500 행을 반환.
    앱의 종목 검색 / 태깅 UI 가 이 파일을 가지고 검색 인덱스 만든다.
    """
    if fdr is None:
        return {"as_of": "", "count": 0, "tickers": [], "error": "fdr unavailable"}

    try:
        df = fdr.StockListing("KRX")
    except Exception as exc:  # noqa: BLE001
        return {"as_of": "", "count": 0, "tickers": [], "error": f"StockListing fail: {exc}"}

    # 컬럼명은 fdr 버전에 따라 'Code'/'Symbol' 등 변할 수 있어 정규화.
    code_col = next((c for c in ["Code", "Symbol", "code"] if c in df.columns), None)
    name_col = next((c for c in ["Name", "name"] if c in df.columns), None)
    market_col = next((c for c in ["Market", "market"] if c in df.columns), None)
    sector_col = next((c for c in ["Sector", "sector"] if c in df.columns), None)
    industry_col = next((c for c in ["Industry", "industry"] if c in df.columns), None)
    marcap_col = next((c for c in ["Marcap", "marcap", "MarketCap", "market_cap"] if c in df.columns), None)
    close_col_uni = next((c for c in ["Close", "close", "종가"] if c in df.columns), None)
    volume_col_uni = next((c for c in ["Volume", "거래량"] if c in df.columns), None)
    if not code_col or not name_col:
        return {
            "as_of": "",
            "count": 0,
            "tickers": [],
            "error": f"unexpected columns: {list(df.columns)}",
        }

    def _clean(v) -> str:
        if v is None:
            return ""
        s = str(v).strip()
        return "" if s.lower() in ("nan", "none") else s

    tickers = []
    seen_codes: set[str] = set()
    marcap_map: dict[str, int] = {}
    tv_map: dict[str, int] = {}  # trading_value 근사 (Close × Volume)
    for _, row in df.iterrows():
        code = _clean(row[code_col])
        name = _clean(row[name_col])
        market = _clean(row[market_col]) if market_col else ""
        sector = _clean(row[sector_col]) if sector_col else ""
        industry = _clean(row[industry_col]) if industry_col else ""
        if not code or not name:
            continue
        # 우선주 / SPAC / ETN 등은 일단 포함. 향후 필터링 옵션 추가.
        entry: dict = {"code": code, "name": name, "market": market, "type": "stock"}
        if sector:
            entry["sector"] = sector
        if industry:
            entry["industry"] = industry
        # 시가총액 + 거래대금 추정 (build_extended_assets 가 cutoff 에 사용).
        if marcap_col:
            try:
                mc = row[marcap_col]
                if pd.notna(mc):
                    mc_int = int(float(mc))
                    if mc_int > 0:
                        marcap_map[code] = mc_int
                        entry["marcap"] = mc_int
            except Exception:  # noqa: BLE001
                pass
        if close_col_uni and volume_col_uni:
            try:
                cv, vv = row[close_col_uni], row[volume_col_uni]
                if pd.notna(cv) and pd.notna(vv):
                    tv = int(float(cv) * float(vv))
                    if tv > 0:
                        tv_map[code] = tv
            except Exception:  # noqa: BLE001
                pass
        tickers.append(entry)
        seen_codes.add(code)
    # 모듈 글로벌에 임시 보관 (build_extended_assets 가 사용).
    globals()['_LAST_MARCAP_MAP'] = marcap_map
    globals()['_LAST_TV_MAP'] = tv_map

    # ETF 추가 (KRX ETF 시장 ~900개). pykrx 의 get_etf_ticker_list / get_etf_ticker_name 사용.
    etf_added = 0
    try:
        # 최근 거래일 기준 — fetch.py 메인이 health 의 trade_date 를 알지만 build_universe 는 독립.
        # pykrx 가 inactive date 면 빈 리스트 줄 수 있어 weekday loop 으로 fallback.
        from datetime import timedelta as _td
        for back in range(0, 10):
            d = datetime.now(KST).date() - _td(days=back)
            ymd = d.strftime("%Y%m%d")
            try:
                etf_codes = stock.get_etf_ticker_list(ymd)
            except Exception:
                etf_codes = []
            if etf_codes:
                for code in etf_codes:
                    code = str(code).strip()
                    if not code or code in seen_codes:
                        continue
                    try:
                        name = stock.get_etf_ticker_name(code)
                    except Exception:
                        name = ""
                    if not name:
                        continue
                    tickers.append({"code": code, "name": name, "market": "ETF", "type": "etf"})
                    seen_codes.add(code)
                    etf_added += 1
                break
        print(f"[INFO] universe ETF added: {etf_added}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] universe ETF enrichment fail: {exc}", file=sys.stderr)

    now = datetime.now(KST)
    return {
        "as_of": now.isoformat(timespec="seconds"),
        "count": len(tickers),
        "tickers": tickers,
    }


def build_extended_assets(
    universe: dict,
    curated: list[dict],
    hawk_eye_codes: set[str],
    marcap_cutoff: int = 500_000_000_000,  # 5,000억
    tv_cutoff: int = 5_000_000_000,         # 50억 (단일일 추정)
) -> list[dict]:
    """분석 모집단을 시총+거래대금 cutoff 로 자동 확장.
    - cutoff: marcap (FDR Marcap) AND tv (FDR Close * Volume 추정)
    - cutoff 통과 + classify_stock 으로 sector 분류 (etc/빈값 제외) 종목 추가
    - 큐레이션 ASSETS + 매의 눈 픽은 cutoff 무시하고 강제 포함
    - 같은 코드는 큐레이션 sectorIds 우선 (자동 분류 sector 는 보조)
    """
    # late import — classify_sectors 가 같은 dir 에 있음.
    try:
        from classify_sectors import classify_stock
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] classify_sectors import fail: {exc} — 자동 확장 skip", file=sys.stderr)
        return list(curated)

    marcap_map: dict[str, int] = globals().get('_LAST_MARCAP_MAP') or {}
    tv_map: dict[str, int] = globals().get('_LAST_TV_MAP') or {}

    # by_code 에 큐레이션 + 매의 눈 강제 포함부터 채움.
    by_code: dict[str, dict] = {}
    for a in curated:
        c = a.get("code")
        if c:
            by_code[c] = dict(a)
            by_code[c].setdefault("type", "stock")

    # universe stocks 순회.
    debug_count = {"candidates": 0, "passed": 0, "classified": 0}
    for t in universe.get("tickers", []):
        if t.get("type") == "etf":
            continue
        code = t.get("code")
        name = t.get("name", "")
        if not code:
            continue
        # 강제 포함 (이미 들어가 있어도 sectorIds 결손이면 채움)
        is_forced = code in by_code or code in hawk_eye_codes
        if not is_forced:
            mc = marcap_map.get(code, 0) or t.get("marcap", 0)
            tv = tv_map.get(code, 0)
            debug_count["candidates"] += 1
            if mc < marcap_cutoff or tv < tv_cutoff:
                continue
            debug_count["passed"] += 1
            sector_id, _conf, _kw = classify_stock(name)
            if not sector_id or sector_id == "etc":
                continue
            debug_count["classified"] += 1
            by_code[code] = {
                "code": code,
                "name": name,
                "type": "stock",
                "sectorIds": [sector_id],
                "auto": True,
            }
        else:
            # 강제 포함 → 큐레이션 sectorIds 없으면 auto 채워줌.
            entry = by_code.get(code)
            if entry and not entry.get("sectorIds") and not entry.get("sectorId"):
                sector_id, _c, _k = classify_stock(name)
                if sector_id and sector_id != "etc":
                    entry["sectorIds"] = [sector_id]
                    entry["auto"] = True

    print(f"[INFO] extended_assets: candidates={debug_count['candidates']}, passed cutoff={debug_count['passed']}, "
          f"classified={debug_count['classified']}, total={len(by_code)}", file=sys.stderr)

    extended = list(by_code.values())
    # 디버그 파일.
    try:
        write_json(PUBLIC_DIR / "assets_extended.json", {
            "as_of": datetime.now(KST).isoformat(timespec="seconds"),
            "count": len(extended),
            "cutoff": {"marcap_won": marcap_cutoff, "trading_value_won": tv_cutoff},
            "debug": debug_count,
            "assets": extended,
        })
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] write assets_extended.json fail: {exc}", file=sys.stderr)

    return extended


def build_quotes(ohlcv_map: dict[str, pd.DataFrame], trade_date_str: str) -> dict:
    """KRX 전종목 (KOSPI+KOSDAQ) 마지막 종가/등락률/거래대금을 한 파일에 모음.
    pykrx 의 get_market_ohlcv(date, market=...) 배치 API 1콜씩이라 빠름.
    실패 시 큐레이션 ohlcv_map 으로 폴백."""
    quotes: dict[str, dict] = {}
    debug: dict = {"trade_date": trade_date_str, "batch_attempts": []}

    # 입력 trade_date_str 은 "YYYY-MM-DD" → pykrx 는 "YYYYMMDD" 받음.
    yyyymmdd = trade_date_str.replace("-", "") if trade_date_str else ""

    batch_fail_reasons: list[str] = []

    # 우선순위 1: fdr.StockListing("KRX") 한 방으로 전종목 종가+등락률.
    # pykrx batch 가 KRX 응답 스키마 변경에 취약해서 unstable → fdr 가 더 안정적.
    if fdr is not None:
        try:
            fdr_df = fdr.StockListing("KRX")
            close_col = next((c for c in ["Close", "close", "종가"] if c in fdr_df.columns), None)
            pct_col = next((c for c in ["ChagesRatio", "ChangesRatio", "ChangeRatio", "등락률", "Change"] if c in fdr_df.columns), None)
            code_col = next((c for c in ["Code", "Symbol", "code"] if c in fdr_df.columns), None)
            vol_col = next((c for c in ["Volume", "거래량"] if c in fdr_df.columns), None)
            mcap_col = next((c for c in ["Marcap", "시가총액"] if c in fdr_df.columns), None)
            debug["fdr_cols"] = {"close": close_col, "pct": pct_col, "code": code_col, "vol": vol_col, "mcap": mcap_col}
            if code_col and close_col:
                added = 0
                for _, row in fdr_df.iterrows():
                    try:
                        code = str(row[code_col]).strip()
                        if not code:
                            continue
                        close_v = row[close_col]
                        if close_v is None or pd.isna(close_v):
                            continue
                        close = int(float(close_v))
                        if close <= 0:
                            continue
                        pct = float(row[pct_col]) if pct_col and not pd.isna(row[pct_col]) else 0.0
                        # ChagesRatio 가 보통 % (예: 5.4) 로 옴 — 그대로 사용.
                        # 거래대금 = close × volume (fdr 는 보통 거래대금 컬럼 없음).
                        vol = int(row[vol_col]) if vol_col and not pd.isna(row[vol_col]) else 0
                        tv = close * vol
                        quotes[code] = {
                            "close": close,
                            "change_pct": round(pct, 2),
                            "trade_date": trade_date_str,
                            "trading_value": tv,
                        }
                        added += 1
                    except Exception:  # noqa: BLE001
                        continue
                debug["fdr_added"] = added
                print(f"[INFO] fdr StockListing batch: {added} quotes", file=sys.stderr)
            else:
                batch_fail_reasons.append(f"fdr cols missing: {list(fdr_df.columns)[:10]}")
        except Exception as exc:  # noqa: BLE001
            batch_fail_reasons.append(f"fdr StockListing fail: {exc}")
            debug["fdr_error"] = repr(exc)

    # 우선순위 2: pykrx batch (보조 — fdr 가 누락한 종목 메우기).
    if yyyymmdd:
        # pykrx 버전/함수명 확인용 — 디버그 파일에 기록.
        debug["pykrx_funcs_present"] = {
            n: hasattr(stock, n) for n in (
                "get_market_ohlcv", "get_market_ohlcv_by_ticker",
                "get_market_ohlcv_by_date", "get_etf_ohlcv_by_ticker",
            )
        }
        for market in ("KOSPI", "KOSDAQ"):
            df = None
            attempt = {"market": market, "tries": []}
            # 1차: 명시적 by_ticker (pykrx 1.0.45+ 표준 batch).
            try:
                df = stock.get_market_ohlcv_by_ticker(yyyymmdd, market=market)
                attempt["tries"].append({"fn": "by_ticker", "ok": True, "rows": int(len(df)) if df is not None else 0, "cols": list(df.columns) if df is not None and not df.empty else []})
            except Exception as exc:  # noqa: BLE001
                attempt["tries"].append({"fn": "by_ticker", "ok": False, "err": repr(exc)})
                batch_fail_reasons.append(f"by_ticker {market}: {exc}")
                # 2차: 래퍼 시도 (구버전 호환).
                try:
                    df = stock.get_market_ohlcv(yyyymmdd, market=market)
                    attempt["tries"].append({"fn": "wrapper", "ok": True, "rows": int(len(df)) if df is not None else 0})
                except Exception as exc2:  # noqa: BLE001
                    attempt["tries"].append({"fn": "wrapper", "ok": False, "err": repr(exc2)})
                    batch_fail_reasons.append(f"wrapper {market}: {exc2}")
                    debug["batch_attempts"].append(attempt)
                    continue
            debug["batch_attempts"].append(attempt)
            if df is None or df.empty:
                batch_fail_reasons.append(f"{market}: empty df")
                continue
            print(f"[INFO] batch {market} {yyyymmdd}: {len(df)} rows", file=sys.stderr)
            for code, row in df.iterrows():
                try:
                    close = int(row["종가"])
                    if close <= 0:
                        continue
                    pct = float(row["등락률"]) if "등락률" in df.columns else 0.0
                    tv = int(row["거래대금"]) if "거래대금" in df.columns else 0
                    quotes[str(code)] = {
                        "close": close,
                        "change_pct": round(pct, 2),
                        "trade_date": trade_date_str,
                        "trading_value": tv,
                    }
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] quotes row {code} skip: {exc}", file=sys.stderr)
    if batch_fail_reasons:
        print(f"[WARN] batch quote fails: {'; '.join(batch_fail_reasons[:5])}", file=sys.stderr)

    # ETF 는 pykrx 의 별도 API. 직전 빌드에서 12개만 잡혔던 원인 — 단일 trade_date_str 가
    # ETF 응답에 비어 있는 경우. 며칠 거슬러 올라가며 재시도.
    if yyyymmdd:
        from datetime import timedelta as _td
        etf_df = None
        for back in range(0, 7):
            d = datetime.strptime(yyyymmdd, "%Y%m%d") - _td(days=back)
            try_ymd = d.strftime("%Y%m%d")
            try:
                cand = stock.get_etf_ohlcv_by_ticker(try_ymd)
            except Exception as exc:  # noqa: BLE001
                cand = None
                if back == 0:
                    print(f"[WARN] get_etf_ohlcv_by_ticker {try_ymd} fail: {exc}", file=sys.stderr)
            if cand is not None and not cand.empty and len(cand) > 50:
                etf_df = cand
                debug["etf_trade_date"] = try_ymd
                break
        debug["etf_count"] = int(len(etf_df)) if etf_df is not None else 0
        if etf_df is not None and not etf_df.empty:
            print(f"[INFO] ETF batch {yyyymmdd}: {len(etf_df)} rows", file=sys.stderr)
            for code, row in etf_df.iterrows():
                try:
                    close = int(row["종가"])
                    if close <= 0:
                        continue
                    pct = float(row["등락률"]) if "등락률" in etf_df.columns else 0.0
                    tv = int(row["거래대금"]) if "거래대금" in etf_df.columns else 0
                    quotes[str(code)] = {
                        "close": close,
                        "change_pct": round(pct, 2),
                        "trade_date": trade_date_str,
                        "trading_value": tv,
                    }
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] etf row {code} skip: {exc}", file=sys.stderr)

    # 큐레이션 종목이 배치에서 누락된 경우(신상장/거래정지 등) 폴백.
    for code, df in ohlcv_map.items():
        if code in quotes or df is None or df.empty:
            continue
        last = df.iloc[-1]
        prev_close = float(df.iloc[-2]["종가"]) if len(df) >= 2 else float(last["종가"])
        close = int(last["종가"])
        change_pct = ((close - prev_close) / prev_close) * 100 if prev_close > 0 else 0.0
        idx = df.index[-1]
        td = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
        quotes[code] = {
            "close": close,
            "change_pct": round(change_pct, 2),
            "trade_date": td,
            "trading_value": int(last["거래대금"]) if "거래대금" in df.columns else 0,
        }

    debug["quotes_count"] = len(quotes)
    debug["fail_reasons"] = batch_fail_reasons
    try:
        write_json(PUBLIC_DIR / "quotes_debug.json", debug)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] write quotes_debug fail: {exc}", file=sys.stderr)

    return {
        "as_of": f"{trade_date_str}T15:30:00+09:00" if trade_date_str else "",
        "source": "pykrx",
        "quotes": quotes,
    }


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
# Phase 1 백엔드 — 시계열 누적 + 상관/lead-lag 기반 next_cycle
# (후행 ranker = rotation_map.json 그대로 유지. 이 블록은 선행 분석용 신규.)
# --------------------------------------------------------------------------

def update_score_history(
    rows: list[dict],
    quotes_data: dict,
    trade_date_str: str,
    rolling_window: int = 120,
) -> None:
    """오늘의 micro sector total + broad market ETF 변동률을 score_history.json 에 append.
    동일 trade_date 가 이미 있으면 교체 (cron 재실행 안전).
    """
    if not trade_date_str:
        print("[WARN] update_score_history: empty trade_date_str — skip", file=sys.stderr)
        return
    path = PUBLIC_DIR / "score_history.json"
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        existing = {"history": []}

    scores: dict[str, float] = {}
    for r in rows:
        sid = r.get("sector", {}).get("id")
        if sid:
            scores[sid] = r["scores"]["total"]

    quotes = (quotes_data or {}).get("quotes", {})
    for label, ticker in BROAD_MARKET_ETFS.items():
        q = quotes.get(ticker) or {}
        chg = q.get("change_pct")
        if chg is not None:
            scores[label] = float(chg)

    today_entry = {"date": trade_date_str, "scores": scores}
    history = [e for e in existing.get("history", []) if e.get("date") != trade_date_str]
    history.insert(0, today_entry)
    history = history[:rolling_window]

    write_json(path, {
        "as_of": f"{trade_date_str}T15:30:00+09:00",
        "window_days": rolling_window,
        "history": history,
    })


def build_next_cycle(
    rows: list[dict],
    lead_lag: dict,
    correlation: dict,
    top_anchors: int = 3,
) -> dict:
    """오늘 강세 micro top N 을 anchor 로 잡고 lead-lag/co-movement 패턴 매칭.
    통계 표본 30일 미만이면 anchor_sectors=[] + 진행 안내.
    """
    n_obs = correlation.get("n_observations", 0)
    progress_note = f"통계 누적 중 ({n_obs}/60거래일). 60일 채워지면 의미 있는 lead-lag 신호 산출."

    # broad market 노드는 anchor 후보에서 제외
    micro_rows = [
        r for r in rows
        if not (r.get("sector", {}).get("id") or "").startswith("_market")
    ]
    sorted_rows = sorted(
        micro_rows,
        key=lambda r: r.get("scores", {}).get("total", 0),
        reverse=True,
    )

    if n_obs < 30 or not lead_lag.get("leaders"):
        return {"anchor_sectors": [], "n_observations": n_obs, "note": progress_note}

    sector_name_by_id = {s["id"]: s.get("name", s["id"]) for s in SECTORS}

    def _enrich(arr: list[dict]) -> list[dict]:
        for f in arr:
            f["sector_name"] = sector_name_by_id.get(f["sector_id"], f["sector_id"])
        return arr

    out_anchors: list[dict] = []
    for row in sorted_rows[:top_anchors]:
        sid = row["sector"]["id"]
        sname = row["sector"].get("name", sid)
        meso_id = MICRO_TO_MESO.get(sid)
        macro_id = MESO_TO_MACRO.get(meso_id, "") if meso_id else ""

        out_anchors.append({
            "sector_id": sid,
            "sector_name": sname,
            "today_score": row["scores"]["total"],
            "meso": meso_id,
            "macro": macro_id,
            "predictions": {
                "co_movement": _enrich(corr_mod.co_movers(correlation, sid, threshold=0.6, top=3)),
                "lag_5d": _enrich(corr_mod.followers_at_lag(lead_lag, sid, target_lag=5, top=3)),
                "lag_10d": _enrich(corr_mod.followers_at_lag(lead_lag, sid, target_lag=10, top=3)),
            },
            "confidence_note": f"통계 표본 {n_obs}일. 상관성 ≠ 인과성. 투자 권유 아님.",
        })

    return {"anchor_sectors": out_anchors, "n_observations": n_obs}


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

    print(f"[INFO] fetch range {start} ~ {end}, curated assets={len(ASSETS)}")

    # 매핑 검증 (KRX 공식명 vs 우리 매핑) — fetch 전에 출력해서 잘못된 코드 잡기.
    # GA 환경에서 KRX 로그인 못 하면 죽을 수 있으니 전체를 try/except 로 격리.
    try:
        audit = verify_asset_names()
        write_json(
            PUBLIC_DIR / "asset_name_audit.json",
            {"generated_at": now.isoformat(timespec="seconds"), "mismatches": audit},
        )
        if audit:
            print(f"[AUDIT] {len(audit)} mismatch(es):", file=sys.stderr)
            for line in audit[:20]:
                print(f"  {line}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] audit step failed (continuing): {exc}", file=sys.stderr)

    # === 1) Universe 먼저 호출. marcap/tv 맵을 globals 에 저장. ===
    universe = build_universe()
    write_json(PUBLIC_DIR / "universe.json", universe)
    print(f"[INFO] universe: {universe.get('count', 0)} tickers")
    code_market: dict[str, str] = {}
    for t in universe.get("tickers", []):
        code_market[str(t.get("code") or "")] = str(t.get("market") or "")

    # === 2) 분석 모집단 자동 확장 (시총+거래대금 cutoff). ===
    curated_assets = list(ASSETS)
    hawk_eye_codes: set[str] = set()
    for picks in HAWK_EYE.values():
        for c in picks:
            hawk_eye_codes.add(c)
    extended_assets = build_extended_assets(universe, curated_assets, hawk_eye_codes)
    # 글로벌 ASSETS 교체 → assets_in_sector / representative_assets / build_compare_per_code 자동 적용.
    globals()['ASSETS'] = extended_assets
    print(f"[INFO] ASSETS replaced: curated {len(curated_assets)} → extended {len(extended_assets)}")

    # === 3) collect_all (extended ASSETS 기반 — 여러 종목 OHLCV) ===
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

    # 기간별 + 시장별 점수 (소분류 49개 micro 기준).
    def _rows_for(market: str) -> dict[int, list[dict]]:
        return {lb: build_sector_rows(ohlcv_map, lookback=lb,
                                       market_filter=market, code_market=code_market)
                for lb in (5, 20, 60)}

    rows_by_period = _rows_for("all")
    rows_by_period_kospi = _rows_for("kospi")
    rows_by_period_kosdaq = _rows_for("kosdaq")
    rows = rows_by_period[20]
    timeline = build_rotation_timeline(ohlcv_map, lookback=20, points=6, gap_days=5)

    # 계층 집계 (meso/macro) per period.
    meso_rows_by_period: dict[int, list[dict]] = {}
    macro_rows_by_period: dict[int, list[dict]] = {}
    if HIERARCHY:
        for lb, micro_rows in rows_by_period.items():
            meso = aggregate_to_meso(micro_rows)
            meso_rows_by_period[lb] = meso
            macro_rows_by_period[lb] = aggregate_to_macro(meso)

    write_json(PUBLIC_DIR / "dashboard.json", build_dashboard(meta, market_summary, rows))
    write_json(PUBLIC_DIR / "rotation_map.json",
               build_rotation_map(meta, rows_by_period, rows_by_period_kospi,
                                  rows_by_period_kosdaq, timeline,
                                  meso_rows_by_period=meso_rows_by_period or None,
                                  macro_rows_by_period=macro_rows_by_period or None))
    # candidates_by_period 도 만들어 NextCycle 에서 period 선택 가능하도록.
    write_json(PUBLIC_DIR / "candidates.json",
               build_candidates(meta, rows, rows_by_period=rows_by_period))
    # quotes 는 KRX 전종목 — health 의 trade_date 기준으로 pykrx 배치.
    quotes_trade_date = health["samsung_005930"]["trade_date"] if health.get("ok") else ""
    quotes_data = build_quotes(ohlcv_map, quotes_trade_date)
    write_json(PUBLIC_DIR / "quotes.json", quotes_data)

    # === Phase 1 백엔드 — 시계열 누적 + 상관/lead-lag + next_cycle ===
    try:
        update_score_history(rows, quotes_data, quotes_trade_date)
        hist_data = json.loads((PUBLIC_DIR / "score_history.json").read_text(encoding="utf-8"))
        hist = hist_data.get("history", [])

        corr_matrix = corr_mod.compute_correlation_matrix(hist, window=60)
        write_json(PUBLIC_DIR / "correlation_matrix.json", {**meta, **corr_matrix})

        ll_matrix = corr_mod.compute_lead_lag_matrix(hist, window=60)
        write_json(PUBLIC_DIR / "lead_lag_matrix.json", {**meta, **ll_matrix})

        nc_payload = build_next_cycle(rows, ll_matrix, corr_matrix, top_anchors=3)
        write_json(PUBLIC_DIR / "next_cycle.json", {**meta, **nc_payload})

        print(
            f"[OK] phase1 backend: history={len(hist)}, corr_n={corr_matrix.get('n_observations', 0)}, "
            f"ll_leaders={len(ll_matrix.get('leaders', {}))}, anchors={len(nc_payload.get('anchor_sectors', []))}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] phase1 backend pipeline failed: {exc}", file=sys.stderr)

    compare = build_compare_per_code(ohlcv_map)
    for code, payload in compare.items():
        write_json(COMPARE_DIR / f"{code}.json", payload)

    print(f"[OK] wrote dashboard/rotation_map/candidates/quotes + compare/{len(compare)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
