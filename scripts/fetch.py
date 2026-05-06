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


def build_rotation_map(meta: dict,
                       rows_by_period: dict[int, list[dict]],
                       rows_by_period_kospi: dict[int, list[dict]],
                       rows_by_period_kosdaq: dict[int, list[dict]],
                       timeline: list[dict]) -> dict:
    """기간별 + 시장별 점수 + 진짜 timeline. rows 는 default(전체+20일) 그대로 둠 (호환성)."""
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

    return {
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
    for _, row in df.iterrows():
        code = _clean(row[code_col])
        name = _clean(row[name_col])
        market = _clean(row[market_col]) if market_col else ""
        sector = _clean(row[sector_col]) if sector_col else ""
        industry = _clean(row[industry_col]) if industry_col else ""
        if not code or not name:
            continue
        # 우선주 / SPAC / ETN 등은 일단 포함. 향후 필터링 옵션 추가.
        entry = {"code": code, "name": name, "market": market}
        if sector:
            entry["sector"] = sector
        if industry:
            entry["industry"] = industry
        tickers.append(entry)

    now = datetime.now(KST)
    return {
        "as_of": now.isoformat(timespec="seconds"),
        "count": len(tickers),
        "tickers": tickers,
    }


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

    # ETF 는 pykrx 의 별도 API 가 필요. 종목 batch 와 동일 shape 으로 합침.
    if yyyymmdd:
        try:
            etf_df = stock.get_etf_ohlcv_by_ticker(yyyymmdd)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] get_etf_ohlcv_by_ticker fail: {exc}", file=sys.stderr)
            etf_df = None
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

    # universe 먼저 — code → market 매핑이 시장별 점수 산출에 필요.
    universe = build_universe()
    write_json(PUBLIC_DIR / "universe.json", universe)
    print(f"[INFO] universe: {universe.get('count', 0)} tickers")
    code_market: dict[str, str] = {}
    for t in universe.get("tickers", []):
        code_market[str(t.get("code") or "")] = str(t.get("market") or "")

    # 기간별 + 시장별 점수.
    def _rows_for(market: str) -> dict[int, list[dict]]:
        return {lb: build_sector_rows(ohlcv_map, lookback=lb,
                                       market_filter=market, code_market=code_market)
                for lb in (5, 20, 60)}

    rows_by_period = _rows_for("all")
    rows_by_period_kospi = _rows_for("kospi")
    rows_by_period_kosdaq = _rows_for("kosdaq")
    rows = rows_by_period[20]
    timeline = build_rotation_timeline(ohlcv_map, lookback=20, points=6, gap_days=5)

    write_json(PUBLIC_DIR / "dashboard.json", build_dashboard(meta, market_summary, rows))
    write_json(PUBLIC_DIR / "rotation_map.json",
               build_rotation_map(meta, rows_by_period, rows_by_period_kospi,
                                  rows_by_period_kosdaq, timeline))
    # candidates_by_period 도 만들어 NextCycle 에서 period 선택 가능하도록.
    write_json(PUBLIC_DIR / "candidates.json",
               build_candidates(meta, rows, rows_by_period=rows_by_period))
    # quotes 는 KRX 전종목 — health 의 trade_date 기준으로 pykrx 배치.
    quotes_trade_date = health["samsung_005930"]["trade_date"] if health.get("ok") else ""
    write_json(PUBLIC_DIR / "quotes.json", build_quotes(ohlcv_map, quotes_trade_date))

    compare = build_compare_per_code(ohlcv_map)
    for code, payload in compare.items():
        write_json(COMPARE_DIR / f"{code}.json", payload)

    print(f"[OK] wrote dashboard/rotation_map/candidates/quotes + compare/{len(compare)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
