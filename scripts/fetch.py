"""Stage 1: pykrx 로 삼성전자(005930) 가장 최근 거래일 종가만 가져와서
public/health.json 으로 떨어뜨리는 검증용 배치.

GitHub Actions 의 cron 으로 매일 17:30 KST 에 실행되며, 결과 JSON 이 같은
레포에 커밋되면 jsDelivr CDN 이 자동 배포한다.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pykrx import stock

KST = timezone(timedelta(hours=9))
SAMSUNG = "005930"
PUBLIC_DIR = Path(__file__).resolve().parent.parent / "public"


def fetch_health() -> dict:
    now = datetime.now(KST)
    end = now.strftime("%Y%m%d")
    # 최근 14일 범위로 받아서 가장 마지막 거래일 행을 사용 — 단일 날짜 호출보다 안정적.
    start = (now - timedelta(days=14)).strftime("%Y%m%d")

    # 1차 — 단일 종목 일별 OHLCV (단일 종목 API 는 거래대금 컬럼이 빠져있음)
    ohlcv = stock.get_market_ohlcv_by_date(start, end, SAMSUNG)
    if ohlcv.empty:
        raise RuntimeError("OHLCV 결과가 비어있음")
    last_idx = ohlcv.index[-1]
    last_row = ohlcv.iloc[-1]
    trade_date_compact = (
        last_idx.strftime("%Y%m%d") if hasattr(last_idx, "strftime") else str(last_idx)
    )
    trade_date = (
        last_idx.strftime("%Y-%m-%d") if hasattr(last_idx, "strftime") else trade_date_compact
    )

    def num(series, key):
        try:
            return int(series[key]) if key in series.index else 0
        except Exception:  # noqa: BLE001
            return 0

    close = num(last_row, "종가")
    volume = num(last_row, "거래량")

    # 2차 — 같은 날짜의 전종목 OHLCV 스냅샷에서 005930 행 뽑기.
    # 이 API 는 거래대금 컬럼을 포함함.
    trading_value = 0
    snapshot_columns: list[str] = []
    try:
        snap = stock.get_market_ohlcv_by_ticker(trade_date_compact)
        snapshot_columns = list(snap.columns)
        if SAMSUNG in snap.index:
            trading_value = num(snap.loc[SAMSUNG], "거래대금")
    except Exception:  # noqa: BLE001
        pass

    # 3차 fallback — 거래대금 = 종가 × 거래량 근사값 (VWAP 가 아니라 정확하진 않지만
    # 같은 자릿수 추정). 위 두 단계가 모두 실패했을 때만 사용.
    approximated = False
    if trading_value == 0 and close > 0 and volume > 0:
        trading_value = close * volume
        approximated = True

    return {
        "ok": True,
        "generated_at": now.isoformat(timespec="seconds"),
        "samsung_005930": {
            "trade_date": trade_date,
            "close_krw": close,
            "volume": volume,
            "trading_value_krw": trading_value,
            "trading_value_approximated": approximated,
        },
        "debug": {
            "ohlcv_columns": list(ohlcv.columns),
            "snapshot_columns": snapshot_columns,
        },
        "source": "pykrx",
    }


def write_json(name: str, payload: dict) -> Path:
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    path = PUBLIC_DIR / name
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def main() -> int:
    try:
        payload = fetch_health()
    except Exception as exc:  # noqa: BLE001
        # 실패해도 last-known JSON 을 덮어쓰지 않도록 별도 파일에 에러 메타만 남긴다.
        err = {
            "ok": False,
            "generated_at": datetime.now(KST).isoformat(timespec="seconds"),
            "error": f"{type(exc).__name__}: {exc}",
        }
        write_json("last_error.json", err)
        print(f"[ERROR] fetch failed: {exc}", file=sys.stderr)
        return 1

    path = write_json("health.json", payload)
    print(f"[OK] wrote {path}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
