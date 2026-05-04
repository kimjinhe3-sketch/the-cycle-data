"""Stage 1: pykrx 로 삼성전자(005930) 가장 최근 거래일 종가만 가져와서
public/health.json 으로 떨어뜨리는 검증용 배치.

GitHub Actions 의 cron 으로 매일 17:30 KST 에 실행되며, 결과 JSON 이 같은
레포에 커밋되면 jsDelivr CDN 이 자동 배포한다.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pykrx import stock

KST = timezone(timedelta(hours=9))
SAMSUNG = "005930"
PUBLIC_DIR = Path(__file__).resolve().parent.parent / "public"


def find_last_trading_day(end_date: datetime) -> str:
    """end_date 부터 거꾸로 최대 10일 거슬러 올라가면서 OHLCV 데이터가 있는
    최초의 영업일을 찾아 YYYYMMDD 형식으로 반환."""
    for i in range(10):
        d = (end_date - timedelta(days=i)).strftime("%Y%m%d")
        df = stock.get_market_ohlcv_by_date(d, d, SAMSUNG)
        if not df.empty:
            return d
    raise RuntimeError("최근 10일 내에 거래일을 찾지 못했습니다")


def fetch_health() -> dict:
    now = datetime.now(KST)
    last = find_last_trading_day(now)
    df = stock.get_market_ohlcv_by_date(last, last, SAMSUNG)
    row = df.iloc[0]
    close = int(row["종가"])
    volume = int(row["거래량"])
    trading_value = int(row.get("거래대금", 0))

    return {
        "ok": True,
        "generated_at": now.isoformat(timespec="seconds"),
        "samsung_005930": {
            "trade_date": f"{last[:4]}-{last[4:6]}-{last[6:]}",
            "close_krw": close,
            "volume": volume,
            "trading_value_krw": trading_value,
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
