# the-cycle-data

더 사이클(The Cycle) 앱의 정적 데이터 소스.

## 구조

- `scripts/fetch.py` — pykrx 로 KRX 일별 시세/시총/수급을 가져와 `public/*.json` 으로 떨어뜨리는 배치
- `.github/workflows/daily.yml` — 매일 17:30 KST 크론 + 수동 트리거. 위 스크립트를 실행하고 변경된 JSON 만 커밋
- `public/` — 클라이언트가 jsDelivr CDN 으로 읽는 정적 응답 파일들

## 클라이언트가 읽는 URL

```
https://cdn.jsdelivr.net/gh/kimjinhe3-sketch/the-cycle-data@main/public/<file>.json
```

jsDelivr 가 GitHub raw 를 캐싱·전세계 CDN 으로 서빙. 별도 호스팅 불필요.

## Stage 1 (현재)

- `public/health.json` — 가장 최근 거래일의 삼성전자(005930) 종가 1개 + 메타데이터
- pykrx 로 실제 KRX 시세를 끌어오는지 검증용

## 확장 예정

- `public/dashboard.json` — 시장 요약 + 강한 섹터
- `public/rotation_map.json` — 섹터 히트맵 + 버블
- `public/candidates.json` — 다음 사이클 후보
- `public/compare/{code}.json` — 종목별 일봉
