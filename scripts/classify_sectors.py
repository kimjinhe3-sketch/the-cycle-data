"""KRX universe 전수 자동 섹터 분류 (v1).

입력: public/universe.json, public/quotes.json
출력: classification.xlsx — 사용자 검토용 (코드/이름/시장/타입/자동섹터/신뢰도/근거)

방법: 종목/ETF 이름 기반 키워드 매칭. 정확도 ~30~50%, 나머지는 unclassified.
신뢰도: high (강한 키워드 매칭) / medium (약한 키워드) / low (광의 패턴)
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = ROOT / "public"

# 섹터 ID → 한글명
SECTOR_NAMES = {
    "auto": "자동차",
    "memory_semi": "메모리반도체",
    "power_semi": "전력반도체",
    "ai": "AI",
    "robotics": "로보틱스",
    "defense": "방산",
    "nuclear": "원전",
    "power_equip": "전력기기",
    "liquid_cooling": "액침냉각",
    "ess": "ESS",
    "solar": "태양광",
    "wind": "풍력",
    "lng": "LNG",
    "shipbuilding": "조선",
    "construction": "건설",
    "telecom": "통신",
    "securities": "증권",
    "department": "백화점",
    "cable": "케이블",
    "comm_equip": "통신장비",
    "bio": "바이오",
    "finance": "금융",
    "chemical": "화학",
    "game": "게임",
    "entertainment": "엔터테인먼트",
    "cosmetics": "화장품",
    "steel": "철강",
    "food": "식음료",
    "transport": "운송",
    "fintech": "핀테크",
    "display": "디스플레이",
    "fashion": "패션의류",
    "mining": "비철금속",
    "real_estate": "리츠",
    "media": "미디어광고",
    "education": "교육",
    "environment": "환경",
    "trading": "종합상사",
    "medical_device": "의료기기",
    "home_appliance": "가전",
    "hotel_tourism": "호텔관광",
    "security": "보안",
    "agri_livestock": "농축산",
    "furniture": "가구인테리어",
    "paper": "종이",
    "cement": "시멘트",
    "holdings": "지주회사",
    "oil": "정유",
    "etc": "기타",
}

# 우선순위: 위에 있을수록 먼저 매칭. 충돌 가능 (예: 통신 vs 통신장비) 시 더 구체적인 게 먼저.
# ★주의★ 키워드는 위에서부터 처음 매칭되는 섹터로 확정. 따라서 더 specific 한 keyword 를 먼저 두어야 함.
RULES: list[tuple[str, list[str], str]] = [
    # (sector_id, keywords, confidence)
    # ── high confidence: 매우 specific 한 키워드 ──
    ("comm_equip",    ["광통신", "통신장비", "광케이블", "5G장비", "라우터", "스위치", "우리넷", "코위버", "대한광통신"], "high"),
    ("memory_semi",   ["삼성전자", "삼성전자우", "반도체", "메모리", "DRAM", "낸드", "NAND", "팹리스", "패키징", "디램", "SK하이닉스", "한미반도체", "리노공업", "원익IPS", "테스", "유진테크", "솔브레인", "동진쎄미켐", "테크윙", "고영", "심텍", "하나마이크론", "해성디에스", "코세스", "엑시콘", "와이씨", "씨아이에스", "티이엠씨", "티에프이", "덕산하이메탈", "티씨케이", "이오테크닉스", "에스앤에스텍", "원익홀딩스", "오로스테크놀로지", "피에스케이", "주성엔지니어링", "에스티아이", "유진테크", "한솔케미칼", "넥스틴", "테크엘", "오킨스전자", "케이씨에스", "DI", "한양이엔지", "후성", "동운아나텍", "이엠텍", "프로텍", "하나머티리얼즈", "월덱스", "리노공업", "심텍홀딩스", "케이엔제이", "코미코", "아이에이"], "high"),
    ("ess",           ["2차전지", "이차전지", "배터리", "양극재", "음극재", "전해액", "분리막", "에너지솔루션", "에코프로", "엘앤에프", "포스코퓨처엠", "삼성SDI", "LG에너지"], "high"),
    ("solar",         ["태양광", "솔라", "OCI홀딩스", "한솔테크닉스", "신성이엔지"], "high"),
    ("wind",          ["풍력", "오션플랜트", "씨에스윈드", "이터닉스", "유니슨", "동국S&C"], "high"),
    ("nuclear",       ["원전", "원자력", "에너빌리티", "한전기술", "한국전력", "비에이치아이", "우진"], "high"),
    ("defense",       ["방산", "방위산업", "디펜스", "에어로스페이스", "항공우주", "한화시스템", "한화에어로", "현대로템", "한국항공우주", "LIG넥스원", "풍산", "스페코", "한일단조"], "high"),
    ("shipbuilding",  ["조선", "한국조선해양", "삼성중공업", "한화오션", "HD현대미포", "HD현대중공업", "STX엔진"], "high"),
    ("robotics",      ["로봇", "로보틱스", "로보티즈", "휴머노이드", "유진로봇", "에스피지"], "high"),
    ("lng",           ["LNG", "가스공사", "SK가스", "지에스이", "E1"], "high"),
    ("cable",         ["전선", "케이블", "광섬유", "일진전기", "가온전선", "대한전선"], "high"),
    ("liquid_cooling",["액침", "냉각", "쿨링", "데이터센터냉각", "케이엔솔"], "high"),
    ("power_equip",   ["변압기", "전력기기", "ELECTRIC", "일렉트릭", "효성중공업", "HD현대일렉", "LS ELECTRIC"], "high"),
    ("power_semi",    ["SiC", "GaN", "전력반도체", "콘덴서", "MLCC", "삼성전기", "삼화콘덴서", "코스모신소재", "DB하이텍", "ISC", "KEC"], "high"),
    # 바이오/제약: 신규 섹터.
    ("bio",           ["바이오", "제약", "약품", "셀트리온", "삼성바이오", "유한양행", "녹십자", "한미사이언스", "한미약품", "휴젤", "알테오젠", "메디톡스", "동아", "광동", "JW중외", "보령", "환인", "안국", "일동", "휴온스", "대원", "유나이티드", "테라젠", "이엔셀", "엔케이맥스", "지놈앤컴퍼니"], "high"),
    # 게임
    ("game",          ["게임즈", "엔씨", "넥슨", "크래프톤", "펄어비스", "넷마블", "웹젠", "네오위즈", "위메이드", "컴투스", "더블유게임", "데브시스터즈", "그라비티", "조이시티", "엠게임", "라이브러"], "high"),
    # 엔터테인먼트
    ("entertainment", ["하이브", "JYP", "에스엠", "와이지", "CJ ENM", "CJ씨지브이", "SBS", "스튜디오드래곤", "쇼박스", "NEW", "키이스트", "FNC", "큐브엔터", "RBW", "미스터블루", "탑선"], "high"),
    # 화장품
    ("cosmetics",     ["화장품", "뷰티", "아모레", "LG생활건강", "콜마", "코스맥스", "코스메", "잇츠한불", "클리오", "토니모리", "한국화장품"], "high"),
    # 철강
    ("steel",         ["POSCO", "포스코홀딩스", "포스코퓨처엠", "포스코인터내셔널", "제철", "고려제강", "동국홀딩스", "동국제강", "강관", "특수강", "동부제철", "휴스틸", "세아", "DS단석", "금속"], "high"),
    # 식음료
    ("food",          ["제일제당", "농심", "오리온", "롯데웰", "롯데칠성", "대상", "오뚜기", "동원F&B", "동원산업", "삼양식품", "빙그레", "남양유업", "매일유업", "샘표", "팔도", "사조", "하림", "에그", "하이트진로", "MGC커피"], "high"),
    # 운송 (해운/항공/물류)
    ("transport",     ["대한항공", "아시아나", "제주항공", "에어부산", "에어서울", "티웨이", "HMM", "팬오션", "흥아해운", "대한해운", "글로비스", "대한통운", "한진"], "high"),
    # 금융 (은행/보험)
    ("finance",       ["KB금융", "신한지주", "하나금융", "우리금융", "BNK", "DGB", "JB금융", "기업은행", "제주은행", "삼성생명", "한화생명", "동양생명", "미래에셋생명", "삼성화재", "DB손해보험", "현대해상", "메리츠금융", "삼성카드", "하나캐피탈", "비씨카드", "교보생명"], "high"),
    # 화학
    ("chemical",      ["롯데케미칼", "케미칼", "화학", "첨단소재", "한솔케미칼", "코오롱인더", "KCC", "OCI", "OCI 머티리얼", "효성첨단", "효성화학", "송원산업", "남해화학", "BNK"], "high"),
    # 증권/자산운용
    ("securities",    ["증권", "자산운용", "금융지주", "투자증권", "키움", "미래에셋"], "high"),
    ("construction",  ["건설", "이앤씨", "E&A", "엔지니어링", "DL이앤씨", "GS건설", "롯데건설", "동부건설", "두산건설", "코오롱글로벌", "현대산업개발"], "high"),
    ("department",    ["백화점", "쇼핑", "신세계", "현대홈쇼핑", "롯데쇼핑", "GS리테일", "이마트"], "high"),
    # ── 신규 9개 섹터 ──
    ("fintech",       ["카카오뱅크", "카카오페이", "토스뱅크", "NICE평가정보", "한국정보통신", "KG이니시스", "갤럭시아", "다날", "FSN", "KCT"], "high"),
    ("display",       ["디스플레이", "OLED", "덕산네오룩스", "LG디스플레이", "디스플레이텍", "야스", "AP시스템", "동아엘텍", "비아트론"], "high"),
    ("fashion",       ["F&F", "한섬", "휠라", "신성통상", "패션", "의류", "한세실업", "영원무역", "코웰패션", "에프앤", "핸썸"], "high"),
    ("mining",        ["고려아연", "영풍", "비철금속", "광산", "구리", "아연", "니켈", "리튬", "포스코엠텍"], "high"),
    ("real_estate",   ["리츠", "부동산", "REIT", "케이탑", "프롭테크", "스타리츠"], "high"),
    ("media",         ["광고", "이노션", "제일기획", "미디어그룹", "이엠넷", "엔비티"], "high"),
    ("education",     ["교육", "학습", "메가스터디", "대교", "비상교육", "에듀윌", "윌비스", "능률", "삼성출판사", "예림당"], "high"),
    ("environment",   ["환경", "재활용", "폐기물", "수처리", "쓰레기", "인선이엔티", "와이엔텍", "코엔텍", "EM&I"], "high"),
    ("trading",       ["인터내셔널", "네트웍스", "코퍼레이션", "삼성물산", "포스코인터", "LX인터", "GS글로벌", "효성"], "medium"),
    # 의료기기
    ("medical_device",["인바디", "클래시스", "뷰웍스", "리메드", "디알텍", "레이언스", "메디포스트", "수젠텍", "휴비츠", "이루다", "원텍", "디오"], "high"),
    # 가전
    ("home_appliance",["LG전자", "코웨이", "위닉스", "동양매직", "쿠쿠전자", "쿠쿠홀딩스", "신일전자", "캐리어"], "high"),
    # 호텔관광/카지노
    ("hotel_tourism", ["호텔신라", "강원랜드", "GKL", "파라다이스", "카지노", "롯데관광", "하나투어", "모두투어", "노랑풍선", "참좋은여행"], "high"),
    # 보안
    ("security",      ["안랩", "윈스", "시큐브", "이글루", "라온시큐어", "지니언스", "지란지교", "닉스테크", "한컴위드"], "high"),
    # 농축산
    ("agri_livestock",["팜스코", "마니커", "사조", "하림", "체리부로", "정다운", "이지바이오", "한일사료", "동우", "선진"], "high"),
    # 가구/인테리어
    ("furniture",     ["한샘", "LX하우시스", "현대리바트", "에넥스", "퍼시스", "코아스", "동성코퍼레이션"], "high"),
    # 종이
    ("paper",         ["페이퍼", "제지", "한솔홀딩스", "깨끗한나라", "신풍제지"], "high"),
    # 시멘트
    ("cement",        ["시멘트", "한일홀딩스", "성신양회", "삼표", "쌍용씨앤이", "고려시멘트"], "high"),
    # 지주회사 (단일 모기업 이름이거나 'XX홀딩스' 형태)
    ("holdings",      ["홀딩스", "지주회사"], "medium"),
    # 정유
    ("oil",           ["S-Oil", "에쓰오일", "SK이노베이션", "현대오일뱅크", "GS칼텍스", "흥구석유", "중앙에너비스", "극동유화", "S-OIL"], "high"),
    # 케이블/통신장비/디스플레이 보조 (자주 누락되는 종목명)
    ("comm_equip",    ["쏠리드", "오이솔루션", "이수페타시스", "케이엠더블유", "RFHIC", "필옵틱스", "라이콤", "텔레필드", "에이스테크", "다보링크", "에이엠텍", "지아이텔콤", "다산네트웍스"], "high"),
    ("display",       ["LG디스플레이", "덕산네오룩스", "솔루엠", "이녹스첨단소재", "야스", "비아트론", "AP시스템", "동아엘텍", "주성엔지니어링"], "high"),
    # 보안 (CCTV, 영상보안 포함)
    ("security",      ["한화비전", "에스원", "ITX-AI"], "high"),
    # 핀테크/결제
    ("fintech",       ["NHN KCP", "코나아이", "헥토파이낸셜", "아톤", "다날", "갤럭시아", "아이씨티케이"], "high"),
    # 조선엔진 + 마린솔루션
    ("shipbuilding",  ["한화엔진", "HD현대마린", "LS마린", "HJ중공업"], "high"),
    # 의료기기 추가 픽
    ("medical_device",["메디포스트", "디오", "리센스메디컬", "휴비츠", "원텍", "리브스메드", "로킷헬스케어", "씨어스"], "high"),
    # ── medium / 광의 ──
    ("auto",          ["자동차", "모터스", "현대차", "현대모비스", "한온시스템", "오토에버", "차량부품", "자율주행", "만도", "에스엘", "현대위아", "성우하이텍", "에스엘티", "타이어", "기아", "엘리베이터", "현대엘리베이터", "켐트로닉스", "아진산업", "대동기어", "진성티이씨", "파인엠텍", "성창오토텍", "S&T모티브", "넥센", "한국타이어", "금호타이어"], "medium"),
    ("telecom",       ["SK텔레콤", "텔레콤", "LG유플러스", "유플러스", "KT가입", "이동통신", "통신서비스", "KT$"], "medium"),
    ("ai",            ["인공지능", "AI", "빅데이터", "클라우드", "데이터센터", "챗봇", "한컴", "솔트룩스", "NAVER", "카카오", "씨엔에스", "마음AI", "이스트소프트", "포티투마루", "삼성에스디에스", "삼성SDS"], "medium"),
]

# ETF 별도 룰 — ETF 는 보통 이름에 섹터 키워드가 직접 들어 있어 매칭 정확도 높음.
ETF_RULES: list[tuple[str, list[str]]] = [
    ("memory_semi",   ["반도체", "200IT", "TIGER Fn반도체", "SOL 차세대반도체"]),
    ("ai",            ["AI", "인공지능", "빅테크", "FANG", "빅데이터", "메타버스"]),
    ("ess",           ["2차전지", "배터리", "K뉴딜", "전기차"]),
    ("solar",         ["태양광", "신재생", "그린뉴딜"]),
    ("wind",          ["풍력"]),
    ("nuclear",       ["원전", "원자력"]),
    ("defense",       ["방산", "K방산"]),
    ("shipbuilding",  ["조선"]),
    ("robotics",      ["로봇", "로보틱스"]),
    ("lng",           ["LNG", "천연가스", "에너지화학"]),
    ("construction",  ["건설", "건축"]),
    ("telecom",       ["통신서비스", "통신"]),
    ("securities",    ["증권"]),
    ("department",    ["소비재", "유통"]),
    ("auto",          ["자동차"]),
    ("bio",           ["바이오", "헬스케어", "제약"]),
    ("finance",       ["은행", "보험", "금융"]),
    ("chemical",      ["화학"]),
    ("game",          ["게임"]),
    ("entertainment", ["미디어", "엔터", "콘텐츠", "K-POP"]),
    ("cosmetics",     ["화장품", "뷰티"]),
    ("steel",         ["철강", "POSCO"]),
    ("food",          ["음식료", "식품", "푸드"]),
    ("transport",     ["운송", "물류", "해운", "항공"]),
]


def classify_stock(name: str) -> tuple[str, str, str]:
    """이름 → (sector_id, confidence, matched_keyword). 매칭 실패 시 'etc' 반환 (100% 커버)."""
    n = name
    for sector_id, keywords, conf in RULES:
        for kw in keywords:
            if kw in n or kw.lower() in n.lower():
                return sector_id, conf, kw
    return "etc", "low", ""


def classify_etf(name: str) -> tuple[str, str, str]:
    n = name
    for sector_id, keywords in ETF_RULES:
        for kw in keywords:
            if kw in n or kw.lower() in n.lower():
                return sector_id, "high", kw
    return "etc", "low", ""


def main() -> int:
    universe = json.loads((PUBLIC / "universe.json").read_text(encoding="utf-8"))
    try:
        quotes = json.loads((PUBLIC / "quotes.json").read_text(encoding="utf-8")).get("quotes", {})
    except Exception:
        quotes = {}

    rows: list[dict] = []
    seen: set[str] = set()
    # universe 먼저 (KOSPI/KOSDAQ/KONEX 종목)
    for t in universe.get("tickers", []):
        code = t.get("code")
        if not code or code in seen:
            continue
        seen.add(code)
        name = t.get("name", "")
        ttype = t.get("type", "stock")
        market = t.get("market", "")
        sector_id, conf, matched = (
            classify_etf(name) if ttype == "etf" else classify_stock(name)
        )
        q = quotes.get(code, {})
        rows.append({
            "코드": code,
            "종목명": name,
            "시장": market,
            "타입": ttype,
            "자동섹터ID": sector_id or "",
            "자동섹터명": SECTOR_NAMES.get(sector_id or "", ""),
            "신뢰도": conf,
            "매칭키워드": matched,
            "종가": q.get("close", ""),
            "등락률(%)": q.get("change_pct", ""),
            "거래대금": q.get("trading_value", ""),
        })

    # quotes 에는 있지만 universe 에 없는 코드 (대부분 ETF). 별도 처리.
    extra_codes = [c for c in quotes.keys() if c not in seen]
    # 큐레이션 ASSETS 에서 이름 lookup
    try:
        curated = json.loads((ROOT / "data" / "assets.json").read_text(encoding="utf-8"))
        curated_map = {a["code"]: a for a in curated}
    except Exception:
        curated_map = {}

    for code in extra_codes:
        meta = curated_map.get(code, {})
        name = meta.get("name", "(unknown)")
        ttype = meta.get("type", "etf")
        sector_id, conf, matched = (
            classify_etf(name) if ttype == "etf" else classify_stock(name)
        )
        # 큐레이션이면 그 sectorIds 도 같이 표기
        curated_sectors = ",".join(meta.get("sectorIds", []) or [])
        q = quotes.get(code, {})
        rows.append({
            "코드": code,
            "종목명": name,
            "시장": "ETF" if ttype == "etf" else "?",
            "타입": ttype,
            "자동섹터ID": sector_id or "",
            "자동섹터명": SECTOR_NAMES.get(sector_id or "", ""),
            "신뢰도": conf,
            "매칭키워드": matched,
            "종가": q.get("close", ""),
            "등락률(%)": q.get("change_pct", ""),
            "거래대금": q.get("trading_value", ""),
            "큐레이션섹터": curated_sectors,
        })

    df = pd.DataFrame(rows)
    # 통계 출력
    total = len(df)
    classified_real = (df["자동섹터ID"].ne("") & df["자동섹터ID"].ne("etc")).sum()
    etc_count = (df["자동섹터ID"] == "etc").sum()
    print(f"[INFO] 총 {total} / 실분류 {classified_real} ({classified_real/total*100:.1f}%) / 기타 {etc_count}")
    print()
    print("섹터별 분류 수:")
    counts = df[df["자동섹터ID"] != ""]["자동섹터명"].value_counts()
    for k, v in counts.items():
        print(f"  {k:12s} {v}")

    out_path = ROOT / "classification.xlsx"
    # 거래대금 컬럼은 string 일 수도 있어 안전하게 numeric.
    df["거래대금_정렬용"] = pd.to_numeric(df["거래대금"], errors="coerce").fillna(0)

    # 시트1: 전체 — 자동섹터명 → 거래대금 desc 순. (검토 시 같은 섹터 묶어서 보기 편함)
    full = df.sort_values(by=["자동섹터명", "거래대금_정렬용"], ascending=[True, False]).drop(columns=["거래대금_정렬용"])
    # 시트2: 기타만 — 거래대금 desc.  (룰 보강 candidates 우선순위)
    etc_df = df[df["자동섹터ID"] == "etc"].sort_values(by="거래대금_정렬용", ascending=False).drop(columns=["거래대금_정렬용"])
    # 시트3: 섹터 카운트 요약
    summary = df["자동섹터명"].value_counts().reset_index()
    summary.columns = ["섹터", "종목수"]

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="요약")
        full.to_excel(writer, index=False, sheet_name="전체분류")
        etc_df.to_excel(writer, index=False, sheet_name="기타_검토대상")
    print(f"[OK] saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
