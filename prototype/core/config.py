"""상수·경로·CSS·RNG 글로벌."""
from pathlib import Path
import numpy as np

# ----- 난수 -----
RNG = np.random.default_rng(42)

# ----- 섹터 (합성 모드) -----
N_SECTORS = 12
SECTOR_NAMES = [
    "반도체", "2차전지", "바이오", "자동차", "화학", "철강",
    "금융", "건설", "유통", "엔터", "통신", "조선",
]

# ----- 합성 종목명 -----
FAKE_NAMES = [
    f"{s}{['전자','산업','중공업','제약','케미칼','테크','홀딩스','파이낸셜','글로벌'][i%9]}"
    for i, s in enumerate([
        "가나", "다라", "마바", "사아", "자차", "카타", "파하", "나다",
        "라마", "바사", "아자", "차카", "타파", "하나", "다라", "마바",
        "사아", "자차", "카타", "파하", "나다", "라마", "바사", "아자",
    ])
]

# ----- 모델 피처 -----
FEATS = [
    "ret_1d", "ret_5d", "ret_20d", "vol_5d", "vol_20d", "ma_ratio",
    "vol_z_20", "surge_5d", "drawdown_20",
    "news_sent_lag1", "news_sent_ma5",
    "disclosure_lag1", "bad_disc_20d", "good_disc_20d", "disc_severity_20d",
    "regime_lag1",
]

# ----- 톤다운 색상 팔레트 (차트·UI 공통) -----
COLORS = {
    "green":  "#66BB6A",   # 호재·상승
    "red":    "#E57373",   # 악재·리스크
    "orange": "#FFB74D",   # 주의
    "blue":   "#5B8DEF",   # 정보
    "gray":   "#BDBDBD",   # 중립
    "dark":   "#424242",   # 텍스트
    "muted":  "#9E9E9E",   # 캡션
    "bg":     "#FCFCFC",   # 카드 배경
    "border": "#ECECEC",   # 보더
}

# 분류 카테고리 색상 (matrix scatter용)
CATEGORY_COLORS = {
    "PRIORITY":  "#81C784",
    "HIGH-RISK": "#FFB74D",
    "HOLD":      "#B0BEC5",
    "AVOID":     "#E57373",
}

FEAT_KOR = {
    "ret_1d": "1일 수익률", "ret_5d": "5일 수익률", "ret_20d": "20일 수익률",
    "vol_5d": "5일 변동성", "vol_20d": "20일 변동성",
    "ma_ratio": "MA5/MA20 비율", "vol_z_20": "거래량 z-score(20)",
    "surge_5d": "5일 30%+ 급등 플래그", "drawdown_20": "20일 드로다운",
    "news_sent_lag1": "전일 뉴스 감성", "news_sent_ma5": "5일 뉴스 감성 평균",
    "disclosure_lag1": "전일 공시 이벤트(룰베이스)",
    "bad_disc_20d": "20일 악재 공시 수", "good_disc_20d": "20일 호재 공시 수",
    "disc_severity_20d": "20일 공시 위험도 합계",
    "regime_lag1": "시장 국면(0상승/1횡보/2하락)",
}

# ----- 데이터 경로 -----
_PROTO_ROOT = Path(__file__).resolve().parent.parent
REAL_PANEL_PATH = _PROTO_ROOT / "_data" / "real_kospi_top100.pkl"
TICKER_META_PATH = _PROTO_ROOT / "_data" / "ticker_meta.json"

# ----- 페이지 CSS — 프로덕션 앱 스타일 -----
# 디자인 원칙:
#   1. 미니멀 톤다운 (파스텔 + 회색 베이스)
#   2. 카드 기반 레이아웃 (그림자 부드럽게)
#   3. 일관된 타이포그래피 (h1~h4, 본문 0.95em)
#   4. 탭 sticky + 부드러운 강조
CSS_BLOCK = """
<style>
/* ===== 본문 베이스 ===== */
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Malgun Gothic', sans-serif; }

/* ===== 헤더 ===== */
h1 { color: #1A1A1A; font-weight: 700; letter-spacing: -0.5px; }
h2 { color: #2E2E2E; font-weight: 600; margin-top: 1.2rem; }
h3 { color: #333; font-weight: 600; }
h4 { color: #424242; font-weight: 600; }

/* ===== 탭 — 더 부드럽게 ===== */
.stTabs [data-baseweb="tab-list"] {
  gap: 4px;
  background: #F7F9FC;
  padding: 4px;
  border-radius: 10px;
}
.stTabs [data-baseweb="tab"] {
  background: transparent;
  border-radius: 7px;
  padding: 8px 14px;
  font-weight: 500;
  color: #5C5C5C;
  border: none;
}
.stTabs [aria-selected="true"] {
  background: #FFFFFF !important;
  color: #1565C0 !important;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

/* ===== 메트릭 색상 ===== */
.metric-up      { color: #5CB874; font-weight: 600; }
.metric-risk    { color: #E07A7A; font-weight: 600; }
.metric-neutral { color: #888;    font-weight: 600; }

/* ===== 카드 ===== */
.card {
  border: 1px solid #ECECEC;
  border-radius: 12px;
  padding: 16px 18px;
  background: #FFFFFF;
  margin-bottom: 10px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.03);
  transition: box-shadow 0.2s;
}
.card:hover { box-shadow: 0 2px 6px rgba(0,0,0,0.06); }
.card h4 { margin: 0 0 6px 0; font-weight: 600; color: #2E2E2E; }
.card small { color: #999; font-weight: 400; }

/* ===== 분류 태그 (파스텔) ===== */
.tag-priority { background: #C8E6C9; color: #2E5933; padding: 3px 10px;
                border-radius: 10px; font-size: 0.78em; font-weight: 600; }
.tag-highrisk { background: #FFE0B2; color: #7A4E13; padding: 3px 10px;
                border-radius: 10px; font-size: 0.78em; font-weight: 600; }
.tag-hold     { background: #ECEFF1; color: #546E7A; padding: 3px 10px;
                border-radius: 10px; font-size: 0.78em; font-weight: 600; }
.tag-avoid    { background: #FFCDD2; color: #8B2D2D; padding: 3px 10px;
                border-radius: 10px; font-size: 0.78em; font-weight: 600; }

/* ===== 면책 박스 ===== */
.disclaimer {
  background: #FFF8E1;
  padding: 10px 14px;
  border-left: 3px solid #FFCA28;
  border-radius: 6px;
  font-size: 0.85em;
  color: #6B6B6B;
  margin: 8px 0 14px 0;
}

/* ===== 상단 헤더 바 ===== */
.app-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  background: linear-gradient(135deg, #5B8DEF 0%, #1565C0 100%);
  color: white;
  border-radius: 12px;
  margin-bottom: 16px;
  box-shadow: 0 2px 8px rgba(21,101,192,0.15);
}
.app-header h1 { color: white !important; margin: 0; font-size: 1.6em; }
.app-header .subtitle { color: rgba(255,255,255,0.85); font-size: 0.9em; margin: 4px 0 0 0; }
.app-header .right { display: flex; gap: 8px; align-items: center; }
.app-header .pill {
  background: rgba(255,255,255,0.2);
  padding: 4px 10px;
  border-radius: 10px;
  font-size: 0.78em;
  font-weight: 600;
}

/* ===== Divider ===== */
hr { border-top: 1px solid #EEEEEE !important; margin: 1.2rem 0 !important; }

/* ===== 데이터프레임 ===== */
.stDataFrame { font-size: 0.92em; }

/* ===== 사이드바 ===== */
[data-testid="stSidebar"] {
  background: #F7F9FC;
  border-right: 1px solid #ECECEC;
}
[data-testid="stSidebar"] h3 { color: #424242; font-size: 1em; font-weight: 600; }

/* ===== 버튼 ===== */
.stButton button {
  border-radius: 8px;
  font-weight: 500;
  transition: all 0.15s;
}
.stButton button[kind="primary"] {
  background: #5B8DEF;
  border: none;
  box-shadow: 0 1px 3px rgba(91,141,239,0.3);
}
.stButton button[kind="primary"]:hover {
  background: #4A7BD8;
  box-shadow: 0 2px 6px rgba(91,141,239,0.4);
}

/* ===== Expander ===== */
.streamlit-expanderHeader {
  font-weight: 500;
  color: #424242;
  background: #FAFAFA;
  border-radius: 8px;
}
</style>
"""
