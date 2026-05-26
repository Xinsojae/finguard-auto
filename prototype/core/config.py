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

# ----- 페이지 CSS — 프로덕션 앱 디자인 시스템 -----
# 영감: Linear / Notion / Stripe / Bloomberg Terminal
# 토큰: CSS variables로 light/dark 분리
CSS_BLOCK = """
<style>
/* ============================================================
   디자인 토큰 (Light 기본값)
   ============================================================ */
:root {
  --bg-base:        #FFFFFF;
  --bg-elevated:    #FAFBFC;
  --bg-card:        #FFFFFF;
  --bg-hover:       #F5F7FA;
  --bg-subtle:      #F0F3F8;

  --border-subtle:  #EAECEF;
  --border-default: #D9DCE1;
  --border-strong:  #B8BDC7;

  --text-primary:   #0F1419;
  --text-secondary: #4B5563;
  --text-muted:     #9CA3AF;
  --text-inverse:   #FFFFFF;

  --accent:         #2563EB;
  --accent-hover:   #1D4ED8;
  --accent-soft:    #DBEAFE;
  --accent-glow:    rgba(37, 99, 235, 0.12);

  --success:        #059669;
  --success-soft:   #D1FAE5;
  --warning:        #D97706;
  --warning-soft:   #FEF3C7;
  --danger:         #DC2626;
  --danger-soft:    #FEE2E2;

  --shadow-sm:      0 1px 2px rgba(15, 20, 25, 0.04);
  --shadow-md:      0 4px 12px rgba(15, 20, 25, 0.06);
  --shadow-lg:      0 12px 32px rgba(15, 20, 25, 0.08);
  --shadow-glow:    0 0 0 1px var(--accent-glow), 0 4px 16px var(--accent-glow);

  --radius-sm:      6px;
  --radius:         10px;
  --radius-lg:      14px;
  --radius-xl:      18px;

  --ease:           cubic-bezier(0.4, 0, 0.2, 1);
  --dur:            180ms;
}

/* ============================================================
   타이포그래피
   ============================================================ */
html, body, .stApp {
  font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display',
               'Inter', 'Pretendard Variable', 'Pretendard',
               'Segoe UI Variable', 'Malgun Gothic', sans-serif !important;
  font-feature-settings: "cv11", "ss01", "ss03", "tnum";
  color: var(--text-primary);
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
}

h1 { font-weight: 700; letter-spacing: -0.025em; color: var(--text-primary); }
h2 { font-weight: 600; letter-spacing: -0.02em; color: var(--text-primary); margin-top: 1.4rem; }
h3 { font-weight: 600; letter-spacing: -0.015em; color: var(--text-primary); }
h4 { font-weight: 600; letter-spacing: -0.01em; color: var(--text-primary); }
p, li, span, label { color: var(--text-secondary); }

/* 숫자 tabular */
[data-testid="stMetricValue"] {
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.02em;
  font-weight: 600;
  color: var(--text-primary);
}

/* ============================================================
   본문 베이스
   ============================================================ */
.block-container {
  padding-top: 1.4rem !important;
  padding-bottom: 4rem !important;
  max-width: 1400px;
}
.stApp { background: var(--bg-base); }

/* ============================================================
   상단 헤더 바 — gradient + glow
   ============================================================ */
.app-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 22px 28px;
  background: linear-gradient(135deg, #1E40AF 0%, #2563EB 50%, #3B82F6 100%);
  color: white;
  border-radius: var(--radius-lg);
  margin-bottom: 20px;
  box-shadow: 0 8px 24px rgba(37, 99, 235, 0.18), 0 1px 2px rgba(0,0,0,0.08);
  position: relative;
  overflow: hidden;
}
.app-header::before {
  content: ""; position: absolute; top: -50%; right: -10%;
  width: 320px; height: 320px;
  background: radial-gradient(circle, rgba(255,255,255,0.12) 0%, rgba(255,255,255,0) 70%);
  pointer-events: none;
}
.app-header .brand h1 {
  color: white !important; margin: 0; font-size: 1.65em;
  letter-spacing: -0.025em; font-weight: 700;
}
.app-header .subtitle {
  color: rgba(255,255,255,0.85); font-size: 0.88em;
  margin: 4px 0 0 0; font-weight: 400; letter-spacing: -0.01em;
}
.app-header .right { display: flex; gap: 8px; align-items: center; }
.app-header .pill {
  background: rgba(255,255,255,0.16);
  backdrop-filter: blur(8px);
  border: 1px solid rgba(255,255,255,0.2);
  padding: 5px 12px;
  border-radius: 100px;
  font-size: 0.76em;
  font-weight: 600;
  letter-spacing: 0.02em;
  color: white;
}

/* ============================================================
   탭 — Linear / Notion 스타일 세그먼티드
   ============================================================ */
.stTabs [data-baseweb="tab-list"] {
  gap: 2px;
  background: var(--bg-elevated);
  padding: 5px;
  border-radius: var(--radius);
  border: 1px solid var(--border-subtle);
  box-shadow: var(--shadow-sm);
}
.stTabs [data-baseweb="tab"] {
  background: transparent;
  border-radius: 7px;
  padding: 9px 16px;
  font-weight: 500;
  font-size: 0.92em;
  color: var(--text-secondary);
  border: none;
  transition: all var(--dur) var(--ease);
  letter-spacing: -0.01em;
}
.stTabs [data-baseweb="tab"]:hover {
  background: var(--bg-hover);
  color: var(--text-primary);
}
.stTabs [aria-selected="true"] {
  background: var(--bg-card) !important;
  color: var(--accent) !important;
  box-shadow: var(--shadow-sm), 0 0 0 1px var(--accent-glow);
  font-weight: 600;
}

/* ============================================================
   카드 — soft shadow + hover lift
   ============================================================ */
.card {
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-lg);
  padding: 18px 20px;
  background: var(--bg-card);
  margin-bottom: 12px;
  box-shadow: var(--shadow-sm);
  transition: transform var(--dur) var(--ease),
              box-shadow var(--dur) var(--ease),
              border-color var(--dur) var(--ease);
}
.card:hover {
  box-shadow: var(--shadow-md);
  border-color: var(--border-default);
  transform: translateY(-1px);
}
.card h4 {
  margin: 0 0 8px 0; font-weight: 600; font-size: 1.02em;
  color: var(--text-primary); letter-spacing: -0.01em;
}
.card small { color: var(--text-muted); font-weight: 400; }

/* 분류 태그 ---------- */
.tag-priority, .tag-highrisk, .tag-hold, .tag-avoid {
  display: inline-block;
  padding: 4px 11px;
  border-radius: 100px;
  font-size: 0.74em;
  font-weight: 600;
  letter-spacing: 0.02em;
  border: 1px solid transparent;
}
.tag-priority { background: var(--success-soft); color: var(--success); border-color: rgba(5,150,105,0.2); }
.tag-highrisk { background: var(--warning-soft); color: var(--warning); border-color: rgba(217,119,6,0.2); }
.tag-hold     { background: #F1F4F8; color: #475569; border-color: #D9DCE1; }
.tag-avoid    { background: var(--danger-soft); color: var(--danger); border-color: rgba(220,38,38,0.2); }

/* 메트릭 색상 ---------- */
.metric-up      { color: var(--success); font-weight: 600; }
.metric-risk    { color: var(--danger);  font-weight: 600; }
.metric-neutral { color: var(--text-muted); font-weight: 600; }

/* 면책 박스 ---------- */
.disclaimer {
  background: var(--warning-soft);
  padding: 11px 16px;
  border-left: 3px solid var(--warning);
  border-radius: var(--radius-sm);
  font-size: 0.85em;
  color: var(--text-secondary);
  margin: 10px 0 16px 0;
  line-height: 1.55;
}

/* ============================================================
   Divider — 절제된 라인
   ============================================================ */
hr {
  border: none !important;
  border-top: 1px solid var(--border-subtle) !important;
  margin: 1.4rem 0 !important;
}

/* ============================================================
   사이드바
   ============================================================ */
[data-testid="stSidebar"] {
  background: var(--bg-elevated) !important;
  border-right: 1px solid var(--border-subtle) !important;
}
[data-testid="stSidebar"] h3 {
  color: var(--text-primary) !important;
  font-size: 0.95em;
  font-weight: 600;
  letter-spacing: -0.01em;
  margin-top: 0.4rem;
}
[data-testid="stSidebar"] .stMetric { padding: 4px 0; }
[data-testid="stSidebar"] [data-testid="stMetricValue"] { font-size: 1.15em; }
[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
  color: var(--text-muted);
  font-size: 0.82em;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  font-weight: 500;
}

/* ============================================================
   버튼 — primary는 glow, secondary는 보더
   ============================================================ */
.stButton button {
  border-radius: var(--radius-sm);
  font-weight: 500;
  font-size: 0.92em;
  padding: 8px 16px;
  transition: all var(--dur) var(--ease);
  border: 1px solid var(--border-default);
  background: var(--bg-card);
  color: var(--text-primary);
  letter-spacing: -0.005em;
}
.stButton button:hover {
  border-color: var(--accent);
  color: var(--accent);
  transform: translateY(-1px);
  box-shadow: var(--shadow-sm);
}
.stButton button[kind="primary"] {
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%);
  border: 1px solid var(--accent-hover);
  color: white;
  box-shadow: 0 1px 3px var(--accent-glow), 0 4px 12px var(--accent-glow);
}
.stButton button[kind="primary"]:hover {
  background: linear-gradient(135deg, var(--accent-hover) 0%, #1E3A8A 100%);
  color: white;
  transform: translateY(-1px);
  box-shadow: 0 2px 6px var(--accent-glow), 0 8px 20px var(--accent-glow);
}

/* ============================================================
   Expander / Slider / Input
   ============================================================ */
.streamlit-expanderHeader {
  font-weight: 500;
  color: var(--text-primary);
  background: var(--bg-elevated);
  border: 1px solid var(--border-subtle) !important;
  border-radius: var(--radius-sm);
  padding: 10px 14px !important;
  transition: all var(--dur) var(--ease);
}
.streamlit-expanderHeader:hover {
  background: var(--bg-hover);
  border-color: var(--border-default) !important;
}

.stTextInput input, .stTextArea textarea, .stNumberInput input {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-default) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-primary) !important;
  font-family: inherit !important;
  transition: all var(--dur) var(--ease);
}
.stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-glow) !important;
}

/* ============================================================
   DataFrame
   ============================================================ */
.stDataFrame {
  font-size: 0.9em;
  border-radius: var(--radius);
  overflow: hidden;
  border: 1px solid var(--border-subtle);
}

/* ============================================================
   알림/Info/Success/Warning 박스 통일
   ============================================================ */
.stAlert {
  border-radius: var(--radius);
  border: 1px solid var(--border-subtle);
  background: var(--bg-elevated);
}

/* ============================================================
   라이트 모드 위젯 프리미엄 보정
   ============================================================ */

/* ---- Multiselect / Selectbox (사이드바·본문) ---- */
.stMultiSelect [data-baseweb="select"] > div,
.stSelectbox [data-baseweb="select"] > div {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-default) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-primary) !important;
  transition: all var(--dur) var(--ease);
  min-height: 38px;
}
.stMultiSelect [data-baseweb="select"]:hover > div,
.stSelectbox [data-baseweb="select"]:hover > div {
  border-color: var(--border-strong) !important;
}
.stMultiSelect [data-baseweb="select"]:focus-within > div,
.stSelectbox [data-baseweb="select"]:focus-within > div {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-glow) !important;
}

/* 선택된 multiselect 태그 (chip) — 라이트 */
[data-baseweb="tag"] {
  background: var(--accent-soft) !important;
  border: 1px solid rgba(37, 99, 235, 0.2) !important;
  color: var(--accent) !important;
  font-weight: 500 !important;
  border-radius: 6px !important;
}
[data-baseweb="tag"] span,
[data-baseweb="tag"] div { color: var(--accent) !important; }
[data-baseweb="tag"] svg { fill: var(--accent) !important; }

/* 드롭다운 옵션 리스트 (popover) */
[data-baseweb="popover"] [data-baseweb="menu"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: var(--radius) !important;
  box-shadow: var(--shadow-lg) !important;
}
[data-baseweb="popover"] [role="option"] {
  color: var(--text-primary) !important;
  font-size: 0.92em;
  padding: 8px 12px !important;
}
[data-baseweb="popover"] [role="option"]:hover {
  background: var(--accent-soft) !important;
  color: var(--accent) !important;
}

/* 드롭다운 화살표 SVG */
[data-baseweb="select"] svg {
  fill: var(--text-muted) !important;
}

/* multiselect 내부 검색 input */
.stMultiSelect input, .stSelectbox input {
  background: transparent !important;
  color: var(--text-primary) !important;
  caret-color: var(--accent) !important;
}

/* ---- Tab 좌우 스크롤 버튼 ---- */
.stTabs [data-baseweb="tab-list"] button:not([role="tab"]),
.stTabs button[data-testid="stTabsScrollButton"],
button[data-testid="stTabsScrollButton"],
[data-testid="stTabsScrollButtonContainer"] button {
  background: var(--bg-card) !important;
  color: var(--text-secondary) !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: var(--radius-sm) !important;
  transition: all var(--dur) var(--ease) !important;
}
.stTabs [data-baseweb="tab-list"] button:not([role="tab"]):hover,
.stTabs button[data-testid="stTabsScrollButton"]:hover,
button[data-testid="stTabsScrollButton"]:hover,
[data-testid="stTabsScrollButtonContainer"] button:hover {
  background: var(--accent-soft) !important;
  color: var(--accent) !important;
  border-color: var(--accent) !important;
}
.stTabs [data-baseweb="tab-list"] button svg,
button[data-testid="stTabsScrollButton"] svg {
  fill: currentColor !important;
}

/* ---- Number input -/+ stepper ---- */
.stNumberInput button,
[data-testid="stNumberInput"] button,
[data-testid="stNumberInputStepDown"],
[data-testid="stNumberInputStepUp"] {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border-subtle) !important;
  color: var(--text-secondary) !important;
  transition: all var(--dur) var(--ease) !important;
}
.stNumberInput button:hover,
[data-testid="stNumberInputStepDown"]:hover,
[data-testid="stNumberInputStepUp"]:hover {
  background: var(--accent-soft) !important;
  border-color: var(--accent) !important;
  color: var(--accent) !important;
}
.stNumberInput button svg,
[data-testid="stNumberInputStepDown"] svg,
[data-testid="stNumberInputStepUp"] svg {
  fill: currentColor !important;
}

/* ---- Primary 버튼 글씨 화이트 보장 ---- */
.stButton button[kind="primary"],
.stButton button[kind="primary"] *,
[data-testid="baseButton-primary"],
[data-testid="baseButton-primary"] * {
  color: #FFFFFF !important;
}

/* ---- Secondary 버튼 — 라이트 정련 ---- */
.stButton button:not([kind="primary"]) {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-default) !important;
  color: var(--text-primary) !important;
}
.stButton button:not([kind="primary"]):hover {
  background: var(--accent-soft) !important;
  color: var(--accent) !important;
  border-color: var(--accent) !important;
}

/* ---- Download 버튼 — secondary 톤 ---- */
[data-testid="stDownloadButton"] button,
.stDownloadButton button {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-default) !important;
  color: var(--text-primary) !important;
  transition: all var(--dur) var(--ease) !important;
}
[data-testid="stDownloadButton"] button:hover,
.stDownloadButton button:hover {
  background: var(--accent-soft) !important;
  color: var(--accent) !important;
  border-color: var(--accent) !important;
  transform: translateY(-1px);
  box-shadow: var(--shadow-sm);
}

/* ---- Radio (AI Lab/운영 sub-nav) ---- */
.stRadio label, .stRadio div { color: var(--text-primary) !important; }
.stRadio [data-baseweb="radio"] div[role="radiogroup"] label {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-sm);
  padding: 6px 14px;
  transition: all var(--dur) var(--ease);
}
.stRadio [data-baseweb="radio"] div[role="radiogroup"] label:hover {
  background: var(--accent-soft);
  border-color: var(--accent);
}

/* ---- 슬라이더 ---- */
.stSlider [data-baseweb="slider"] [role="slider"] {
  background: var(--accent) !important;
  border: 2px solid #FFFFFF !important;
  box-shadow: 0 1px 3px var(--accent-glow);
}
.stSlider [data-baseweb="slider"] > div > div:first-child > div {
  background: var(--accent) !important;
}

/* ---- Toggle 스위치 ---- */
.stCheckbox [data-baseweb="checkbox"] span:first-child {
  border-radius: 4px;
  border-color: var(--border-default) !important;
}
[data-baseweb="toggle"] {
  filter: saturate(0.85);
}

/* ---- Sidebar 메트릭 정렬 ---- */
[data-testid="stSidebar"] .stMetric {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-sm);
  padding: 10px 12px !important;
  margin-bottom: 6px;
  transition: all var(--dur) var(--ease);
}
[data-testid="stSidebar"] .stMetric:hover {
  border-color: var(--border-default);
  box-shadow: var(--shadow-sm);
}

/* ---- File uploader ---- */
[data-testid="stFileUploader"] section {
  background: var(--bg-elevated) !important;
  border: 2px dashed var(--border-default) !important;
  border-radius: var(--radius) !important;
  transition: all var(--dur) var(--ease);
}
[data-testid="stFileUploader"] section:hover {
  border-color: var(--accent) !important;
  background: var(--accent-soft) !important;
}

/* ---- Spinner ---- */
.stSpinner > div { border-color: var(--accent) transparent var(--accent) transparent !important; }

/* ---- Progress bar ---- */
.stProgress > div > div > div { background: var(--accent) !important; }
.stProgress > div > div { background: var(--bg-subtle) !important; }
</style>
"""
