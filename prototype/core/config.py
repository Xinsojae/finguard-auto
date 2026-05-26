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

# ----- 페이지 CSS (톤다운 팔레트, 눈 피로 줄임) -----
# 색상 원칙: 채도·명도 모두 낮춤. 의미 색상(녹·적·주황·청)은 유지하되 부드럽게.
CSS_BLOCK = """
<style>
/* 본문 베이스 */
.block-container { padding-top: 1.5rem; }

/* 메트릭 색상 */
.metric-up      { color: #5CB874; font-weight: 600; }
.metric-risk    { color: #E07A7A; font-weight: 600; }
.metric-neutral { color: #888;    font-weight: 600; }

/* 카드 */
.card {
  border: 1px solid #ECECEC;
  border-radius: 10px;
  padding: 16px 18px;
  background: #FCFCFC;
  margin-bottom: 8px;
}
.card h4 { margin: 0 0 6px 0; font-weight: 600; color: #333; }
.card small { color: #999; font-weight: 400; }

/* 분류 태그 (파스텔) */
.tag-priority { background: #C8E6C9; color: #2E5933; padding: 3px 10px;
                border-radius: 12px; font-size: 0.78em; font-weight: 600; }
.tag-highrisk { background: #FFE0B2; color: #7A4E13; padding: 3px 10px;
                border-radius: 12px; font-size: 0.78em; font-weight: 600; }
.tag-hold     { background: #ECEFF1; color: #546E7A; padding: 3px 10px;
                border-radius: 12px; font-size: 0.78em; font-weight: 600; }
.tag-avoid    { background: #FFCDD2; color: #8B2D2D; padding: 3px 10px;
                border-radius: 12px; font-size: 0.78em; font-weight: 600; }

/* 면책 박스 */
.disclaimer {
  background: #FFF8E1;
  padding: 10px 14px;
  border-left: 3px solid #FFCA28;
  border-radius: 6px;
  font-size: 0.85em;
  color: #6B6B6B;
  margin: 8px 0 14px 0;
}

/* divider 부드럽게 */
hr { border-top: 1px solid #EEEEEE !important; margin: 1.2rem 0 !important; }

/* 헤더 색상 통일 */
h2, h3, h4 { color: #333; font-weight: 600; }

/* 데이터프레임 폰트 살짝 작게 */
.stDataFrame { font-size: 0.92em; }
</style>
"""
