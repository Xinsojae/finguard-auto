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

# ----- 페이지 CSS -----
CSS_BLOCK = """
<style>
.metric-up   { color: #2E7D32; font-weight: 700; }
.metric-risk { color: #C62828; font-weight: 700; }
.metric-neutral { color: #666; font-weight: 700; }
.card { border: 1px solid #E0E0E0; border-radius: 8px; padding: 12px; background: #FAFAFA; }
.tag-priority { background: #2E7D32; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; }
.tag-highrisk { background: #F57C00; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; }
.tag-hold     { background: #9E9E9E; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; }
.tag-avoid    { background: #C62828; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; }
.disclaimer { background: #FFF3E0; padding: 8px; border-left: 4px solid #F57C00;
              border-radius: 4px; font-size: 0.85em; color: #666; }
</style>
"""
