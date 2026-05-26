"""FinGuard Auto - Streamlit 대시보드 진입점.

실행:
    pip install -r requirements.txt
    streamlit run streamlit_app.py

구조:
    streamlit_app.py       → 페이지 설정 + 사이드바 + 데이터 로드 + 탭 dispatch
    core/                  → 데이터·피처·모델·UI 유틸 (재사용 가능)
    tabs/                  → 5개 탭별 render() (개별 모듈)
    disclosure_analyzer.py → 30유형 룰베이스 공시 분류
    news_sentiment.py      → TF-IDF + KR-FinBERT 뉴스 감성
    font_setup.py          → 한글 폰트 (번들 NanumGothic)
"""
import streamlit as st

# 한글 폰트 등록 (matplotlib rcParams 자동 설정)
from font_setup import KFONT_FP  # noqa: F401

from core import (
    FEATS, REAL_PANEL_PATH,
    gen_panel, load_real_panel_bundled, inject_disclosure_signals,
    latest_snapshot, make_features, train_models,
    train_anomaly_detector, score_snapshot,
    compute_confidence_for_snap,
    classify, apply_css,
)
from tabs import AppCtx
from tabs import stocks as tab_stocks
from tabs import matrix as tab_matrix
from tabs import news as tab_news
from tabs import backtest as tab_backtest
from tabs import disclosure as tab_disclosure
from tabs import paper as tab_paper
from tabs import portfolio as tab_portfolio


# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(page_title="FinGuard Auto", page_icon="🛡️", layout="wide")
apply_css()

st.title("🛡️ FinGuard Auto")
st.caption("개인투자자를 위한 설명 가능한 AI 리스크 분석·투자 학습·모의 검증 플랫폼")


# ============================================================
# 데이터 모드 선택
# ============================================================
_real_available = REAL_PANEL_PATH.exists()
_default_mode_idx = 0 if _real_available else 1
data_mode = st.radio(
    "데이터 모드",
    ["🇰🇷 실데이터 (KOSPI 시총 상위 100, 2021~2025)", "🎲 합성 데이터 (빠른 데모)"],
    index=_default_mode_idx, horizontal=True,
    help="실데이터: FinanceDataReader로 사전 다운로드한 KOSPI 99종목 일봉. "
         "합성: GARCH형 패널 (속도 우선).",
)
USE_REAL = data_mode.startswith("🇰🇷")
if USE_REAL and not _real_available:
    st.error(f"실데이터 파일 없음: {REAL_PANEL_PATH}. 합성 모드로 자동 전환.")
    USE_REAL = False


# ============================================================
# 사이드바 — 고급 설정 (데이터 로드 전)
# ============================================================
with st.sidebar:
    with st.expander("⚙️ 고급 설정 (인터랙티브)", expanded=False):
        if not USE_REAL:
            CFG_N_STOCKS = st.slider("합성 종목 수", 50, 200, 120, step=10)
            CFG_N_DAYS = st.slider("합성 일자 수", 300, 1000, 600, step=50)
        else:
            CFG_N_STOCKS = None
            CFG_N_DAYS = None
            st.caption("실데이터 모드: 종목·일자 고정 (99종목 × 1224일)")
        st.divider()
        st.caption("**분류 임계값** (2×2 매트릭스)")
        CFG_UP_TH = st.slider("우선 관심 상승 점수 기준", 30, 80, 50, step=5)
        CFG_RISK_TH = st.slider("위험 분류 점수 기준", 30, 80, 50, step=5)
        st.divider()
        st.caption("**백테스트 파라미터**")
        CFG_K_TOP = st.slider("리밸런스 picks (k_top)", 5, 50, 20, step=5)
        CFG_RISK_PCT = st.slider("리스크 필터 분위 (B 전략)", 0.50, 0.95, 0.70, step=0.05)
        CFG_HOLD = st.slider("보유 일수 (비중첩)", 3, 20, 5, step=1)


# ============================================================
# 면책 / 데이터 모드 안내
# ============================================================
if USE_REAL:
    st.markdown(
        "<div class='disclaimer'>⚠️ 실데이터 모드: KOSPI 시총 상위 99종목(2021-01-04 ~ 2025-12-31). "
        "공시·뉴스 NLP 피처는 룰베이스 분류기로 주입(KF-DeBERTa는 확장 계획). "
        "본 분석은 매수·매도 추천이 아닌 의사결정 보조이며, 최종 판단·책임은 사용자에게 있습니다.</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"<div class='disclaimer'>⚠️ 합성 데이터 모드: GARCH형 패널 "
        f"({CFG_N_STOCKS}종목 × {CFG_N_DAYS}일). "
        "실제 매수·매도 추천이 아닙니다. 최종 투자 판단·책임은 사용자에게 있습니다.</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# 데이터 로드 + 모델 학습 + 스냅샷
# ============================================================
with st.spinner(("실데이터 로드" if USE_REAL else "합성 데이터 생성")
                + " + 공시 분류 주입 + 모델 학습 중..."):
    if USE_REAL:
        panel = load_real_panel_bundled()
    else:
        panel = gen_panel(n_stocks=CFG_N_STOCKS, n_days=CFG_N_DAYS)
    panel = inject_disclosure_signals(panel, n_events_per_stock=4)
    panel = make_features(panel)
    m_up, m_cr, metrics = train_models(panel)
    iso_model = train_anomaly_detector(panel)
    snap, latest_date = latest_snapshot(panel)
    snap["score_up_p"] = m_up.predict_proba(snap[FEATS])[:, 1]
    snap["score_cr_p"] = m_cr.predict_proba(snap[FEATS])[:, 1]
    snap["score_up"] = (snap["score_up_p"] * 100).round(0).astype(int)
    snap["score_risk"] = (snap["score_cr_p"] * 100).round(0).astype(int)
    snap["category"] = snap.apply(
        lambda r: classify(r["score_up"], r["score_risk"], CFG_UP_TH, CFG_RISK_TH),
        axis=1)
    # AI 신뢰도 5요소 (§11.4)
    conf_df = compute_confidence_for_snap(
        snap=snap, panel=panel, model_auc=metrics["up_auc"])
    for col in conf_df.columns:
        snap[col] = conf_df[col].values
    # 기존 confidence 컬럼 호환 유지 (rank 기반 → 5요소 overall로 교체)
    snap["confidence"] = snap["confidence_overall"]
    # Isolation Forest 이상 탐지 (§10.2 MVP #3)
    snap["anomaly_score"] = score_snapshot(iso_model, snap)


# ============================================================
# 사이드바 — 워치리스트 + 시장 요약 + 모델 성능
# ============================================================
mkt_risk = int(snap["score_risk"].mean())
mkt_label = "낮음" if mkt_risk < 35 else "중간" if mkt_risk < 55 else "높음"

with st.sidebar:
    st.subheader("📋 워치리스트")
    default_picks = (snap.nlargest(8, "score_up").head(5)["name"].tolist()
                     + snap.nlargest(5, "score_risk").head(2)["name"].tolist()
                     + [snap.iloc[0]["name"]])
    picks = st.multiselect(
        "관심 종목", snap["name"].tolist(),
        default=list(dict.fromkeys(default_picks))[:6])
    st.divider()
    st.subheader("🌡️ 오늘의 시장")
    st.metric("기준일", str(latest_date.date()))
    st.metric("시장 위험도", f"{mkt_risk}/100 ({mkt_label})")
    st.metric("우선 관심 후보 수", int((snap["category"] == "PRIORITY").sum()))
    st.metric("회피 후보 수", int((snap["category"] == "AVOID").sum()))
    n_anom = int((snap["anomaly_score"] >= 70).sum())
    st.metric("이상치 의심 종목", f"{n_anom} (Isolation Forest)",
              help="anomaly_score ≥ 70. 거래량·변동성·드로다운 패턴이 평소와 다른 종목.")
    st.divider()
    st.subheader("📊 모델 성능 (out-of-time)")
    st.metric("상승 ROC-AUC", f"{metrics['up_auc']:.3f}")
    st.metric("급락 ROC-AUC", f"{metrics['cr_auc']:.3f}")
    st.caption("PoC 베이스라인. KF-DeBERTa 추가 시 향상 예상.")


# ============================================================
# 컨텍스트 생성 + 탭 dispatch
# ============================================================
ctx = AppCtx(
    panel=panel, snap=snap, latest_date=latest_date,
    m_up=m_up, m_cr=m_cr, metrics=metrics,
    picks=picks, use_real=USE_REAL,
    up_th=CFG_UP_TH, risk_th=CFG_RISK_TH,
    k_top=CFG_K_TOP, hold_days=CFG_HOLD, risk_pct=CFG_RISK_PCT,
    kfont_fp=KFONT_FP,
)

t1, t2, t3, t4, t5, t6, t7 = st.tabs([
    "🎯 종목 분석", "🗺️ 2×2 매트릭스", "📰 공시·뉴스",
    "📈 백테스트", "🔍 공시 분석기", "💼 모의투자", "📊 포트폴리오",
])
with t1: tab_stocks.render(ctx)
with t2: tab_matrix.render(ctx)
with t3: tab_news.render(ctx)
with t4: tab_backtest.render(ctx)
with t5: tab_disclosure.render(ctx)
with t6: tab_paper.render(ctx)
with t7: tab_portfolio.render(ctx)

st.divider()
st.caption("FinGuard Auto · AI 개론 프로젝트 · 2026.05 · 본 프로토타입은 합성 데이터 기반 학술 데모입니다.")
