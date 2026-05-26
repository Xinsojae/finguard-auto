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
    train_quantile_model, predict_quantile,
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
from tabs import ai_lab as tab_ai_lab
from tabs import ops as tab_ops
from tabs import compare as tab_compare
from core import mocks
from core.tour import render_tour_widget


# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(page_title="FinGuard Auto", page_icon="🛡️", layout="wide")
apply_css()

# ----- 다크모드 (CSS variables override — 고급 다크 팔레트) -----
# 영감: Linear, Vercel, Bloomberg Terminal. 미드나잇 블루-슬레이트.
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False
if st.session_state["dark_mode"]:
    st.markdown("""
    <style>
    :root {
      --bg-base:        #0A0E14;
      --bg-elevated:    #11161F;
      --bg-card:        #161C28;
      --bg-hover:       #1C2433;
      --bg-subtle:      #0F1420;

      --border-subtle:  #1F2937;
      --border-default: #2D3548;
      --border-strong:  #3F4859;

      --text-primary:   #F1F5F9;
      --text-secondary: #94A3B8;
      --text-muted:     #64748B;
      --text-inverse:   #0A0E14;

      --accent:         #60A5FA;
      --accent-hover:   #93C5FD;
      --accent-soft:    rgba(96, 165, 250, 0.12);
      --accent-glow:    rgba(96, 165, 250, 0.18);

      --success:        #34D399;
      --success-soft:   rgba(52, 211, 153, 0.12);
      --warning:        #FBBF24;
      --warning-soft:   rgba(251, 191, 36, 0.12);
      --danger:         #F87171;
      --danger-soft:    rgba(248, 113, 113, 0.12);

      --shadow-sm:      0 1px 2px rgba(0, 0, 0, 0.4);
      --shadow-md:      0 4px 16px rgba(0, 0, 0, 0.45);
      --shadow-lg:      0 16px 48px rgba(0, 0, 0, 0.55);
    }

    .stApp { background: var(--bg-base) !important; }

    /* 헤더 바 — 다크에서 더 깊이있게 */
    .app-header {
      background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #1E3A8A 100%) !important;
      box-shadow: 0 8px 28px rgba(0,0,0,0.5),
                  0 0 0 1px rgba(96,165,250,0.1) !important;
    }
    .app-header .pill {
      background: rgba(96, 165, 250, 0.15) !important;
      border: 1px solid rgba(96, 165, 250, 0.3) !important;
      color: #DBEAFE !important;
    }

    /* Streamlit 위젯 라벨/캡션 */
    label, .stSelectbox label, .stRadio label, .stSlider label,
    .stMultiSelect label, .stTextInput label, .stTextArea label,
    .stNumberInput label, .stToggle label, .stCheckbox label,
    .stFileUploader label, .stDateInput label, p, span, li {
      color: var(--text-secondary) !important;
    }
    .stCaption, [data-testid="stCaptionContainer"] {
      color: var(--text-muted) !important;
    }

    /* Selectbox/Multiselect/Radio 다크 */
    .stSelectbox > div > div, .stMultiSelect > div > div {
      background: var(--bg-card) !important;
      border-color: var(--border-default) !important;
      color: var(--text-primary) !important;
    }

    /* DataFrame 다크 */
    .stDataFrame, .stDataFrame table {
      background: var(--bg-card) !important;
      color: var(--text-primary) !important;
    }
    .stDataFrame [data-testid="stDataFrameResizable"] {
      background: var(--bg-card) !important;
    }
    .stDataFrame th {
      background: var(--bg-elevated) !important;
      color: var(--text-primary) !important;
      border-color: var(--border-subtle) !important;
    }
    .stDataFrame td {
      color: var(--text-secondary) !important;
      border-color: var(--border-subtle) !important;
    }

    /* st.info/warning/success */
    .stAlert {
      background: var(--bg-card) !important;
      border-color: var(--border-subtle) !important;
    }
    .stAlert p { color: var(--text-primary) !important; }

    /* 코드 블록 */
    code, pre {
      background: var(--bg-elevated) !important;
      color: var(--accent-hover) !important;
      border: 1px solid var(--border-subtle) !important;
    }

    /* 슬라이더 핸들 */
    .stSlider [data-baseweb="slider"] [role="slider"] {
      background: var(--accent) !important;
      border-color: var(--accent-hover) !important;
    }

    /* ===== 입력 위젯 — 사이드바: 더 어두운 채도 + 흰 글씨 ===== */
    /* 종목명 검색 (text input) */
    [data-testid="stSidebar"] .stTextInput input {
      background: #060A12 !important;
      border: 1px solid #1A2030 !important;
      color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] .stTextInput input::placeholder {
      color: #6B7280 !important;
    }

    /* 섹터 필터 / 관심 종목 (multiselect) — closed state */
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] > div,
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
      background: #060A12 !important;
      border: 1px solid #1A2030 !important;
      color: #FFFFFF !important;
    }

    /* 비교할 종목 (main 영역 multiselect) */
    .main .stMultiSelect [data-baseweb="select"] > div,
    .main .stSelectbox [data-baseweb="select"] > div {
      background: #0B1018 !important;
      border: 1px solid #1F2937 !important;
      color: #FFFFFF !important;
    }

    /* 선택된 multiselect 태그 (chip) - 사이드바·본문 공통 */
    [data-baseweb="tag"] {
      background: #1A2030 !important;
      border: 1px solid #2D3548 !important;
      color: #FFFFFF !important;
    }
    [data-baseweb="tag"] span,
    [data-baseweb="tag"] div { color: #FFFFFF !important; }
    [data-baseweb="tag"] svg { fill: #FFFFFF !important; }

    /* 라벨 — 관심 종목 / 비교할 종목 / 종목명 검색 / 섹터 필터 — 흰색 */
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stSelectbox label,
    .main .stMultiSelect label,
    .main .stSelectbox label {
      color: #FFFFFF !important;
      font-weight: 500 !important;
    }

    /* 드롭다운 옵션 리스트 */
    [data-baseweb="popover"] [data-baseweb="menu"] {
      background: #11161F !important;
      border: 1px solid #2D3548 !important;
    }
    [data-baseweb="popover"] [role="option"] {
      color: #E2E8F0 !important;
    }
    [data-baseweb="popover"] [role="option"]:hover {
      background: #1A2030 !important;
      color: #FFFFFF !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 상단 헤더 바 (프로덕션 스타일)
st.markdown(
    """
    <div class='app-header'>
      <div>
        <h1>🛡️ FinGuard Auto</h1>
        <div class='subtitle'>개인투자자를 위한 설명 가능한 AI 리스크 분석·투자 학습·모의 검증 플랫폼</div>
      </div>
      <div class='right'>
        <span class='pill'>v0.7 prototype</span>
        <span class='pill'>학술 데모</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


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
    q_model = train_quantile_model(panel, alpha=0.10)
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
    # Quantile 손실 구간 (§11.2 마지막 요소)
    snap["var_5d_p10"] = predict_quantile(q_model, snap)


# ============================================================
# 알림 (st.toast) — 첫 진입 1회 노출
# ============================================================
if "alerts_shown" not in st.session_state:
    for alert in mocks.generate_alerts(snap)[:4]:
        st.toast(f"{alert['icon']} {alert['msg']}")
    st.session_state["alerts_shown"] = True


# ============================================================
# 사이드바 — 워치리스트 + 시장 요약 + 모델 성능
# ============================================================
mkt_risk = int(snap["score_risk"].mean())
mkt_label = "낮음" if mkt_risk < 35 else "중간" if mkt_risk < 55 else "높음"

with st.sidebar:
    # ----- 데모 투어 + 테마 -----
    render_tour_widget()
    theme_col1, theme_col2 = st.columns([3, 1])
    with theme_col1:
        st.caption("🎨 테마")
    with theme_col2:
        new_dark = st.toggle("🌙", value=st.session_state.get("dark_mode", False),
                             key="theme_toggle", help="다크모드")
        if new_dark != st.session_state.get("dark_mode"):
            st.session_state["dark_mode"] = new_dark
            st.rerun()
    st.divider()

    # ----- 검색·필터 (워치리스트 위) -----
    st.subheader("🔍 검색 · 필터")
    search_q = st.text_input("종목명 검색", "", placeholder="예: 삼성, 하이닉스",
                             key="sb_search")
    sector_options = sorted(snap["sector"].dropna().unique().tolist())
    sector_filter = st.multiselect("섹터 필터", sector_options,
                                   default=[], key="sb_sector")
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        min_up = st.slider("상승 ≥", 0, 100, 0, step=10, key="sb_min_up")
    with f_col2:
        max_risk = st.slider("리스크 ≤", 0, 100, 100, step=10, key="sb_max_risk")

    # 필터 적용된 후보 풀
    filtered = snap.copy()
    if search_q.strip():
        q = search_q.strip().lower()
        filtered = filtered[filtered["name"].str.lower().str.contains(q)]
    if sector_filter:
        filtered = filtered[filtered["sector"].isin(sector_filter)]
    filtered = filtered[(filtered["score_up"] >= min_up)
                        & (filtered["score_risk"] <= max_risk)]
    st.caption(f"필터 결과: **{len(filtered)}** / {len(snap)}종목")

    st.divider()
    st.subheader("📋 워치리스트")
    pool = filtered["name"].tolist() if not filtered.empty else snap["name"].tolist()
    default_picks = (snap.nlargest(8, "score_up").head(5)["name"].tolist()
                     + snap.nlargest(5, "score_risk").head(2)["name"].tolist()
                     + [snap.iloc[0]["name"]])
    # default는 풀에 있는 것만
    default_picks = [n for n in dict.fromkeys(default_picks) if n in pool][:6]
    picks = st.multiselect("관심 종목", pool, default=default_picks)
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
    st.divider()
    # ----- 알림 (사이드바 상단 영역) -----
    alerts = mocks.generate_alerts(snap)
    st.subheader(f"🔔 알림 ({len(alerts)})")
    if alerts:
        for a in alerts[:5]:
            st.markdown(f"<div style='padding:6px 10px;background:#FAFAFA;"
                        f"border-left:3px solid #FFB74D;border-radius:4px;"
                        f"margin:4px 0;font-size:0.85em;'>"
                        f"{a['icon']} {a['msg']}</div>",
                        unsafe_allow_html=True)
    else:
        st.caption("주요 알림 없음.")


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

t1, t2, t3, t4, t5, t6, t7, t8, t9, t10 = st.tabs([
    "🎯 종목 분석", "🗺️ 매트릭스", "🔄 비교", "📰 공시·뉴스",
    "📈 백테스트", "🔍 공시 분석기", "💼 모의투자", "📊 포트폴리오",
    "🧪 AI Lab", "🛠️ 운영",
])
with t1: tab_stocks.render(ctx)
with t2: tab_matrix.render(ctx)
with t3: tab_compare.render(ctx)
with t4: tab_news.render(ctx)
with t5: tab_backtest.render(ctx)
with t6: tab_disclosure.render(ctx)
with t7: tab_paper.render(ctx)
with t8: tab_portfolio.render(ctx)
with t9: tab_ai_lab.render(ctx)
with t10: tab_ops.render(ctx)

st.divider()
st.caption("FinGuard Auto · AI 개론 프로젝트 · 2026.05 · 본 프로토타입은 합성 데이터 기반 학술 데모입니다.")
