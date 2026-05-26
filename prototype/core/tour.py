"""발표 데모 투어 — 단계별 가이드.

session_state["tour_active"]로 진행. 우측 하단 floating panel에
현재 단계 + 설명 + 다음/종료 버튼.
"""
import streamlit as st


TOUR_STEPS = [
    {
        "title": "1. 환영합니다",
        "body": "FinGuard Auto는 개인투자자를 위한 설명 가능한 AI 리스크 분석 플랫폼입니다. "
                "이 투어는 약 2분 — 핵심 7개 기능을 보여드립니다.",
        "tab": None,
    },
    {
        "title": "2. 사이드바 — 검색·필터",
        "body": "왼쪽 사이드바에서 종목을 검색하거나 섹터·점수 범위로 필터링할 수 있습니다. "
                "필터링된 종목만 워치리스트 후보로 사용됩니다.",
        "tab": None,
    },
    {
        "title": "3. 종목 분석 (Tab 1)",
        "body": "선택한 종목의 상승/리스크/신뢰도 점수 + TreeSHAP 판단 근거 + "
                "가격·거래량 차트 + 룰베이스 공시 분석을 한 화면에 표시합니다.",
        "tab": "🎯 종목 분석",
    },
    {
        "title": "4. 2×2 매트릭스 (Tab 2)",
        "body": "전체 종목을 상승×리스크 평면에 산점도로 표시. "
                "우상단=우선관심, 좌상단=고위험관심, 좌하단=회피.",
        "tab": "🗺️ 매트릭스",
    },
    {
        "title": "5. 종목 비교 (Tab 3)",
        "body": "2~4종목을 나란히 비교 — 점수·가격추세·SHAP 기여도. "
                "포트폴리오 후보 선정에 활용.",
        "tab": "🔄 비교",
    },
    {
        "title": "6. 백테스트 (Tab 5)",
        "body": "Walk-forward 3-fold로 A(상승만) vs B(상승+리스크필터) 전략 비교. "
                "5일 비중첩 리밸런스, 거래비용 0.3% 반영.",
        "tab": "📈 백테스트",
    },
    {
        "title": "7. 공시 분석기 (Tab 6)",
        "body": "30개 공시 유형 룰베이스 분류 + 자연어 해석 + 투자자 체크포인트 + 유사 사례.",
        "tab": "🔍 공시 분석기",
    },
    {
        "title": "8. 모의투자 + Kill Switch (Tab 7)",
        "body": "가상 자금 1,000만원으로 자동/수동 매수. "
                "리스크 60+ 종목은 매수 차단되며, 시장 평균 리스크 70 이상이면 Kill Switch 활성.",
        "tab": "💼 모의투자",
    },
    {
        "title": "9. AI Lab (Tab 9)",
        "body": "HyperCLOVA X, KoBigBird, PatchTST, Whisper, Conformal Prediction, OCR 등 "
                "확장 모델의 DEMO MODE 시뮬레이션.",
        "tab": "🧪 AI Lab",
    },
    {
        "title": "10. 운영 (Tab 10)",
        "body": "MLflow 실험 추적, 이벤트 캘린더, 주간 리스크 리포트 (실 HTML 다운로드 가능).",
        "tab": "🛠️ 운영",
    },
    {
        "title": "✅ 투어 완료",
        "body": "지금까지 10개 탭의 핵심 기능을 살펴봤습니다. "
                "각 탭을 직접 클릭하며 슬라이더·필터를 조정해 보세요. "
                "본 프로토타입은 매수·매도 추천이 아닌 의사결정 보조용 학술 데모입니다.",
        "tab": None,
    },
]


def render_tour_widget() -> None:
    """투어 시작 버튼 + 단계별 패널."""
    if "tour_step" not in st.session_state:
        st.session_state["tour_step"] = -1  # -1 = 비활성

    step = st.session_state["tour_step"]
    total = len(TOUR_STEPS)

    if step < 0:
        # 시작 버튼만
        if st.button("📖 5분 데모 투어", key="tour_start",
                     use_container_width=True):
            st.session_state["tour_step"] = 0
            st.rerun()
        return

    # 진행 중
    cur = TOUR_STEPS[step]
    progress = (step + 1) / total
    st.markdown(
        f"<div style='background:var(--warning-soft);padding:12px 14px;"
        f"border:1px solid var(--warning);border-radius:8px;margin:8px 0;'>"
        f"<div style='display:flex;justify-content:space-between;"
        f"font-size:0.8em;color:var(--warning);margin-bottom:6px;font-weight:600;'>"
        f"<span>📖 데모 투어</span><span>{step + 1} / {total}</span></div>"
        f"<div style='font-weight:600;color:var(--text-primary);margin-bottom:6px;'>{cur['title']}</div>"
        f"<div style='color:var(--text-secondary);font-size:0.88em;line-height:1.5;'>{cur['body']}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.progress(progress)
    if cur["tab"]:
        st.caption(f"👉 상단 **{cur['tab']}** 탭을 클릭해 보세요.")

    col1, col2, col3 = st.columns(3)
    with col1:
        if step > 0 and st.button("◀ 이전", key=f"tour_prev_{step}",
                                  use_container_width=True):
            st.session_state["tour_step"] = step - 1
            st.rerun()
    with col2:
        if step < total - 1:
            if st.button("다음 ▶", key=f"tour_next_{step}", type="primary",
                         use_container_width=True):
                st.session_state["tour_step"] = step + 1
                st.rerun()
        else:
            if st.button("✓ 완료", key="tour_finish", type="primary",
                         use_container_width=True):
                st.session_state["tour_step"] = -1
                st.rerun()
    with col3:
        if st.button("✕ 종료", key=f"tour_close_{step}",
                     use_container_width=True):
            st.session_state["tour_step"] = -1
            st.rerun()
