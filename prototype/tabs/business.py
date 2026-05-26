"""Tab: 사업·운영 — §18 + §20.3 + §21.2 통합.

발표 평가의 3대 영역 (사업화 / 운영 비용 / 사회적 가치) 단일 화면.
"""
import streamlit as st
import pandas as pd

from core.ui_kit import section_header, info_card, demo_badge
from tabs import AppCtx


def render(ctx: AppCtx) -> None:
    st.subheader("🏢 사업·운영")
    st.markdown(
        f"기획서 §18 사업화 · §20.3 운영 비용 · §21.2 사회적 가치 통합. "
        f"{demo_badge('수치 추정')}",
        unsafe_allow_html=True,
    )

    sub = st.radio(
        "섹션",
        ["💳 구독·애드온·크레딧 (§18)",
         "💰 운영 비용 추정 (§20.3)",
         "🌍 사회적 가치 정량 (§21.2)"],
        horizontal=True,
    )

    if sub.startswith("💳"):
        _render_pricing()
    elif sub.startswith("💰"):
        _render_ops_cost()
    elif sub.startswith("🌍"):
        _render_social_value()


# ============================================================
def _render_pricing() -> None:
    section_header("사업 포지셔닝 + 수익 모델",
                   "B2C 구독 + 애드온 + 크레딧 + 일회성 리포트 + B2B 화이트라벨",
                   icon="💳")
    info_card(
        "사업 포지셔닝 (§18.1)",
        "FinGuard Auto는 종목 추천이 아닌 <b>리스크 분석·투자 학습·모의 검증</b> 플랫폼. "
        "사용자는 '수익 보장'이 아닌 <b>'큰 손실 회피'와 '판단 근거 이해'</b>에 결제.",
        color="#5B8DEF",
    )

    st.markdown("### 1. 개인 구독 플랜 (8단)")
    sub = pd.DataFrame([
        ["Free", "0원", "체험 사용자", "기본 점수·매트릭스"],
        ["Starter", "29,000원", "초보 투자자", "+ 공시 분석·뉴스 감성"],
        ["Pro", "79,000원", "적극적 개인투자자", "+ SHAP 설명·신뢰도 분해"],
        ["Premium", "149,000원", "단기·스윙 투자자", "+ 백테스트·모의투자"],
        ["Quant", "299,000원", "전략 검증 사용자", "+ 고급 백테스트·전략 비교"],
        ["Flagship", "590,000원", "파워유저·소규모 팀", "+ 포트폴리오 Risk·API Lite"],
        ["Flagship Plus", "990,000원", "투자 동아리·운용팀", "+ 팀 워크스페이스·리포트"],
        ["Ultra Lab", "1,990,000원", "프로슈머 팀·고급 워크스페이스", "+ 전 기능·우선 지원"],
    ], columns=["플랜", "월 가격", "대상", "주요 기능"])
    st.dataframe(sub, use_container_width=True, hide_index=True)

    st.markdown("### 2. 기능별 애드온")
    addon = pd.DataFrame([
        ["뉴스·공시 Pro", "+39,000원", "KF-DeBERTa 감성·신뢰도"],
        ["포트폴리오 Risk", "+69,000원", "VaR·상관관계·기여도"],
        ["Backtest Lab", "+99,000원", "고급 전략 백테스트"],
        ["AI 설명 카드 Pro", "+49,000원", "HyperCLOVA X 자연어"],
        ["이벤트 캘린더 Pro", "+39,000원", "실시간 알림·필터"],
        ["Quant Screener", "+149,000원", "대량 종목 스크리닝"],
        ["API Lite", "+199,000원", "외부 시스템 연동"],
        ["리포트 자동 생성", "+99,000원", "주간/월간 HTML"],
    ], columns=["애드온", "월 가격", "주요 기능"])
    st.dataframe(addon, use_container_width=True, hide_index=True)

    st.markdown("### 3. 크레딧 시스템 (무거운 연산 종량제)")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**크레딧 패키지**")
        cr_pkg = pd.DataFrame([
            ["10크레딧", "29,000원"], ["30크레딧", "79,000원"],
            ["100크레딧", "199,000원"], ["300크레딧", "499,000원"],
            ["1,000크레딧", "1,490,000원"], ["3,000크레딧", "3,990,000원"],
        ], columns=["패키지", "가격"])
        st.dataframe(cr_pkg, use_container_width=True, hide_index=True)
    with cc2:
        st.markdown("**크레딧 소모 예시**")
        cr_use = pd.DataFrame([
            ["종목 상세 분석", 1], ["공시 상세 해석", 1],
            ["뉴스 묶음 분석", 2], ["유사 사례 검색", 3],
            ["포트폴리오 진단", 10], ["전략 백테스트", 10],
            ["고급 전략 비교", 30], ["월간 종합 리포트", 50],
            ["대량 종목 스크리닝", 100],
        ], columns=["기능", "소모 크레딧"])
        st.dataframe(cr_use, use_container_width=True, hide_index=True)

    st.markdown("### 4. 일회성 리포트")
    rep = pd.DataFrame([
        ["종목 10개 위험 진단", "99,000원"],
        ["종목 30개 스크리닝", "199,000원"],
        ["포트폴리오 진단 Basic", "199,000원"],
        ["포트폴리오 진단 Pro", "499,000원"],
        ["전략 백테스트 리포트", "399,000원"],
        ["고급 전략 검증 패키지", "990,000원"],
        ["월간 리스크 종합 리포트", "790,000원"],
        ["투자 학습 패키지", "299,000원"],
    ], columns=["상품", "가격"])
    st.dataframe(rep, use_container_width=True, hide_index=True)

    st.markdown("### 5. B2B 모델")
    b2b = pd.DataFrame([
        ["Education Basic", "월 300만원", "대학·동아리"],
        ["Education Pro", "월 700만원", "금융 교육기관"],
        ["API Basic", "월 500만원", "핀테크 스타트업"],
        ["API Pro", "월 1,500만원", "중견 금융사"],
        ["API Enterprise", "월 3,000만원+", "증권사·자산운용사"],
        ["White Label 구축", "1억~3억원 + 월 유지비", "맞춤 브랜드"],
        ["Enterprise Custom", "연 3억~10억원+", "대형 금융기관"],
    ], columns=["상품", "가격", "대상"])
    st.dataframe(b2b, use_container_width=True, hide_index=True)


# ============================================================
def _render_ops_cost() -> None:
    section_header("운영 비용 추정 (월 기준)",
                   "기획서 §20.3 — 사용자 규모 1,000 가정. 실제는 사용량에 따라 변동.",
                   icon="💰")

    cost = pd.DataFrame([
        ["서버 (FastAPI + PostgreSQL)", "10~30만원", "AWS t3.medium + RDS small"],
        ["캐시 (Redis)", "3~5만원", "ElastiCache small"],
        ["데이터 수집 (KRX·DART)", "0원", "무료 공개 API"],
        ["뉴스 API (네이버 검색)", "0~10만원", "무료 한도 초과 시"],
        ["LLM API (HyperCLOVA X)", "30~150만원", "종목당 일 1회 × 1,000종목"],
        ["GPU (모델 학습·재학습)", "10~30만원", "spot 인스턴스"],
        ["저장소 (S3)", "1~3만원", "데이터·모델 버전"],
        ["모니터링 (Prometheus + Grafana)", "1~3만원", "self-hosted"],
        ["로깅·에러추적 (Sentry)", "0~5만원", "free tier 가능"],
    ], columns=["항목", "추정 비용", "비고"])
    st.dataframe(cost, use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("최소 (월)", "약 55만원", help="GPU·LLM 절약 시")
    c2.metric("평균 (월)", "약 130만원", help="중간 사용량")
    c3.metric("최대 (월)", "약 230만원", help="LLM·GPU 풀 활용")

    info_card(
        "LLM 비용 절감 전략 (§20.3 주의사항)",
        "설명 카드를 매 호출마다 실시간 생성하면 비용 폭증. "
        "<b>캐싱 + 배치 생성 + 점수 변화 시에만 재생성</b> 전략으로 <b>90% 이상 절감</b> 가능. "
        "현재 프로토타입은 if-else 템플릿으로 LLM 호출 0건 (DEMO).",
        color="#FFB74D",
    )

    st.markdown("### 손익분기 (BEP) 추정")
    bep = pd.DataFrame([
        ["Free 100% (무료만)", "월 0원", "❌ 적자"],
        ["Pro 100명", "월 7,900,000원", "✅ 충분 흑자"],
        ["Pro 50명 + Premium 20명", "월 6,930,000원", "✅ 흑자"],
        ["Pro 30명 + 크레딧 10건/일", "월 4,500,000원", "✅ 가능"],
    ], columns=["시나리오", "월 매출", "운영비 130만 대비"])
    st.dataframe(bep, use_container_width=True, hide_index=True)


# ============================================================
def _render_social_value() -> None:
    section_header("사회적 가치 정량 추정",
                   "기획서 §21.2 — 보수적 가정 기반 사회적 손실 절감 잠재력",
                   icon="🌍")

    info_card(
        "산출 공식",
        "<pre style='margin:0;font-family:Consolas,monospace;font-size:0.92em;line-height:1.6;'>"
        "한국 개인투자자 (실질):     약 1,400만 명\n"
        "× 본 서비스 도달률 (5년 내):  3%  =  42만 명\n"
        "× 평균 연간 거래 손실 추정: 100만원/인\n"
        "× 본 시스템 위험 회피 효과:  10% (PoC 기반 보수 가정)\n"
        "───────────────────────────────────────────────\n"
        "<b>연간 잠재 손실 절감 추정:   ≈ 420억원</b>"
        "</pre>",
        color="#34D399",
    )

    st.markdown("### 가정별 시나리오 분석")
    scn = pd.DataFrame([
        ["보수", "1%", "14만 명", "50만원", "5%", "35억원"],
        ["기본", "3%", "42만 명", "100만원", "10%", "420억원"],
        ["낙관", "5%", "70만 명", "150만원", "15%", "1,575억원"],
        ["적극", "10%", "140만 명", "200만원", "20%", "5,600억원"],
    ], columns=["시나리오", "도달률", "사용자 수", "평균 손실", "회피율", "연간 절감"])
    st.dataframe(scn, use_container_width=True, hide_index=True)

    st.markdown("### 정성적 사회적 기여 (§21.2)")
    cols = st.columns(2)
    with cols[0]:
        info_card("개인투자자 정보 비대칭 완화",
                  "공시·뉴스·시장지표를 통합 해석 → 정보 격차 축소.", color="#5B8DEF")
        info_card("무분별한 추격매수 방지",
                  "급등 + 리스크 동시 평가 (2×2 매트릭스) → 단순 모멘텀 추종 억제.",
                  color="#5B8DEF")
        info_card("설명 가능한 금융 AI 제공",
                  "TreeSHAP + HyperCLOVA X (확장) → 블랙박스 AI 신뢰도 문제 해소.",
                  color="#5B8DEF")
    with cols[1]:
        info_card("투자 교육 효과",
                  "공시 30유형 분류 + 자연어 해석 → 초보 투자자 학습 도구.",
                  color="#34D399")
        info_card("책임 있는 자동화 방향 제시",
                  "Kill Switch + 모의투자 한정 → '자동매매로 돈 벌기' 위험 회피.",
                  color="#34D399")
        info_card("규제 친화 모델",
                  "자본시장법 §101 회피 (불특정 다수 정보 제공 구조) → "
                  "리딩방·유사투자자문업 대안.",
                  color="#34D399")

    st.caption("※ 본 수치는 보수적 가정 기반 잠재력 추정이며, "
               "실제 효과는 사용자 행동·시장 환경에 따라 다를 수 있음.")
