"""Tab 5: 공시 분석기 (30유형 룰베이스 분류 + 해석 + 체크포인트)."""
import streamlit as st

from tabs import AppCtx


def render(ctx: AppCtx) -> None:
    try:
        import disclosure_analyzer as da
    except Exception as e:
        st.error(f"disclosure_analyzer 모듈 로드 실패: {e}")
        return

    st.subheader("🔍 공시 분석기 - 30개 이벤트 유형 자동 분류 + 쉬운 해석")
    st.caption("OpenDART 공시를 30개 유형으로 분류하고 위험도·체크포인트·유사 사례를 제공합니다. "
               "발표 데모용 Mock 데이터로 작동, 실데이터 전환은 disclosure_analyzer.py 가이드 참조.")

    mode = st.radio("입력 방식",
                    ["📋 Mock 공시 목록에서 선택", "✍️ 텍스트 직접 입력"],
                    horizontal=True)

    if mode == "📋 Mock 공시 목록에서 선택":
        title, body, corp, date = _input_from_mock(da)
    else:
        title, body, corp, date = _input_from_text()

    if st.button("🔍 분석하기", type="primary"):
        _run_analysis(da, title, body)

    st.markdown("---")
    _render_classifier_stats(da)


def _input_from_mock(da):
    mock = da.load_mock_disclosures(n_days=20)
    colA, colB = st.columns([1, 2])
    with colA:
        cname = st.selectbox("기업 선택",
                             ["전체"] + sorted(mock["corp_name"].unique().tolist()))
        view = mock if cname == "전체" else mock[mock["corp_name"] == cname]
        st.caption(f"총 {len(view)}건")
        view_disp = view.copy()
        view_disp["label"] = view_disp.apply(
            lambda r: f"[{r['rcept_dt']}] {r['corp_name']} - {r['report_nm']}", axis=1)
        pick = st.selectbox("공시 선택", view_disp["label"].tolist())
        row = view_disp[view_disp["label"] == pick].iloc[0]
    with colB:
        st.markdown(f"#### 📄 {row['corp_name']} ({row['rcept_dt']})")
        st.markdown(f"**제목**: {row['report_nm']}")
        st.markdown("**본문**:")
        st.text_area("", row["report_body"], height=120,
                     label_visibility="collapsed", key="body_view")
    return row["report_nm"], row["report_body"], row["corp_name"], row["rcept_dt"]


def _input_from_text():
    corp = st.text_input("기업명 (선택)", "삼성전자")
    date = st.date_input("공시일").strftime("%Y-%m-%d")
    title = st.text_input("공시 제목", "주요사항보고서(유상증자결정)")
    body = st.text_area(
        "공시 본문",
        "당사는 운영자금 확보를 위해 주주배정 후 실권주 일반공모 방식의 유상증자를 결정하였습니다. "
        "발행주식 1,000,000주, 발행가액 12,000원, 시설투자 자금으로 사용 예정.",
        height=140,
    )
    return title, body, corp, date


def _run_analysis(da, title, body) -> None:
    with st.spinner("공시 분류 중..."):
        full_text = (title or "") + " " + (body or "")
        results = da.classify(full_text)

    if not results:
        st.warning("매칭되는 공시 유형이 없습니다.")
        return

    top = results[0]
    st.markdown("---")
    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        st.markdown(f"### {top.name}")
        st.markdown(f"**카테고리**: {top.category}")
    with c2:
        st.markdown(
            f"<h3 style='color:{top.risk_color};margin:0;'>위험도: {top.risk_label}</h3>",
            unsafe_allow_html=True)
        st.markdown(f"**신뢰도**: {top.confidence:.0%}")
    with c3:
        st.markdown("**매칭 키워드**:")
        st.markdown(", ".join([f"`{k}`" for k in top.matched_keywords]))

    st.markdown("### 📖 쉬운 해석")
    st.info(top.explanation)

    st.markdown("### ✅ 투자자 체크포인트")
    for i, cp in enumerate(top.checkpoints, 1):
        st.markdown(f"{i}. {cp}")

    if top.similar_cases:
        st.markdown("### 📚 유사 과거 사례")
        for sc in top.similar_cases:
            st.markdown(f"- {sc}")

    if len(results) > 1:
        with st.expander(f"🔎 추가 매칭 {len(results)-1}건 보기"):
            for r in results[1:]:
                st.markdown(
                    f"**{r.name}** ({r.category}) - 위험도: "
                    f"<span style='color:{r.risk_color};font-weight:700;'>{r.risk_label}</span>"
                    f" / 신뢰도: {r.confidence:.0%}",
                    unsafe_allow_html=True)
                st.caption(f"매칭: {', '.join(r.matched_keywords)}")


def _render_classifier_stats(da) -> None:
    st.markdown("### 📊 분류기 작동 통계 (Mock 데이터)")
    mock_all = da.load_mock_disclosures(n_days=20)
    stats = {}
    for _, r in mock_all.iterrows():
        res = da.classify(r["report_nm"] + " " + r["report_body"])
        key = "미분류" if not res else res[0].risk_label
        stats[key] = stats.get(key, 0) + 1
    cols = st.columns(4)
    items = list(stats.items())
    for i, (label, n) in enumerate(items):
        cols[i % 4].metric(label, n)
    total = len(mock_all)
    classified = total - stats.get("미분류", 0)
    coverage = classified / total if total > 0 else 0
    st.caption(f"분류 커버리지: {coverage:.0%} ({classified}/{total}건)")
