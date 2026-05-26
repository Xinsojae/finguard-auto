"""Tab 3: 공시·뉴스 패널 + 뉴스 텍스트 감성 분석 (NLP 모델)."""
import streamlit as st
import matplotlib.pyplot as plt

from tabs import AppCtx


def render(ctx: AppCtx) -> None:
    st.subheader("📰 공시·뉴스 패널")

    snap = ctx.snap
    panel = ctx.panel
    options = ctx.picks if ctx.picks else snap["name"].tolist()[:5]
    sel = st.selectbox("종목 선택", options, key="disc_sel")
    if sel:
        sid = snap[snap["name"] == sel]["stock_id"].iloc[0]
        hist = panel[panel["stock_id"] == sid].tail(60).copy()
        _render_event_list(sel, hist)
        _render_sentiment_chart(sel, hist, ctx.kfont_fp)

    st.divider()
    _render_sentiment_analyzer()


def _render_event_list(sel, hist) -> None:
    events = hist[hist["disc_code"] != ""].tail(5)
    st.markdown(f"**{sel} - 최근 60일 공시 이벤트 (룰베이스 30유형 분류)**")
    if len(events) == 0:
        st.success("최근 60일 내 분류된 공시 이벤트 없음 (긍정 시그널)")
        return
    for _, e in events.iterrows():
        risk_val = int(e["disclosure"])
        if risk_val < 0:
            emoji, color = "🔴", "#C62828"
        elif risk_val > 0:
            emoji, color = "🟢", "#2E7D32"
        else:
            emoji, color = "⚪", "#666"
        st.markdown(
            f"- **{e['date'].date()}** {emoji} "
            f"<b style='color:{color};'>[{e['disc_name']}]</b> "
            f"위험도: {e['disc_risk_label']} (수익률 영향 {e['return']*100:+.1f}%)",
            unsafe_allow_html=True,
        )


def _render_sentiment_chart(sel, hist, kfont_fp) -> None:
    st.markdown(f"**최근 뉴스 감성 추세** (합성)")
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(hist["date"], hist["news_sent"], color="#1565C0", lw=1)
    ax.axhline(0, color="#999", ls="--")
    ax.fill_between(hist["date"], hist["news_sent"], 0,
                    where=hist["news_sent"] > 0, color="#2E7D32", alpha=0.2)
    ax.fill_between(hist["date"], hist["news_sent"], 0,
                    where=hist["news_sent"] < 0, color="#C62828", alpha=0.2)
    ax.set_ylabel("News Sentiment")
    ax.set_title(f"{sel} 뉴스 감성 (60일)", fontproperties=kfont_fp)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)


def _render_sentiment_analyzer() -> None:
    """뉴스 텍스트 → TF-IDF/KR-FinBERT 감성 점수."""
    st.markdown("### 📝 뉴스 텍스트 감성 분석 (NLP 모델)")
    st.caption("기획서 MVP #4 베이스라인 (TF-IDF + LogReg) + 확장 옵션 (KR-FinBERT). "
               "입력 텍스트 → 감성 점수 [-1, +1] 출력.")

    ns_col1, ns_col2 = st.columns([1, 2])
    with ns_col1:
        backend_label = st.radio(
            "NLP 모델",
            ["TF-IDF + LogReg (베이스라인)", "KR-FinBERT (HuggingFace)"],
            help="KR-FinBERT는 transformers+torch 필요. Cloud 환경에서 메모리 부족 시 "
                 "자동으로 TF-IDF로 fallback됩니다.",
        )
    with ns_col2:
        user_text = st.text_area(
            "분석할 뉴스 텍스트",
            "당사는 자사주 300만주를 매입하기로 결정하였습니다. "
            "유통주식 수 감소로 EPS 상승 효과가 기대됩니다.",
            height=120,
        )

    if not st.button("🔍 감성 분석 실행", type="primary", key="ns_run"):
        return

    backend_key = "krfinbert" if "KR-FinBERT" in backend_label else "tfidf"
    with st.spinner(f"{backend_label} 로드 및 분석 중..."):
        try:
            import news_sentiment as ns
            model, err = ns.get_sentiment_model(backend_key)
            if err:
                st.warning(err)
            texts_in = [t.strip() for t in user_text.split("\n") if t.strip()]
            if not texts_in:
                st.warning("입력 텍스트가 비어 있습니다.")
                return
            scores = model.analyze(texts_in)
            st.markdown(f"**사용 모델**: `{model.label}`")
            for t, s in zip(texts_in, scores):
                if s > 0.2:
                    color, tag = "#2E7D32", "🟢 호재"
                elif s < -0.2:
                    color, tag = "#C62828", "🔴 악재"
                else:
                    color, tag = "#9E9E9E", "⚪ 중립"
                st.markdown(
                    f"<div style='border-left:4px solid {color};padding:8px 12px;"
                    f"margin:6px 0;background:#FAFAFA;border-radius:4px;'>"
                    f"<b style='color:{color};'>{tag}</b> "
                    f"<span style='color:{color};font-weight:700;'>score {s:+.3f}</span>"
                    f"<br><small>{t}</small></div>",
                    unsafe_allow_html=True,
                )
            with st.expander("ℹ️ 모델 정보"):
                st.json(model.info())
        except Exception as e:
            st.error(f"감성 분석 실패: {e}")
