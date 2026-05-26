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
    # 실데이터 모드는 news_sent 모두 0 (placeholder) — 의미 없는 직선 대신 안내
    sent_var = float(hist["news_sent"].std() or 0)
    if sent_var < 1e-6:
        st.info(
            f"**{sel} 뉴스 감성 추세** — 실데이터 모드에서는 뉴스 텍스트 수집·KF-DeBERTa "
            "감성 분석이 아직 미연동입니다. 아래 '뉴스 텍스트 감성 분석' 박스에서 "
            "직접 입력한 뉴스로 TF-IDF/KR-FinBERT 점수를 확인할 수 있습니다."
        )
        return
    st.markdown(f"**{sel} 최근 60일 뉴스 감성 추세** (합성)")
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.plot(hist["date"], hist["news_sent"], color="#5B8DEF", lw=1.2)
    ax.axhline(0, color="#BDBDBD", ls="--", lw=0.8)
    ax.fill_between(hist["date"], hist["news_sent"], 0,
                    where=hist["news_sent"] > 0, color="#81C784", alpha=0.25)
    ax.fill_between(hist["date"], hist["news_sent"], 0,
                    where=hist["news_sent"] < 0, color="#E57373", alpha=0.25)
    ax.set_ylabel("Sentiment", color="#666")
    ax.tick_params(colors="#666")
    for s in ax.spines.values():
        s.set_color("#E0E0E0")
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
                    f"<div style='padding:10px 14px 10px 18px;"
                    f"margin:6px 0;background:var(--bg-elevated);"
                    f"border:1px solid var(--border-subtle);"
                    f"border-left:4px solid {color};"
                    f"border-radius:8px;'>"
                    f"<b style='color:{color};'>{tag}</b> "
                    f"<span style='color:{color};font-weight:700;'>score {s:+.3f}</span>"
                    f"<br><small style='color:var(--text-secondary);'>{t}</small></div>",
                    unsafe_allow_html=True,
                )
            with st.expander("ℹ️ 모델 정보"):
                st.json(model.info())
        except Exception as e:
            st.error(f"감성 분석 실패: {e}")
