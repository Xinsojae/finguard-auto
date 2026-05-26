"""Tab: AI Lab — 확장 모델 데모 모음.

mocks: HyperCLOVA X, KoBigBird, PatchTST, Whisper, Conformal, OCR.
각 출력은 DEMO MODE 배지로 명시.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from core import mocks
from core.ui_kit import demo_badge, section_header, info_card
from core.plotly_theme import layout_kwargs, palette
from tabs import AppCtx


def render(ctx: AppCtx) -> None:
    st.subheader("🧪 AI Lab — 확장 모델 데모")
    st.markdown(
        f"기획서 §10.3·§10.4 확장 계획 모델 미리보기. {demo_badge('모든 카드 시뮬레이션')} · "
        "실제 모델 연동은 향후 단계.",
        unsafe_allow_html=True,
    )

    sub = st.radio(
        "데모 모듈",
        ["💬 HyperCLOVA X 설명", "📄 KoBigBird 장문 요약",
         "📈 PatchTST 시계열 예측", "🎤 Whisper 컨퍼런스콜",
         "🎯 Conformal Prediction", "📑 OCR 공시 PDF",
         "🔬 Permutation Importance", "📉 ALE Plot",
         "🔗 KoSimCSE 유사사례"],
        horizontal=True,
    )

    if sub.startswith("💬"):
        _render_clova(ctx)
    elif sub.startswith("📄"):
        _render_bigbird()
    elif sub.startswith("📈"):
        _render_patchtst(ctx)
    elif sub.startswith("🎤"):
        _render_whisper()
    elif sub.startswith("🎯"):
        _render_conformal(ctx)
    elif sub.startswith("📑"):
        _render_ocr()
    elif sub.startswith("🔬"):
        _render_perm_importance(ctx)
    elif sub.startswith("📉"):
        _render_ale(ctx)
    elif sub.startswith("🔗"):
        _render_kosimcse()


# ============================================================
def _render_clova(ctx: AppCtx) -> None:
    section_header("HyperCLOVA X 자연어 설명 카드",
                   "기획서 §10.3 — TreeSHAP 결과를 자연어로 풀어 설명. "
                   "현재는 if-else 템플릿 시뮬레이션 (실제 LLM 호출 시 이 함수만 교체).",
                   icon="💬", demo=True)
    snap = ctx.snap
    if snap.empty:
        st.warning("데이터 없음.")
        return
    sel = st.selectbox("종목 선택", snap["name"].tolist(), key="clova_sel")
    row = snap[snap["name"] == sel].iloc[0]
    conf = "HIGH" if row.get("confidence", 0) > 0.66 else \
           "MEDIUM" if row.get("confidence", 0) > 0.33 else "LOW"
    # top SHAP feature — 단순화: ret_5d 또는 vol_z_20 등
    top_feat_kor = "최근 5일 수익률" if row["score_up"] >= 50 else "20일 드로다운"
    text = mocks.hyperclova_explanation(
        name=sel, sector=row["sector"], cat=row["category"],
        up=int(row["score_up"]), risk=int(row["score_risk"]),
        conf=conf, top_feat=top_feat_kor,
    )
    if st.button("✨ 설명 생성", type="primary", key="clova_run"):
        import time
        ph = st.empty()
        buf = ""
        for ch in text:
            buf += ch
            ph.markdown(f"<div style='padding:14px 18px;background:var(--bg-elevated);"
                        f"border:1px solid var(--border-subtle);"
                        f"color:var(--text-primary);"
                        f"border-radius:8px;line-height:1.7;font-size:0.95em;'>"
                        f"{buf}▍</div>", unsafe_allow_html=True)
            if len(buf) % 4 == 0:
                time.sleep(0.012)
        ph.markdown(f"<div style='padding:14px 18px;background:#F5F7FA;"
                    f"border-radius:8px;line-height:1.7;font-size:0.95em;'>"
                    f"{buf}</div>", unsafe_allow_html=True)
        st.caption("⚙️ 실제 HyperCLOVA X 호출 시: prompt = SHAP top-5 + 점수 + 신뢰도 → "
                   "API 응답 (~1.5초). 본 데모는 템플릿 + typewriter 흉내.")


# ============================================================
def _render_bigbird() -> None:
    section_header("KoBigBird 장문 공시 요약",
                   "기획서 §10.3 — 4,096토큰 컨텍스트. 현재는 TextRank 근사로 추출 요약.",
                   icon="📄", demo=True)
    default = (
        "당사는 운영자금 확보를 위해 주주배정 후 실권주 일반공모 방식의 유상증자를 "
        "결정하였습니다. 발행주식 1,000,000주, 발행가액 12,000원입니다. "
        "조달 자금은 시설투자에 사용될 예정입니다. "
        "기존 주주의 청약권은 보장되며, 청약 기간은 2026년 6월 15일부터 6월 20일입니다. "
        "본 결정은 이사회 만장일치로 의결되었습니다. "
        "발행 주식은 기존 발행주식 총수의 약 8%에 해당합니다. "
        "유상증자 후 자본금은 기존 대비 약 8% 증가하며, 부채비율은 5%포인트 하락할 전망입니다. "
        "발행 가액 12,000원은 최근 1개월 평균 종가 대비 15% 할인 가격입니다. "
        "기관투자자 수요예측은 2026년 6월 10일에 실시될 예정입니다. "
        "공시 후 주가는 단기 약세를 보일 수 있으나, 시설투자에 따른 장기 매출 증가가 기대됩니다."
    )
    text = st.text_area("장문 공시 본문", default, height=200, key="bb_text")
    n = st.slider("요약 문장 수", 2, 5, 3, key="bb_n")
    if st.button("📌 요약 실행", type="primary", key="bb_run"):
        out = mocks.kobigbird_summarize(text, n_sent=n)
        info_card("요약 결과 (mock)", out["summary"], color="#5B8DEF")
        c1, c2 = st.columns(2)
        c1.metric("입력 문장 수", out["n_input_sent"])
        c2.metric("출력 문장 수", out.get("n_output_sent", n))
        if out.get("key_terms"):
            st.caption("핵심 키워드: " + ", ".join(f"`{k}`" for k in out["key_terms"]))


# ============================================================
def _render_patchtst(ctx: AppCtx) -> None:
    section_header("PatchTST 시계열 예측",
                   "기획서 §10.4 — 미래 5일 가격 예측. 현재는 단순 추세+노이즈 근사. "
                   "실제는 transformer 시계열 모델.",
                   icon="📈", demo=True)
    snap = ctx.snap
    sel = st.selectbox("종목 선택", snap["name"].tolist(), key="pt_sel")
    sid = snap[snap["name"] == sel]["stock_id"].iloc[0]
    hist = ctx.panel[ctx.panel["stock_id"] == sid].tail(60)
    if hist.empty:
        st.warning("데이터 없음.")
        return
    horizon = st.slider("예측 horizon (일)", 3, 10, 5, key="pt_h")
    if st.button("🔮 예측 실행", type="primary", key="pt_run"):
        fc = mocks.patchtst_forecast(hist["close"], horizon=horizon)
        if fc.empty:
            st.warning("히스토리 부족 (20일 미만)")
            return
        past_x = list(range(-len(hist), 0))
        future_x = list(range(1, horizon + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=past_x, y=hist["close"].values, mode="lines",
            name="과거 종가", line=dict(color="#5B8DEF", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=future_x, y=fc["predicted_close"], mode="lines+markers",
            name="예측 (mock)", line=dict(color="#FFB74D", width=2),
            marker=dict(size=8),
        ))
        fig.add_trace(go.Scatter(
            x=future_x + future_x[::-1],
            y=list(fc["upper_80"]) + list(fc["lower_80"])[::-1],
            fill="toself", fillcolor="rgba(255,183,77,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="80% 구간", showlegend=True, hoverinfo="skip",
        ))
        p = palette()
        fig.add_vline(x=0, line_dash="dash", line_color=p["axis_line_color"])
        lk = layout_kwargs(height=400)
        lk["xaxis"].update(title="일 (현재=0)")
        lk["yaxis"].update(title="가격 (원)")
        lk["legend"] = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        fig.update_layout(**lk)
        st.plotly_chart(fig, use_container_width=True)
        # 표
        disp = fc.copy()
        for c in ["predicted_close", "lower_80", "upper_80"]:
            disp[c] = disp[c].apply(lambda v: f"{v:,.0f}원")
        st.dataframe(disp, use_container_width=True, hide_index=True)


# ============================================================
def _render_whisper() -> None:
    section_header("Whisper 컨퍼런스콜 분석",
                   "기획서 §10.4 — ASR 변환 후 NLP 감성. "
                   "현재는 mock transcript + TF-IDF 감성 점수.",
                   icon="🎤", demo=True)
    if st.button("▶️ Mock 컨퍼런스콜 로드 + 분석", type="primary", key="ws_run"):
        lines = mocks.whisper_mock_transcript()
        df = mocks.whisper_sentiment_scores(lines)
        st.markdown("**Transcript + 감성 점수**")
        st.dataframe(df, use_container_width=True, hide_index=True)
        avg = df["점수"].mean()
        if avg > 0.1:
            tone = ("긍정", "#2E5933", "#C8E6C9")
        elif avg < -0.1:
            tone = ("부정", "#8B2D2D", "#FFCDD2")
        else:
            tone = ("중립", "#546E7A", "#ECEFF1")
        st.markdown(
            f"<div style='padding:12px 16px;background:{tone[2]};"
            f"border-radius:6px;color:{tone[1]};font-weight:600;'>"
            f"전체 톤: {tone[0]} (평균 {avg:+.3f}) — {len(df)}문장</div>",
            unsafe_allow_html=True,
        )


# ============================================================
def _render_conformal(ctx: AppCtx) -> None:
    section_header("Conformal Prediction 예측 구간",
                   "기획서 §11.4 — 신뢰도 6번째 요소. "
                   "현재는 ±width 근사. 실제는 calibration set의 nonconformity 분위.",
                   icon="🎯", demo=True)
    snap = ctx.snap
    sel = st.selectbox("종목 선택", snap["name"].tolist(), key="cf_sel")
    row = snap[snap["name"] == sel].iloc[0]
    p = float(row["score_up_p"])
    alpha = st.slider("유의수준 α", 0.05, 0.20, 0.10, step=0.05, key="cf_alpha")
    out = mocks.conformal_interval(p, alpha=alpha)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("점 추정 (상승 확률)", f"{out['point']*100:.1f}%")
    c2.metric(f"하한 ({out['confidence']*100:.0f}%)", f"{out['lower']*100:.1f}%")
    c3.metric(f"상한 ({out['confidence']*100:.0f}%)", f"{out['upper']*100:.1f}%")
    c4.metric("구간 폭", f"{out['width']*100:.1f}%pt")
    # bar 시각화
    fig, ax = plt.subplots(figsize=(8, 1.5))
    ax.barh([0], [out['upper'] - out['lower']], left=[out['lower']],
            color="#FFB74D", alpha=0.5, height=0.4)
    ax.scatter([out['point']], [0], color="#E57373", s=120, zorder=3, label="점추정")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("상승 확률", color="#666")
    ax.tick_params(colors="#666")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)
    st.caption(f"calibration set 크기 (mock): {out['n_calibration']}건")


# ============================================================
def _render_ocr() -> None:
    section_header("Donut / LayoutLMv3 — 공시 PDF OCR",
                   "기획서 §10.4 — OCR-free 문서 이해. "
                   "현재는 mock 추출 결과 표시.",
                   icon="📑", demo=True)
    up = st.file_uploader("공시 PDF 업로드 (mock — 어떤 파일이든 가능)",
                          type=["pdf", "png", "jpg"], key="ocr_up")
    fn = up.name if up else "샘플_공시.pdf"
    if st.button("🔍 OCR + 표·수치 추출 실행", type="primary", key="ocr_run"):
        out = mocks.ocr_mock_pdf_analyze(fn)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("페이지 수", out["n_pages"])
        c2.metric("추출 표", out["extracted_tables"])
        c3.metric("추출 차트", out["extracted_figures"])
        c4.metric("OCR 신뢰도", f"{out['ocr_confidence']*100:.0f}%")
        info_card("추출 텍스트 미리보기 (mock)",
                  f"<pre style='margin:0;font-family:Consolas,monospace;font-size:0.85em;'>"
                  f"{out['extracted_text_preview']}</pre>",
                  color="#81C784")
        st.caption(f"처리 시간 (mock): {out['processing_time_sec']:.1f}초 · 파일: {fn}")


# ============================================================
def _render_perm_importance(ctx: AppCtx) -> None:
    section_header("Permutation Importance",
                   "기획서 §14.2 — 각 피처를 셔플했을 때 예측 성능 감소량. "
                   "실제 sklearn.permutation_importance 사용 (검증셋 일부).",
                   icon="🔬")
    from core.config import FEATS
    target = st.radio("타깃", ["target_up (상승)", "target_crash (급락)"],
                      horizontal=True, key="pi_target")
    target_col = "target_up" if "상승" in target else "target_crash"
    model = ctx.m_up if target_col == "target_up" else ctx.m_cr
    if st.button("🚀 Permutation Importance 계산", type="primary", key="pi_run"):
        with st.spinner("셔플 + 재예측 (n_repeats=3, max_rows=5000)..."):
            df = mocks.permutation_importance_quick(model, ctx.panel, FEATS, target_col)
        if "error" in df.columns:
            st.error(df["error"].iloc[0])
            return
        # 한글 피처명 매핑
        from core.config import FEAT_KOR
        df["feature_kor"] = df["feature"].map(lambda f: FEAT_KOR.get(f, f))
        fig = go.Figure(go.Bar(
            x=df["importance_mean"], y=df["feature_kor"],
            orientation="h",
            error_x=dict(type="data", array=df["importance_std"], color="#94A3B8"),
            marker=dict(color="#5B8DEF", opacity=0.85),
            text=[f"{v:.4f}" for v in df["importance_mean"]],
            textposition="outside",
            hovertemplate="%{y}: %{x:.4f} ± %{error_x.array:.4f}<extra></extra>",
        ))
        lk = layout_kwargs(height=max(360, len(df) * 28))
        lk["xaxis"].update(title="ROC-AUC 감소량 (셔플 시)")
        fig.update_layout(**lk)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"상위 3개 피처: {', '.join(df['feature_kor'].head(3).tolist())} — "
                   "이들 피처를 셔플하면 모델 성능이 가장 크게 떨어짐 = 가장 중요.")


# ============================================================
def _render_ale(ctx: AppCtx) -> None:
    section_header("ALE Plot (Accumulated Local Effects)",
                   "기획서 §14.2 — 단일 피처 값 변경 시 예측 변화. "
                   "Partial Dependence보다 상관 피처에 robust.",
                   icon="📉")
    from core.config import FEATS, FEAT_KOR
    sel = st.selectbox("피처 선택", FEATS,
                       format_func=lambda f: f"{FEAT_KOR.get(f, f)} ({f})",
                       key="ale_feat")
    target = st.radio("타깃", ["target_up (상승)", "target_crash (급락)"],
                      horizontal=True, key="ale_target")
    model = ctx.m_up if "상승" in target else ctx.m_cr
    if st.button("📊 ALE 계산", type="primary", key="ale_run"):
        with st.spinner(f"{FEAT_KOR.get(sel, sel)} 20개 구간 평균 예측..."):
            df = mocks.ale_1d(model, ctx.panel, sel, FEATS,
                              target_col="target_up" if "상승" in target else "target_crash")
        if df.empty:
            st.warning("계산 실패. 데이터 부족.")
            return
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["grid_value"], y=df["effect"],
            mode="lines+markers",
            line=dict(color="#5B8DEF", width=2),
            marker=dict(size=6),
            fill="tozeroy", fillcolor="rgba(91,141,239,0.15)",
            hovertemplate="값: %{x:.4f}<br>effect: %{y:+.4f}<extra></extra>",
        ))
        p = palette()
        fig.add_hline(y=0, line_dash="dash", line_color=p["axis_line_color"])
        lk = layout_kwargs(height=360)
        lk["xaxis"].update(title=f"{FEAT_KOR.get(sel, sel)} 값 (분위)")
        lk["yaxis"].update(title="예측 확률 변화 (centered)")
        fig.update_layout(**lk)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("0보다 큰 구간 = 해당 피처값이 상승 확률을 높임. "
                   "0보다 작은 구간 = 낮춤. 곡선 기울기 = 민감도.")


# ============================================================
def _render_kosimcse() -> None:
    section_header("KoSimCSE 유사 사례 검색",
                   "기획서 §14.2 — 한국어 문장 임베딩으로 과거 유사 공시 패턴 검색. "
                   "현재는 mock 7건 + 키워드 매칭 부스트.",
                   icon="🔗", demo=True)
    q = st.text_input(
        "현재 종목의 상황 또는 공시 키워드",
        "유상증자 시설투자 자금 조달",
        key="ksc_q",
    )
    top_k = st.slider("Top K", 3, 7, 5, key="ksc_topk")
    if st.button("🔍 유사 사례 검색", type="primary", key="ksc_run"):
        df = mocks.kosimcse_similar_cases(q, top_k=top_k)
        st.markdown("**유사도 순 정렬 결과**")
        for _, r in df.iterrows():
            sim_pct = r["sim"] * 100
            bar = "█" * int(sim_pct / 10) + "░" * (10 - int(sim_pct / 10))
            info_card(
                f"{r['name']} — {r['event']} ({r['date']})",
                f"<b style='color:#5B8DEF;'>유사도 {sim_pct:.0f}%</b> "
                f"<span style='font-family:monospace;color:#94A3B8;'>{bar}</span>"
                f"<br><span style='color:var(--text-secondary);'>"
                f"<b>결과:</b> {r['outcome']}</span>",
                color="#5B8DEF",
            )
        st.caption(f"실제 KoSimCSE: 한국어 SimCSE finetune 모델 → 임베딩 → 코사인 유사도. "
                   f"현재는 mock {len(df)}건.")
