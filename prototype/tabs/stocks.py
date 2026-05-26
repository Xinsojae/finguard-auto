"""Tab 1: 종목 분석 (워치리스트 + 상세 + SHAP + 공시 분석 카드)."""
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from core.config import FEATS, FEAT_KOR
from core.ui import tag_html, tag_plain
from tabs import AppCtx


def render(ctx: AppCtx) -> None:
    if not ctx.picks:
        st.warning("좌측 사이드바에서 관심 종목을 선택하세요.")
        return

    snap = ctx.snap
    panel = ctx.panel

    # ----- 워치리스트 카드 -----
    cols = st.columns(min(len(ctx.picks), 3))
    for i, name in enumerate(ctx.picks):
        with cols[i % 3]:
            row = snap[snap["name"] == name].iloc[0]
            cat = row["category"]
            conf_lbl = ("HIGH" if row["confidence"] > 0.66
                        else "MED" if row["confidence"] > 0.33 else "LOW")
            st.markdown(f"""
            <div class='card'>
              <h4>{name} <small>({row['sector']})</small></h4>
              {tag_html(cat)}
              <p style='margin-top:8px;'>
                <span class='metric-up'>상승 {row['score_up']}</span> &nbsp;
                <span class='metric-risk'>리스크 {row['score_risk']}</span>
              </p>
              <p style='font-size:0.85em; color:#777;'>신뢰도: {conf_lbl}</p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ----- 상세 분석 -----
    sel = st.selectbox("상세 분석할 종목", ctx.picks)
    if not sel:
        return

    row = snap[snap["name"] == sel].iloc[0]
    cat = row["category"]
    conf = ("HIGH" if row["confidence"] > 0.66
            else "MEDIUM" if row["confidence"] > 0.33 else "LOW")

    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        st.markdown(f"### {sel}")
        st.markdown(tag_html(cat), unsafe_allow_html=True)
        st.markdown(f"**섹터**: {row['sector']}")
        st.markdown(f"**기준일**: {ctx.latest_date.date()}")
        st.markdown(f"**현재가**: {row['close']:,.0f}원")
    with c2:
        st.metric("상승 가능성", f"{row['score_up']}/100",
                  delta=f"{row['score_up'] - 50:+d} vs 중립")
        st.metric("급락 위험", f"{row['score_risk']}/100",
                  delta=f"{row['score_risk'] - 50:+d} vs 중립",
                  delta_color="inverse")
        st.metric("AI 신뢰도", conf)
    with c3:
        _render_shap_chart(row, ctx.m_up, ctx.kfont_fp)

    _render_disclosure_card(panel, row)
    _render_explanation_box(sel, cat, conf, row)


def _render_shap_chart(row, m_up, kfont_fp) -> None:
    x_row = row[FEATS].values.reshape(1, -1)
    # LightGBM Booster의 raw 예측 분해 (TreeSHAP과 동등한 SHAP value)
    contrib = m_up.booster_.predict(x_row, pred_contrib=True)[0][:-1]
    top_idx = np.argsort(-np.abs(contrib))[:5]
    st.markdown("**🔍 판단 근거 (TreeSHAP Top-5)**")
    fig, ax = plt.subplots(figsize=(5, 3))
    names = [FEAT_KOR[FEATS[i]] for i in top_idx][::-1]
    values = [contrib[i] for i in top_idx][::-1]
    colors = ["#2E7D32" if v > 0 else "#C62828" for v in values]
    ax.barh(names, values, color=colors)
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_xlabel("SHAP contribution to 상승 score", fontproperties=kfont_fp)
    ax.tick_params(axis="y", labelsize=9)
    if kfont_fp is not None:
        for lbl in ax.get_yticklabels():
            lbl.set_fontproperties(kfont_fp)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_disclosure_card(panel, row) -> None:
    sid = row["stock_id"]
    recent = panel[(panel["stock_id"] == sid) & (panel["disc_code"] != "")
                   ].sort_values("date").tail(3)
    if len(recent) == 0:
        return
    st.divider()
    st.markdown("**📑 최근 공시 분석 (룰베이스 30유형 분류 · disclosure_analyzer 연동)**")
    for _, d in recent.iterrows():
        risk_val = int(d["disclosure"])
        color = ("#C62828" if risk_val < 0
                 else "#2E7D32" if risk_val > 0 else "#666")
        st.markdown(
            f"<div style='border-left:4px solid {color};padding:8px 12px;"
            f"margin:6px 0;background:#FAFAFA;border-radius:4px;'>"
            f"<b>{d['date'].date()}</b> · "
            f"<span style='color:{color};font-weight:700;'>"
            f"[{d['disc_name']}] {d['disc_risk_label']}</span> "
            f"(risk_score {risk_val:+d})"
            f"<br><small style='color:#555;'>{d['disc_explanation']}</small>"
            f"</div>",
            unsafe_allow_html=True,
        )
    bad = int((recent["disclosure"] < 0).sum())
    good = int((recent["disclosure"] > 0).sum())
    st.caption(f"최근 분류 이벤트: 악재 {bad}건 / 호재 {good}건 / 총 {len(recent)}건")


def _render_explanation_box(sel, cat, conf, row) -> None:
    st.divider()
    st.markdown("**🤖 AI 설명 카드 (자연어, V3에서 HyperCLOVA X로 생성 예정)**")
    sign = "상승 시그널" if cat in ["PRIORITY", "HIGH-RISK"] else "관망 또는 회피"
    risk_note = ("리스크 신호도 동시에 강하므로 모의투자로 검증 권장"
                 if row["score_risk"] >= 50 else "리스크 신호는 낮음")
    action = {
        "HIGH": "백테스트로 사후 검증",
        "MEDIUM": "모의투자 5거래일 후 재평가",
        "LOW": "데이터 추가 확보 후 재평가",
    }[conf]
    st.info(
        f"**{sel}**은(는) **{tag_plain(cat)}**(으)로 분류됩니다. "
        f"{sign}이 우세하고, {risk_note}. "
        f"AI 신뢰도는 **{conf}**이며, 권장 행동: {action}. "
        f"본 결과는 매수·매도 추천이 아닌 의사결정 보조 정보입니다."
    )
