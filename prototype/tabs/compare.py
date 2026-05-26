"""Tab: 종목 비교 — 2~4종목 나란히 비교.

상승·리스크 점수, 신뢰도 5요소, SHAP Top-5, 가격 차트 overlay.
"""
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from core.config import FEATS, FEAT_KOR
from tabs import AppCtx


def render(ctx: AppCtx) -> None:
    st.subheader("🔄 종목 비교 모드")
    st.caption("최대 4개 종목을 나란히 비교 — 점수 / 신뢰도 / SHAP / 가격 추세")

    snap = ctx.snap
    sel = st.multiselect(
        "비교할 종목 (2~4개)", snap["name"].tolist(),
        default=snap.nlargest(3, "score_up")["name"].tolist(),
        max_selections=4,
    )
    if len(sel) < 2:
        st.info("2개 이상 선택하세요.")
        return

    rows = [snap[snap["name"] == n].iloc[0] for n in sel]

    # ----- 1. 점수 비교 표 -----
    st.markdown("### 1. 점수 비교")
    cols = st.columns(len(rows))
    for col, r in zip(cols, rows):
        col.markdown(f"#### {r['name']}")
        col.caption(f"섹터: {r['sector']}")
        col.metric("상승", f"{int(r['score_up'])}/100")
        col.metric("리스크", f"{int(r['score_risk'])}/100",
                   delta_color="inverse")
        col.metric("신뢰도", f"{r.get('conf_label','-')}")
        anom = int(r.get("anomaly_score", 0))
        col.metric("이상치", f"{anom}/100")
        var_5d = float(r.get("var_5d_p10", 0)) * 100
        col.metric("최악손실(5d)", f"{var_5d:+.2f}%")

    st.divider()

    # ----- 2. 가격 비교 (정규화) -----
    st.markdown("### 2. 가격 추세 비교 (시작=100 정규화, 최근 60일)")
    fig = go.Figure()
    palette = ["#5B8DEF", "#E57373", "#81C784", "#FFB74D"]
    for i, r in enumerate(rows):
        sid = r["stock_id"]
        hist = ctx.panel[ctx.panel["stock_id"] == sid].tail(60).copy()
        if hist.empty:
            continue
        norm = (hist["close"] / hist["close"].iloc[0] * 100)
        fig.add_trace(go.Scatter(
            x=hist["date"], y=norm, mode="lines",
            name=r["name"], line=dict(color=palette[i % 4], width=2),
            hovertemplate=f"<b>{r['name']}</b><br>%{{x|%Y-%m-%d}}<br>지수: %{{y:.1f}}<extra></extra>",
        ))
    fig.add_hline(y=100, line_dash="dash", line_color="#BDBDBD", line_width=1)
    fig.update_layout(
        height=340,
        xaxis=dict(gridcolor="#F0F0F0"),
        yaxis=dict(title="정규화 가격 (시작=100)", gridcolor="#F0F0F0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0),
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        hovermode="x unified",
        font=dict(family="Malgun Gothic, sans-serif", color="#555"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ----- 3. SHAP Top-5 비교 -----
    st.markdown("### 3. SHAP Top-5 기여 요인 비교")
    fig = go.Figure()
    for i, r in enumerate(rows):
        x_row = r[FEATS].values.reshape(1, -1)
        contrib = ctx.m_up.booster_.predict(x_row, pred_contrib=True)[0][:-1]
        top_idx = np.argsort(-np.abs(contrib))[:5]
        names = [FEAT_KOR[FEATS[idx]] for idx in top_idx]
        values = [contrib[idx] for idx in top_idx]
        fig.add_trace(go.Bar(
            x=names, y=values, name=r["name"],
            marker=dict(color=palette[i % 4], opacity=0.75),
            hovertemplate=f"<b>{r['name']}</b><br>%{{x}}<br>SHAP: %{{y:+.3f}}<extra></extra>",
        ))
    fig.add_hline(y=0, line_color="#BDBDBD", line_width=1)
    fig.update_layout(
        barmode="group", height=380,
        xaxis=dict(gridcolor="#F0F0F0", tickangle=-15),
        yaxis=dict(title="SHAP contribution", gridcolor="#F0F0F0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0),
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Malgun Gothic, sans-serif", color="#555"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.caption("팁: 가격 추세 정규화로 상대 성과 직관적 비교. SHAP 비교로 점수 차이 원인 추적.")
