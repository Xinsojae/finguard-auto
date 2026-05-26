"""Tab 2: 2×2 기회-위험 매트릭스."""
import streamlit as st
import plotly.graph_objects as go

from core.config import CATEGORY_COLORS as _COLOR_MAP
from core.plotly_theme import layout_kwargs, palette
from tabs import AppCtx


def render(ctx: AppCtx) -> None:
    st.subheader("2×2 기회-위험 매트릭스")
    st.caption(
        "점 = 종목. **우측 하단** = 상승↑ · 리스크↓ → 우선 관심 / "
        "우측 상단 = 상승↑ · 리스크↑ → 고위험 관심 / "
        "좌측 하단 = 상승↓ · 리스크↓ → 관망 / 좌측 상단 = 상승↓ · 리스크↑ → 회피"
    )

    snap = ctx.snap
    fig = go.Figure()
    for cat, c in _COLOR_MAP.items():
        sub = snap[snap["category"] == cat]
        if sub.empty:
            continue
        hover = [
            f"<b>{r['name']}</b><br>섹터: {r['sector']}<br>"
            f"상승: {r['score_up']} · 리스크: {r['score_risk']}<br>"
            f"현재가: {r['close']:,.0f}원"
            for _, r in sub.iterrows()
        ]
        fig.add_trace(go.Scatter(
            x=sub["score_up"], y=sub["score_risk"],
            mode="markers", name=cat,
            marker=dict(color=c, size=11, opacity=0.7,
                        line=dict(color="white", width=1)),
            text=hover, hoverinfo="text",
        ))
    # 선택된 종목 강조
    if ctx.picks:
        picked = snap[snap["name"].isin(ctx.picks)]
        fig.add_trace(go.Scatter(
            x=picked["score_up"], y=picked["score_risk"],
            mode="markers+text", name="선택",
            marker=dict(color="rgba(0,0,0,0)", size=20,
                        line=dict(color="#424242", width=2)),
            text=picked["name"], textposition="top center",
            textfont=dict(size=10, color="#424242"),
            hoverinfo="skip",
        ))
    p = palette()
    fig.add_hline(y=ctx.risk_th, line_dash="dash", line_color=p["axis_line_color"])
    fig.add_vline(x=ctx.up_th, line_dash="dash", line_color=p["axis_line_color"])
    lk = layout_kwargs(height=520)
    lk["xaxis"].update(title=f"Upside Score (분류 ≥ {ctx.up_th})", range=[0, 100])
    lk["yaxis"].update(title=f"Risk Score (위험 ≥ {ctx.risk_th})", range=[0, 100])
    lk["legend"] = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    fig.update_layout(**lk)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### 카테고리별 종목 수")
    counts = snap["category"].value_counts()
    cols = st.columns(4)
    for i, (cat, label) in enumerate(zip(
            ["PRIORITY", "HIGH-RISK", "HOLD", "AVOID"],
            ["우선 관심", "고위험 관심", "관망", "회피"])):
        cols[i].metric(label, int(counts.get(cat, 0)))
