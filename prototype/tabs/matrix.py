"""Tab 2: 2×2 기회-위험 매트릭스."""
import streamlit as st
import matplotlib.pyplot as plt

from core.config import CATEGORY_COLORS as _COLOR_MAP
from tabs import AppCtx


def render(ctx: AppCtx) -> None:
    st.subheader("2×2 기회-위험 매트릭스")
    st.caption("점 = 종목. 우측 상단: 상승 가능성 높음 + 리스크 낮음 = 우선 관심 후보")

    snap = ctx.snap
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for cat, c in _COLOR_MAP.items():
        sub = snap[snap["category"] == cat]
        ax.scatter(sub["score_up"], sub["score_risk"], c=c, alpha=0.55, s=50,
                   edgecolor="white", linewidth=0.6, label=cat)
    for name in ctx.picks:
        r = snap[snap["name"] == name].iloc[0]
        ax.scatter([r["score_up"]], [r["score_risk"]], facecolor="none",
                   edgecolor="#424242", linewidth=1.6, s=140, marker="o")
        ax.annotate(name, (r["score_up"], r["score_risk"]),
                    fontsize=8, color="#424242",
                    xytext=(5, 5), textcoords="offset points",
                    fontproperties=ctx.kfont_fp)
    ax.axhline(ctx.risk_th, color="#BDBDBD", ls="--", lw=1)
    ax.axvline(ctx.up_th, color="#BDBDBD", ls="--", lw=1)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xlabel(f"Upside Score (분류 ≥ {ctx.up_th})", color="#666")
    ax.set_ylabel(f"Risk Score (위험 ≥ {ctx.risk_th})", color="#666")
    ax.tick_params(colors="#666")
    for s in ax.spines.values():
        s.set_color("#E0E0E0")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.grid(True, alpha=0.15)
    st.pyplot(fig); plt.close(fig)

    st.divider()
    st.markdown("### 카테고리별 종목 수")
    counts = snap["category"].value_counts()
    cols = st.columns(4)
    for i, (cat, label) in enumerate(zip(
            ["PRIORITY", "HIGH-RISK", "HOLD", "AVOID"],
            ["우선 관심", "고위험 관심", "관망", "회피"])):
        cols[i].metric(label, int(counts.get(cat, 0)))
