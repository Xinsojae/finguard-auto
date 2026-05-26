"""Tab 2: 2×2 기회-위험 매트릭스."""
import streamlit as st
import matplotlib.pyplot as plt

from tabs import AppCtx

_COLOR_MAP = {
    "PRIORITY":  "#2E7D32",
    "HIGH-RISK": "#F57C00",
    "HOLD":      "#9E9E9E",
    "AVOID":     "#C62828",
}


def render(ctx: AppCtx) -> None:
    st.subheader("2×2 기회-위험 매트릭스")
    st.caption("점 = 종목. 우측 상단: 상승 가능성 높음 + 리스크 낮음 = 우선 관심 후보")

    snap = ctx.snap
    fig, ax = plt.subplots(figsize=(9, 7))
    for cat, c in _COLOR_MAP.items():
        sub = snap[snap["category"] == cat]
        ax.scatter(sub["score_up"], sub["score_risk"], c=c, alpha=0.65, s=55,
                   edgecolor="white", linewidth=0.5, label=cat)
    # 선택된 종목 강조
    for name in ctx.picks:
        r = snap[snap["name"] == name].iloc[0]
        ax.scatter([r["score_up"]], [r["score_risk"]], facecolor="none",
                   edgecolor="black", linewidth=2, s=150, marker="o")
        ax.annotate(name, (r["score_up"], r["score_risk"]),
                    fontsize=8, xytext=(5, 5), textcoords="offset points",
                    fontproperties=ctx.kfont_fp)
    ax.axhline(ctx.risk_th, color="#444", ls="--", lw=1)
    ax.axvline(ctx.up_th, color="#444", ls="--", lw=1)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xlabel(f"Upside Score (분류 ≥ {ctx.up_th})")
    ax.set_ylabel(f"Risk Score (위험 ≥ {ctx.risk_th})")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.25)
    st.pyplot(fig); plt.close(fig)

    st.divider()
    st.markdown("### 카테고리별 종목 수")
    counts = snap["category"].value_counts()
    cols = st.columns(4)
    for i, (cat, label) in enumerate(zip(
            ["PRIORITY", "HIGH-RISK", "HOLD", "AVOID"],
            ["우선 관심", "고위험 관심", "관망", "회피"])):
        cols[i].metric(label, int(counts.get(cat, 0)))
