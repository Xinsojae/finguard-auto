"""Tab 7: 포트폴리오 분석 (§15.2 포트폴리오 화면)."""
import streamlit as st
import matplotlib.pyplot as plt

from core.paper_trading import Portfolio
from core.portfolio import (
    sector_concentration, return_correlation,
    portfolio_var, volatility_contribution,
)
from tabs import AppCtx


def _get_pf() -> Portfolio:
    if "paper_pf" not in st.session_state:
        st.session_state["paper_pf"] = Portfolio()
    return st.session_state["paper_pf"]


def render(ctx: AppCtx) -> None:
    st.subheader("📊 포트폴리오 분석")
    st.caption("기획서 §15.2 — 보유 종목의 섹터 쏠림 · 종목 상관관계 · "
               "예상 최대 손실(VaR) · 변동성 기여도. "
               "데이터 소스: 💼 모의투자 탭에서 만든 가상 포지션.")

    pf = _get_pf()
    snap = ctx.snap
    panel = ctx.panel
    prices = {str(r["stock_id"]): float(r["close"]) for _, r in snap.iterrows()}
    held_ids = [sid for sid, p in pf.positions.items() if p.qty > 0]

    if not held_ids:
        st.info("보유 종목이 없습니다. '💼 모의투자' 탭에서 매수 후 다시 확인하세요.")
        return

    _render_sector(pf, prices, snap, ctx.kfont_fp)
    st.divider()
    _render_correlation(panel, snap, held_ids, ctx.kfont_fp)
    st.divider()
    _render_var(pf, prices, panel)
    st.divider()
    _render_vol_contrib(pf, prices, panel)


def _render_sector(pf, prices, snap, kfont_fp) -> None:
    st.markdown("### 1. 섹터 쏠림")
    sec_dict = sector_concentration(pf.positions, prices, snap)
    if not sec_dict:
        st.caption("데이터 없음")
        return
    secs = list(sec_dict.keys())
    weights = [v * 100 for v in sec_dict.values()]
    colors = ["#C62828" if v > 40 else "#F57C00" if v > 30 else "#2E7D32"
              for v in weights]
    fig, ax = plt.subplots(figsize=(8, max(2, len(secs) * 0.4)))
    ax.barh(secs, weights, color=colors)
    ax.axvline(40, color="#C62828", ls="--", lw=0.8, label="한도 40%")
    ax.set_xlabel("비중 %")
    if kfont_fp is not None:
        for lbl in ax.get_yticklabels():
            lbl.set_fontproperties(kfont_fp)
        ax.set_xlabel("비중 %", fontproperties=kfont_fp)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)


def _render_correlation(panel, snap, held_ids, kfont_fp) -> None:
    st.markdown("### 2. 종목 간 수익률 상관관계 (최근 60일)")
    if len(held_ids) < 2:
        st.caption("보유 종목 2개 이상 필요")
        return
    corr = return_correlation(panel, held_ids, window=60)
    name_map = {str(r["stock_id"]): r["name"] for _, r in snap.iterrows()}
    corr.index = [name_map.get(str(c), str(c)) for c in corr.index]
    corr.columns = [name_map.get(str(c), str(c)) for c in corr.columns]
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(corr.values, cmap="RdYlGn_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    if kfont_fp is not None:
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontproperties(kfont_fp)
    plt.colorbar(im, ax=ax, label="상관계수")
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)
    n = len(corr)
    avg_off_diag = (corr.values.sum() - n) / (n * (n - 1)) if n > 1 else 0
    st.caption(f"평균 상관계수 (대각선 제외): **{avg_off_diag:.2f}** "
               "— 1에 가까울수록 분산효과 약함")


def _render_var(pf, prices, panel) -> None:
    st.markdown("### 3. 예상 최대 손실 (Historical VaR)")
    var = portfolio_var(pf.positions, prices, panel, window=60)
    c1, c2, c3 = st.columns(3)
    c1.metric("VaR 95%", f"{var['VaR_95']:,.0f}원",
              help="95% 신뢰수준 일일 최대 손실 추정 (역사적 시뮬레이션)")
    c2.metric("VaR 99%", f"{var['VaR_99']:,.0f}원",
              help="99% 신뢰수준 일일 최대 손실 추정")
    c3.metric("실현 최대 손실 (최근)", f"{var['expected_max_loss']:,.0f}원")
    st.caption(f"기준: 최근 {var['n_days']}일 수익률 분포 · "
               f"총 평가액 {var['total_mv']:,.0f}원")


def _render_vol_contrib(pf, prices, panel) -> None:
    st.markdown("### 4. 변동성 기여도")
    vc = volatility_contribution(pf.positions, prices, panel, window=60)
    if vc.empty:
        st.caption("데이터 없음")
        return
    disp = vc.copy()
    disp["비중%"] = (disp["weight"] * 100).apply(lambda v: f"{v:.2f}%")
    disp["변동성%"] = (disp["vol"] * 100).apply(lambda v: f"{v:.2f}%")
    disp["기여도%"] = (disp["contribution"] * 100).apply(lambda v: f"{v:.3f}%")
    disp = disp.rename(columns={"name": "종목"})[["종목", "비중%", "변동성%", "기여도%"]]
    st.dataframe(disp, use_container_width=True, hide_index=True)
    st.caption("기여도 = 비중 × 변동성 (단순 추정). 높은 종목이 포트 변동성을 주도.")
