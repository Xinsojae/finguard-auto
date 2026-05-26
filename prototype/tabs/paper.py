"""Tab 6: 모의투자 (Paper Trading).

기획서 §15.2(모의투자 화면) + §16.1(모의 자동운용 흐름):
  - 보유 종목 / 모의 주문 내역 / 손익 현황 / 리스크 경고
"""
import streamlit as st

from core.paper_trading import (
    Portfolio, INITIAL_CASH, MAX_RISK_FOR_BUY,
    place_buy, place_sell, auto_buy_priority,
    positions_dataframe, trades_dataframe,
)
from core.risk import (
    check_kill_switch, summarize_rules, position_concentration,
    auto_close_check,
)
from core.ui_kit import download_csv_button
from tabs import AppCtx


def _get_portfolio() -> Portfolio:
    """세션 단위 가상 포트폴리오."""
    if "paper_pf" not in st.session_state:
        st.session_state["paper_pf"] = Portfolio()
    return st.session_state["paper_pf"]


def render(ctx: AppCtx) -> None:
    st.subheader("💼 모의투자 (Paper Trading)")
    st.caption("기획서 §15.2 + §16.1 — 가상 1,000만원 자금. "
               f"리스크 점수 ≥ {MAX_RISK_FOR_BUY} 종목 매수 차단 (§12.2). "
               "실거래 아님, 학습·검증 전용.")

    pf = _get_portfolio()
    snap = ctx.snap

    # 현재가 매핑 (snap 기준 최신 종가)
    prices = {str(r["stock_id"]): float(r["close"]) for _, r in snap.iterrows()}

    # ----- 손익 요약 -----
    total_equity = pf.total_equity(prices)
    total_pnl = pf.total_pnl(prices)
    roi = (total_equity / INITIAL_CASH - 1) * 100

    # ----- 리스크 엔진 + Kill Switch (§12) -----
    mkt_avg_risk = float(snap["score_risk"].mean())
    daily_pnl_pct = total_pnl / INITIAL_CASH
    ks = check_kill_switch(market_avg_risk=mkt_avg_risk,
                           daily_pnl_pct=daily_pnl_pct)
    pf.kill_switch_active = ks.active
    if ks.active:
        st.error(ks.banner_text())
    else:
        st.success(ks.banner_text())
    with st.expander("📐 리스크 엔진 규칙 (§12.2 + §12.3)", expanded=False):
        for rule in summarize_rules():
            st.markdown(f"- {rule}")
        st.caption(f"현재: 시장 평균 리스크 {mkt_avg_risk:.0f}/100 · "
                   f"일일 손익 {daily_pnl_pct*100:+.2f}%")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("초기 자금", f"{INITIAL_CASH:,.0f}원")
    c2.metric("총 평가액", f"{total_equity:,.0f}원",
              delta=f"{total_pnl:+,.0f}")
    c3.metric("현금 잔액", f"{pf.cash:,.0f}원")
    c4.metric("총 수익률", f"{roi:+.2f}%", delta=f"vs 0%")

    # ----- 자동 모의운용 -----
    st.divider()
    st.markdown("### 🤖 자동 모의운용 (§16.1)")
    auto_col1, auto_col2, auto_col3 = st.columns([2, 2, 2])
    with auto_col1:
        k = st.number_input("PRIORITY 자동매수 종목 수", 1, 10, 5, key="paper_auto_k")
    with auto_col2:
        cash_pct = st.slider("종목당 매수 비중 (%)", 5, 30, 10, step=5,
                             key="paper_auto_pct")
    with auto_col3:
        st.write("")
        st.write("")
        if st.button("🚀 자동 매수 실행", type="primary", key="paper_auto_run"):
            msgs = auto_buy_priority(pf, snap, k=k,
                                     cash_per_pick_pct=cash_pct / 100)
            for m in msgs:
                st.write(f"- {m}")

    # ----- 수동 주문 -----
    st.divider()
    st.markdown("### ✋ 수동 주문")
    o1, o2, o3, o4, o5 = st.columns([2, 1, 1, 1, 1])
    with o1:
        sel_name = st.selectbox("종목", snap["name"].tolist(), key="paper_buy_name")
    sel_row = snap[snap["name"] == sel_name].iloc[0]
    sel_sid = str(sel_row["stock_id"])
    sel_price = float(sel_row["close"])
    sel_risk = int(sel_row["score_risk"])
    with o2:
        st.metric("현재가", f"{sel_price:,.0f}")
    with o3:
        st.metric("리스크", f"{sel_risk}/100")
    with o4:
        qty = st.number_input("수량", 1, 1000, 10, key="paper_buy_qty")
    with o5:
        st.write("")
        st.write("")
        col_b, col_s = st.columns(2)
        with col_b:
            if st.button("매수", key="paper_buy_btn"):
                err = place_buy(pf, sel_sid, sel_name, qty, sel_price,
                                risk_score=sel_risk, reason="수동")
                if err:
                    st.error(err)
                else:
                    st.success(f"{sel_name} {qty}주 매수 완료")
                    st.rerun()
        with col_s:
            if st.button("매도", key="paper_sell_btn"):
                err = place_sell(pf, sel_sid, qty, sel_price, reason="수동")
                if err:
                    st.error(err)
                else:
                    st.success(f"{sel_name} {qty}주 매도 완료")
                    st.rerun()

    # ----- 보유 종목 -----
    st.divider()
    st.markdown("### 📊 보유 종목")
    pos_df = positions_dataframe(pf, prices)
    if pos_df.empty:
        st.info("보유 종목 없음. 위에서 자동 또는 수동 매수.")
    else:
        st.dataframe(pos_df, use_container_width=True, hide_index=True)

    # ----- 리스크 경고 (보유 종목 score_risk + 집중도) -----
    risk_warnings = []
    for sid, pos in pf.positions.items():
        if pos.qty == 0:
            continue
        snap_row = snap[snap["stock_id"].astype(str) == sid]
        if not snap_row.empty:
            r_score = int(snap_row.iloc[0]["score_risk"])
            if r_score >= MAX_RISK_FOR_BUY:
                risk_warnings.append((pos.name, r_score))
    if risk_warnings:
        st.warning("⚠️ **보유 종목 리스크 경고**")
        for n, r in risk_warnings:
            st.markdown(f"- **{n}**: 현재 리스크 {r}/100 (매도 검토 권장)")
    conc = position_concentration(pf.positions, prices, snap)
    if conc:
        st.warning("⚠️ **포지션 집중도 경고**")
        for w in conc:
            st.markdown(f"- {w}")

    # ----- 자동 청산 검토 (손절·익절·보유만료) -----
    sell_candidates = []
    for sid, pos in pf.positions.items():
        if pos.qty == 0:
            continue
        cur = prices.get(sid, pos.avg_price)
        # days_held: 마지막 BUY 시각 기반 단순 추정 (실시간 dummy = 0일)
        last_buy = next(
            (t for t in reversed(pf.trades)
             if t.side == "BUY" and t.stock_id == sid), None)
        days_held = 0  # session 안에선 일수 추적 어려움 — 손절/익절만 활성
        should, reason = auto_close_check(pos.avg_price, cur, days_held)
        if should:
            sell_candidates.append((sid, pos, cur, reason))
    if sell_candidates:
        st.warning(f"🤖 **자동 청산 검토 ({len(sell_candidates)}건)** — 손절·익절 조건 발생")
        for sid, pos, cur, reason in sell_candidates:
            st.markdown(f"- **{pos.name}** {pos.qty}주 @ {cur:,.0f}원 → {reason}")
        if st.button("🤖 일괄 자동 청산 실행", key="paper_auto_close"):
            for sid, pos, cur, reason in sell_candidates:
                place_sell(pf, sid, pos.qty, cur, reason=f"자동: {reason}")
            st.success(f"{len(sell_candidates)}건 청산 완료")
            st.rerun()

    # ----- 거래내역 -----
    st.divider()
    st.markdown("### 📜 거래 내역")
    tr_df = trades_dataframe(pf)
    if tr_df.empty:
        st.caption("거래 내역 없음.")
    else:
        st.dataframe(tr_df, use_container_width=True, hide_index=True)
        st.caption(f"총 {len(pf.trades)}건 · 차단된 매수 시도 {pf.blocked_buys}건")
        download_csv_button(tr_df, "거래내역 CSV 다운로드",
                            "paper_trades.csv", key="paper_dl_trades")

    # ----- 리셋 -----
    st.divider()
    with st.expander("⚠️ 포트폴리오 초기화"):
        st.caption("자금·보유·거래내역 모두 삭제하고 처음으로 되돌립니다.")
        if st.button("초기화 실행", key="paper_reset"):
            pf.reset()
            st.success("초기화 완료")
            st.rerun()
