"""Tab 4: walk-forward 백테스트 A vs B."""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from core.models import walk_forward_backtest
from core.ui_kit import download_csv_button
from core.plotly_theme import layout_kwargs, palette
from tabs import AppCtx


def render(ctx: AppCtx) -> None:
    st.subheader("📈 백테스트: A(상승만) vs B(상승+리스크 필터)")
    st.caption("walk-forward 3-fold · 비중첩 5일 보유 리밸런스 · 거래비용 0.3% (진입+청산)")

    spinner_msg = (f"walk-forward 백테스트 실행 중 "
                   f"(k_top={ctx.k_top}, hold={ctx.hold_days}d, risk_pct={ctx.risk_pct})...")
    with st.spinner(spinner_msg):
        ra_s, rb_s, avoided, per_fold = walk_forward_backtest(
            ctx.panel, n_folds=3, k_top=ctx.k_top, hold_days=ctx.hold_days,
            cost=0.003, risk_pct=ctx.risk_pct)
        cum_a = (1 + ra_s).cumprod()
        cum_b = (1 + rb_s).cumprod()

    if len(cum_a) == 0:
        st.warning(
            "백테스트 가능한 fold가 없습니다. 패널 크기가 너무 작거나 "
            "일별 종목 수가 부족할 수 있습니다 (k_top×2 미만)."
        )
    else:
        _render_curve(cum_a, cum_b, per_fold)
        _render_metrics(ra_s, rb_s, cum_a, cum_b, avoided)
        _render_fold_table(per_fold)

    _render_caveat()
    _render_interpretation(ctx.use_real)


def _render_curve(cum_a, cum_b, per_fold) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cum_a.index, y=cum_a.values, mode="lines",
        name=f"A: 상승만 (누적 {cum_a.iloc[-1] - 1:+.1%})",
        line=dict(color="#E57373", width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>누적: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=cum_b.index, y=cum_b.values, mode="lines",
        name=f"B: 상승+리스크 필터 (누적 {cum_b.iloc[-1] - 1:+.1%})",
        line=dict(color="#81C784", width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>누적: %{y:.3f}<extra></extra>",
    ))
    p = palette()
    fig.add_hline(y=1.0, line_dash="dash", line_color=p["axis_line_color"], line_width=1)
    for rec in per_fold[1:]:
        fig.add_vline(x=pd.Timestamp(rec["test_start"]),
                      line_dash="dot", line_color=p["axis_line_color"],
                      line_width=1, opacity=0.6)
    lk = layout_kwargs(height=420)
    lk["hovermode"] = "x unified"
    lk["title"] = dict(text="Cumulative Return (walk-forward, 5d non-overlap)",
                       font=dict(color=p["title_color"], size=14))
    lk["xaxis"].update(rangeslider=dict(visible=True, thickness=0.05))
    lk["yaxis"].update(title="누적 자산 (start=1.0)")
    lk["legend"] = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    lk["margin"] = dict(l=10, r=10, t=50, b=10)
    fig.update_layout(**lk)
    st.plotly_chart(fig, use_container_width=True)


def _render_metrics(ra_s, rb_s, cum_a, cum_b, avoided) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("A 5일 평균 수익률", f"{ra_s.mean() * 100:+.2f}%")
    c2.metric("B 5일 평균 수익률", f"{rb_s.mean() * 100:+.2f}%")
    c3.metric("A MDD", f"{(cum_a / cum_a.cummax() - 1).min() * 100:.1f}%")
    c4.metric("리스크 필터 회피 picks", f"{avoided:,}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("총 리밸런스 횟수", f"{len(ra_s)}")
    c6.metric("A 승률", f"{(ra_s > 0).mean() * 100:.1f}%")
    c7.metric("B 승률", f"{(rb_s > 0).mean() * 100:.1f}%")
    sharpe_a = ra_s.mean() / (ra_s.std() + 1e-9) * np.sqrt(52)
    c8.metric("A Sharpe (연환산)", f"{sharpe_a:.2f}")


def _render_fold_table(per_fold) -> None:
    st.divider()
    st.markdown("**🧩 Walk-forward fold별 결과**")
    fold_df = pd.DataFrame(per_fold).rename(columns={
        "fold": "Fold", "train_end": "Train 종료",
        "test_start": "Test 시작", "test_end": "Test 종료",
        "n_picks": "리밸런스 횟수",
        "A_mean": "A 평균(5일)", "B_mean": "B 평균(5일)",
    })
    for col in ["A 평균(5일)", "B 평균(5일)"]:
        fold_df[col] = fold_df[col].apply(lambda v: f"{v * 100:+.2f}%")
    st.dataframe(fold_df, use_container_width=True, hide_index=True)
    download_csv_button(fold_df, "Fold 결과 CSV 다운로드",
                        "backtest_folds.csv", key="bt_dl_folds")


def _render_caveat() -> None:
    st.divider()
    st.warning(
        "📐 **방법론 caveat**\n\n"
        "- **walk-forward 3-fold**: 시작 40%를 warmup, 나머지 60%를 3분할. fold별 재학습 → out-of-time 평가.\n"
        "- **비중첩 5일 보유**: 5거래일마다 리밸런스. 이전 합성 코드의 *일별 picks를 일별 컴파운드*하던 버그(5일 중첩 fwd_ret) 수정.\n"
        "- **거래비용**: 진입+청산 단순 0.3% 차감. 슬리피지·시장충격·세금 미반영.\n"
        "- **kill-switch 미적용**: 손절(-3%)·익절(+5%)·일일 손실 한도 미시뮬레이션 (기획서 §12 확장)."
    )


def _render_interpretation(use_real: bool) -> None:
    st.markdown("**📝 해석**")
    if use_real:
        st.info(
            "**실데이터 모드**: KOSPI 99종목 / 2021~2025 / NLP 피처는 룰베이스 공시 분류만 주입 "
            "(KF-DeBERTa 뉴스·공시 모델은 확장 계획). 급락 모델 ROC-AUC는 합성 0.58 → 실 0.63 개선 "
            "확인됨. B(리스크 필터)가 A를 상회하는 정도는 fold별 변동성이 큽니다."
        )
    else:
        st.info(
            "**합성 데이터 모드**: GARCH형 패널. 급락 신호가 약해(ROC-AUC ~0.58) 리스크 필터(B)가 "
            "A 대비 유의미한 개선을 보이지 못할 수 있습니다. **실데이터 모드 + KF-DeBERTa 추가 시 "
            "B가 A를 상회할 것으로 가설**합니다. 발표에서 '솔직한 베이스라인 + 개선 방향'으로 제시 가능."
        )
