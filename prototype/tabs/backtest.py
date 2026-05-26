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
    st.caption(
        f"walk-forward 3-fold · 비중첩 {ctx.hold_days}일 리밸런스 · "
        f"수익률 측정 = fwd_ret_5d (고정) · 거래비용 0.3% (진입+청산)"
    )
    if ctx.hold_days != 5:
        st.warning(
            f"⚠️ hold_days={ctx.hold_days} 설정 — 수익률 측정 horizon은 5일 고정. "
            f"({ctx.hold_days} < 5 시 forward-label embargo 부족 · "
            f"{ctx.hold_days} > 5 시 보유 표시와 실 수익 불일치). "
            f"권장: 5"
        )

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
        _render_experiments(per_fold, avoided)

    _render_caveat(ctx.hold_days)
    _render_interpretation(ctx.use_real, per_fold)


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


def _render_experiments(per_fold, avoided) -> None:
    """§13.1~13.4 실험별 분리 표."""
    st.divider()
    st.markdown("### 📊 §13 검증 실험 분리 결과")
    st.caption("기획서 §13.1~13.4 실험 4종 — fold별 모델 메트릭 + 결합 효과 + 자동화 검증.")

    df = pd.DataFrame(per_fold)
    if df.empty:
        st.info("fold 결과 없음.")
        return

    with st.expander("§13.1 실험 1 — 상승 가능성 모델 검증", expanded=False):
        exp1 = df[["fold", "test_start", "test_end",
                   "up_auc", "up_pr", "up_topk_prec"]].copy()
        exp1.columns = ["Fold", "Test 시작", "Test 종료",
                        "ROC-AUC", "PR-AUC", "Top-20% Precision"]
        for c in ["ROC-AUC", "PR-AUC", "Top-20% Precision"]:
            exp1[c] = exp1[c].apply(lambda v: f"{v:.3f}")
        st.dataframe(exp1, use_container_width=True, hide_index=True)
        avg_auc = df["up_auc"].mean()
        avg_pr = df["up_pr"].mean()
        st.caption(f"평균 ROC-AUC: **{avg_auc:.3f}** · 평균 PR-AUC: **{avg_pr:.3f}** "
                   f"(베이스라인 0.5 / 양성비율 ≈ 0.30 대비 평가)")

    with st.expander("§13.2 실험 2 — 급락 위험 모델 검증", expanded=False):
        exp2 = df[["fold", "test_start", "test_end",
                   "cr_auc", "cr_pr", "cr_topk_prec"]].copy()
        exp2.columns = ["Fold", "Test 시작", "Test 종료",
                        "ROC-AUC", "PR-AUC", "Top-20% Precision"]
        for c in ["ROC-AUC", "PR-AUC", "Top-20% Precision"]:
            exp2[c] = exp2[c].apply(lambda v: f"{v:.3f}")
        st.dataframe(exp2, use_container_width=True, hide_index=True)
        avg_auc = df["cr_auc"].mean()
        avg_pr = df["cr_pr"].mean()
        st.caption(f"평균 ROC-AUC: **{avg_auc:.3f}** · 평균 PR-AUC: **{avg_pr:.3f}** "
                   f"(양성비율 ≈ 0.13 대비 평가 — PR-AUC가 더 의미있는 지표)")

    with st.expander("§13.3 실험 3 — 리스크 결합 효과 (A vs B)", expanded=False):
        exp3 = df[["fold", "test_start", "test_end",
                   "n_picks", "A_mean", "B_mean"]].copy()
        exp3.columns = ["Fold", "Test 시작", "Test 종료",
                        "리밸런스 횟수", "A 평균(5d)", "B 평균(5d)"]
        for c in ["A 평균(5d)", "B 평균(5d)"]:
            exp3[c] = exp3[c].apply(lambda v: f"{v*100:+.2f}%")
        st.dataframe(exp3, use_container_width=True, hide_index=True)
        b_minus_a = (df["B_mean"] - df["A_mean"]).mean() * 100
        st.caption(f"B - A 평균 차: **{b_minus_a:+.3f}%pt** · "
                   f"리스크 필터 회피 picks 누적: **{avoided:,}건**. "
                   "차이가 양수면 리스크 필터가 효과적 (급락 회피로 평균 수익률 개선).")

    with st.expander("§13.4 실험 4 — 모의 자동화 검증", expanded=False):
        st.markdown("**자동 매수 → 보유 → 자동 청산 시뮬레이션**은 "
                    "**💼 모의투자 탭에서 직접 실행** 가능.")
        st.markdown("- **자동 매수 흐름**: PRIORITY 종목 상위 k → 리스크 60+ 차단 → "
                    "현금 배분 매수")
        st.markdown("- **자동 청산**: 손절 -3% / 익절 +5% / 보유 5일 만료 자동 매도")
        st.markdown(f"- **Kill Switch**: 시장 평균 리스크 ≥ 70 또는 일일 손실 ≤ -5% 시 "
                    f"신규 매수 즉시 중단")
        st.caption("실험 4의 본 백테스트 통합 평가는 §13.3에 반영됨 (walk-forward A/B). "
                   "세션 단위 라이브 시연은 모의투자 탭 참조.")


def _render_caveat(hold_days: int = 5) -> None:
    st.divider()
    st.warning(
        f"📐 **방법론 caveat**\n\n"
        f"- **walk-forward 3-fold**: 시작 40%를 warmup, 나머지 60%를 3분할. fold별 재학습 → out-of-time 평가.\n"
        f"- **비중첩 {hold_days}일 보유**: {hold_days}거래일마다 리밸런스. 이전 합성 코드의 *일별 picks를 일별 컴파운드*하던 버그 수정.\n"
        f"- **타깃 horizon 고정**: 수익률 측정은 항상 fwd_ret_5d (5일 후 수익률). "
        f"hold_days를 5보다 작게 설정하면 forward-label 누수 가능 — 5 권장.\n"
        f"- **거래비용**: 진입+청산 단순 0.3% 차감. 슬리피지·시장충격·세금 미반영.\n"
        f"- **kill-switch 미적용**: 손절·익절·일일 손실 한도는 모의투자 탭에서 별도 시뮬레이션."
    )


def _render_interpretation(use_real: bool, per_fold: list = None) -> None:
    """per_fold에서 실측 평균 ROC-AUC 계산 후 표시 (고정 문구 X)."""
    st.markdown("**📝 해석**")
    # 실측 평균 fold ROC-AUC
    avg_up = avg_cr = None
    if per_fold:
        ups = [r.get("up_auc", 0) for r in per_fold if r.get("up_auc")]
        crs = [r.get("cr_auc", 0) for r in per_fold if r.get("cr_auc")]
        if ups:
            avg_up = sum(ups) / len(ups)
        if crs:
            avg_cr = sum(crs) / len(crs)
    auc_text = ""
    if avg_up is not None and avg_cr is not None:
        auc_text = f" (현재 fold 평균 ROC-AUC — 상승: **{avg_up:.3f}**, 급락: **{avg_cr:.3f}**)"
    if use_real:
        st.info(
            "**실데이터 모드**: KOSPI 99종목 / 2021~2025. "
            "공시·뉴스·시장국면 NLP는 시뮬레이션, 가격·거래량은 실데이터. "
            "(KF-DeBERTa·OpenDART는 확장 계획)" + auc_text + ". "
            "B(리스크 필터)가 A를 상회하는 정도는 fold·파라미터에 따라 변동."
        )
    else:
        st.info(
            "**합성 데이터 모드**: GARCH형 패널" + auc_text + ". "
            "**실데이터 모드 + KF-DeBERTa 추가 시 B가 A를 상회할 것으로 가설**. "
            "발표에서 '솔직한 베이스라인 + 개선 방향'으로 제시 가능."
        )
