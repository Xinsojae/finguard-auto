"""Tab: 운영 (MLOps · 캘린더 · 주간 리포트).

mocks: MLflow 실험 추적, 이벤트 캘린더, 주간 HTML 리포트.
"""
from datetime import date
import streamlit as st
import pandas as pd

from core import mocks
from core.ui_kit import demo_badge, section_header, info_card, download_csv_button
from tabs import AppCtx


def render(ctx: AppCtx) -> None:
    st.subheader("🛠️ 운영 · MLOps · 리포트")
    st.markdown(
        f"기획서 §20 · §8.2 운영 기능 미리보기. {demo_badge('MLflow/캘린더는 mock')} · "
        "주간 리포트는 실제 HTML 생성.",
        unsafe_allow_html=True,
    )

    sub = st.radio(
        "운영 모듈",
        ["📊 MLflow 실험 추적", "📅 이벤트 캘린더", "📝 주간 리스크 리포트"],
        horizontal=True,
    )

    if sub.startswith("📊"):
        _render_mlflow()
    elif sub.startswith("📅"):
        _render_calendar(ctx)
    elif sub.startswith("📝"):
        _render_report(ctx)


# ============================================================
def _render_mlflow() -> None:
    section_header("MLflow 실험 추적",
                   "기획서 §20.2 — 모델 버전·성능·파라미터 이력. "
                   "현재는 mock 8건 (실제 MLflow는 SQLite/S3 백엔드 필요).",
                   icon="📊", demo=True)
    df = mocks.mlflow_mock_experiments()
    # 상태별 카운트
    counts = df["status"].value_counts()
    cols = st.columns(len(counts))
    for col, (status, n) in zip(cols, counts.items()):
        col.metric(status, n)

    st.divider()
    st.dataframe(df, use_container_width=True, hide_index=True)
    download_csv_button(df, "MLflow 실험 로그 CSV 다운로드",
                        "mlflow_experiments.csv", key="op_dl_mlflow")
    st.caption(
        "실제 MLflow CLI: `mlflow ui --backend-store-uri ./mlruns` → "
        "localhost:5000에서 실험 비교, 파라미터 sweep 결과 시각화. 본 데모는 mock."
    )

    # ROC-AUC 비교 차트 (실측 있는 것만)
    import matplotlib.pyplot as plt
    real_rows = df[df["ROC_AUC_up"].notna()].copy()
    if not real_rows.empty:
        st.markdown("**실험별 ROC-AUC 비교 (상승 모델)**")
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.barh(real_rows["model"], real_rows["ROC_AUC_up"],
                color="#81C784", alpha=0.85)
        ax.axvline(0.5, color="#BDBDBD", ls="--", lw=0.8, label="random")
        ax.set_xlim(0.5, 0.65)
        ax.set_xlabel("ROC-AUC", color="#666")
        ax.tick_params(colors="#666")
        for sp in ax.spines.values():
            sp.set_color("#E0E0E0")
        ax.legend(loc="best", frameon=False)
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)


# ============================================================
def _render_calendar(ctx: AppCtx) -> None:
    section_header("이벤트 캘린더",
                   "기획서 §8.2 V2 — 다가오는 공시·실적·배당·주총 일정. "
                   "OpenDART 일정 API 연동 시 실데이터로 교체.",
                   icon="📅", demo=True)
    snap = ctx.snap
    df = mocks.event_calendar_mock(snap, n=12)
    if df.empty:
        st.info("이벤트 데이터 없음.")
        return
    # 가까운 순으로
    c1, c2, c3 = st.columns(3)
    c1.metric("총 이벤트", len(df))
    c2.metric("D-7 이내", int((df["예정일"] <= (date(2026, 5, 27) +
                                         pd.Timedelta(days=7)).date()).sum()))
    c3.metric("D-30 이내", len(df))
    st.divider()
    st.dataframe(df, use_container_width=True, hide_index=True)
    download_csv_button(df, "캘린더 CSV 다운로드",
                        "event_calendar.csv", key="op_dl_cal")
    st.caption("실데이터 연동: OpenDART 공시 일정 API + KRX 휴장일 API. 현재는 mock 12건.")


# ============================================================
def _render_report(ctx: AppCtx) -> None:
    section_header("주간 리스크 리포트",
                   "기획서 §8.2 V2 — 주간 리스크 요약 HTML 자동 생성. "
                   "다운로드 후 메일·Slack 발송 가능.",
                   icon="📝")
    snap = ctx.snap
    html = mocks.weekly_report_html(snap)
    info_card("리포트 요약 (당주)",
              f"- 우선 관심 후보: {int((snap['category']=='PRIORITY').sum())}건<br>"
              f"- 고위험 관심: {int((snap['category']=='HIGH-RISK').sum())}건<br>"
              f"- 회피 후보: {int((snap['category']=='AVOID').sum())}건<br>"
              f"- 시장 평균 리스크: {snap['score_risk'].mean():.1f}/100",
              color="#FFB74D")
    st.download_button(
        label="📥 주간 리포트 HTML 다운로드",
        data=html.encode("utf-8"),
        file_name=f"FinGuard_weekly_{date.today()}.html",
        mime="text/html",
        type="primary",
    )
    with st.expander("리포트 미리보기 (HTML 렌더링)"):
        st.components.v1.html(html, height=700, scrolling=True)
