"""공통 UI 컴포넌트 — DEMO 배지, 카드, 섹션 헤더 등.

탭별 중복 줄이고 디자인 일관성 확보.
"""
import streamlit as st


def demo_badge(label: str = "DEMO MODE") -> str:
    """노란 배지 — mock 출력 명시. HTML 문자열 반환."""
    return (
        f"<span style='background:#FFF59D;color:#5C5018;padding:3px 10px;"
        f"border-radius:10px;font-size:0.74em;font-weight:600;"
        f"border:1px solid #F9A825;'>🧪 {label}</span>"
    )


def section_header(title: str, subtitle: str = "",
                   icon: str = "", demo: bool = False) -> None:
    """일관된 섹션 헤더. 좌측 색 바 + 타이틀 + 부제."""
    badge = demo_badge() if demo else ""
    icon_html = f"<span style='font-size:1.4em;margin-right:8px;'>{icon}</span>" if icon else ""
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;"
        f"padding:14px 0 6px 0;border-bottom:2px solid #F0F0F0;margin-bottom:12px;'>"
        f"{icon_html}"
        f"<h3 style='margin:0;color:#333;font-weight:600;'>{title}</h3>"
        f"{badge}"
        f"</div>",
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(
            f"<p style='color:#888;font-size:0.9em;margin:0 0 14px 0;'>{subtitle}</p>",
            unsafe_allow_html=True,
        )


def info_card(title: str, body: str, color: str = "#5B8DEF") -> None:
    """좌측 색 바 + 본문 카드."""
    st.markdown(
        f"<div style='border-left:4px solid {color};background:#FAFAFA;"
        f"padding:12px 16px;border-radius:6px;margin:8px 0;'>"
        f"<b style='color:#424242;'>{title}</b>"
        f"<div style='color:#555;font-size:0.92em;margin-top:6px;line-height:1.5;'>{body}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def status_pill(text: str, kind: str = "info") -> str:
    """상태 알약 (small badge). kind: ok/warn/risk/info."""
    palette = {
        "ok":   ("#C8E6C9", "#2E5933"),
        "warn": ("#FFE0B2", "#7A4E13"),
        "risk": ("#FFCDD2", "#8B2D2D"),
        "info": ("#E3F2FD", "#1565C0"),
    }
    bg, fg = palette.get(kind, palette["info"])
    return (
        f"<span style='background:{bg};color:{fg};padding:2px 10px;"
        f"border-radius:10px;font-size:0.78em;font-weight:600;'>{text}</span>"
    )


def metric_row(metrics: list) -> None:
    """대시보드용 메트릭 가로 정렬. metrics: list of (label, value, delta_opt)."""
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        if len(m) == 3:
            col.metric(m[0], m[1], delta=m[2])
        else:
            col.metric(m[0], m[1])
