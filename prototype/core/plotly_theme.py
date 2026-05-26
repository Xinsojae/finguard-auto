"""Plotly 차트 다크/라이트 테마 헬퍼.

각 차트는 layout_kwargs(dark) 결과를 fig.update_layout(**...)에 전개.
"""
from __future__ import annotations
import streamlit as st


def is_dark() -> bool:
    return bool(st.session_state.get("dark_mode", False))


# Light / Dark 토큰
LIGHT = {
    "paper_bgcolor": "#FFFFFF",
    "plot_bgcolor": "#FFFFFF",
    "font_color": "#4B5563",
    "title_color": "#0F1419",
    "grid_color": "#EAECEF",
    "axis_line_color": "#D9DCE1",
    "zero_line_color": "#D9DCE1",
}
DARK = {
    "paper_bgcolor": "#11161F",
    "plot_bgcolor": "#11161F",
    "font_color": "#94A3B8",
    "title_color": "#F1F5F9",
    "grid_color": "#1F2937",
    "axis_line_color": "#2D3548",
    "zero_line_color": "#2D3548",
}


def palette(dark: bool = None) -> dict:
    if dark is None:
        dark = is_dark()
    return DARK if dark else LIGHT


def layout_kwargs(height: int = 400, dark: bool = None) -> dict:
    """fig.update_layout(**layout_kwargs(...)) 용 dict."""
    p = palette(dark)
    return dict(
        height=height,
        paper_bgcolor=p["paper_bgcolor"],
        plot_bgcolor=p["plot_bgcolor"],
        font=dict(family="-apple-system, 'Pretendard', 'Malgun Gothic', sans-serif",
                  color=p["font_color"], size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(gridcolor=p["grid_color"],
                   linecolor=p["axis_line_color"],
                   zerolinecolor=p["zero_line_color"]),
        yaxis=dict(gridcolor=p["grid_color"],
                   linecolor=p["axis_line_color"],
                   zerolinecolor=p["zero_line_color"]),
    )


# 시각화 카테고리 색상 (다크/라이트 둘 다 적합한 톤)
CHART_COLORS = {
    "accent":     "#5B8DEF",
    "accent_dim": "#3B82F6",
    "success":    "#34D399",
    "warning":    "#FBBF24",
    "danger":     "#F87171",
    "neutral":    "#94A3B8",
    "ma":         "#FBBF24",
    "series_4":   ["#5B8DEF", "#F87171", "#34D399", "#FBBF24"],
}
