"""UI 유틸: 분류 라벨링·태그 HTML·CSS 주입."""
import streamlit as st
from core.config import CSS_BLOCK


def apply_css() -> None:
    """페이지 진입 시 1회 호출. 카드/태그 스타일 주입."""
    st.markdown(CSS_BLOCK, unsafe_allow_html=True)


def classify(up: int, risk: int, up_th: int = 50, risk_th: int = 50) -> str:
    """상승·리스크 점수 → 카테고리.

    PRIORITY: 상승≥up_th 리스크<risk_th
    HIGH-RISK: 상승≥up_th 리스크≥risk_th
    HOLD: 상승<up_th 리스크<risk_th
    AVOID: 그 외
    """
    if up >= up_th and risk < risk_th:
        return "PRIORITY"
    if up >= up_th and risk >= risk_th:
        return "HIGH-RISK"
    if up < up_th and risk < risk_th:
        return "HOLD"
    return "AVOID"


_TAG_MAP = {
    "PRIORITY":  "<span class='tag-priority'>우선 관심</span>",
    "HIGH-RISK": "<span class='tag-highrisk'>고위험 관심</span>",
    "HOLD":      "<span class='tag-hold'>관망</span>",
    "AVOID":     "<span class='tag-avoid'>회피</span>",
}

_TAG_PLAIN = {
    "PRIORITY":  "우선 관심 후보",
    "HIGH-RISK": "고위험 관심 후보",
    "HOLD":      "관망 후보",
    "AVOID":     "회피 후보",
}


def tag_html(cat: str) -> str:
    return _TAG_MAP[cat]


def tag_plain(cat: str) -> str:
    """HTML 없는 평문 라벨 (info 박스 등에서 사용)."""
    return _TAG_PLAIN[cat]
