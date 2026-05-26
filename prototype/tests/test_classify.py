"""분류·태그·신뢰도 헬퍼 테스트."""
import pytest
from core.ui import classify, tag_html


# ---------- classify ----------
@pytest.mark.parametrize("up,risk,expected", [
    (80, 20, "PRIORITY"),
    (80, 70, "HIGH-RISK"),
    (20, 20, "HOLD"),
    (20, 70, "AVOID"),
    (50, 49, "PRIORITY"),   # 경계: up=th, risk<th
    (50, 50, "HIGH-RISK"),  # 경계: up=th, risk=th
    (49, 49, "HOLD"),       # 경계: up<th, risk<th
    (49, 50, "AVOID"),      # 경계: up<th, risk=th
])
def test_classify_quadrants(up, risk, expected):
    assert classify(up, risk) == expected


def test_classify_custom_thresholds():
    assert classify(40, 40, up_th=30, risk_th=30) == "HIGH-RISK"
    assert classify(40, 40, up_th=60, risk_th=60) == "HOLD"


# ---------- tag_html ----------
@pytest.mark.parametrize("cat", ["PRIORITY", "HIGH-RISK", "HOLD", "AVOID"])
def test_tag_html_has_class(cat):
    html = tag_html(cat)
    assert "tag-" in html
    assert "<span" in html


def test_tag_html_unknown_raises():
    with pytest.raises(KeyError):
        tag_html("INVALID")
