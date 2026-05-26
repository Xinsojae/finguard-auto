"""risk 엔진 + Kill Switch 단위 테스트."""
import pytest
from core.risk import (
    check_kill_switch, auto_close_check, position_concentration,
    summarize_rules,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT, MAX_HOLD_DAYS,
    DAILY_LOSS_LIMIT_PCT, MARKET_CRASH_THRESHOLD,
)


# ---------- Kill Switch ----------
def test_kill_switch_inactive_normal():
    ks = check_kill_switch(market_avg_risk=40, daily_pnl_pct=0.0)
    assert not ks.active
    assert ks.reasons == []


def test_kill_switch_active_market_crash():
    ks = check_kill_switch(market_avg_risk=75, daily_pnl_pct=0.0)
    assert ks.active
    assert any("시장 평균 리스크" in r for r in ks.reasons)


def test_kill_switch_active_daily_loss():
    ks = check_kill_switch(market_avg_risk=40, daily_pnl_pct=-0.08)
    assert ks.active
    assert any("일일 손익" in r for r in ks.reasons)


def test_kill_switch_active_data_error():
    ks = check_kill_switch(market_avg_risk=40, daily_pnl_pct=0.0,
                           data_error=True)
    assert ks.active
    assert any("데이터" in r for r in ks.reasons)


def test_kill_switch_multiple_reasons():
    ks = check_kill_switch(market_avg_risk=80, daily_pnl_pct=-0.10,
                           model_error=True)
    assert ks.active
    assert len(ks.reasons) == 3


def test_kill_switch_threshold_boundary():
    """경계값 정확성: market_avg_risk == MARKET_CRASH_THRESHOLD → active."""
    ks = check_kill_switch(market_avg_risk=MARKET_CRASH_THRESHOLD,
                           daily_pnl_pct=0.0)
    assert ks.active


# ---------- 자동 청산 ----------
def test_stop_loss_triggers():
    should, reason = auto_close_check(avg_price=10000,
                                       current_price=9690,  # -3.1%
                                       days_held=2)
    assert should
    assert "손절" in reason


def test_take_profit_triggers():
    should, reason = auto_close_check(avg_price=10000,
                                       current_price=10510,  # +5.1%
                                       days_held=2)
    assert should
    assert "익절" in reason


def test_hold_expiry_triggers():
    should, reason = auto_close_check(avg_price=10000,
                                       current_price=10100,  # +1%
                                       days_held=MAX_HOLD_DAYS)
    assert should
    assert "보유기간" in reason


def test_no_close_in_normal_range():
    should, reason = auto_close_check(avg_price=10000,
                                       current_price=10200,  # +2%
                                       days_held=2)
    assert not should
    assert reason == ""


def test_no_close_avg_price_zero():
    """avg_price 0이면 ZeroDivisionError 없이 안전 return."""
    should, reason = auto_close_check(avg_price=0.0,
                                       current_price=10000,
                                       days_held=2)
    assert not should


# ---------- 포지션 집중도 ----------
class _MockPos:
    def __init__(self, sid, name, qty, avg_price):
        self.stock_id = sid
        self.name = name
        self.qty = qty
        self.avg_price = avg_price

    def market_value(self, cur):
        return self.qty * cur


def test_concentration_no_violation():
    import pandas as pd
    positions = {
        "A": _MockPos("A", "A주", 10, 10000),  # 100,000
        "B": _MockPos("B", "B주", 10, 20000),  # 200,000
        "C": _MockPos("C", "C주", 10, 30000),  # 300,000
    }
    prices = {"A": 10000, "B": 20000, "C": 30000}
    snap = pd.DataFrame({
        "stock_id": ["A", "B", "C"],
        "sector": ["반도체", "금융", "바이오"],
    })
    warnings = position_concentration(positions, prices, snap)
    # 50% C 비중 → SINGLE_NAME_MAX_PCT 20% 위반
    assert any("단일 종목" in w for w in warnings)


def test_concentration_empty_safe():
    import pandas as pd
    warnings = position_concentration({}, {}, pd.DataFrame())
    assert warnings == []


# ---------- 규칙 요약 ----------
def test_summarize_rules_returns_list():
    rules = summarize_rules()
    assert isinstance(rules, list)
    assert len(rules) >= 6  # §12.2 핵심 규칙 최소 6개
