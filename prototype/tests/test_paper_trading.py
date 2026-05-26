"""paper_trading 거래 가드 + 장부 무결성 (codex #6 회귀 방지)."""
import pytest
from core.paper_trading import (
    Portfolio, INITIAL_CASH, MAX_RISK_FOR_BUY,
    place_buy, place_sell, auto_buy_priority,
)


@pytest.fixture
def pf():
    return Portfolio()


# ---------- 매수 가드 ----------
def test_buy_risk_score_blocks(pf):
    err = place_buy(pf, "005930", "삼성전자", 10, 50000,
                    risk_score=70, reason="test")
    assert err, "리스크 70 종목 매수 차단되지 않음"
    assert pf.blocked_buys == 1
    assert pf.cash == INITIAL_CASH
    assert "005930" not in pf.positions


def test_buy_risk_score_allows_low(pf):
    err = place_buy(pf, "005930", "삼성전자", 10, 50000,
                    risk_score=30, reason="test")
    assert not err
    assert pf.positions["005930"].qty == 10


def test_buy_kill_switch_blocks(pf):
    pf.kill_switch_active = True
    err = place_buy(pf, "005930", "삼성전자", 10, 50000, risk_score=10)
    assert "Kill Switch" in err
    assert pf.cash == INITIAL_CASH


def test_buy_negative_qty_rejected(pf):
    err = place_buy(pf, "005930", "삼성전자", -5, 50000, risk_score=10)
    assert err == "수량·가격은 양수여야 함"


def test_buy_insufficient_cash(pf):
    err = place_buy(pf, "005930", "삼성전자",
                    qty=1_000_000, price=50000, risk_score=10)
    assert "현금 부족" in err


def test_buy_avg_price_recalculated(pf):
    place_buy(pf, "005930", "삼성전자", 10, 50000, risk_score=10)
    place_buy(pf, "005930", "삼성전자", 10, 60000, risk_score=10)
    pos = pf.positions["005930"]
    assert pos.qty == 20
    assert abs(pos.avg_price - 55000) < 0.01


# ---------- 매도 가드 (codex #6 핵심) ----------
def test_sell_negative_qty_rejected(pf):
    """codex #6: place_sell에 음수 검증 없어 -2주 매도로 보유 증가하던 버그."""
    place_buy(pf, "005930", "삼성전자", 10, 50000, risk_score=10)
    err = place_sell(pf, "005930", -2, 50000)
    assert err == "수량·가격은 양수여야 함"
    assert pf.positions["005930"].qty == 10, \
        "음수 매도가 보유량 증가시킴 — 장부 무결성 깨짐"


def test_sell_zero_price_rejected(pf):
    place_buy(pf, "005930", "삼성전자", 10, 50000, risk_score=10)
    err = place_sell(pf, "005930", 5, 0)
    assert err == "수량·가격은 양수여야 함"
    assert pf.positions["005930"].qty == 10


def test_sell_more_than_held(pf):
    place_buy(pf, "005930", "삼성전자", 10, 50000, risk_score=10)
    err = place_sell(pf, "005930", 15, 50000)
    assert "매도 수량" in err
    assert pf.positions["005930"].qty == 10


def test_sell_no_position(pf):
    err = place_sell(pf, "999999", 5, 50000)
    assert "보유 수량 없음" in err


def test_sell_realizes_pnl(pf):
    place_buy(pf, "005930", "삼성전자", 10, 50000, risk_score=10)
    err = place_sell(pf, "005930", 10, 55000)
    assert not err
    assert pf.positions["005930"].qty == 0
    assert pf.positions["005930"].realized_pnl == 50000  # (55000-50000)*10
    # 현금 = 초기 - 매수 + 매도
    assert pf.cash == INITIAL_CASH - 500000 + 550000


# ---------- 포트폴리오 누적 ----------
def test_total_equity_no_position(pf):
    assert pf.total_equity({}) == INITIAL_CASH
    assert pf.total_pnl({}) == 0


def test_total_equity_with_position(pf):
    place_buy(pf, "005930", "삼성전자", 10, 50000, risk_score=10)
    prices = {"005930": 55000}
    equity = pf.total_equity(prices)
    # 현금(초기-50만) + 평가액(10주×55000=55만) = 초기+5만
    assert equity == INITIAL_CASH + 50000
    assert pf.total_pnl(prices) == 50000


def test_reset_clears_state(pf):
    place_buy(pf, "005930", "삼성전자", 10, 50000, risk_score=10)
    place_buy(pf, "000660", "SK하이닉스", 5, 100000, risk_score=20)
    pf.reset()
    assert pf.cash == INITIAL_CASH
    assert len(pf.positions) == 0
    assert len(pf.trades) == 0
    assert pf.blocked_buys == 0
    assert pf.kill_switch_active is False
