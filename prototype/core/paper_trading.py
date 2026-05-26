"""모의투자 (Paper Trading) 엔진.

기획서 §15.2(모의투자 화면) + §16.1(모의 자동운용 흐름) 구현.

특징:
  - 가상 자금 1,000만원으로 시작
  - 종목 매수/매도 → 가상 포지션 관리
  - 손익 계산: (현재가 - 평균단가) × 수량
  - 리스크 점수 60+ 종목 매수 차단 (§12.2)
  - 거래내역 로그 (CSV 직렬화 가능)
  - st.session_state 기반 — 페이지 리프레시 후에도 세션 내에서는 유지

세부 구조 (§12 리스크 엔진은 core/risk.py에서 별도 통합 예정).
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime

INITIAL_CASH = 10_000_000  # 1,000만원
MAX_RISK_FOR_BUY = 60      # 리스크 60+ 매수 차단 (§12.2)


@dataclass
class Position:
    """단일 종목 보유 정보."""
    stock_id: str
    name: str
    qty: int = 0
    avg_price: float = 0.0          # 평균 매수단가
    realized_pnl: float = 0.0        # 청산 손익 누적

    def cost_basis(self) -> float:
        return self.qty * self.avg_price

    def market_value(self, current_price: float) -> float:
        return self.qty * current_price

    def unrealized_pnl(self, current_price: float) -> float:
        return (current_price - self.avg_price) * self.qty


@dataclass
class Trade:
    """단일 거래 기록."""
    ts: str               # 체결 시각 (str로 직렬화 용이)
    side: str             # "BUY" / "SELL"
    stock_id: str
    name: str
    qty: int
    price: float
    amount: float         # qty * price
    reason: str = ""      # "수동" / "자동 (PRIORITY)" / "익절" 등


@dataclass
class Portfolio:
    """모의 포트폴리오 전체 상태."""
    cash: float = INITIAL_CASH
    positions: Dict[str, Position] = field(default_factory=dict)
    trades: List[Trade] = field(default_factory=list)
    blocked_buys: int = 0   # 리스크 차단된 매수 시도 수
    kill_switch_active: bool = False

    def total_equity(self, prices: Dict[str, float]) -> float:
        mv = sum(p.market_value(prices.get(p.stock_id, p.avg_price))
                 for p in self.positions.values())
        return self.cash + mv

    def total_pnl(self, prices: Dict[str, float]) -> float:
        return self.total_equity(prices) - INITIAL_CASH

    def reset(self) -> None:
        self.cash = INITIAL_CASH
        self.positions.clear()
        self.trades.clear()
        self.blocked_buys = 0
        self.kill_switch_active = False


def place_buy(pf: Portfolio, stock_id: str, name: str, qty: int,
              price: float, risk_score: int = 0, reason: str = "수동") -> str:
    """모의 매수. 성공 시 빈 문자열, 차단 시 사유 반환."""
    if pf.kill_switch_active:
        return "Kill Switch 활성화 — 신규 매수 차단"
    if risk_score >= MAX_RISK_FOR_BUY:
        pf.blocked_buys += 1
        return f"리스크 점수 {risk_score} ≥ {MAX_RISK_FOR_BUY} — 매수 차단 (§12.2)"
    if qty <= 0 or price <= 0:
        return "수량·가격은 양수여야 함"
    amount = qty * price
    if amount > pf.cash:
        return f"현금 부족: 필요 {amount:,.0f}원 > 보유 {pf.cash:,.0f}원"

    pf.cash -= amount
    pos = pf.positions.get(stock_id)
    if pos is None:
        pf.positions[stock_id] = Position(
            stock_id=stock_id, name=name, qty=qty, avg_price=price)
    else:
        total_cost = pos.cost_basis() + amount
        pos.qty += qty
        pos.avg_price = total_cost / pos.qty
    pf.trades.append(Trade(
        ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        side="BUY", stock_id=stock_id, name=name,
        qty=qty, price=price, amount=amount, reason=reason,
    ))
    return ""


def place_sell(pf: Portfolio, stock_id: str, qty: int, price: float,
               reason: str = "수동") -> str:
    if qty <= 0 or price <= 0:
        return "수량·가격은 양수여야 함"
    pos = pf.positions.get(stock_id)
    if pos is None or pos.qty == 0:
        return f"{stock_id} 보유 수량 없음"
    if qty > pos.qty:
        return f"매도 수량 {qty} > 보유 {pos.qty}"
    amount = qty * price
    realized = (price - pos.avg_price) * qty
    pos.realized_pnl += realized
    pos.qty -= qty
    pf.cash += amount
    pf.trades.append(Trade(
        ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        side="SELL", stock_id=stock_id, name=pos.name,
        qty=qty, price=price, amount=amount, reason=reason,
    ))
    if pos.qty == 0:
        pos.avg_price = 0.0
    return ""


def auto_buy_priority(pf: Portfolio, snap: pd.DataFrame, k: int = 5,
                      cash_per_pick_pct: float = 0.10) -> List[str]:
    """PRIORITY 종목 상위 k개 자동 매수 (§16.1 모의 자동운용).

    cash_per_pick_pct: 종목당 매수 금액 = 초기 현금 × 비율.
    반환: 결과 메시지 리스트 (성공/차단 사유).
    """
    if pf.kill_switch_active:
        return ["Kill Switch 활성화 — 자동 매수 중단"]
    priority = snap[snap["category"] == "PRIORITY"].nlargest(k, "score_up")
    if priority.empty:
        return ["PRIORITY 종목 없음 — 자동 매수 보류"]
    budget_per = INITIAL_CASH * cash_per_pick_pct
    msgs = []
    for _, r in priority.iterrows():
        price = float(r["close"])
        qty = max(int(budget_per // price), 1)
        sid = str(r["stock_id"])
        err = place_buy(pf, sid, r["name"], qty, price,
                        risk_score=int(r["score_risk"]),
                        reason=f"자동 PRIORITY (up={r['score_up']})")
        msgs.append(f"[{r['name']}] " + (err if err else f"매수 {qty}주 @ {price:,.0f}원"))
    return msgs


def positions_dataframe(pf: Portfolio, prices: Dict[str, float]) -> pd.DataFrame:
    """보유 종목 표시용 DataFrame."""
    if not pf.positions:
        return pd.DataFrame(columns=[
            "종목", "수량", "평균단가", "현재가", "평가액", "평가손익", "수익률%",
        ])
    rows = []
    for sid, pos in pf.positions.items():
        if pos.qty == 0:
            continue
        cur = prices.get(sid, pos.avg_price)
        upnl = pos.unrealized_pnl(cur)
        roi = (cur / pos.avg_price - 1) * 100 if pos.avg_price > 0 else 0
        rows.append({
            "종목": pos.name, "수량": pos.qty,
            "평균단가": f"{pos.avg_price:,.0f}",
            "현재가": f"{cur:,.0f}",
            "평가액": f"{pos.market_value(cur):,.0f}",
            "평가손익": f"{upnl:+,.0f}",
            "수익률%": f"{roi:+.2f}%",
        })
    return pd.DataFrame(rows)


def trades_dataframe(pf: Portfolio) -> pd.DataFrame:
    if not pf.trades:
        return pd.DataFrame(columns=["시각", "구분", "종목", "수량", "가격", "금액", "사유"])
    rows = []
    for t in pf.trades:
        rows.append({
            "시각": t.ts, "구분": t.side,
            "종목": t.name, "수량": t.qty,
            "가격": f"{t.price:,.0f}",
            "금액": f"{t.amount:,.0f}",
            "사유": t.reason,
        })
    return pd.DataFrame(rows)
