"""리스크 엔진 + Kill Switch.

기획서 §12 구현:
  §12.2 주요 규칙
    - 리스크 점수 ≥ 60 → 모의 매수 차단 (paper_trading.place_buy 내장)
    - 5일 30%+ 급등 → 고위험 (surge_5d 피처)
    - 악재 공시 → 회피
    - 시장 급락 → 신규 매수 제한 (Kill Switch)
    - 손절 -3%, 익절 +5%, 보유 5거래일
    - 일일 손실 한도 → 자동 중단 (Kill Switch)
    - 데이터 실패 → 예측 중단

  §12.3 Kill Switch
    - 데이터 오류 / 시장 급락 / 일일 손실 한도 / 모델 실패 / API 오류
      → 자동화 즉시 중단
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import pandas as pd

# ===== 임계값 (기획서 §12.2 명시) =====
STOP_LOSS_PCT = -0.03           # 손절 -3%
TAKE_PROFIT_PCT = 0.05          # 익절 +5%
MAX_HOLD_DAYS = 5               # 최대 보유 5거래일
DAILY_LOSS_LIMIT_PCT = -0.05    # 일일 손실 한도 -5%
MARKET_CRASH_THRESHOLD = 70     # 시장 평균 risk ≥ 70 → 시장 급락 판단
SECTOR_MAX_PCT = 0.40           # 단일 섹터 노출 한도 40%
SINGLE_NAME_MAX_PCT = 0.20      # 단일 종목 노출 한도 20%


@dataclass
class KillSwitchStatus:
    active: bool = False
    reasons: List[str] = field(default_factory=list)

    def banner_text(self) -> str:
        if not self.active:
            return "✅ 정상 운영 — Kill Switch 비활성"
        return "🛑 **Kill Switch 활성** — " + " / ".join(self.reasons)


def check_kill_switch(market_avg_risk: float,
                      daily_pnl_pct: float = 0.0,
                      data_error: bool = False,
                      model_error: bool = False) -> KillSwitchStatus:
    """§12.3 Kill Switch 평가. 하나라도 true면 active."""
    reasons = []
    if data_error:
        reasons.append("데이터 수집 오류")
    if model_error:
        reasons.append("모델 예측 오류")
    if market_avg_risk >= MARKET_CRASH_THRESHOLD:
        reasons.append(
            f"시장 평균 리스크 {market_avg_risk:.0f} ≥ {MARKET_CRASH_THRESHOLD} (시장 급락)")
    if daily_pnl_pct <= DAILY_LOSS_LIMIT_PCT:
        reasons.append(
            f"일일 손익 {daily_pnl_pct*100:+.1f}% ≤ 한도 {DAILY_LOSS_LIMIT_PCT*100:.0f}%")
    return KillSwitchStatus(active=bool(reasons), reasons=reasons)


def overheating_flag(snap_row) -> bool:
    """§12.2: 5일 30%+ 급등 → 고위험. snap_row['surge_5d'] 활용."""
    return bool(snap_row.get("surge_5d", 0))


def negative_disclosure_recent(snap_row) -> bool:
    """§12.2: 최근 20일 악재 공시 존재 → 회피 신호."""
    bad = snap_row.get("bad_disc_20d", 0)
    return bad is not None and bad > 0


def auto_close_check(avg_price: float, current_price: float,
                     days_held: int) -> Tuple[bool, str]:
    """§12.2: 손절/익절/보유기간 만료 자동 청산 판정.

    Returns (should_sell, reason).
    """
    if avg_price <= 0:
        return False, ""
    pnl_pct = current_price / avg_price - 1
    if pnl_pct <= STOP_LOSS_PCT:
        return True, f"손절 ({pnl_pct*100:+.2f}%)"
    if pnl_pct >= TAKE_PROFIT_PCT:
        return True, f"익절 ({pnl_pct*100:+.2f}%)"
    if days_held >= MAX_HOLD_DAYS:
        return True, f"보유기간 {days_held}일 만료"
    return False, ""


def position_concentration(positions: dict, prices: dict,
                           snap: pd.DataFrame) -> List[str]:
    """§12.2: 단일 종목·섹터 노출 한도 검사. 위반 메시지 리스트 반환."""
    warnings = []
    total_mv = sum(p.market_value(prices.get(p.stock_id, p.avg_price))
                   for p in positions.values() if p.qty > 0)
    if total_mv <= 0:
        return warnings
    # 단일 종목 한도
    for sid, pos in positions.items():
        if pos.qty == 0:
            continue
        mv = pos.market_value(prices.get(sid, pos.avg_price))
        pct = mv / total_mv
        if pct > SINGLE_NAME_MAX_PCT:
            warnings.append(
                f"단일 종목 한도 위반: {pos.name} {pct*100:.0f}% "
                f"(한도 {SINGLE_NAME_MAX_PCT*100:.0f}%)")
    # 섹터 한도
    sector_mv = {}
    for sid, pos in positions.items():
        if pos.qty == 0:
            continue
        row = snap[snap["stock_id"].astype(str) == str(sid)]
        sector = row.iloc[0]["sector"] if not row.empty else "기타"
        mv = pos.market_value(prices.get(sid, pos.avg_price))
        sector_mv[sector] = sector_mv.get(sector, 0) + mv
    for sec, mv in sector_mv.items():
        pct = mv / total_mv
        if pct > SECTOR_MAX_PCT:
            warnings.append(
                f"섹터 한도 위반: {sec} {pct*100:.0f}% "
                f"(한도 {SECTOR_MAX_PCT*100:.0f}%)")
    return warnings


def summarize_rules() -> List[str]:
    """발표/UI 표시용 규칙 요약."""
    return [
        f"리스크 점수 ≥ 60 → 매수 차단",
        f"5일 30%+ 급등 → 고위험 표시",
        f"악재 공시 (최근 20일) → 회피 신호",
        f"시장 평균 리스크 ≥ {MARKET_CRASH_THRESHOLD} → Kill Switch (신규 매수 중단)",
        f"손절 {STOP_LOSS_PCT*100:.0f}% / 익절 +{TAKE_PROFIT_PCT*100:.0f}% / 보유 {MAX_HOLD_DAYS}일",
        f"일일 손실 {DAILY_LOSS_LIMIT_PCT*100:.0f}% 초과 → Kill Switch",
        f"단일 종목 노출 ≤ {SINGLE_NAME_MAX_PCT*100:.0f}% / 단일 섹터 ≤ {SECTOR_MAX_PCT*100:.0f}%",
    ]
