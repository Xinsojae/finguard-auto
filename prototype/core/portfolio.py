"""포트폴리오 분석 (§15.2 포트폴리오 화면).

섹터 쏠림 / 종목 상관관계 / 예상 최대 손실(VaR) / 변동성 기여도.
모의투자 Tab에서 만든 가상 포지션을 대상으로 분석.
"""
from __future__ import annotations
from typing import Dict, List
import numpy as np
import pandas as pd


def sector_concentration(positions: dict, prices: dict,
                         snap: pd.DataFrame) -> Dict[str, float]:
    """섹터별 비중 dict {sector: weight in [0,1]}."""
    total_mv = sum(p.market_value(prices.get(p.stock_id, p.avg_price))
                   for p in positions.values() if p.qty > 0)
    if total_mv <= 0:
        return {}
    out: Dict[str, float] = {}
    for sid, pos in positions.items():
        if pos.qty == 0:
            continue
        row = snap[snap["stock_id"].astype(str) == str(sid)]
        sector = row.iloc[0]["sector"] if not row.empty else "기타"
        mv = pos.market_value(prices.get(sid, pos.avg_price))
        out[sector] = out.get(sector, 0.0) + mv / total_mv
    return out


def return_correlation(panel: pd.DataFrame, stock_ids: List,
                       window: int = 60) -> pd.DataFrame:
    """최근 window일 수익률 상관 행렬."""
    ids = [str(s) for s in stock_ids]
    sub = panel[panel["stock_id"].astype(str).isin(ids)]
    pivot = sub.pivot_table(index="date", columns="stock_id", values="return")
    pivot = pivot.tail(window)
    return pivot.corr()


def portfolio_var(positions: dict, prices: dict, panel: pd.DataFrame,
                  window: int = 60) -> dict:
    """포트폴리오 일일 VaR (historical simulation)."""
    held = [(str(sid), pos) for sid, pos in positions.items() if pos.qty > 0]
    if not held:
        return {"VaR_95": 0.0, "VaR_99": 0.0,
                "expected_max_loss": 0.0, "n_days": 0, "total_mv": 0.0}
    total_mv = sum(p.market_value(prices.get(s, p.avg_price)) for s, p in held)
    weights = {s: p.market_value(prices.get(s, p.avg_price)) / total_mv
               for s, p in held}
    ids = [s for s, _ in held]
    sub = panel[panel["stock_id"].astype(str).isin(ids)]
    pivot = sub.pivot_table(index="date", columns="stock_id", values="return").tail(window)
    w_arr = np.array([weights.get(str(c), 0.0) for c in pivot.columns])
    port_ret = pivot.fillna(0).values @ w_arr
    # VaR는 손실액(양수)이어야 함 — quantile이 양수(이익만 있음)면 손실 0으로 clip
    var95 = max(-float(np.quantile(port_ret, 0.05)), 0.0) if len(port_ret) > 0 else 0.0
    var99 = max(-float(np.quantile(port_ret, 0.01)), 0.0) if len(port_ret) > 0 else 0.0
    max_loss = max(-float(port_ret.min()), 0.0) if len(port_ret) > 0 else 0.0
    return {
        "VaR_95": var95 * total_mv,
        "VaR_99": var99 * total_mv,
        "expected_max_loss": max_loss * total_mv,
        "n_days": len(port_ret),
        "total_mv": total_mv,
    }


def volatility_contribution(positions: dict, prices: dict,
                            panel: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """종목별 변동성 기여도 = 비중 × 표준편차 (단순 추정)."""
    held = [(str(sid), pos) for sid, pos in positions.items() if pos.qty > 0]
    if not held:
        return pd.DataFrame()
    total_mv = sum(p.market_value(prices.get(s, p.avg_price)) for s, p in held)
    rows = []
    for sid, pos in held:
        cur = prices.get(sid, pos.avg_price)
        mv = pos.market_value(cur)
        weight = mv / total_mv if total_mv > 0 else 0.0
        sub = panel[panel["stock_id"].astype(str) == sid].tail(window)
        std = float(sub["return"].std()) if len(sub) > 1 else 0.0
        rows.append({
            "stock_id": sid, "name": pos.name,
            "weight": weight, "vol": std,
            "contribution": weight * std,
        })
    df = pd.DataFrame(rows).sort_values("contribution", ascending=False)
    return df
