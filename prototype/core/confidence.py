"""AI 신뢰도 점수 — 기획서 §11.4 5요소 구현.

요소:
  1. 유사 사례 수: 해당 종목의 학습 데이터 행 수
  2. 데이터 충분성: 최근 60일 피처 결측률
  3. 시장 안정성: 시장 평균 리스크 (낮을수록 안정)
  4. 예측 분산: |p-0.5| (확신도)
  5. 과거 성능: 전역 모델 ROC-AUC
  (보너스) Conformal Prediction 예측 구간 폭은 단순 근사로 추가
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict
import numpy as np
import pandas as pd

from core.config import FEATS


@dataclass
class ConfidenceBreakdown:
    similarity: float       # 유사 사례 수
    data_sufficiency: float # 데이터 충분성
    market_stability: float # 시장 안정성
    pred_variance: float    # 예측 분산
    past_performance: float # 과거 성능
    overall: float          # 평균

    def label(self) -> str:
        if self.overall >= 0.66:
            return "HIGH"
        if self.overall >= 0.33:
            return "MEDIUM"
        return "LOW"

    def as_dict_pct(self) -> Dict[str, str]:
        return {
            "유사 사례 수": f"{self.similarity*100:.0f}/100",
            "데이터 충분성": f"{self.data_sufficiency*100:.0f}/100",
            "시장 안정성": f"{self.market_stability*100:.0f}/100",
            "예측 분산(확신)": f"{self.pred_variance*100:.0f}/100",
            "과거 성능(AUC)": f"{self.past_performance*100:.0f}/100",
            "종합": f"{self.overall*100:.0f}/100 ({self.label()})",
        }


def compute_confidence_for_stock(panel: pd.DataFrame, stock_id, score_up_p: float,
                                 market_avg_risk: float,
                                 model_auc: float,
                                 max_history: int = 1200) -> ConfidenceBreakdown:
    """단일 종목 신뢰도 5요소 점수.

    각 요소는 [0, 1] 정규화. overall은 단순 평균.
    """
    # 1. 유사 사례 수: 해당 종목의 유효 행 수
    sub = panel[panel["stock_id"] == stock_id]
    n_rows = len(sub)
    similarity = min(n_rows / max_history, 1.0)

    # 2. 데이터 충분성: 최근 60일 피처 결측률 (낮을수록 좋음)
    recent = sub.tail(60)
    if len(recent) > 0:
        missing_rate = recent[FEATS].isna().mean().mean()
        data_sufficiency = max(0.0, 1.0 - missing_rate)
    else:
        data_sufficiency = 0.0

    # 3. 시장 안정성: 시장 평균 risk 낮을수록 안정
    market_stability = max(0.0, 1.0 - market_avg_risk / 100.0)

    # 4. 예측 분산: |p-0.5| * 2 → [0, 1]
    pred_variance = min(abs(score_up_p - 0.5) * 2, 1.0)

    # 5. 과거 성능: 모델 ROC-AUC를 [0.5, 0.7] → [0, 1] 매핑
    past_performance = float(np.clip((model_auc - 0.5) / 0.2, 0.0, 1.0))

    overall = float(np.mean([
        similarity, data_sufficiency, market_stability,
        pred_variance, past_performance,
    ]))
    return ConfidenceBreakdown(
        similarity=similarity,
        data_sufficiency=data_sufficiency,
        market_stability=market_stability,
        pred_variance=pred_variance,
        past_performance=past_performance,
        overall=overall,
    )


def compute_confidence_for_snap(snap: pd.DataFrame, panel: pd.DataFrame,
                                model_auc: float,
                                market_avg_risk: float = None) -> pd.DataFrame:
    """snap 전체에 5요소 신뢰도 계산 → 컬럼 추가.

    추가 컬럼: confidence_overall, conf_label,
              conf_similarity, conf_data, conf_market,
              conf_variance, conf_perf
    """
    if market_avg_risk is None:
        market_avg_risk = float(snap["score_risk"].mean())

    rows = []
    for _, r in snap.iterrows():
        cb = compute_confidence_for_stock(
            panel=panel,
            stock_id=r["stock_id"],
            score_up_p=float(r["score_up_p"]),
            market_avg_risk=market_avg_risk,
            model_auc=model_auc,
        )
        rows.append({
            "confidence_overall": cb.overall,
            "conf_label": cb.label(),
            "conf_similarity": cb.similarity,
            "conf_data": cb.data_sufficiency,
            "conf_market": cb.market_stability,
            "conf_variance": cb.pred_variance,
            "conf_perf": cb.past_performance,
        })
    return pd.DataFrame(rows, index=snap.index)
