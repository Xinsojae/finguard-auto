"""Quantile Regression 기반 예상 최악 손실 구간.

기획서 §11.2 리스크 점수 8요소 중 마지막 '예상 최악 손실 구간':
LightGBM quantile objective로 fwd_ret_5d의 하위 분위(예: 10%)를 예측.

해석:
  var_5d_p10 = -0.05 → "5일 후 하위 10% 시나리오에서 -5% 손실 예상"
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import lightgbm as lgb

from core.config import FEATS


QUANTILE_EMBARGO_DAYS = 5


@st.cache_resource
def train_quantile_model(panel: pd.DataFrame, alpha: float = 0.10):
    """fwd_ret_5d의 alpha 분위(낮은 쪽) 예측 — unique date 70/30 + embargo.

    이전 버그: iloc[:70%] 행 분할 → 종목군 분할이라 시간 분할 아니었음.
    """
    df = panel.dropna(subset=FEATS + ["fwd_ret_5d"]).reset_index(drop=True)
    if len(df) < 200:
        return None
    dates = np.array(sorted(df["date"].unique()))
    n_dates = len(dates)
    if n_dates < 30:
        tr = df
    else:
        cut_idx = int(n_dates * 0.7)
        train_end_idx = max(cut_idx - QUANTILE_EMBARGO_DAYS, 1)
        train_dates = dates[:train_end_idx]
        tr = df[df["date"].isin(train_dates)]
    if len(tr) < 100:
        return None
    model = lgb.LGBMRegressor(
        objective="quantile", alpha=alpha,
        num_leaves=31, learning_rate=0.05,
        n_estimators=300, min_data_in_leaf=200,
        verbose=-1, random_state=42,
    )
    model.fit(tr[FEATS], tr["fwd_ret_5d"])
    return model


def predict_quantile(model, snap: pd.DataFrame) -> np.ndarray:
    """예상 5일 분위 수익률 (보통 음수). 빈 모델 시 0."""
    if model is None:
        return np.zeros(len(snap))
    return model.predict(snap[FEATS].fillna(0))
