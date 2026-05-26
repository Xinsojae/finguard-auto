"""Isolation Forest 이상 탐지.

기획서 §10.2 MVP 모델 #3 (이상 탐지: Isolation Forest).
거래량·변동성·수익률 패턴이 평소와 다른 종목을 탐지하여
"비정상 행동" 신호로 활용.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest

# walk-forward 누수 차단용 embargo
ANOMALY_EMBARGO_DAYS = 5

ANOMALY_FEATS = [
    "ret_1d", "ret_5d", "vol_5d", "vol_20d",
    "vol_z_20", "drawdown_20", "ma_ratio",
]


@st.cache_resource
def train_anomaly_detector(panel: pd.DataFrame, contamination: float = 0.05):
    """unique date 70/30 + embargo 적용 IsolationForest.

    이전 버그: iloc[:70%] 행 분할 → (stock_id, date) 정렬 panel에서
    종목 70% 학습 / 30% 테스트가 되어 시간 분할 아니었음. snap 최신일
    평가 시 학습/미학습 종목 혼재.
    """
    df = panel.dropna(subset=ANOMALY_FEATS)
    if df.empty:
        return None
    dates = np.array(sorted(df["date"].unique()))
    n_dates = len(dates)
    if n_dates < 30:
        # 너무 짧으면 전체로 fit (그래도 시간순 fit)
        tr = df[ANOMALY_FEATS]
    else:
        cut_idx = int(n_dates * 0.7)
        train_end_idx = max(cut_idx - ANOMALY_EMBARGO_DAYS, 1)
        train_dates = dates[:train_end_idx]
        tr = df[df["date"].isin(train_dates)][ANOMALY_FEATS]
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples=min(10000, len(tr)),
        random_state=42, n_jobs=-1,
    )
    model.fit(tr)
    return model


def score_snapshot(model, snap: pd.DataFrame) -> np.ndarray:
    """각 종목당 anomaly score (높을수록 이상). 0~100 정규화."""
    if model is None:
        return np.zeros(len(snap))
    df = snap[ANOMALY_FEATS].fillna(0)
    # decision_function: 높을수록 정상, 낮을수록 이상
    raw = -model.decision_function(df)
    # min-max 정규화 → 0~100
    if raw.max() == raw.min():
        return np.zeros(len(snap))
    return ((raw - raw.min()) / (raw.max() - raw.min()) * 100).round(0).astype(int)
