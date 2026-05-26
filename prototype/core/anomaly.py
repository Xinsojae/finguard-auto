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

ANOMALY_FEATS = [
    "ret_1d", "ret_5d", "vol_5d", "vol_20d",
    "vol_z_20", "drawdown_20", "ma_ratio",
]


@st.cache_resource
def train_anomaly_detector(panel: pd.DataFrame, contamination: float = 0.05):
    """패널 전체로 IsolationForest 학습.

    contamination: 이상치 비율 가정 (기본 5%).
    """
    df = panel.dropna(subset=ANOMALY_FEATS)
    if df.empty:
        return None
    # 시간순 학습 (랜덤 분할 금지)
    n = len(df)
    cut = int(n * 0.7)
    tr = df.iloc[:cut][ANOMALY_FEATS]
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
