"""기술지표·뉴스·공시·시장국면 → 학습용 피처 생성."""
import pandas as pd
import streamlit as st


@st.cache_data
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """패널에 lag/rolling 피처 + 학습 타깃 추가.

    데이터 누수 방지: 모든 피처는 .shift(1) (전일까지의 정보만 사용).
    타깃 fwd_ret_5d는 .shift(-5) (5일 후 수익률).
    """
    df = df.sort_values(["stock_id", "date"]).copy()
    g = df.groupby("stock_id", group_keys=False)

    df["ret_1d"] = g["return"].transform(lambda s: s.shift(1))
    df["ret_5d"] = g["close"].transform(lambda s: s.pct_change(5).shift(1))
    df["ret_20d"] = g["close"].transform(lambda s: s.pct_change(20).shift(1))
    df["vol_5d"] = g["return"].transform(lambda s: s.rolling(5).std().shift(1))
    df["vol_20d"] = g["return"].transform(lambda s: s.rolling(20).std().shift(1))
    df["ma_ratio"] = g["close"].transform(
        lambda s: (s.rolling(5).mean() / s.rolling(20).mean() - 1).shift(1))
    df["vol_z_20"] = g["volume"].transform(
        lambda s: ((s - s.rolling(20).mean()) / s.rolling(20).std()).shift(1))
    df["surge_5d"] = g["close"].transform(
        lambda s: (s.pct_change(5).shift(1) > 0.30)).astype(int)
    df["drawdown_20"] = g["close"].transform(
        lambda s: (s / s.rolling(20).max() - 1).shift(1))
    df["news_sent_lag1"] = g["news_sent"].transform(lambda s: s.shift(1))
    df["news_sent_ma5"] = g["news_sent"].transform(
        lambda s: s.rolling(5).mean().shift(1))
    df["disclosure_lag1"] = g["disclosure"].transform(lambda s: s.shift(1))
    # 룰베이스 분류 결과 (-3~+3 범위) 호환: 부호 기반 집계
    df["bad_disc_20d"] = g["disclosure"].transform(
        lambda s: (s < 0).rolling(20).sum().shift(1))
    df["good_disc_20d"] = g["disclosure"].transform(
        lambda s: (s > 0).rolling(20).sum().shift(1))
    df["disc_severity_20d"] = g["disclosure"].transform(
        lambda s: s.abs().rolling(20).sum().shift(1))
    df["regime_lag1"] = g["regime"].transform(lambda s: s.shift(1))
    df["fwd_ret_5d"] = g["close"].transform(lambda s: s.pct_change(5).shift(-5))
    df["target_up"] = (df["fwd_ret_5d"] > df["fwd_ret_5d"].quantile(0.70)).astype(int)
    df["target_crash"] = (df["fwd_ret_5d"] < -0.05).astype(int)
    return df
