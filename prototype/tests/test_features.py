"""features.make_features 테스트 — codex #1 회귀 방지."""
import numpy as np
import pandas as pd
from core.features import make_features
from core.config import FEATS


def test_target_nan_for_last_5_days(small_panel):
    """fwd_ret_5d NaN 행은 target_up·target_crash도 NaN이어야 함.

    이전 버그: NaN < 임계값 = False → .astype(int) = 0 → 가짜 음성 라벨로 학습됨.
    """
    df = make_features(small_panel)
    n_stocks = df["stock_id"].nunique()
    expected_nan = n_stocks * 5  # 종목별 마지막 5일

    assert df["target_up"].isna().sum() == expected_nan, \
        f"target_up NaN 개수 불일치 (예상 {expected_nan}, 실제 {df['target_up'].isna().sum()})"
    assert df["target_crash"].isna().sum() == expected_nan, \
        f"target_crash NaN 개수 불일치 (예상 {expected_nan}, 실제 {df['target_crash'].isna().sum()})"


def test_target_nan_aligned_with_fwd_ret(small_panel):
    """target NaN과 fwd_ret_5d NaN이 정확히 일치해야 함."""
    df = make_features(small_panel)
    fwd_na = df["fwd_ret_5d"].isna()
    up_na = df["target_up"].isna()
    crash_na = df["target_crash"].isna()
    assert (fwd_na == up_na).all(), "fwd_ret_5d NaN과 target_up NaN 비정렬"
    assert (fwd_na == crash_na).all(), "fwd_ret_5d NaN과 target_crash NaN 비정렬"


def test_target_binary_values(small_panel):
    """NaN 제외하면 모든 라벨은 0 또는 1."""
    df = make_features(small_panel)
    up_valid = df["target_up"].dropna().unique()
    crash_valid = df["target_crash"].dropna().unique()
    assert set(up_valid).issubset({0.0, 1.0}), f"target_up 비정상 값: {up_valid}"
    assert set(crash_valid).issubset({0.0, 1.0}), f"target_crash 비정상 값: {crash_valid}"


def test_features_lag_no_leakage(small_panel):
    """모든 피처는 shift(1) 적용 → 첫 행은 반드시 NaN."""
    df = make_features(small_panel)
    df_sorted = df.sort_values(["stock_id", "date"])
    first_rows = df_sorted.groupby("stock_id").head(1)
    for feat in ["ret_1d", "ret_5d", "vol_5d", "ma_ratio"]:
        assert first_rows[feat].isna().all(), \
            f"{feat} 첫 행이 NaN 아님 — shift(1) 누락 가능"


def test_all_feats_columns_exist(small_panel):
    """FEATS 리스트의 모든 컬럼이 생성됨."""
    df = make_features(small_panel)
    missing = [f for f in FEATS if f not in df.columns]
    assert not missing, f"누락 컬럼: {missing}"
