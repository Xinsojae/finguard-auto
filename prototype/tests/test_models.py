"""models.train_models · walk_forward_backtest 테스트 (codex #2, #4 회귀 방지)."""
import numpy as np
import pandas as pd
import pytest
from core.features import make_features
from core.models import train_models, walk_forward_backtest, EMBARGO_DAYS


def test_train_models_time_based_split(med_panel):
    """codex #2: 행 분할(종목군) 아닌 unique date 분할이어야 함.

    검증: train 최대 날짜 < val 최소 날짜 (embargo 만큼 격차)
    """
    panel = make_features(med_panel)
    m_up, m_cr, metrics = train_models(panel)
    train_max = pd.Timestamp(metrics["train_date_max"]).date()
    val_min = pd.Timestamp(metrics["val_date_min"]).date()
    val_max = pd.Timestamp(metrics["val_date_max"]).date()
    assert train_max < val_min, \
        f"train 최대일({train_max}) ≥ val 최소일({val_min}) — 시간분할 실패"
    assert val_min < val_max, "val 구간이 단일 날짜로 축소됨"


def test_train_models_embargo_gap(med_panel):
    """train 끝 ~ val 시작 사이 5일(EMBARGO_DAYS) 격차 존재해야 함."""
    panel = make_features(med_panel)
    _, _, metrics = train_models(panel)
    train_max = pd.Timestamp(metrics["train_date_max"]).date()
    val_min = pd.Timestamp(metrics["val_date_min"]).date()
    gap_days = (val_min - train_max).days
    # 영업일 5일 = 달력일 약 7일, 최소 EMBARGO_DAYS-1 이상이면 OK
    assert gap_days >= EMBARGO_DAYS - 1, \
        f"embargo 격차 부족: {gap_days}일 (기대 ≥ {EMBARGO_DAYS-1})"


def test_train_models_metrics_keys(med_panel):
    """metrics dict에 필수 키 존재."""
    panel = make_features(med_panel)
    _, _, metrics = train_models(panel)
    required = {"up_auc", "up_pr", "cr_auc", "cr_pr",
                "train_date_max", "val_date_min", "val_date_max",
                "n_train_rows", "n_val_rows"}
    missing = required - set(metrics.keys())
    assert not missing, f"누락 키: {missing}"


def test_train_models_rows_consistent(med_panel):
    """학습/검증 행 수 합이 dropna 후 panel 크기보다 작거나 같아야 함 (embargo로 일부 제외)."""
    panel = make_features(med_panel)
    _, _, metrics = train_models(panel)
    valid_rows = panel.dropna(subset=["target_up", "target_crash"]).shape[0]
    train_rows = metrics["n_train_rows"]
    val_rows = metrics["n_val_rows"]
    # embargo로 행이 train·val 어디에도 속하지 않음 → train+val ≤ valid
    assert train_rows + val_rows <= valid_rows, \
        "train+val > valid_rows — 누수 가능성"
    # embargo 제외율이 30% 이하여야 함 (정상적으로 5~10% 정도 예상)
    excluded_pct = (valid_rows - train_rows - val_rows) / valid_rows
    assert 0 <= excluded_pct <= 0.30, \
        f"embargo 제외율 비정상: {excluded_pct:.1%} (기대 ≤ 30%)"


def test_walk_forward_returns_structure(med_panel):
    """walk_forward_backtest 반환 구조 검증."""
    panel = make_features(med_panel)
    ra, rb, avoided, per_fold = walk_forward_backtest(
        panel, n_folds=3, k_top=5, hold_days=5)
    assert isinstance(ra, pd.Series)
    assert isinstance(rb, pd.Series)
    assert isinstance(avoided, int)
    assert isinstance(per_fold, list)
    if per_fold:
        rec = per_fold[0]
        for key in ["fold", "train_end", "test_start", "test_end",
                    "n_picks", "A_mean", "B_mean",
                    "up_auc", "up_pr", "cr_auc", "cr_pr"]:
            assert key in rec, f"per_fold[0] 누락: {key}"


def test_walk_forward_fold_dates_monotonic(med_panel):
    """각 fold의 train_end < test_start, fold 순서대로 시간 진행."""
    panel = make_features(med_panel)
    _, _, _, per_fold = walk_forward_backtest(panel, n_folds=3)
    for rec in per_fold:
        assert rec["train_end"] < rec["test_start"], \
            f"fold {rec['fold']}: train_end({rec['train_end']}) ≥ test_start({rec['test_start']})"
    # fold 순서대로 test_start 증가
    for i in range(1, len(per_fold)):
        assert per_fold[i]["test_start"] > per_fold[i-1]["test_start"]
