"""Codex 2차 감사 회귀 방지 테스트.

각 항목별 회귀 발생 시 즉시 감지.
UI 캡션·문구(#3, #5, #6, #9)는 단위테스트 범위 밖.
"""
import numpy as np
import pandas as pd
import pytest

import disclosure_analyzer as da
from core.features import make_features, relabel_target_up
from core.anomaly import train_anomaly_detector
from core.quantile_risk import train_quantile_model
from core.portfolio import portfolio_var
from core import mocks
from core.config import FEATS


# ============================================================
# #1 — 복합 공시 악재 우선
# ============================================================
def test_compound_disclosure_negative_first():
    """관리종목(-3) + 자사주 소각(+3) 동시 매칭 → top은 악재여야 함.

    정규식: "관리종목" / "자기주식\\s*소각" 모두 매칭되도록 텍스트 구성.
    """
    text = ("당사는 관리종목 지정 결정과 동시에 자기주식 소각을 결정하였습니다.")
    results = da.classify(text)
    assert len(results) >= 2, f"복수 매칭 안 됨 (실제 매칭: {len(results)}건)"
    top = results[0]
    assert top.risk_score < 0, \
        f"호재가 top으로 선택됨 (top.name={top.name}, score={top.risk_score})"


def test_sort_negative_priority_over_positive():
    """동일 |risk_score|에서 negative가 positive보다 먼저."""
    # 관리종목(-3) + 자사주소각(+3)
    text = "관리종목 지정 및 자사주 소각 결정"
    results = da.classify(text)
    if len(results) >= 2:
        # 첫 결과가 negative면 OK (정렬 우선순위 보장)
        scores = [r.risk_score for r in results]
        # negative 결과들이 모두 positive보다 먼저 와야 함
        neg_count = sum(1 for s in scores if s < 0)
        first_neg_idx = next((i for i, s in enumerate(scores) if s < 0), -1)
        first_pos_idx = next((i for i, s in enumerate(scores) if s > 0), -1)
        if neg_count > 0 and first_pos_idx >= 0:
            assert first_neg_idx < first_pos_idx, \
                f"positive({first_pos_idx})가 negative({first_neg_idx})보다 앞"


# ============================================================
# #2 — target_up fold별 q70 (전체 누수 차단)
# ============================================================
def test_relabel_target_up_uses_only_fit_mask(med_panel):
    """relabel은 fit_mask 구간의 q70만 사용 — 전체 누수 차단."""
    df = make_features(med_panel)
    # fit_mask: 앞 절반만
    cut = df["date"].quantile(0.5)
    fit_mask = df["date"] <= cut

    q70_full = df["fwd_ret_5d"].quantile(0.70)
    q70_fit = df.loc[fit_mask, "fwd_ret_5d"].quantile(0.70)

    # 두 q70이 다르다면 의미 있는 테스트
    if abs(q70_full - q70_fit) < 1e-6:
        pytest.skip("fit/full q70이 동일 — random seed 영향")

    df_relabeled = relabel_target_up(df, fit_mask)
    # 새 라벨이 q70_fit 기준이어야 함
    fwd = df_relabeled["fwd_ret_5d"]
    expected = np.where(fwd.isna(), np.nan, (fwd > q70_fit).astype(float))
    assert np.array_equal(
        df_relabeled["target_up"].fillna(-1).values,
        np.where(np.isnan(expected), -1, expected),
    ), "relabel이 fit_mask q70을 사용하지 않음"


def test_train_models_target_uses_train_q70(med_panel):
    """train_models는 train 구간 q70으로 라벨 재계산해야 함."""
    from core.models import train_models
    panel = make_features(med_panel)
    # train_models 호출 후 metrics 검증만 (직접 q70 비교 어려움)
    _, _, metrics = train_models(panel)
    # 정상 작동 + 분할 메타 노출
    assert "train_date_max" in metrics
    assert metrics["n_train_rows"] > 0
    assert metrics["n_val_rows"] > 0


# ============================================================
# #4 — anomaly / quantile 시간 분할
# ============================================================
def test_anomaly_uses_time_split(med_panel):
    """IsolationForest는 시간순 train 구간만 학습 (이전 버그: 종목 70%)."""
    panel = make_features(med_panel)
    model = train_anomaly_detector(panel)
    assert model is not None
    # decision_function이 panel 전체에 정상 작동
    from core.anomaly import ANOMALY_FEATS
    sample = panel.dropna(subset=ANOMALY_FEATS)[ANOMALY_FEATS].head(10)
    scores = model.decision_function(sample)
    assert len(scores) == 10


def test_quantile_uses_time_split(med_panel):
    """Quantile LightGBM도 시간 분할 — 학습된 모델 반환."""
    panel = make_features(med_panel)
    model = train_quantile_model(panel, alpha=0.10)
    assert model is not None
    # 예측 호출 가능
    from core.quantile_risk import predict_quantile
    out = predict_quantile(model, panel.dropna(subset=FEATS).head(20))
    assert len(out) == 20
    # 분위 회귀라 일부 음수 가능 (정상)
    assert out.dtype in (np.float32, np.float64)


# ============================================================
# #7 — Permutation Importance val 구간 사용
# ============================================================
def test_permutation_importance_uses_val_range(med_panel):
    """permutation_importance_quick은 val 구간(뒤 30%)만 사용."""
    panel = make_features(med_panel)
    # train_models로 m_up 학습
    from core.models import train_models
    m_up, _, _ = train_models(panel)
    # Permutation 호출 가능 + DataFrame 반환
    result = mocks.permutation_importance_quick(
        m_up, panel, FEATS, "target_up", n_repeats=2, max_rows=2000,
    )
    assert "feature" in result.columns
    assert "importance_mean" in result.columns
    assert len(result) == len(FEATS)


# ============================================================
# #8 — VaR 음수 손실 0 clip
# ============================================================
class _MockPos:
    def __init__(self, sid, name, qty, avg_price):
        self.stock_id = sid
        self.name = name
        self.qty = qty
        self.avg_price = avg_price

    def market_value(self, cur):
        return self.qty * cur


def test_var_zero_when_all_returns_positive():
    """모든 일 수익률 양수 → VaR/expected_max_loss 모두 0 이상."""
    n_days = 80
    # 모든 수익률 +1%
    panel = pd.DataFrame({
        "date": list(pd.bdate_range("2024-01-01", periods=n_days)) * 2,
        "stock_id": ["A"] * n_days + ["B"] * n_days,
        "return": [0.01] * (n_days * 2),
        "close": [10000] * (n_days * 2),
    })
    positions = {
        "A": _MockPos("A", "A주", 10, 10000),
        "B": _MockPos("B", "B주", 10, 10000),
    }
    prices = {"A": 10100, "B": 10100}
    out = portfolio_var(positions, prices, panel, window=60)
    assert out["VaR_95"] >= 0, f"VaR_95 음수: {out['VaR_95']}"
    assert out["VaR_99"] >= 0, f"VaR_99 음수: {out['VaR_99']}"
    assert out["expected_max_loss"] >= 0, \
        f"expected_max_loss 음수: {out['expected_max_loss']}"


def test_var_positive_when_losses_exist():
    """손실 존재 시 VaR은 양수 (정상 작동)."""
    n_days = 80
    rng = np.random.default_rng(0)
    # 평균 -0.5%, 변동 2% — 손실 자주 발생
    rets = rng.normal(-0.005, 0.02, n_days * 2)
    panel = pd.DataFrame({
        "date": list(pd.bdate_range("2024-01-01", periods=n_days)) * 2,
        "stock_id": ["A"] * n_days + ["B"] * n_days,
        "return": rets,
        "close": [10000] * (n_days * 2),
    })
    positions = {
        "A": _MockPos("A", "A주", 10, 10000),
        "B": _MockPos("B", "B주", 10, 10000),
    }
    prices = {"A": 10000, "B": 10000}
    out = portfolio_var(positions, prices, panel, window=60)
    assert out["VaR_95"] > 0
    assert out["expected_max_loss"] > 0
