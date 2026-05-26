"""pytest 공통 설정.

streamlit 의존성을 가벼운 stub으로 대체 → core/ 모듈을 streamlit 없이 import 가능.
공통 fixture (synthetic panel) 제공.
"""
import sys
import types
from pathlib import Path
import pytest
import numpy as np
import pandas as pd


# ---- streamlit stub (모듈 import 시점) ----
def _passthrough_decorator(*args, **kwargs):
    """@cache_data 또는 @cache_data(...) 양쪽 지원하는 패스스루."""
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return lambda f: f


_st_stub = types.SimpleNamespace(
    cache_data=_passthrough_decorator,
    cache_resource=_passthrough_decorator,
    session_state={},
)
sys.modules.setdefault("streamlit", _st_stub)

# tests/ 폴더에서 prototype 루트 import 가능하게
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---- 공통 fixture ----
@pytest.fixture
def small_panel():
    """5종목 × 50일 합성 panel — features.make_features 입력용."""
    n_stocks, n_days = 5, 50
    rng = np.random.default_rng(0)
    rows = []
    for sid in range(n_stocks):
        dates = pd.bdate_range("2024-01-01", periods=n_days)
        close = 10000 * np.exp(np.cumsum(rng.normal(0, 0.01, n_days)))
        ret = pd.Series(close).pct_change().fillna(0).values
        rows.append(pd.DataFrame({
            "date": dates, "stock_id": sid,
            "name": f"종목{sid:02d}", "sector": "테스트",
            "close": close, "return": ret,
            "volume": rng.uniform(1e4, 1e6, n_days),
            "news_sent": 0.0, "disclosure": 0, "regime": 0,
        }))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def med_panel():
    """30종목 × 250일 — train_models·walk-forward 시간분할 검증용."""
    n_stocks, n_days = 30, 250
    rng = np.random.default_rng(42)
    rows = []
    for sid in range(n_stocks):
        dates = pd.bdate_range("2023-01-02", periods=n_days)
        close = 10000 * np.exp(np.cumsum(rng.normal(0.0001, 0.012, n_days)))
        ret = pd.Series(close).pct_change().fillna(0).values
        rows.append(pd.DataFrame({
            "date": dates, "stock_id": sid,
            "name": f"종목{sid:03d}", "sector": "테스트",
            "close": close, "return": ret,
            "volume": rng.uniform(1e4, 1e6, n_days),
            "news_sent": 0.0, "disclosure": 0, "regime": 0,
        }))
    return pd.concat(rows, ignore_index=True)
