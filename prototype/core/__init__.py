"""core 패키지 — 공용 상수·데이터·모델·UI 유틸 re-export."""
from core.config import (
    FEATS, FEAT_KOR, SECTOR_NAMES, FAKE_NAMES,
    REAL_PANEL_PATH, TICKER_META_PATH, CSS_BLOCK, RNG,
)
from core.data import (
    gen_panel, load_real_panel_bundled, inject_disclosure_signals,
    latest_snapshot,
)
from core.features import make_features
from core.models import train_models, walk_forward_backtest
from core.anomaly import train_anomaly_detector, score_snapshot
from core.ui import classify, tag_html, apply_css

__all__ = [
    "FEATS", "FEAT_KOR", "SECTOR_NAMES", "FAKE_NAMES",
    "REAL_PANEL_PATH", "TICKER_META_PATH", "CSS_BLOCK", "RNG",
    "gen_panel", "load_real_panel_bundled", "inject_disclosure_signals",
    "latest_snapshot",
    "make_features",
    "train_models", "walk_forward_backtest",
    "train_anomaly_detector", "score_snapshot",
    "classify", "tag_html", "apply_css",
]
