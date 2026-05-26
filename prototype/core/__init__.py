"""core 패키지 — 공용 상수·데이터·모델·UI 유틸 re-export."""
from core.config import (
    FEATS, FEAT_KOR, SECTOR_NAMES, FAKE_NAMES,
    REAL_PANEL_PATH, TICKER_META_PATH, CSS_BLOCK, RNG,
    COLORS, CATEGORY_COLORS,
)
from core.data import (
    gen_panel, load_real_panel_bundled, inject_disclosure_signals,
    latest_snapshot,
)
from core.features import make_features
from core.models import train_models, walk_forward_backtest
from core.anomaly import train_anomaly_detector, score_snapshot
from core.quantile_risk import train_quantile_model, predict_quantile
from core.confidence import (
    compute_confidence_for_snap, compute_confidence_for_stock,
    ConfidenceBreakdown,
)
from core.ui import classify, tag_html, apply_css
from core.ui_kit import (
    demo_badge, section_header, info_card, status_pill, metric_row,
    format_won, download_csv_button,
)
from core.plotly_theme import (
    is_dark, palette, layout_kwargs, CHART_COLORS,
)

__all__ = [
    "FEATS", "FEAT_KOR", "SECTOR_NAMES", "FAKE_NAMES",
    "REAL_PANEL_PATH", "TICKER_META_PATH", "CSS_BLOCK", "RNG",
    "COLORS", "CATEGORY_COLORS",
    "gen_panel", "load_real_panel_bundled", "inject_disclosure_signals",
    "latest_snapshot",
    "make_features",
    "train_models", "walk_forward_backtest",
    "train_anomaly_detector", "score_snapshot",
    "train_quantile_model", "predict_quantile",
    "compute_confidence_for_snap", "compute_confidence_for_stock",
    "ConfidenceBreakdown",
    "classify", "tag_html", "apply_css",
]
