"""패널 데이터 생성·로드·공시 주입.

주요 함수:
    gen_panel(n_stocks, n_days)        → GARCH 합성 패널
    load_real_panel_bundled()          → KOSPI 99종목 실데이터 pkl 로드
    inject_disclosure_signals(panel)   → 룰베이스 공시 분류 결과 주입
    latest_snapshot(panel)             → 최신 일자 스냅샷
"""
import json
import numpy as np
import pandas as pd
import streamlit as st

from core.config import (
    RNG, N_SECTORS, SECTOR_NAMES, FAKE_NAMES,
    REAL_PANEL_PATH, TICKER_META_PATH,
)


@st.cache_data
def load_real_panel_bundled() -> pd.DataFrame:
    """KOSPI 시총 상위 100종목 일봉 (2021-01-04 ~ 2025-12-31).

    poc_finguard_real.py로 사전 다운로드한 패널 (FinanceDataReader).
    파일 부재 시 빈 DataFrame.
    """
    if not REAL_PANEL_PATH.exists():
        return pd.DataFrame()
    panel = pd.read_pickle(REAL_PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])
    if TICKER_META_PATH.exists():
        meta = json.loads(TICKER_META_PATH.read_text(encoding="utf-8"))
        name_map = meta.get("name", {})
        sector_map = meta.get("sector", {})
        panel["name"] = panel["stock_id"].map(lambda t: name_map.get(str(t), str(t)))
        if sector_map:
            panel["sector"] = panel["stock_id"].map(
                lambda t: sector_map.get(str(t), "기타") or "기타")
        else:
            panel["sector"] = panel["sector"].replace({"Unknown": "기타"})
    else:
        panel["name"] = panel["stock_id"].astype(str)
    return panel


@st.cache_data
def gen_panel(n_stocks: int = 150, n_days: int = 600) -> pd.DataFrame:
    """GARCH형 합성 패널 + 시장 국면(HMM) + 섹터 충격 + 호재/악재 이벤트."""
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    trans = np.array([[0.97, 0.02, 0.01],
                      [0.02, 0.96, 0.02],
                      [0.02, 0.04, 0.94]])
    drifts = np.array([0.0010, 0.0002, -0.0008])
    states = np.zeros(n_days, dtype=int)
    s = 0
    for t in range(n_days):
        states[t] = s
        s = RNG.choice(3, p=trans[s])
    sector_shock = RNG.normal(0, 0.008, (N_SECTORS, n_days))
    rows = []
    for sid in range(n_stocks):
        sector = sid % N_SECTORS
        idio_drift = RNG.normal(0, 0.0004)
        idio_vol = RNG.uniform(0.012, 0.025)
        eps = RNG.normal(0, 1, n_days)
        vol = np.zeros(n_days)
        vol[0] = idio_vol
        for t in range(1, n_days):
            vol[t] = np.sqrt(0.94 * vol[t-1]**2 + 0.06 * (eps[t-1] * idio_vol)**2)
        ret = drifts[states] + idio_drift + vol * eps + sector_shock[sector]
        bad = RNG.choice(n_days, size=int(n_days * 0.004), replace=False)
        ret[bad] -= RNG.uniform(0.06, 0.14, len(bad))
        good = RNG.choice(n_days, size=int(n_days * 0.005), replace=False)
        ret[good] += RNG.uniform(0.03, 0.08, len(good))
        price = 10000 * np.exp(np.cumsum(ret))
        volume = np.exp(RNG.normal(13, 0.7, n_days)) * (1 + 6 * np.abs(ret))
        news_sent = 0.3 * np.sign(np.roll(ret, 1)) + RNG.normal(0, 0.5, n_days)
        disc = np.zeros(n_days, dtype=int)
        disc[good] = 1
        disc[bad] = -1
        rows.append(pd.DataFrame({
            "date": dates, "stock_id": sid,
            "name": FAKE_NAMES[sid % len(FAKE_NAMES)] + f"{sid:03d}",
            "sector": SECTOR_NAMES[sector],
            "close": price, "return": ret, "volume": volume,
            "news_sent": news_sent, "disclosure": disc, "regime": states,
        }))
    return pd.concat(rows, ignore_index=True)


@st.cache_data
def inject_disclosure_signals(panel: pd.DataFrame,
                              n_events_per_stock: int = 4) -> pd.DataFrame:
    """합성 disclosure를 disclosure_analyzer.classify() 결과로 교체.

    Mock 공시 64건을 일괄 분류 → 각 종목에 무작위 시점으로 매핑.
    disclosure 컬럼: 룰베이스 risk_score(-3~+3) 정수.
    disc_code/disc_name/disc_risk_label/disc_explanation 컬럼 추가 (UI 표시용).
    """
    import disclosure_analyzer as da
    events = da.classify_mock_all()
    if not events:
        return panel

    panel = panel.copy()
    panel["disclosure"] = 0
    panel["disc_code"] = ""
    panel["disc_name"] = ""
    panel["disc_risk_label"] = ""
    panel["disc_explanation"] = ""

    rng = np.random.default_rng(123)
    for sid in panel["stock_id"].unique():
        mask_sid = panel["stock_id"] == sid
        sub_dates = panel.loc[mask_sid, "date"].values
        if len(sub_dates) < 30:
            continue
        # warmup 25일 + 마지막 5일 제외 (look-ahead 차단)
        usable = sub_dates[25:-5]
        if len(usable) < n_events_per_stock:
            continue
        chosen_dates = rng.choice(usable, size=n_events_per_stock, replace=False)
        chosen_events = rng.choice(len(events), size=n_events_per_stock, replace=True)
        for d, ev_idx in zip(chosen_dates, chosen_events):
            ev = events[ev_idx]
            m = mask_sid & (panel["date"] == d)
            panel.loc[m, "disclosure"] = int(ev["risk_score"])
            panel.loc[m, "disc_code"] = ev["code"]
            panel.loc[m, "disc_name"] = ev["name"]
            panel.loc[m, "disc_risk_label"] = ev["risk_label"]
            panel.loc[m, "disc_explanation"] = ev["explanation"]
    return panel


@st.cache_data
def latest_snapshot(panel: pd.DataFrame):
    """최신 일자의 종목별 스냅샷 + 그 날짜 반환."""
    from core.config import FEATS  # 지연 import (순환 방지)
    df = panel.dropna(subset=FEATS).copy()
    latest_date = df["date"].max()
    snap = df[df["date"] == latest_date].copy()
    return snap, latest_date
