"""LightGBM 모델 학습 + walk-forward 백테스트."""
import numpy as np
import pandas as pd
import streamlit as st
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
)

from core.config import FEATS


@st.cache_resource
def train_models(panel: pd.DataFrame):
    """단일 70/30 holdout으로 상승·급락 LightGBM 학습 (대시보드 SHAP·점수용).

    walk-forward 평가는 walk_forward_backtest() 별도.
    """
    df = panel.dropna(subset=FEATS + ["target_up", "target_crash"]).reset_index(drop=True)
    n = len(df)
    cut = int(n * 0.7)
    tr = df.iloc[:cut]
    va = df.iloc[cut:]
    m_up = lgb.LGBMClassifier(
        objective="binary", num_leaves=31, learning_rate=0.05,
        n_estimators=300, min_data_in_leaf=200, verbose=-1, random_state=42)
    m_up.fit(tr[FEATS], tr["target_up"],
             eval_set=[(va[FEATS], va["target_up"])],
             callbacks=[lgb.early_stopping(40, verbose=False)])
    m_cr = lgb.LGBMClassifier(
        objective="binary", num_leaves=31, learning_rate=0.05,
        n_estimators=300, min_data_in_leaf=200, verbose=-1, random_state=42)
    m_cr.fit(tr[FEATS], tr["target_crash"],
             eval_set=[(va[FEATS], va["target_crash"])],
             callbacks=[lgb.early_stopping(40, verbose=False)])
    p_up = m_up.predict_proba(va[FEATS])[:, 1]
    p_cr = m_cr.predict_proba(va[FEATS])[:, 1]
    metrics = dict(
        up_auc=roc_auc_score(va["target_up"], p_up),
        up_pr=average_precision_score(va["target_up"], p_up),
        cr_auc=roc_auc_score(va["target_crash"], p_cr),
        cr_pr=average_precision_score(va["target_crash"], p_cr),
    )
    return m_up, m_cr, metrics


@st.cache_data(show_spinner=False)
def walk_forward_backtest(panel: pd.DataFrame, n_folds: int = 3, k_top: int = 20,
                          hold_days: int = 5, cost: float = 0.003,
                          risk_pct: float = 0.70):
    """Walk-forward n_folds + 비중첩 hold_days 리밸런스.

    Returns (ra_s, rb_s, avoided_total, per_fold_records):
        ra_s, rb_s : 보유기간 평균 수익률 시계열
        avoided_total : 리스크 필터 회피 picks 누적
        per_fold_records : fold별 학습/테스트 구간·n_picks·A/B 평균

    설계:
      - 시작 40%는 항상 train(warmup), 나머지 60%를 n_folds로 분할
      - fold k의 train = D[:warmup_end + k*fold_len], test = 그 다음 fold_len
      - 각 fold에서 LightGBM 재학습 → out-of-time 예측
      - 리밸런스: hold_days 간격 (5일 중첩 fwd_ret 컴파운드 버그 회피)
      - 거래비용 cost는 진입+청산 단순 차감
    """
    df = panel.dropna(subset=FEATS + ["target_up", "target_crash", "fwd_ret_5d"])
    df = df.reset_index(drop=True)
    dates = np.array(sorted(df["date"].unique()))
    N = len(dates)
    if N < 60:
        return (pd.Series(dtype=float), pd.Series(dtype=float), 0, [])

    warmup_end = max(int(N * 0.40), 30)
    remain = N - warmup_end
    fold_len = max(remain // n_folds, hold_days * 2)

    rets_a, rets_b, dates_bt = [], [], []
    avoided_total = 0
    per_fold = []

    for k in range(n_folds):
        train_end = warmup_end + k * fold_len
        test_start = train_end
        test_end = min(train_end + fold_len, N)
        if test_end - test_start < hold_days + 1:
            continue
        train_dates = dates[:train_end]
        test_dates = dates[test_start:test_end]
        tr = df[df["date"].isin(train_dates)]
        te = df[df["date"].isin(test_dates)].copy()
        if len(tr) < 200 or len(te) < 50:
            continue

        m_up_k = lgb.LGBMClassifier(
            objective="binary", num_leaves=31, learning_rate=0.05,
            n_estimators=300, min_data_in_leaf=200, verbose=-1, random_state=42)
        m_up_k.fit(tr[FEATS], tr["target_up"])
        m_cr_k = lgb.LGBMClassifier(
            objective="binary", num_leaves=31, learning_rate=0.05,
            n_estimators=300, min_data_in_leaf=200, verbose=-1, random_state=42)
        m_cr_k.fit(tr[FEATS], tr["target_crash"])

        te["score_up"] = m_up_k.predict_proba(te[FEATS])[:, 1]
        te["score_cr"] = m_cr_k.predict_proba(te[FEATS])[:, 1]

        # 실험 1·2 — 모델 검증 메트릭 (per-fold)
        try:
            up_auc = float(roc_auc_score(te["target_up"], te["score_up"]))
            up_pr = float(average_precision_score(te["target_up"], te["score_up"]))
            cr_auc = float(roc_auc_score(te["target_crash"], te["score_cr"]))
            cr_pr = float(average_precision_score(te["target_crash"], te["score_cr"]))
            # Top-20% precision
            top_n = max(int(len(te) * 0.20), 1)
            top_up = te.nlargest(top_n, "score_up")
            top_cr = te.nlargest(top_n, "score_cr")
            up_topk_prec = float(top_up["target_up"].mean())
            cr_topk_prec = float(top_cr["target_crash"].mean())
        except Exception:
            up_auc = up_pr = cr_auc = cr_pr = up_topk_prec = cr_topk_prec = 0.0

        fold_a, fold_b, fold_dates = [], [], []
        fold_avoided = 0
        for i in range(0, len(test_dates), hold_days):
            d = test_dates[i]
            sub = te[te["date"] == d]
            if len(sub) < k_top * 2:
                continue
            a = sub.nlargest(k_top, "score_up")
            ra = a["fwd_ret_5d"].mean() - cost
            cr_cut = sub["score_cr"].quantile(risk_pct)
            sub_lr = sub[sub["score_cr"] <= cr_cut]
            b = sub_lr.nlargest(k_top, "score_up")
            fold_avoided += len(set(a["stock_id"]) - set(b["stock_id"]))
            rb = b["fwd_ret_5d"].mean() - cost if len(b) > 0 else 0.0
            fold_a.append(float(ra))
            fold_b.append(float(rb))
            fold_dates.append(pd.Timestamp(d))

        rets_a.extend(fold_a)
        rets_b.extend(fold_b)
        dates_bt.extend(fold_dates)
        avoided_total += fold_avoided
        per_fold.append({
            "fold": k + 1,
            "train_end": pd.Timestamp(train_dates[-1]).date(),
            "test_start": pd.Timestamp(test_dates[0]).date(),
            "test_end": pd.Timestamp(test_dates[-1]).date(),
            "n_picks": len(fold_a),
            "A_mean": float(np.mean(fold_a)) if fold_a else 0.0,
            "B_mean": float(np.mean(fold_b)) if fold_b else 0.0,
            # §13.1·13.2 모델 검증 메트릭
            "up_auc": up_auc, "up_pr": up_pr, "up_topk_prec": up_topk_prec,
            "cr_auc": cr_auc, "cr_pr": cr_pr, "cr_topk_prec": cr_topk_prec,
        })

    ra_s = pd.Series(rets_a, index=dates_bt)
    rb_s = pd.Series(rets_b, index=dates_bt)
    return ra_s, rb_s, avoided_total, per_fold
