"""DEMO MODE 시뮬레이션 — 확장 계획 기능의 placeholder 출력.

원칙: 모든 mock 함수 출력에 "MOCK" / "DEMO" 라벨 부착하여
평가자 오해 차단. 실제 모델 연동 시 이 모듈만 교체.
"""
from __future__ import annotations
import random
import time
from datetime import date, timedelta
from typing import List, Dict
import numpy as np
import pandas as pd


# ============================================================
# 1. HyperCLOVA X 설명 카드 (자연어)
# ============================================================
_CLOVA_TEMPLATES = {
    "PRIORITY": (
        "**{name}**(섹터: {sector})은 현재 우선 관심 후보로 분류됩니다.\n\n"
        "최근 가격 추세가 안정적이며 (상승 점수 {up}/100), 급락 리스크 신호도 "
        "낮은 편입니다 (리스크 {risk}/100). TreeSHAP 분석 결과 가장 큰 기여 요인은 "
        "**{top_feat}**으로, 이는 단기 모멘텀이 형성되고 있음을 시사합니다.\n\n"
        "다만 AI 신뢰도가 {conf}이므로, 즉시 매수보다는 **백테스트 사후 검증** 또는 "
        "**모의투자 5일 관찰**을 권장합니다. 본 분석은 매수 추천이 아닌 의사결정 보조입니다."
    ),
    "HIGH-RISK": (
        "**{name}**은 상승 신호와 리스크 신호가 동시에 강한 **고위험 관심** 후보입니다.\n\n"
        "상승 점수 {up}/100은 단기 모멘텀을 시사하지만, 리스크 점수 {risk}/100은 "
        "변동성·악재 신호를 동반합니다. SHAP 기여도 1위 요인은 **{top_feat}**입니다.\n\n"
        "**모의투자로 충분히 검증 후** 진입 여부 판단을 권장합니다. "
        "리스크 점수가 60 이상으로 올라가면 자동 매수가 차단됩니다 (Kill Switch §12.2)."
    ),
    "HOLD": (
        "**{name}**은 현재 **관망** 단계입니다.\n\n"
        "상승 신호({up}/100)와 리스크 신호({risk}/100) 모두 명확하지 않아 "
        "포지션 진입 근거가 약합니다. 신뢰도 {conf} 상태에서 무리한 진입보다는 "
        "**추가 데이터 누적 후 재평가**가 합리적입니다."
    ),
    "AVOID": (
        "**{name}**은 **회피 후보**로 분류됩니다.\n\n"
        "리스크 점수가 높고({risk}/100) 상승 신호도 약합니다. "
        "최근 SHAP 분석에서 **{top_feat}**이 부정적으로 작용했습니다. "
        "신규 진입을 피하고, 기존 보유 시 매도 검토를 권장합니다."
    ),
}


def hyperclova_explanation(name: str, sector: str, cat: str,
                           up: int, risk: int, conf: str,
                           top_feat: str) -> str:
    """HyperCLOVA X 자연어 설명 카드 (mock).

    실제로는 if-else 기반 템플릿 + 점수 삽입. 실제 LLM 호출처럼
    구조화된 출력 형태로 흉내.
    """
    tpl = _CLOVA_TEMPLATES.get(cat, _CLOVA_TEMPLATES["HOLD"])
    return tpl.format(name=name, sector=sector, up=up, risk=risk,
                      conf=conf, top_feat=top_feat)


# ============================================================
# 2. KoBigBird 장문 공시 요약 (TextRank 근사)
# ============================================================
def kobigbird_summarize(text: str, n_sent: int = 3) -> dict:
    """장문 공시 요약 (mock). 문장 단위 단순 추출 — TextRank 근사.

    실제로는 KoBigBird embedding + PageRank가 정석.
    여기서는 길이·키워드 빈도 기반 단순 점수.
    """
    import re
    text = text.strip()
    if not text:
        return {"summary": "", "n_input_sent": 0, "key_terms": []}
    sents = [s.strip() for s in re.split(r"[.\n!?]+", text) if len(s.strip()) > 5]
    if not sents:
        return {"summary": text[:200], "n_input_sent": 1, "key_terms": []}
    # 키워드 빈도 추정
    words = re.findall(r"[가-힣A-Za-z]+", text)
    from collections import Counter
    common = Counter(words).most_common(20)
    common_words = {w for w, c in common if len(w) >= 2 and c >= 2}
    # 문장 점수 = 키워드 포함 수 + 위치 가중치 (앞쪽 우대)
    scored = []
    for i, s in enumerate(sents):
        kw_hits = sum(1 for w in common_words if w in s)
        position = 1.0 / (1 + i * 0.1)
        scored.append((kw_hits + position, i, s))
    top = sorted(scored, reverse=True)[:n_sent]
    top = sorted(top, key=lambda x: x[1])  # 원문 순서로
    summary = ". ".join([s[2] for s in top]) + "."
    return {
        "summary": summary,
        "n_input_sent": len(sents),
        "n_output_sent": len(top),
        "key_terms": [w for w, _ in common[:8] if len(w) >= 2],
    }


# ============================================================
# 3. PatchTST 시계열 예측 (mock — LightGBM 결과 + 노이즈)
# ============================================================
def patchtst_forecast(hist_close: pd.Series, horizon: int = 5,
                      seed: int = 42) -> pd.DataFrame:
    """미래 5일 가격 예측 (mock).

    실제 PatchTST는 transformer 시계열 모델. 여기서는 최근 20일 추세 +
    잔차 ARIMA-like 보정으로 근사.
    """
    if len(hist_close) < 20:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    recent = hist_close.tail(20)
    returns = recent.pct_change().dropna()
    drift = returns.mean()
    sigma = returns.std()
    last_price = float(recent.iloc[-1])
    forecast = []
    p = last_price
    for h in range(1, horizon + 1):
        eps = rng.normal(0, sigma)
        p = p * (1 + drift + eps * 0.5)  # 변동성 절반으로 (예측 안정)
        forecast.append({
            "step": h,
            "predicted_close": p,
            "lower_80": p * (1 - 1.28 * sigma),
            "upper_80": p * (1 + 1.28 * sigma),
        })
    return pd.DataFrame(forecast)


# ============================================================
# 4. Whisper 컨퍼런스콜 (mock transcript + 감성)
# ============================================================
_MOCK_CONFCALL = [
    "CEO: 이번 분기 매출은 전년 동기 대비 12% 증가했습니다.",
    "CEO: 신사업 부문 진출에 따른 초기 비용이 일시적으로 영업이익에 영향을 주었습니다.",
    "CFO: 영업이익률은 8.5%로 가이던스 범위 내입니다.",
    "Analyst Q: 중국 시장 수요 둔화에 대한 대응 계획은?",
    "CEO: 동남아 시장 다변화로 리스크 분산을 진행 중입니다.",
    "CFO: 다음 분기 가이던스는 보수적으로 제시합니다.",
    "Analyst Q: AI 투자 확대 계획은?",
    "CEO: AI 인프라 투자는 향후 3년간 1조원 규모로 확대 예정입니다.",
]


def whisper_mock_transcript() -> List[str]:
    """모의 컨퍼런스콜 transcript 8문장."""
    return list(_MOCK_CONFCALL)


def whisper_sentiment_scores(lines: List[str]) -> pd.DataFrame:
    """문장별 감성 점수 (mock — TF-IDF 모델 활용)."""
    from news_sentiment import TfidfSentiment
    m = TfidfSentiment()
    scores = m.analyze(lines)
    rows = []
    for i, (line, s) in enumerate(zip(lines, scores)):
        if s > 0.2:
            tag = "🟢 긍정"
        elif s < -0.2:
            tag = "🔴 부정"
        else:
            tag = "⚪ 중립"
        rows.append({"#": i + 1, "발언": line, "감성": tag, "점수": round(s, 3)})
    return pd.DataFrame(rows)


# ============================================================
# 5. MLflow 실험 추적 (mock — JSON 실험 로그)
# ============================================================
def mlflow_mock_experiments() -> pd.DataFrame:
    """가상 실험 추적 로그 8건."""
    rows = [
        {"run_id": "a1b2c3", "model": "LightGBM-v1", "data": "synth-120s",
         "ROC_AUC_up": 0.541, "ROC_AUC_crash": 0.582, "PR_AUC_up": 0.328,
         "params": "lr=0.05, leaves=31, n_est=300",
         "ts": "2026-05-10 14:22", "status": "✅ archived"},
        {"run_id": "d4e5f6", "model": "LightGBM-v2 (+sector)", "data": "synth-120s",
         "ROC_AUC_up": 0.547, "ROC_AUC_crash": 0.591, "PR_AUC_up": 0.335,
         "params": "lr=0.05, leaves=31, n_est=300, +sector",
         "ts": "2026-05-13 09:41", "status": "✅ archived"},
        {"run_id": "g7h8i9", "model": "LightGBM-real", "data": "KOSPI-100",
         "ROC_AUC_up": 0.559, "ROC_AUC_crash": 0.628, "PR_AUC_up": 0.357,
         "params": "lr=0.05, leaves=31, n_est=400",
         "ts": "2026-05-21 17:08", "status": "✅ production"},
        {"run_id": "j1k2l3", "model": "LightGBM+TFIDF-news", "data": "KOSPI-100",
         "ROC_AUC_up": 0.564, "ROC_AUC_crash": 0.631, "PR_AUC_up": 0.362,
         "params": "lr=0.05, leaves=31, n_est=400, +news_sent (TFIDF)",
         "ts": "2026-05-24 11:15", "status": "🧪 staging"},
        {"run_id": "m4n5o6", "model": "LightGBM+KF-DeBERTa", "data": "KOSPI-100",
         "ROC_AUC_up": None, "ROC_AUC_crash": None, "PR_AUC_up": None,
         "params": "(planned)",
         "ts": "2026-06-01 (계획)", "status": "📋 planned"},
        {"run_id": "p7q8r9", "model": "IsolationForest", "data": "KOSPI-100",
         "ROC_AUC_up": None, "ROC_AUC_crash": None, "PR_AUC_up": None,
         "params": "contamination=0.05, n=200",
         "ts": "2026-05-26 22:30", "status": "✅ production"},
        {"run_id": "s1t2u3", "model": "Quantile-LightGBM", "data": "KOSPI-100",
         "ROC_AUC_up": None, "ROC_AUC_crash": None, "PR_AUC_up": None,
         "params": "alpha=0.10",
         "ts": "2026-05-26 23:10", "status": "✅ production"},
        {"run_id": "v4w5x6", "model": "PatchTST", "data": "KOSPI-100",
         "ROC_AUC_up": None, "ROC_AUC_crash": None, "PR_AUC_up": None,
         "params": "(planned, V3 확장)",
         "ts": "2026-Q3 (계획)", "status": "📋 planned"},
    ]
    return pd.DataFrame(rows)


# ============================================================
# 6. 이벤트 캘린더 (mock — 다가오는 공시·실적·배당)
# ============================================================
def event_calendar_mock(snap, n: int = 12) -> pd.DataFrame:
    """다가오는 이벤트 mock — snap 종목 일부에 배정."""
    today = date.today()
    rng = random.Random(20260527)
    event_types = [
        ("📅 정기보고서 제출", "INFO_QUARTER"),
        ("💰 분기 배당 결정", "POS_DIV_UP"),
        ("📊 실적 발표", "INFO_QUARTER"),
        ("🏢 주주총회", "GOV_CEO"),
        ("📝 IR 컨퍼런스콜", "INFO_QUARTER"),
        ("⚠️ 관리종목 사유 발표 예정", "REG_DELIST_WARN"),
        ("📋 자사주 매입 만료", "POS_TREASURY"),
        ("⚖️ 1심 판결 예정", "LEG_LAWSUIT"),
    ]
    if len(snap) == 0:
        return pd.DataFrame()
    sample_names = snap["name"].sample(min(n, len(snap)), random_state=42).tolist()
    rows = []
    for i, name in enumerate(sample_names):
        days_ahead = rng.randint(1, 30)
        ev = rng.choice(event_types)
        rows.append({
            "예정일": today + timedelta(days=days_ahead),
            "종목": name,
            "이벤트": ev[0],
            "관련 코드": ev[1],
            "D-day": f"D-{days_ahead}",
        })
    return pd.DataFrame(rows).sort_values("예정일").reset_index(drop=True)


# ============================================================
# 7. 주간 리포트 (HTML 생성)
# ============================================================
def weekly_report_html(snap, ra_summary: dict = None) -> str:
    """주간 리스크 리포트 HTML (mock)."""
    today = date.today()
    week_start = today - timedelta(days=6)
    n_priority = int((snap["category"] == "PRIORITY").sum())
    n_avoid = int((snap["category"] == "AVOID").sum())
    n_highrisk = int((snap["category"] == "HIGH-RISK").sum())
    mkt_risk = float(snap["score_risk"].mean())
    top_priority = snap[snap["category"] == "PRIORITY"].nlargest(5, "score_up")
    top_avoid = snap.nlargest(5, "score_risk")
    html = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<title>FinGuard Auto 주간 리스크 리포트 ({today})</title>
<style>
  body {{ font-family: 'Malgun Gothic', sans-serif; max-width: 800px; margin: 40px auto; color: #333; line-height: 1.6; }}
  h1 {{ color: #1565C0; border-bottom: 2px solid #1565C0; padding-bottom: 8px; }}
  h2 {{ color: #424242; margin-top: 32px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ padding: 8px 12px; border-bottom: 1px solid #E0E0E0; text-align: left; }}
  th {{ background: #F5F5F5; }}
  .summary {{ background: #FFF8E1; padding: 16px; border-left: 4px solid #FFCA28; border-radius: 6px; }}
  .disclaimer {{ font-size: 0.85em; color: #888; margin-top: 32px; }}
</style></head><body>
<h1>🛡️ FinGuard Auto — 주간 리스크 리포트</h1>
<p><b>발행:</b> {today} | <b>대상 주간:</b> {week_start} ~ {today}</p>

<div class="summary">
  <b>요약</b>: 우선 관심 후보 {n_priority}건 · 고위험 관심 {n_highrisk}건 · 회피 후보 {n_avoid}건.
  시장 평균 리스크 점수 {mkt_risk:.1f}/100.
</div>

<h2>1. 우선 관심 후보 Top 5</h2>
<table><tr><th>종목</th><th>섹터</th><th>상승</th><th>리스크</th></tr>
{"".join(f"<tr><td>{r['name']}</td><td>{r['sector']}</td><td>{r['score_up']}</td><td>{r['score_risk']}</td></tr>" for _, r in top_priority.iterrows())}
</table>

<h2>2. 회피 후보 Top 5</h2>
<table><tr><th>종목</th><th>섹터</th><th>상승</th><th>리스크</th></tr>
{"".join(f"<tr><td>{r['name']}</td><td>{r['sector']}</td><td>{r['score_up']}</td><td>{r['score_risk']}</td></tr>" for _, r in top_avoid.iterrows())}
</table>

<p class="disclaimer">
본 리포트는 FinGuard Auto 시스템이 자동 생성한 분석 결과이며, 매수·매도 추천이 아닙니다.
최종 투자 판단과 책임은 사용자에게 있습니다. 본 시스템은 학술 데모이며, 실 운영 서비스가 아닙니다.
</p>
</body></html>"""
    return html


# ============================================================
# 8. 알림 (st.toast 메시지 생성)
# ============================================================
def generate_alerts(snap) -> List[dict]:
    """리스크 60+ 종목, 우선관심 신규, Kill Switch, 임박 이벤트 알림 묶음."""
    alerts = []
    high_risk = snap[snap["score_risk"] >= 60].nlargest(3, "score_risk")
    for _, r in high_risk.iterrows():
        alerts.append({
            "icon": "⚠️",
            "msg": f"{r['name']} 리스크 점수 {int(r['score_risk'])}/100 — 매수 차단",
        })
    priority = snap[snap["category"] == "PRIORITY"].nlargest(2, "score_up")
    for _, r in priority.iterrows():
        alerts.append({
            "icon": "🟢",
            "msg": f"{r['name']} 우선 관심 후보 (상승 {int(r['score_up'])})",
        })
    mkt = float(snap["score_risk"].mean())
    if mkt >= 70:
        alerts.append({
            "icon": "🛑",
            "msg": f"시장 평균 리스크 {mkt:.0f} — Kill Switch 활성 권장",
        })
    # D-7 이내 캘린더 이벤트 통합
    try:
        cal = event_calendar_mock(snap, n=12)
        today = date.today()
        soon = cal[cal["예정일"] <= today + timedelta(days=7)]
        for _, e in soon.head(3).iterrows():
            d = (e["예정일"] - today).days
            alerts.append({
                "icon": "📅",
                "msg": f"D-{d} {e['종목']} {e['이벤트']}",
            })
    except Exception:
        pass
    return alerts


# ============================================================
# 14. 데이터 품질 진단 (실 panel 분석)
# ============================================================
def data_quality_report(panel) -> dict:
    """panel 결측·이상치·기간·종목 요약 (§9.2)."""
    if panel.empty:
        return {"empty": True}
    n_rows = len(panel)
    n_stocks = panel["stock_id"].nunique()
    n_days = panel["date"].nunique()
    date_min = panel["date"].min()
    date_max = panel["date"].max()
    missing = panel.isna().sum()
    missing_rate = (missing / n_rows * 100).round(3)
    # 가격 이상치: 0 이하 또는 일일 변동률 30% 초과
    bad_price = int((panel["close"] <= 0).sum()) if "close" in panel.columns else 0
    if "return" in panel.columns:
        bad_return = int((panel["return"].abs() > 0.30).sum())
    else:
        bad_return = 0
    # 종목별 데이터 수 분포 (생존편향 체크)
    per_stock = panel.groupby("stock_id").size()
    stock_min = int(per_stock.min())
    stock_max = int(per_stock.max())
    stock_median = int(per_stock.median())
    return {
        "empty": False,
        "n_rows": n_rows,
        "n_stocks": n_stocks,
        "n_days": n_days,
        "date_min": date_min,
        "date_max": date_max,
        "missing": missing.to_dict(),
        "missing_rate": missing_rate.to_dict(),
        "bad_price": bad_price,
        "bad_return": bad_return,
        "stock_min_rows": stock_min,
        "stock_max_rows": stock_max,
        "stock_median_rows": stock_median,
    }


# ============================================================
# 9. Conformal Prediction (mock 예측 구간)
# ============================================================
def conformal_interval(score_up_p: float, n_calibration: int = 500,
                       alpha: float = 0.10) -> dict:
    """예측 구간 [lower, upper] mock.

    실제로는 calibration set의 nonconformity score 90% quantile.
    여기서는 단순 ±width 근사.
    """
    # 신뢰수준 90% (alpha=0.10)
    width = 0.15  # 약 ±0.15 cushion
    lower = max(0.0, score_up_p - width)
    upper = min(1.0, score_up_p + width)
    return {
        "alpha": alpha,
        "confidence": 1 - alpha,
        "point": score_up_p,
        "lower": lower,
        "upper": upper,
        "width": upper - lower,
        "n_calibration": n_calibration,
    }


# ============================================================
# 11. Permutation Importance (실제 sklearn, 빠른 1-shot)
# ============================================================
import streamlit as _st  # 캐싱용 (모듈 전역 1회)


@_st.cache_data(show_spinner=False)
def permutation_importance_quick(_model, panel, feats: list,
                                 target_col: str = "target_up",
                                 n_repeats: int = 3, max_rows: int = 5000) -> pd.DataFrame:
    """sklearn.permutation_importance — 시간 분할 val 구간(뒤 30%) 기반.

    이전 버그: df.tail(5000) → panel이 (stock_id, date) 정렬이라
    마지막 일부 종목의 전 기간 샘플이 됨. 검증셋 평가가 아니라
    일부 종목 평가가 되어 모델 설명이 오인 가능.
    수정: unique date의 마지막 30%만 사용 → train_models의 val과 동일 구간.
    """
    from sklearn.inspection import permutation_importance
    df = panel.dropna(subset=feats + [target_col])
    # 시간 분할 val 구간 (뒤 30%)
    dates = np.array(sorted(df["date"].unique()))
    if len(dates) >= 10:
        cut_idx = int(len(dates) * 0.7)
        val_dates = dates[cut_idx:]
        df = df[df["date"].isin(val_dates)]
    if len(df) > max_rows:
        df = df.tail(max_rows)
    X, y = df[feats], df[target_col]
    try:
        r = permutation_importance(_model, X, y, n_repeats=n_repeats,
                                   random_state=42, n_jobs=1)
        out = pd.DataFrame({
            "feature": feats,
            "importance_mean": r.importances_mean,
            "importance_std": r.importances_std,
        }).sort_values("importance_mean", ascending=False)
        return out
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})


# ============================================================
# 12. ALE Plot (단일 피처 1D, 실제 계산)
# ============================================================
@_st.cache_data(show_spinner=False)
def ale_1d(_model, panel, feat: str, all_feats: list,
           target_col: str = "target_up", n_bins: int = 20,
           max_rows: int = 3000) -> pd.DataFrame:
    """1D Accumulated Local Effects — 단일 피처 변경 시 예측 변화."""
    df = panel.dropna(subset=all_feats + [target_col])
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=42)
    if feat not in df.columns:
        return pd.DataFrame()
    quantiles = np.linspace(0.05, 0.95, n_bins + 1)
    grid = df[feat].quantile(quantiles).values
    # 각 grid 값으로 모든 행을 대체 → 예측 평균
    rows = []
    for v in grid:
        X = df[all_feats].copy()
        X[feat] = v
        pred = _model.predict_proba(X)[:, 1].mean()
        rows.append({"grid_value": v, "predicted_prob": pred})
    out = pd.DataFrame(rows)
    # ALE는 centered effect — 평균 차감
    out["effect"] = out["predicted_prob"] - out["predicted_prob"].mean()
    return out


# ============================================================
# 13. KoSimCSE 유사 사례 검색 (mock)
# ============================================================
_SIM_CASES = [
    {"date": "2024-03-15", "name": "A전자", "event": "유상증자 발표",
     "outcome": "공시 후 5일 -8.2%, 30일 -12.5%", "sim": 0.92},
    {"date": "2023-11-22", "name": "B케미칼", "event": "주요 거래처 이탈",
     "outcome": "공시 후 5일 -6.1%, 회복 시간 약 2개월", "sim": 0.87},
    {"date": "2024-07-08", "name": "C바이오", "event": "임상 3상 결과 발표",
     "outcome": "공시 후 5일 +18.4%, 30일 +25.1%", "sim": 0.84},
    {"date": "2023-09-19", "name": "D금융", "event": "자사주 매입·소각",
     "outcome": "공시 후 5일 +4.2%, 안정적 상승", "sim": 0.81},
    {"date": "2024-01-30", "name": "E산업", "event": "분기 흑자전환",
     "outcome": "공시 후 5일 +12.7%, 30일 +18.3%", "sim": 0.78},
    {"date": "2023-12-14", "name": "F홀딩스", "event": "최대주주 변경",
     "outcome": "공시 후 5일 +6.5%, 변동성 확대", "sim": 0.76},
    {"date": "2024-05-02", "name": "G테크", "event": "M&A 인수 결정",
     "outcome": "공시 후 5일 -3.8%, 통합 리스크 우려", "sim": 0.74},
]


def kosimcse_similar_cases(query: str, top_k: int = 5) -> pd.DataFrame:
    """KoSimCSE 임베딩 유사 사례 검색 (mock).

    query 길이 기반으로 가짜 유사도 미세 조정.
    """
    cases = _SIM_CASES[:top_k]
    # query 키워드 매칭 시 sim 가산
    boost_kw = {"유상증자": 0, "거래처": 1, "임상": 2, "자사주": 3,
                "흑자": 4, "최대주주": 5, "인수": 6}
    for kw, idx in boost_kw.items():
        if kw in query and idx < len(cases):
            cases[idx] = {**cases[idx], "sim": min(0.98, cases[idx]["sim"] + 0.05)}
    return pd.DataFrame(sorted(cases, key=lambda r: -r["sim"]))


# ============================================================
# 10. OCR 공시 PDF (mock)
# ============================================================
def ocr_mock_pdf_analyze(filename: str = "공시.pdf") -> dict:
    """OCR + LayoutLMv3 처리 mock 결과."""
    return {
        "filename": filename,
        "n_pages": 4,
        "extracted_text_preview": (
            "1. 회사명: ○○전자\n"
            "2. 보고서명: 분기보고서\n"
            "3. 매출액: 12,345억원 (전년 동기 +18%)\n"
            "4. 영업이익: 1,234억원 (전년 동기 +12%)\n"
            "5. 당기순이익: 987억원 (흑자 유지)\n"
            "6. 자기자본비율: 62.5%"
        ),
        "extracted_tables": 3,
        "extracted_figures": 2,
        "ocr_confidence": 0.93,
        "processing_time_sec": 4.2,
    }
