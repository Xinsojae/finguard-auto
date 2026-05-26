"""
FinGuard Auto - Streamlit 대시보드 프로토타입
============================================
실행:
    pip install streamlit lightgbm shap pandas numpy scikit-learn matplotlib
    streamlit run streamlit_app.py

기능:
  - 워치리스트 (좌측 사이드바)
  - 종목 상세: 상승/리스크 점수, AI 신뢰도, 예측 구간
  - TreeSHAP 설명 카드 (Top-5 근거)
  - 2x2 기회-위험 매트릭스 (인터랙티브)
  - 공시·뉴스 패널
  - 백테스트 누적 수익률 그래프

데이터: 합성 패널 (실데이터 전환은 load_synthetic_panel 교체)
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

# -------- 한글 폰트 (Streamlit Cloud: fonts-nanum, Windows: 맑은고딕) --------
_KFONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/malgun.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
KFONT_PATH = None
KFONT_FP = None
for _f in _KFONT_CANDIDATES:
    if os.path.exists(_f):
        font_manager.fontManager.addfont(_f)
        _name = font_manager.FontProperties(fname=_f).get_name()
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [_name] + plt.rcParams["font.sans-serif"]
        KFONT_PATH = _f
        KFONT_FP = font_manager.FontProperties(fname=_f)
        break
plt.rcParams["axes.unicode_minus"] = False

# -------- 스타일 --------
st.set_page_config(page_title="FinGuard Auto", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.metric-up   { color: #2E7D32; font-weight: 700; }
.metric-risk { color: #C62828; font-weight: 700; }
.metric-neutral { color: #666; font-weight: 700; }
.card { border: 1px solid #E0E0E0; border-radius: 8px; padding: 12px; background: #FAFAFA; }
.tag-priority { background: #2E7D32; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; }
.tag-highrisk { background: #F57C00; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; }
.tag-hold     { background: #9E9E9E; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; }
.tag-avoid    { background: #C62828; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; }
.disclaimer { background: #FFF3E0; padding: 8px; border-left: 4px solid #F57C00;
              border-radius: 4px; font-size: 0.85em; color: #666; }
</style>
""", unsafe_allow_html=True)

# -------- 데이터/모델 (캐싱) --------
RNG = np.random.default_rng(42)
N_SECTORS = 12
SECTOR_NAMES = ["반도체","2차전지","바이오","자동차","화학","철강",
                "금융","건설","유통","엔터","통신","조선"]
FEATS = ["ret_1d","ret_5d","ret_20d","vol_5d","vol_20d","ma_ratio","vol_z_20",
         "surge_5d","drawdown_20","news_sent_lag1","news_sent_ma5",
         "disclosure_lag1","bad_disc_20d","good_disc_20d","regime_lag1"]
FEAT_KOR = {
    "ret_1d":"1일 수익률","ret_5d":"5일 수익률","ret_20d":"20일 수익률",
    "vol_5d":"5일 변동성","vol_20d":"20일 변동성",
    "ma_ratio":"MA5/MA20 비율","vol_z_20":"거래량 z-score(20)",
    "surge_5d":"5일 30%+ 급등 플래그","drawdown_20":"20일 드로다운",
    "news_sent_lag1":"전일 뉴스 감성","news_sent_ma5":"5일 뉴스 감성 평균",
    "disclosure_lag1":"전일 공시 이벤트","bad_disc_20d":"20일 악재 공시 수",
    "good_disc_20d":"20일 호재 공시 수","regime_lag1":"시장 국면(0상승/1횡보/2하락)",
}

# 종목 합성 이름 (재미용 - 실제 종목 아님)
FAKE_NAMES = [
    f"{s}{['전자','산업','중공업','제약','케미칼','테크','홀딩스','파이낸셜','글로벌'][i%9]}"
    for i, s in enumerate(["가나","다라","마바","사아","자차","카타","파하","나다","라마","바사","아자","차카",
                            "타파","하나","다라","마바","사아","자차","카타","파하","나다","라마","바사","아자"])
]

@st.cache_data
def gen_panel(n_stocks=150, n_days=600):
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    trans = np.array([[0.97,0.02,0.01],[0.02,0.96,0.02],[0.02,0.04,0.94]])
    drifts = np.array([0.0010, 0.0002, -0.0008])
    states = np.zeros(n_days, dtype=int)
    s = 0
    for t in range(n_days):
        states[t] = s; s = RNG.choice(3, p=trans[s])
    sector_shock = RNG.normal(0, 0.008, (N_SECTORS, n_days))
    rows = []
    for sid in range(n_stocks):
        sector = sid % N_SECTORS
        idio_drift = RNG.normal(0, 0.0004)
        idio_vol = RNG.uniform(0.012, 0.025)
        eps = RNG.normal(0, 1, n_days)
        vol = np.zeros(n_days); vol[0] = idio_vol
        for t in range(1, n_days):
            vol[t] = np.sqrt(0.94*vol[t-1]**2 + 0.06*(eps[t-1]*idio_vol)**2)
        ret = drifts[states] + idio_drift + vol*eps + sector_shock[sector]
        bad = RNG.choice(n_days, size=int(n_days*0.004), replace=False)
        ret[bad] -= RNG.uniform(0.06, 0.14, len(bad))
        good = RNG.choice(n_days, size=int(n_days*0.005), replace=False)
        ret[good] += RNG.uniform(0.03, 0.08, len(good))
        price = 10000*np.exp(np.cumsum(ret))
        volume = np.exp(RNG.normal(13, 0.7, n_days))*(1+6*np.abs(ret))
        news_sent = 0.3*np.sign(np.roll(ret,1)) + RNG.normal(0, 0.5, n_days)
        disc = np.zeros(n_days, dtype=int); disc[good]=1; disc[bad]=-1
        rows.append(pd.DataFrame({
            "date": dates, "stock_id": sid,
            "name": FAKE_NAMES[sid % len(FAKE_NAMES)] + f"{sid:03d}",
            "sector": SECTOR_NAMES[sector],
            "close": price, "return": ret, "volume": volume,
            "news_sent": news_sent, "disclosure": disc, "regime": states,
        }))
    return pd.concat(rows, ignore_index=True)

@st.cache_data
def make_features(df):
    df = df.sort_values(["stock_id","date"]).copy()
    g = df.groupby("stock_id", group_keys=False)
    df["ret_1d"] = g["return"].transform(lambda s: s.shift(1))
    df["ret_5d"] = g["close"].transform(lambda s: s.pct_change(5).shift(1))
    df["ret_20d"] = g["close"].transform(lambda s: s.pct_change(20).shift(1))
    df["vol_5d"] = g["return"].transform(lambda s: s.rolling(5).std().shift(1))
    df["vol_20d"] = g["return"].transform(lambda s: s.rolling(20).std().shift(1))
    df["ma_ratio"] = g["close"].transform(lambda s: (s.rolling(5).mean()/s.rolling(20).mean()-1).shift(1))
    df["vol_z_20"] = g["volume"].transform(lambda s: ((s-s.rolling(20).mean())/s.rolling(20).std()).shift(1))
    df["surge_5d"] = g["close"].transform(lambda s: (s.pct_change(5).shift(1)>0.30)).astype(int)
    df["drawdown_20"] = g["close"].transform(lambda s: (s/s.rolling(20).max()-1).shift(1))
    df["news_sent_lag1"] = g["news_sent"].transform(lambda s: s.shift(1))
    df["news_sent_ma5"] = g["news_sent"].transform(lambda s: s.rolling(5).mean().shift(1))
    df["disclosure_lag1"] = g["disclosure"].transform(lambda s: s.shift(1))
    df["bad_disc_20d"] = g["disclosure"].transform(lambda s: (s==-1).rolling(20).sum().shift(1))
    df["good_disc_20d"] = g["disclosure"].transform(lambda s: (s==1).rolling(20).sum().shift(1))
    df["regime_lag1"] = g["regime"].transform(lambda s: s.shift(1))
    df["fwd_ret_5d"] = g["close"].transform(lambda s: s.pct_change(5).shift(-5))
    df["target_up"] = (df["fwd_ret_5d"]>df["fwd_ret_5d"].quantile(0.70)).astype(int)
    df["target_crash"] = (df["fwd_ret_5d"]<-0.05).astype(int)
    return df

@st.cache_resource
def train_models(panel):
    df = panel.dropna(subset=FEATS+["target_up","target_crash"]).reset_index(drop=True)
    n = len(df)
    cut = int(n*0.7)
    tr = df.iloc[:cut]; va = df.iloc[cut:]
    m_up = lgb.LGBMClassifier(objective="binary", num_leaves=31, learning_rate=0.05,
        n_estimators=300, min_data_in_leaf=200, verbose=-1, random_state=42)
    m_up.fit(tr[FEATS], tr["target_up"], eval_set=[(va[FEATS], va["target_up"])],
             callbacks=[lgb.early_stopping(40, verbose=False)])
    m_cr = lgb.LGBMClassifier(objective="binary", num_leaves=31, learning_rate=0.05,
        n_estimators=300, min_data_in_leaf=200, verbose=-1, random_state=42)
    m_cr.fit(tr[FEATS], tr["target_crash"], eval_set=[(va[FEATS], va["target_crash"])],
             callbacks=[lgb.early_stopping(40, verbose=False)])
    # 검증 지표
    p_up = m_up.predict_proba(va[FEATS])[:,1]
    p_cr = m_cr.predict_proba(va[FEATS])[:,1]
    metrics = dict(
        up_auc=roc_auc_score(va["target_up"], p_up),
        up_pr=average_precision_score(va["target_up"], p_up),
        cr_auc=roc_auc_score(va["target_crash"], p_cr),
        cr_pr=average_precision_score(va["target_crash"], p_cr),
    )
    return m_up, m_cr, metrics

@st.cache_data
def latest_snapshot(panel):
    df = panel.dropna(subset=FEATS).copy()
    latest_date = df["date"].max()
    snap = df[df["date"]==latest_date].copy()
    return snap, latest_date

def classify(up, risk):
    if up>=50 and risk<50: return "PRIORITY"
    if up>=50 and risk>=50: return "HIGH-RISK"
    if up<50 and risk<50: return "HOLD"
    return "AVOID"

def tag_html(cat):
    return {"PRIORITY":"<span class='tag-priority'>우선 관심</span>",
            "HIGH-RISK":"<span class='tag-highrisk'>고위험 관심</span>",
            "HOLD":"<span class='tag-hold'>관망</span>",
            "AVOID":"<span class='tag-avoid'>회피</span>"}[cat]

# -------- 메인 --------
st.title("🛡️ FinGuard Auto")
st.caption("개인투자자를 위한 설명 가능한 AI 리스크 분석·투자 학습·모의 검증 플랫폼")
st.markdown("<div class='disclaimer'>⚠️ 본 프로토타입은 합성 데이터 기반 학술 데모입니다. "
            "실제 매수·매도 추천이 아닙니다. 최종 투자 판단·책임은 사용자에게 있습니다.</div>",
            unsafe_allow_html=True)

with st.spinner("합성 데이터 생성 및 모델 학습 중..."):
    panel = gen_panel(n_stocks=120, n_days=600)
    panel = make_features(panel)
    m_up, m_cr, metrics = train_models(panel)
    snap, latest_date = latest_snapshot(panel)
    snap["score_up_p"] = m_up.predict_proba(snap[FEATS])[:,1]
    snap["score_cr_p"] = m_cr.predict_proba(snap[FEATS])[:,1]
    snap["score_up"] = (snap["score_up_p"]*100).round(0).astype(int)
    snap["score_risk"] = (snap["score_cr_p"]*100).round(0).astype(int)
    snap["category"] = snap.apply(lambda r: classify(r["score_up"], r["score_risk"]), axis=1)
    # 신뢰도 (간단 휴리스틱: 예측 확률 0.5와의 거리)
    snap["confidence"] = ((snap["score_up_p"]-0.5).abs() + (snap["score_cr_p"]-0.5).abs()).rank(pct=True)

# 시장 위험도 (전체 평균 risk score)
mkt_risk = int(snap["score_risk"].mean())
mkt_label = "낮음" if mkt_risk<35 else "중간" if mkt_risk<55 else "높음"

# -------- 사이드바 --------
with st.sidebar:
    st.subheader("📋 워치리스트")
    default_picks = snap.nlargest(8, "score_up").head(5)["name"].tolist() + \
                    snap.nlargest(5, "score_risk").head(2)["name"].tolist() + \
                    [snap.iloc[0]["name"]]
    picks = st.multiselect("관심 종목", snap["name"].tolist(),
                           default=list(dict.fromkeys(default_picks))[:6])
    st.divider()
    st.subheader("🌡️ 오늘의 시장")
    st.metric("기준일", str(latest_date.date()))
    st.metric("시장 위험도", f"{mkt_risk}/100 ({mkt_label})")
    st.metric("우선 관심 후보 수", int((snap["category"]=="PRIORITY").sum()))
    st.metric("회피 후보 수", int((snap["category"]=="AVOID").sum()))
    st.divider()
    st.subheader("📊 모델 성능 (out-of-time)")
    st.metric("상승 ROC-AUC", f"{metrics['up_auc']:.3f}")
    st.metric("급락 ROC-AUC", f"{metrics['cr_auc']:.3f}")
    st.caption("PoC 베이스라인. KF-DeBERTa 추가 시 향상 예상.")

# -------- 메인 영역 --------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 종목 분석", "🗺️ 2×2 매트릭스",
                                          "📰 공시·뉴스", "📈 백테스트",
                                          "🔍 공시 분석기"])

with tab1:
    if not picks:
        st.warning("좌측 사이드바에서 관심 종목을 선택하세요.")
    else:
        # 워치리스트 카드
        cols = st.columns(min(len(picks), 3))
        for i, name in enumerate(picks):
            with cols[i % 3]:
                row = snap[snap["name"]==name].iloc[0]
                cat = row["category"]
                st.markdown(f"""
                <div class='card'>
                  <h4>{name} <small>({row['sector']})</small></h4>
                  {tag_html(cat)}
                  <p style='margin-top:8px;'>
                    <span class='metric-up'>상승 {row['score_up']}</span> &nbsp;
                    <span class='metric-risk'>리스크 {row['score_risk']}</span>
                  </p>
                  <p style='font-size:0.85em; color:#777;'>
                    신뢰도: {'HIGH' if row['confidence']>0.66 else 'MED' if row['confidence']>0.33 else 'LOW'}
                  </p>
                </div>
                """, unsafe_allow_html=True)

        st.divider()
        sel = st.selectbox("상세 분석할 종목", picks)
        if sel:
            row = snap[snap["name"]==sel].iloc[0]
            cat = row["category"]
            c1, c2, c3 = st.columns([2, 2, 3])
            with c1:
                st.markdown(f"### {sel}")
                st.markdown(tag_html(cat), unsafe_allow_html=True)
                st.markdown(f"**섹터**: {row['sector']}")
                st.markdown(f"**기준일**: {latest_date.date()}")
                st.markdown(f"**현재가**: {row['close']:,.0f}원")
            with c2:
                st.metric("상승 가능성", f"{row['score_up']}/100",
                          delta=f"{row['score_up']-50:+d} vs 중립")
                st.metric("급락 위험", f"{row['score_risk']}/100",
                          delta=f"{row['score_risk']-50:+d} vs 중립", delta_color="inverse")
                conf = 'HIGH' if row['confidence']>0.66 else 'MEDIUM' if row['confidence']>0.33 else 'LOW'
                st.metric("AI 신뢰도", conf)
            with c3:
                # SHAP-like top features
                x_row = row[FEATS].values.reshape(1, -1)
                # LightGBM Booster의 raw 예측 분해 (SHAP 대용으로 booster predict + contrib)
                contrib = m_up.booster_.predict(x_row, pred_contrib=True)[0][:-1]
                top_idx = np.argsort(-np.abs(contrib))[:5]
                st.markdown("**🔍 판단 근거 (TreeSHAP Top-5)**")
                fig, ax = plt.subplots(figsize=(5, 3))
                names = [FEAT_KOR[FEATS[i]] for i in top_idx][::-1]
                values = [contrib[i] for i in top_idx][::-1]
                colors = ["#2E7D32" if v>0 else "#C62828" for v in values]
                ax.barh(names, values, color=colors)
                ax.axvline(0, color="#444", lw=0.8)
                ax.set_xlabel("SHAP contribution to 상승 score",
                              fontproperties=KFONT_FP)
                ax.tick_params(axis="y", labelsize=9)
                if KFONT_FP is not None:
                    for lbl in ax.get_yticklabels():
                        lbl.set_fontproperties(KFONT_FP)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            st.divider()
            st.markdown("**🤖 AI 설명 카드 (자연어, V3에서 HyperCLOVA X로 생성 예정)**")
            sign = "상승 시그널" if cat in ["PRIORITY","HIGH-RISK"] else "관망 또는 회피"
            risk_note = "리스크 신호도 동시에 강하므로 모의투자로 검증 권장" if row['score_risk']>=50 else "리스크 신호는 낮음"
            st.info(
                f"**{sel}**은(는) **{tag_html(cat)}**(으)로 분류됩니다. "
                f"{sign}이 우세하고, {risk_note}. "
                f"AI 신뢰도는 **{conf}**이며, 권장 행동: "
                f"{'백테스트로 사후 검증' if conf=='HIGH' else '모의투자 5거래일 후 재평가' if conf=='MEDIUM' else '데이터 추가 확보 후 재평가'}. "
                f"본 결과는 매수·매도 추천이 아닌 의사결정 보조 정보입니다.".replace(
                    "<span class='tag-priority'>우선 관심</span>", "우선 관심 후보"
                ).replace(
                    "<span class='tag-highrisk'>고위험 관심</span>", "고위험 관심 후보"
                ).replace(
                    "<span class='tag-hold'>관망</span>", "관망 후보"
                ).replace(
                    "<span class='tag-avoid'>회피</span>", "회피 후보"
                )
            )

with tab2:
    st.subheader("2×2 기회-위험 매트릭스")
    st.caption("점 = 종목. 우측 상단: 상승 가능성 높음 + 리스크 낮음 = 우선 관심 후보")
    fig, ax = plt.subplots(figsize=(9, 7))
    color_map = {"PRIORITY":"#2E7D32","HIGH-RISK":"#F57C00","HOLD":"#9E9E9E","AVOID":"#C62828"}
    for cat, c in color_map.items():
        sub = snap[snap["category"]==cat]
        ax.scatter(sub["score_up"], sub["score_risk"], c=c, alpha=0.65, s=55,
                   edgecolor="white", linewidth=0.5, label=cat)
    # 선택된 종목 강조
    for name in picks:
        r = snap[snap["name"]==name].iloc[0]
        ax.scatter([r["score_up"]], [r["score_risk"]], facecolor="none",
                   edgecolor="black", linewidth=2, s=150, marker="o")
        ax.annotate(name, (r["score_up"], r["score_risk"]),
                    fontsize=8, xytext=(5, 5), textcoords="offset points",
                    fontproperties=KFONT_FP)
    ax.axhline(50, color="#444", ls="--", lw=1)
    ax.axvline(50, color="#444", ls="--", lw=1)
    ax.set_xlim(0,100); ax.set_ylim(0,100)
    ax.set_xlabel("Upside Score"); ax.set_ylabel("Risk Score")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.25)
    st.pyplot(fig); plt.close(fig)

    st.divider()
    st.markdown("### 카테고리별 종목 수")
    counts = snap["category"].value_counts()
    cols = st.columns(4)
    for i, (cat, c) in enumerate(zip(["PRIORITY","HIGH-RISK","HOLD","AVOID"],
                                      ["우선 관심","고위험 관심","관망","회피"])):
        cols[i].metric(c, int(counts.get(cat, 0)))

with tab3:
    st.subheader("📰 공시·뉴스 패널")
    sel = st.selectbox("종목 선택", picks if picks else snap["name"].tolist()[:5], key="disc_sel")
    if sel:
        sid = snap[snap["name"]==sel]["stock_id"].iloc[0]
        hist = panel[panel["stock_id"]==sid].tail(60).copy()
        # 최근 공시 이벤트 추출
        events = hist[hist["disclosure"]!=0].tail(5)
        st.markdown(f"**{sel} - 최근 30일 공시 이벤트**")
        if len(events)==0:
            st.success("최근 30일 내 주요 공시 이벤트 없음 (긍정 시그널)")
        else:
            for _, e in events.iterrows():
                kind = "🟢 호재 공시" if e["disclosure"]==1 else "🔴 악재 공시"
                st.markdown(f"- **{e['date'].date()}** {kind} (수익률 영향 {e['return']*100:+.1f}%)")
        st.markdown(f"**최근 뉴스 감성 추세** (합성)")
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(hist["date"], hist["news_sent"], color="#1565C0", lw=1)
        ax.axhline(0, color="#999", ls="--")
        ax.fill_between(hist["date"], hist["news_sent"], 0,
                        where=hist["news_sent"]>0, color="#2E7D32", alpha=0.2)
        ax.fill_between(hist["date"], hist["news_sent"], 0,
                        where=hist["news_sent"]<0, color="#C62828", alpha=0.2)
        ax.set_ylabel("News Sentiment")
        ax.set_title(f"{sel} 뉴스 감성 (60일)", fontproperties=KFONT_FP)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

with tab4:
    st.subheader("📈 백테스트: A(상승만) vs B(상승+리스크 필터)")
    st.caption("walk-forward 검증 기간, 수수료/슬리피지 0.3% 반영")
    # 간단 백테스트 (snap에서 그대로 데모)
    with st.spinner("백테스트 실행 중..."):
        df = panel.dropna(subset=FEATS+["target_up","target_crash","fwd_ret_5d"]).reset_index(drop=True)
        cut = int(len(df)*0.7)
        va = df.iloc[cut:].copy()
        va["score_up"] = m_up.predict_proba(va[FEATS])[:,1]
        va["score_cr"] = m_cr.predict_proba(va[FEATS])[:,1]
        rets_a, rets_b, dates_bt = [], [], []
        avoided = 0
        for d, sub in va.groupby("date"):
            if len(sub)<60: continue
            a = sub.nlargest(20, "score_up")
            ra = a["fwd_ret_5d"].mean() - 0.003
            cr_cut = sub["score_cr"].quantile(0.70)
            sub_lr = sub[sub["score_cr"]<=cr_cut]
            b_filt = sub_lr.nlargest(20, "score_up")
            avoided += len(set(a["stock_id"]) - set(b_filt["stock_id"]))
            rb = b_filt["fwd_ret_5d"].mean() - 0.003 if len(b_filt)>0 else 0.0
            rets_a.append(ra); rets_b.append(rb); dates_bt.append(d)
        ra_s = pd.Series(rets_a, index=dates_bt)
        rb_s = pd.Series(rets_b, index=dates_bt)
        cum_a = (1+ra_s).cumprod(); cum_b = (1+rb_s).cumprod()

    if len(cum_a) == 0:
        st.warning(
            "백테스트 가능한 날짜 그룹이 없습니다. 패널 크기가 너무 작아 "
            "일별 종목 수가 60개 미만일 가능성이 큽니다. "
            "사이드바에서 n_stocks/n_days를 늘려보세요."
        )
    else:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(cum_a.index, cum_a.values, color="#C62828", lw=1.5,
                label=f"A: 상승만 (누적 {cum_a.iloc[-1]-1:+.1%})")
        ax.plot(cum_b.index, cum_b.values, color="#2E7D32", lw=1.5,
                label=f"B: 상승+리스크 필터 (누적 {cum_b.iloc[-1]-1:+.1%})")
        ax.axhline(1.0, color="#999", ls="--", lw=0.8)
        ax.set_title("Cumulative Return")
        ax.set_ylabel("Cumulative Asset (start=1.0)")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("A 평균 수익률", f"{ra_s.mean()*100:.2f}%/주")
        c2.metric("B 평균 수익률", f"{rb_s.mean()*100:.2f}%/주")
        c3.metric("A MDD", f"{(cum_a/cum_a.cummax()-1).min()*100:.1f}%")
        c4.metric("리스크 필터 회피 후보", f"{avoided:,}")

    st.divider()
    st.markdown("**📝 해석**")
    st.info(
        "본 합성 데이터에서는 급락 모델의 신호가 약해(ROC-AUC ~0.58) 리스크 필터(B)가 A 대비 "
        "유의미한 개선을 보이지 못했습니다. **실제 데이터에서는 KF-DeBERTa 기반 뉴스·공시 "
        "감성·이벤트 피처를 추가하면 급락 신호가 강화되어 B가 A를 상회할 것으로 가설**합니다. "
        "이 결과는 발표에서 '솔직한 베이스라인 + 개선 방향'으로 제시할 수 있는 honest finding입니다."
    )

# ============================================================================
# Tab 5: 공시 분석기 (NEW)
# ============================================================================
with tab5:
    try:
        import disclosure_analyzer as da
    except Exception as e:
        st.error(f"disclosure_analyzer 모듈 로드 실패: {e}")
        da = None

    if da is not None:
        st.subheader("🔍 공시 분석기 - 30개 이벤트 유형 자동 분류 + 쉬운 해석")
        st.caption("OpenDART 공시를 30개 유형으로 분류하고 위험도·체크포인트·유사 사례를 제공합니다. "
                   "발표 데모용 Mock 데이터로 작동, 실데이터 전환은 disclosure_analyzer.py 가이드 참조.")

        mode = st.radio("입력 방식", ["📋 Mock 공시 목록에서 선택", "✍️ 텍스트 직접 입력"], horizontal=True)

        if mode == "📋 Mock 공시 목록에서 선택":
            mock = da.load_mock_disclosures(n_days=20)
            colA, colB = st.columns([1, 2])
            with colA:
                cname = st.selectbox("기업 선택", ["전체"] + sorted(mock["corp_name"].unique().tolist()))
                view = mock if cname == "전체" else mock[mock["corp_name"] == cname]
                st.caption(f"총 {len(view)}건")
                view_disp = view.copy()
                view_disp["label"] = view_disp.apply(
                    lambda r: f"[{r['rcept_dt']}] {r['corp_name']} - {r['report_nm']}", axis=1)
                pick = st.selectbox("공시 선택", view_disp["label"].tolist())
                row = view_disp[view_disp["label"] == pick].iloc[0]
                title = row["report_nm"]; body = row["report_body"]
                corp = row["corp_name"]; date = row["rcept_dt"]
            with colB:
                st.markdown(f"#### 📄 {corp} ({date})")
                st.markdown(f"**제목**: {title}")
                st.markdown("**본문**:")
                st.text_area("", body, height=120, label_visibility="collapsed", key="body_view")
        else:
            corp = st.text_input("기업명 (선택)", "삼성전자")
            date = st.date_input("공시일").strftime("%Y-%m-%d")
            title = st.text_input("공시 제목", "주요사항보고서(유상증자결정)")
            body = st.text_area(
                "공시 본문",
                "당사는 운영자금 확보를 위해 주주배정 후 실권주 일반공모 방식의 유상증자를 결정하였습니다. "
                "발행주식 1,000,000주, 발행가액 12,000원, 시설투자 자금으로 사용 예정.",
                height=140,
            )

        if st.button("🔍 분석하기", type="primary"):
            with st.spinner("공시 분류 중..."):
                full_text = (title or "") + " " + (body or "")
                results = da.classify(full_text)

            if not results:
                st.warning("매칭되는 공시 유형이 없습니다.")
            else:
                top = results[0]
                st.markdown("---")
                c1, c2, c3 = st.columns([2, 2, 3])
                with c1:
                    st.markdown(f"### {top.name}")
                    st.markdown(f"**카테고리**: {top.category}")
                with c2:
                    st.markdown(f"<h3 style='color:{top.risk_color};margin:0;'>위험도: {top.risk_label}</h3>",
                                unsafe_allow_html=True)
                    st.markdown(f"**신뢰도**: {top.confidence:.0%}")
                with c3:
                    st.markdown("**매칭 키워드**:")
                    st.markdown(", ".join([f"`{k}`" for k in top.matched_keywords]))

                st.markdown("### 📖 쉬운 해석")
                st.info(top.explanation)

                st.markdown("### ✅ 투자자 체크포인트")
                for i, cp in enumerate(top.checkpoints, 1):
                    st.markdown(f"{i}. {cp}")

                if top.similar_cases:
                    st.markdown("### 📚 유사 과거 사례")
                    for sc in top.similar_cases:
                        st.markdown(f"- {sc}")

                if len(results) > 1:
                    with st.expander(f"🔎 추가 매칭 {len(results)-1}건 보기"):
                        for r in results[1:]:
                            st.markdown(f"**{r.name}** ({r.category}) - 위험도: "
                                        f"<span style='color:{r.risk_color};font-weight:700;'>{r.risk_label}</span>"
                                        f" / 신뢰도: {r.confidence:.0%}",
                                        unsafe_allow_html=True)
                            st.caption(f"매칭: {', '.join(r.matched_keywords)}")

        st.markdown("---")
        st.markdown("### 📊 분류기 작동 통계 (Mock 데이터)")
        mock_all = da.load_mock_disclosures(n_days=20)
        stats = {}
        for _, r in mock_all.iterrows():
            res = da.classify(r["report_nm"] + " " + r["report_body"])
            key = "미분류" if not res else res[0].risk_label
            stats[key] = stats.get(key, 0) + 1
        cols = st.columns(4)
        items = list(stats.items())
        for i, (label, n) in enumerate(items):
            cols[i % 4].metric(label, n)
        coverage = (len(mock_all) - stats.get("미분류", 0)) / len(mock_all) if len(mock_all) > 0 else 0
        st.caption(f"분류 커버리지: {coverage:.0%} ({len(mock_all)-stats.get('미분류',0)}/{len(mock_all)}건)")

st.divider()
st.caption("FinGuard Auto · AI 개론 프로젝트 · 2026.05 · 본 프로토타입은 합성 데이터 기반 학술 데모입니다.")
