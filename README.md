# FinGuard Auto - 프로젝트 산출물 가이드

> AI 개론 프로젝트 (2026-06-26 발표). 인공지능 기반 개인투자자 리스크 분석·모의 검증 플랫폼.

## 폴더 구성

```
ai_Project/
├── README.md                              ← 이 파일
├── fin_guard_auto_기획서_보강본_v3.md      ← 최종 기획서 (PPT 작성 기반)
├── fin_guard_auto_기획서_보강본.md         ← 중간 버전 (참고용)
├── poc_finguard.py                        ← PoC v1 (9 피처, 기본)
├── poc_finguard_v2.py                     ← PoC v2 (15 피처, 섹터+국면+뉴스+공시, 합성)
├── poc_finguard_real.py                   ← PoC 실데이터 (KOSPI 상위 100, 2021~2025)
├── poc_results.json                       ← v1 결과
├── poc_results_v2.json                    ← v2 결과
├── poc_results_real.json                  ← 실데이터 결과
├── streamlit_app.py                       ← 인터랙티브 대시보드 프로토타입
├── figures/                               ← PPT용 시각화 (PNG)
│   ├── V1_architecture.png
│   ├── V2_pipeline.png
│   ├── V3_score_flow.png
│   ├── V4_2x2_matrix.png
│   ├── V5_dashboard_mockup.png
│   ├── V6_backtest_curve.png
│   └── V_extra_poc_metrics.png
└── 소스파일/                              ← 원본 자료 (참조용)
    ├── fin_guard_auto_기획서_최종정리본.md
    └── 프로젝트 작성할때 주의해야할 것.txt
```

## 빠른 실행

### 1. PoC 실험 재실행 (v2 권장)

```bash
pip install lightgbm pandas numpy scikit-learn matplotlib
python poc_finguard_v2.py
```

결과는 `poc_results_v2.json`에 저장되고, 백테스트 그래프는 `figures/V6_backtest_curve.png`에 생성됩니다.

### 2. Streamlit 대시보드 실행

```bash
pip install streamlit lightgbm pandas numpy scikit-learn matplotlib
streamlit run streamlit_app.py
```

브라우저가 자동으로 열리며 (`http://localhost:8501`), 다음 탭이 표시됩니다:

- **🎯 종목 분석** - 워치리스트 카드, 종목 상세, TreeSHAP Top-5 근거, AI 설명 카드
- **🗺️ 2×2 매트릭스** - 인터랙티브 기회-위험 산점도
- **📰 공시·뉴스** - 종목별 공시 이벤트, 뉴스 감성 추세
- **📈 백테스트** - A(상승만) vs B(상승+리스크 필터) 누적 수익률

### 3. 시각화 한글로 다시 생성하기

샌드박스 환경에 한글 폰트가 없어 그래프가 영문으로 출력되어 있습니다. 로컬(Windows)에서 한글로 다시 생성:

```bash
# Windows에 기본 설치된 맑은 고딕 사용 (또는 Nanum 설치)
# poc_finguard_v2.py 상단 KFONT_CANDIDATES에 추가:
#   "C:/Windows/Fonts/malgun.ttf"
python poc_finguard_v2.py
```

## 실데이터(KOSPI/KOSDAQ)로 전환하기

`poc_finguard_v2.py`의 `load_synthetic_panel()` 함수를 다음으로 교체하면 됩니다:

```python
# 옵션 A: pykrx (최신 버전은 KRX_ID/KRX_PW 환경변수 필요)
from pykrx import stock

def load_real_panel(start="20210104", end="20251231"):
    tickers = stock.get_market_ticker_list(end, market="KOSPI")[:150]
    rows = []
    for t in tickers:
        df = stock.get_market_ohlcv(start, end, t).reset_index()
        df = df.rename(columns={"날짜":"date","종가":"close","거래량":"volume"})
        df["stock_id"] = t
        df["return"] = df["close"].pct_change()
        # 합성 NLP 피처는 0으로 채우고, 실제로는 KF-DeBERTa 점수로 채워야 함
        df["news_sent"] = 0.0
        df["disclosure"] = 0
        df["regime"] = 1  # 초기엔 횡보로 가정
        df["sector"] = "기타"  # KRX 섹터 정보 별도 매핑 필요
        rows.append(df[["date","stock_id","sector","close","return",
                        "volume","news_sent","disclosure","regime"]])
    return pd.concat(rows, ignore_index=True)

# 옵션 B: FinanceDataReader (간편하나 API 변경 자주 발생)
import FinanceDataReader as fdr
def load_real_panel_fdr(start="2021-01-04", end="2025-12-31"):
    krx = fdr.StockListing("KOSPI").head(150)
    rows = []
    for sym in krx["Code"]:
        df = fdr.DataReader(sym, start, end).reset_index()
        df = df.rename(columns={"Date":"date","Close":"close","Volume":"volume"})
        df["stock_id"] = sym
        df["return"] = df["close"].pct_change()
        df["news_sent"] = 0.0; df["disclosure"] = 0; df["regime"] = 1; df["sector"] = "기타"
        rows.append(df[["date","stock_id","sector","close","return",
                        "volume","news_sent","disclosure","regime"]])
    return pd.concat(rows, ignore_index=True)
```

뉴스 감성·공시 이벤트 피처는 별도 파이프라인 필요:
- 뉴스: OpenDART 보도자료 + 네이버 검색 API + KF-DeBERTa fine-tune
- 공시: OpenDART API + 룰베이스 30종 분류 + KoBigBird 장문 처리

## PoC 결과 요약 (v2, 합성 데이터)

| 지표 | 상승 모델 | 급락 모델 |
|---|---|---|
| ROC-AUC (평균) | 0.539 | 0.581 |
| PR-AUC | 0.323 | 0.164 |
| Top-20% Precision | 0.337 | 0.174 |
| Top-20% Recall | 0.228 | 0.263 |

- 양성비율: 상승 0.30, 급락 0.13
- 패널: 150종목 × 1,200일 = 180K 행, 15 피처, 12 섹터
- 백테스트: A 누적 -63.7%, B 누적 -73.2% (합성 데이터 한계 — 실데이터 + KF-DeBERTa 추가 시 개선 예상)

> **honest finding**: 합성 데이터의 급락 신호가 약하여 리스크 필터 B가 A를 상회하지 못했음. 이는 실데이터(KF-DeBERTa로 뉴스·공시 신호 강화) 적용 시 개선될 것으로 가설하며, 발표에서 솔직히 보고할 수 있는 베이스라인 결과.

## PoC 결과 (실데이터, KOSPI 상위 100)

`poc_finguard_real.py` 실행 결과. FinanceDataReader로 KOSPI 시가총액 상위 100종목, 2021-01-04 ~ 2025-12-31 일봉 다운로드 후 v2와 동일 피처·모델 파이프라인 적용.

```bash
pip install lightgbm pandas numpy scikit-learn matplotlib finance-datareader pykrx
python poc_finguard_real.py
```

**데이터 패널**: 99종목 × 1,224일 = 115,838행, 15 피처

| 지표 | 상승 모델 | 급락 모델 | 합성 대비 |
|---|---|---|---|
| ROC-AUC (평균) | **0.559** | **0.628** | 상승 +0.020, 급락 **+0.047** |
| PR-AUC | 0.357 | 0.196 | 상승 +0.034, 급락 +0.032 |
| Top-20% Precision | 0.376 | 0.207 | 상승 +0.039, 급락 +0.033 |
| Top-20% Recall | 0.239 | 0.318 | 상승 +0.011, 급락 +0.055 |

- 양성비율: 상승 0.299, 급락 0.134 (합성과 거의 동일)
- 결과 JSON: `poc_results_real.json`
- 백테스트 그래프: `figures/V6_backtest_real.png`

> **핵심 발견**: 급락 모델 ROC-AUC가 합성 0.581 → **실데이터 0.628**로 가장 크게 개선. NLP 피처가 0인 상태(KF-DeBERTa 미적용)에서도 실제 시장의 급락 패턴이 가격·거래량 기술지표만으로 어느 정도 감지됨을 의미. KF-DeBERTa로 뉴스·공시 신호를 추가하면 추가 개선 여지가 있다는 가설을 뒷받침.

### 실데이터 PoC의 한계 (정직하게 보고)

1. **NLP 피처 미연동**: `news_sent`, `disclosure`는 0 placeholder. KF-DeBERTa 연동 후 재실험 필요. (코드 내 `TODO` 주석 위치 명시)
2. **regime은 0 고정**: KOSPI 지수의 3-state HMM 분류 미적용.
3. **sector는 'Unknown'**: KRX 섹터 매핑 미연동.
4. **생존편향**: pykrx KRX 로그인 실패로 FDR `StockListing("KOSPI")` 현재 시점 기준 사용. 시작일 시점 상위 100과 차이 가능.
5. **백테스트 누적 수익률 과장**: `fwd_ret_5d`(5일 중첩 수익률)를 일별 picks에 적용 후 일별 컴파운드로 합산하여 실제보다 5배 가량 과장. **합성 v2도 동일 구조라 상대 비교에는 영향 없으나, 절대 수치 사용 금지**. 향후 비중첩 holding period 또는 P&L 추적 방식으로 교체 필요.

## PPT 제작 가이드

발표 슬라이드 권장 구성 (20~25장):

1. 타이틀 + Executive Summary 1장 → 기획서 §0
2. 문제 정의 (개인투자자 통계) 1장 → §2
3. 시장 분석 + 기존 서비스 한계 1장 → §5.4
4. 선행 연구 (글로벌·국내 모델) 1~2장 → §5.1, §5.2
5. 솔루션 개요 + 차별성 1장 → §5.3, §6
6. 시스템 아키텍처 1장 → **V1_architecture.png**
7. 데이터 파이프라인 1장 → **V2_pipeline.png**
8. 점수 산출 흐름 1장 → **V3_score_flow.png**
9. 2×2 매트릭스 1장 → **V4_2x2_matrix.png**
10. AI 모델 슬림화 + 근거 1~2장 → §10.2~10.5
11. PoC 실험 결과 1~2장 → **V_extra_poc_metrics.png** + §13.1
12. 백테스트 + 리스크 필터 1장 → **V6_backtest_curve.png**
13. 설명 카드 (TreeSHAP) 1장 → §14
14. 대시보드 시연 1장 → **V5_dashboard_mockup.png** + Streamlit 실시간 시연
15. 사회적 가치 정량화 1장 → §22.2
16. 규제·윤리·편향 1장 → §20, §23.3
17. Plan B / 한계 1장 → §23.1~23.2
18. 사업화 (압축) 1장 → §19
19. 향후 확장 + 결론 1~2장 → §24

## 참고 문헌

기획서 v3 마지막의 References 섹션 (15편) 참조.

## 라이선스 및 면책

- 본 프로젝트는 학술 목적의 데모이며, 매수·매도 추천이 아닙니다.
- 최종 투자 판단·책임은 사용자에게 있습니다.
- 합성 데이터 기반이므로 실데이터 성능은 다를 수 있습니다.
