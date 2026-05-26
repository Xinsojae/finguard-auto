# FinGuard Auto - 프로토타입 (Streamlit)

이 폴더는 단독으로 실행·배포 가능한 프로토타입입니다.

## 파일 구성

```
prototype/
├── streamlit_app.py        # 메인 앱 (5개 탭)
├── disclosure_analyzer.py  # 공시 분석기 모듈 (30개 유형 룰베이스)
├── requirements.txt        # 의존성
├── .streamlit/
│   └── config.toml         # 테마·서버 설정
├── .gitignore
├── figures/
│   └── V5_dashboard_mockup.png  # 참고용 화면 목업
├── poc_results.json        # PoC v1 결과 (참고)
└── poc_results_v2.json     # PoC v2 결과 (참고)
```

## 로컬 실행

```bash
cd prototype
pip install -r requirements.txt
streamlit run streamlit_app.py
```

브라우저가 자동으로 열리며 `http://localhost:8501` 에 접속됩니다.

## 5개 탭

1. **🎯 종목 분석** — LightGBM 상승/리스크 점수, AI 신뢰도, TreeSHAP Top-5
2. **🗺️ 2×2 매트릭스** — 기회-위험 인터랙티브 산점도
3. **📰 공시·뉴스** — 종목별 공시 이벤트와 뉴스 감성 추세
4. **📈 백테스트** — A(상승만) vs B(상승+리스크 필터) 누적 수익률
5. **🔍 공시 분석기** — 30개 공시 유형 자동 분류 + 자연어 해석 + 체크포인트

## 배포

상위 폴더의 `DEPLOY.md`를 참고하세요. **Streamlit Community Cloud** 사용 시:

1. GitHub 저장소에 `prototype/` 폴더 안의 모든 파일을 푸시
2. <https://share.streamlit.io> 에서 **New app**
3. Main file path: `streamlit_app.py` (또는 모노레포면 `prototype/streamlit_app.py`)
4. Deploy 후 URL 공유

## 실데이터 전환

`streamlit_app.py`의 `gen_panel()` 함수와 `disclosure_analyzer.py`의 `load_mock_disclosures()` 함수를 각각 pykrx/FinanceDataReader/OpenDART 호출로 교체하면 실시장 데이터로 작동합니다. 자세한 내용은 각 파일 상단 docstring 참조.

## 면책

본 프로토타입은 학술 데모이며, 매수·매도 추천이 아닙니다. 최종 투자 판단·책임은 사용자에게 있습니다.
