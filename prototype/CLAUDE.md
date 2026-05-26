# FinGuard Auto - Claude Code 프로젝트 컨텍스트

> **목적**: Claude Code가 이 폴더에서 작업할 때 자동으로 읽는 컨텍스트 문서. 프로젝트 배경·현재 상태·코딩 컨벤션·진행할 작업을 한 곳에 모았다.

---

## 1. 프로젝트 개요

- **이름**: FinGuard Auto
- **부제**: 개인투자자를 위한 설명 가능한 AI 리스크 분석·투자 학습·모의 검증 플랫폼
- **맥락**: 대학교 인공지능개론 학기말 프로젝트
- **최종 산출물**: PPT 발표 자료 (라이브 데모 URL 포함)
- **마감일**: 2026-06-26
- **평가자**: 교수 1명 + AI 3명, 총 4명
- **언어**: 모든 산출물·코드 주석·문서는 한국어 (코드 식별자는 영문)

### 핵심 가치 5

1. 위험을 먼저 발견한다
2. 왜 위험한지 설명한다 (TreeSHAP + LLM)
3. 투자 전 검증하게 한다 (백테스트 + 모의투자)
4. AI 신뢰도를 표시해 과신을 막는다
5. 추천이 아닌 의사결정 보조와 학습을 제공한다

### 한 줄 차별성

> 한국형 KF-DeBERTa 공시·뉴스 분석 + 2축(상승·급락) 점수 + TreeSHAP 설명 + 모의 검증 — 이 네 가지를 통합한 첫 학생 프로젝트 시도.

---

## 2. 폴더 구조

```
ai_Project/
├── CLAUDE.md                              ← 이 파일 (Claude Code 컨텍스트)
├── CLAUDE_PROMPTS.md                      ← 다음 작업별 프롬프트 모음
├── README.md                              ← 일반 사용자용 가이드
├── DEPLOY.md                              ← 배포 가이드 (Streamlit Cloud / HF Spaces)
│
├── fin_guard_auto_기획서_보강본_v3.md      ← 최종 기획서 (PPT 작성 기반, ~33KB)
├── fin_guard_auto_기획서_보강본.md         ← 중간 버전 (참고용)
│
├── prototype/                             ← 단독 실행·배포 가능한 프로토타입
│   ├── README.md
│   ├── streamlit_app.py                   ← 메인 앱 (5개 탭, ~520줄)
│   ├── disclosure_analyzer.py             ← 공시 분석기 (30개 유형, ~515줄)
│   ├── requirements.txt
│   ├── .streamlit/config.toml
│   ├── .gitignore
│   ├── figures/V5_dashboard_mockup.png
│   ├── poc_results.json
│   └── poc_results_v2.json
│
├── poc_finguard.py                        ← PoC v1 (9 피처, 합성 데이터)
├── poc_finguard_v2.py                     ← PoC v2 (15 피처, 섹터+국면+뉴스+공시)
├── poc_results.json / poc_results_v2.json
│
├── figures/                               ← PPT용 시각화 (6장 PNG)
│   ├── V1_architecture.png
│   ├── V2_pipeline.png
│   ├── V3_score_flow.png
│   ├── V4_2x2_matrix.png
│   ├── V5_dashboard_mockup.png
│   ├── V6_backtest_curve.png
│   └── V_extra_poc_metrics.png
│
└── 소스파일/                              ← 원본 자료 (참조용, 수정 금지)
    ├── fin_guard_auto_기획서_최종정리본.md
    └── 프로젝트 작성할때 주의해야할 것.txt
```

---

## 3. 현재 진행 상황 (2026-05-26 기준)

### 완료된 작업

- [x] 기획서 v3 (보강본): Executive Summary, 선행 연구 비교표, 데이터 정량 수치, PoC 결과, 시각화 5종 명시, 사회적 가치 정량 추정, References 15편, Plan B, 윤리·편향 단락
- [x] PoC v1: LightGBM 상승/급락 모델, 9 피처, 120 종목 × 1000 일 합성 데이터, ROC-AUC 상승 0.55 / 급락 0.62
- [x] PoC v2: 정교화 (15 피처, 12 섹터, 3-state HMM 시장 국면, 뉴스·공시 피처), 150 종목 × 1200 일, ROC-AUC 상승 0.54 / 급락 0.58
- [x] 시각화 6장 (V1~V5 + V6 백테스트 + V_extra PoC 막대그래프) — 영문 라벨 (한글 폰트 부재로 로컬 재생성 권장)
- [x] Streamlit 프로토타입 5탭: 종목분석·2×2매트릭스·공시뉴스·백테스트·**공시분석기(30개 유형)**
- [x] 공시 분석기: 30개 이벤트 유형 룰베이스 분류 + 30개 자연어 템플릿 + 12개 체크포인트 + Mock 데이터 40건 100% 분류 커버리지
- [x] 배포 환경: requirements.txt, .gitignore, .streamlit/config.toml
- [x] 배포 가이드: DEPLOY.md (Streamlit Cloud + HF Spaces 두 옵션)

### 남은 작업 (우선순위 순)

1. **실 데이터 연동** (3~5일) — pykrx/FinanceDataReader로 KOSPI200+KOSDAQ150 실제 데이터를 받아 PoC 재실행. `prototype/streamlit_app.py`의 `gen_panel()`와 `disclosure_analyzer.py`의 `load_mock_disclosures()`를 실제 API 호출로 교체.
2. **Streamlit Cloud 배포** (1~2일) — GitHub push → Streamlit Cloud에서 `prototype/streamlit_app.py` 지정 → 공개 URL 발급 → QR 코드 생성.
3. **PPT 제작** (1~2주) — 20~25장 슬라이드, 기획서 v3 § 매핑은 `README.md` 참조. python-pptx로 자동 생성 또는 PowerPoint로 수동.
4. **발표 리허설** — 시간 측정(15분 발표 + 5분 Q&A 가정), 핵심 메시지 압축, 라이브 시연 동선 연습.

### 알려진 한계 / 정직하게 보고해야 할 점

- **합성 데이터 PoC**: 합성 데이터의 급락 신호가 약해(ROC-AUC ~0.58) 백테스트에서 리스크 필터(B 전략)가 단순 상승만(A 전략)을 능가하지 못함. **이는 honest finding으로 발표에서 솔직히 보고하고**, 실데이터 + KF-DeBERTa 적용 시 개선될 것이라는 가설로 제시.
- **시각화 영문 라벨**: 샌드박스 환경에 한글 폰트(NanumGothic 등) 부재로 영문 출력. 로컬 Windows에서 `make_figures.py`의 폰트 경로에 `C:/Windows/Fonts/malgun.ttf` 추가 후 재실행 필요.
- **NLP 모듈 placeholder**: 기획서에는 KF-DeBERTa·KoBigBird·HyperCLOVA X 등을 명시하지만 실제 구현은 룰베이스 + TF-IDF만. 발표 시 "확장 계획"으로 명확히 구분.

---

## 4. 코딩·문서 컨벤션

### 언어와 톤

- 사용자 응답·문서·코드 주석: **한국어**
- 코드 식별자·함수명: 영문 (`load_mock_disclosures`, `classify`, `make_features`)
- 통화 단위: 원, 만원, 억원 (KRW)
- 날짜 형식: YYYY-MM-DD

### 정확성 우선 원칙

- 통계 수치를 인용할 때는 "추정", "보고서별 상이" 등 불확실성을 명시
- 출처는 가능한 한 보고서명·발행일·페이지까지 기재 (자본시장연구원, 금감원, 한국예탁결제원 등)
- 모델 성능 수치는 PoC 실측값을 그대로 게재 (낮아도 honest finding)
- 실제 종목명을 사용할 때는 매수·매도 추천이 아님을 항상 명시

### 규제 준수

- "매수 추천", "매도 추천", "수익 보장" 같은 표현 절대 금지
- 사용할 표현: "위험 경보", "관심 후보", "분석 결과", "의사결정 보조", "모의 검증"
- 자본시장법 제101조 (유사투자자문업) 회피: 불특정 다수 대상 정보 제공으로 구조화

### 파일 수정 시 주의

- **`소스파일/` 폴더의 두 파일은 절대 수정하지 말 것** (원본 자료, 참조용)
- 기존 산출물을 수정할 때는 백업 또는 버전 표기 (`_v2`, `_v3` 등)
- 한국어 변수명·함수명을 새로 만들지 말 것 (영문 유지)

### 코드 스타일

- Python: 들여쓰기 4칸, 한 줄 100자 이내 권장
- 함수 docstring 필수 (한국어로 짧게)
- 데이터프레임 컬럼명: snake_case 영문 (`stock_id`, `fwd_ret_5d`, `score_up`)
- 시간순 분할 엄격 적용 (랜덤 분할 금지, walk-forward 사용)

### 데이터 누수 방지 (금융 ML 필수)

- 공시·뉴스: 발표 시각 이후 데이터만 사용
- 재무제표: 공시 발표일 이후부터 반영
- 기술지표: 과거 가격 데이터만 사용 (look-ahead 금지)
- 정규화: train fit, val/test transform only
- 시간순 Train/Val/Test 분할 + Walk-forward validation

---

## 5. 자주 사용하는 명령

```bash
# 프로토타입 로컬 실행
cd prototype
pip install -r requirements.txt
streamlit run streamlit_app.py

# PoC 재실행 (v2 권장)
python poc_finguard_v2.py

# 공시 분석기 단독 테스트
python prototype/disclosure_analyzer.py
```

---

## 6. 평가 기준 (교수님이 강조한 항목)

`소스파일/프로젝트 작성할때 주의해야할 것.txt` 핵심:

1. **새로움 / Novelty**
2. **세상에 기여할 수 있는 가치** (사회적 기여)
3. **AI를 활용한 문제 해결**
4. **충분한 데이터·논리 기반 근거** ← 단순 주장 금지
5. **실현 가능성** (현재 기술 수준 조사 + 구현 가능성)
6. **시각화 자료**

피해야 할 것: 막연한 아이디어, 데이터 없는 주장, AI 생성물 그대로 사용, 출처 불명확.

---

## 7. 외부 참조

- 기획서 v3 전문: `fin_guard_auto_기획서_보강본_v3.md` 참조 (Section 0 Executive Summary, Section 13 PoC, Section 22.2 사회적 가치 정량화, References 15편)
- 배포 절차: `DEPLOY.md`
- 다음 작업별 프롬프트: `CLAUDE_PROMPTS.md`

---

**문서 갱신일**: 2026-05-26 (프로토타입 완성 + 단독 실행 검증 통과 시점)
