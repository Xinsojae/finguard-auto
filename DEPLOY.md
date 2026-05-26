# FinGuard Auto 배포 가이드

발표일(2026-06-26) 평가자가 노트북·휴대폰으로 직접 접속할 수 있는 공개 URL을 만드는 가이드입니다. **Streamlit Community Cloud**(권장)와 **Hugging Face Spaces** 두 가지 방법을 소개합니다.

---

## 사전 준비 (공통)

### 1. GitHub 계정 + 저장소 생성

1. <https://github.com> 에서 계정 생성/로그인
2. **New repository** → 이름: `finguard-auto` (또는 원하는 이름)
3. **Public** 선택 (무료 티어 사용을 위해)
4. README/license는 추가하지 않음 ("Add a README file" 체크 해제)

### 2. 로컬에서 git 푸시

PowerShell 또는 Git Bash에서:

```bash
cd "C:\Users\pma03\OneDrive\문서\Claude\Projects\ai_Project"

# git 초기화 (처음 1회)
git init
git add .
git commit -m "Initial commit: FinGuard Auto prototype"

# 원격 연결 (USERNAME과 REPO를 본인 것으로 교체)
git branch -M main
git remote add origin https://github.com/USERNAME/finguard-auto.git
git push -u origin main
```

> Git이 처음이면 **GitHub Desktop**(<https://desktop.github.com>)을 GUI로 사용해도 됩니다. 폴더 드래그 → Publish repository → Public.

---

## 옵션 A: Streamlit Community Cloud (권장, 가장 간단)

### 장점
- 완전 무료 (1 앱 무제한 실행)
- GitHub 자동 연동 → push만 하면 자동 재배포
- 한국어 UI 깔끔하게 표시
- 1~2분 안에 URL 발급

### 절차

1. <https://share.streamlit.io> 접속, GitHub 계정으로 로그인
2. **Create app** → **From existing repo** 선택
3. 다음 입력:
   - Repository: `USERNAME/finguard-auto`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
   - App URL: `finguard-auto` (또는 원하는 이름) → 최종 URL: `https://finguard-auto.streamlit.app`
4. **Advanced settings** → Python version `3.11` 선택 (선택사항)
5. **Deploy** 클릭

### 빌드 진행

처음 빌드는 3~6분 걸립니다. 의존성 설치(`requirements.txt`) → 앱 시작 순.

빌드 로그에서 에러가 보이면:
- `ModuleNotFoundError` → `requirements.txt`에 해당 패키지 추가하고 다시 git push
- `MemoryError` → 데이터 크기 축소 (`gen_panel(n_stocks=80, n_days=400)`로 변경)

### 발표 당일 사용

평가자에게 URL만 전달하면 됩니다. PPT에 QR 코드로 넣어두면 클릭 한 번에 접속.

---

## 옵션 B: Hugging Face Spaces (대안)

### 장점
- AI/ML 커뮤니티 표준 호스팅
- GPU 무료 티어 가능 (KF-DeBERTa 등 향후 확장 시 유용)
- 도메인이 `huggingface.co/spaces/...` 형식

### 절차

1. <https://huggingface.co> 계정 생성/로그인
2. <https://huggingface.co/new-space>
3. 다음 입력:
   - Space name: `finguard-auto`
   - SDK: **Streamlit** 선택
   - Hardware: **CPU basic (free)**
   - Visibility: Public
4. **Create Space** 클릭 → 빈 Space가 생성됨
5. Space 페이지의 **Files** 탭으로 이동
6. 다음 파일들을 업로드 (드래그앤드롭 또는 git clone 후 push):
   - `streamlit_app.py`
   - `disclosure_analyzer.py`
   - `requirements.txt`
   - `.streamlit/config.toml` (선택)
7. 자동으로 빌드되며 약 5분 후 URL 활성화

### git 방식 업로드

```bash
git clone https://huggingface.co/spaces/USERNAME/finguard-auto
cd finguard-auto
# 필요한 파일 복사
cp ../streamlit_app.py .
cp ../disclosure_analyzer.py .
cp ../requirements.txt .
git add .
git commit -m "Initial deploy"
git push
```

### Spaces용 README 추가 (선택)

루트에 `README.md`를 만들고 다음을 추가하면 메타데이터로 인식됩니다:

```markdown
---
title: FinGuard Auto
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.30.0
app_file: streamlit_app.py
pinned: false
---
```

---

## 트러블슈팅

### 한글 폰트가 깨져 보일 때 (Streamlit Cloud)

`packages.txt` 파일을 루트에 추가하고 아래 내용 입력:

```
fonts-nanum
```

git push 후 재배포. matplotlib 그래프 안의 한글이 정상 출력됩니다.

### "Out of memory" 에러

- 합성 데이터 크기 줄이기: `streamlit_app.py`의 `gen_panel(n_stocks=120, n_days=600)` → `gen_panel(n_stocks=60, n_days=300)`
- LightGBM `n_estimators` 줄이기: 300 → 150

### 빌드는 성공했는데 앱이 안 뜰 때

- 로그 탭에서 Python 에러 확인
- 흔한 원인: `disclosure_analyzer.py`가 같은 폴더에 없음 → 함께 푸시했는지 확인
- import 에러: `requirements.txt`에 패키지 빠짐

### 앱이 너무 느릴 때 (cold start)

Streamlit Cloud는 무료 티어에서 일정 시간 미사용 시 sleep 상태로 들어갑니다.
- 발표 30분 전에 미리 한 번 접속해 깨워두면 됩니다.
- 또는 cron-job.org 등으로 5분마다 ping 자동화

---

## 배포 후 체크리스트

- [ ] URL 접속 시 사이드바와 5개 탭 모두 표시되는가
- [ ] 종목 분석 탭에서 SHAP 그래프가 그려지는가
- [ ] 2×2 매트릭스 산점도가 나오는가
- [ ] 공시 분석 탭에서 Mock 공시 선택 → 분석하기 클릭 → 결과 표시되는가
- [ ] 백테스트 탭에서 누적 수익률 곡선이 보이는가
- [ ] 모바일에서도 접속 가능한가 (PPT QR 코드 대응)
- [ ] 면책 문구(매수 추천이 아님)가 상단/하단에 보이는가

---

## PPT에 넣을 내용 권장

발표 슬라이드에 다음을 포함하세요:

1. **공개 URL** + **QR 코드** (QR 생성: <https://www.qr-code-generator.com>)
2. "지금 휴대폰으로 접속해보세요" 한 줄
3. 라이브 시연 시간 1~2분 확보 (5개 탭 중 핵심 2개만 클릭)
4. 데모 화면 캡처를 슬라이드 배경에 깔아두기 (네트워크 장애 백업)

---

## 보너스: 도메인 커스터마이즈

Streamlit Cloud 무료 티어에서는 `*.streamlit.app` 서브도메인만 가능합니다. 자신만의 도메인을 쓰고 싶다면:

1. Cloudflare Pages + GitHub Actions로 정적 사이트 빌드 → 무료 도메인 연결
2. 또는 본격 운영 시 Streamlit Community Cloud 유료 플랜 / 자체 호스팅 (AWS EC2)

학부 과제 단계에서는 `finguard-auto.streamlit.app`로 충분히 임팩트가 있습니다.
