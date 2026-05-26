"""뉴스 감성 분석 베이스라인 + 한국 금융 BERT.

두 가지 백엔드를 동일 인터페이스로 제공:
    1. TfidfSentiment - 합성 라벨 코퍼스로 학습한 TF-IDF + LogReg
       (기획서 MVP #4 베이스라인. 가벼움, 의존성 sklearn만)
    2. KrFinBertSentiment - snunlp/KR-FinBert-SC HuggingFace 모델
       (실제 한국 금융 BERT. transformers+torch 필요. 메모리 큼)

공통 인터페이스:
    model.analyze(["뉴스 텍스트1", "뉴스 텍스트2"])
        → np.ndarray, 각 원소 [-1, +1] (음수=악재, 양수=호재)
    model.label → 표시용 모델명
"""
from __future__ import annotations

import random
from typing import List, Tuple, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# ============================================================
# 합성 라벨 코퍼스 — 한국 금융 도메인 키워드 기반
# ============================================================
POS_KEYWORDS = [
    "자사주 매입", "자사주 소각", "흑자전환", "최대실적", "사상최대",
    "수주 계약 체결", "신사업 진출", "특허 등록", "배당 증가",
    "M&A 인수", "분기 흑자", "매출 성장", "영업이익 증가",
    "공장 증설", "기술 이전", "정부 지원", "수출 호조",
    "신약 허가", "임상 성공", "대규모 수주", "독점 공급",
]
NEG_KEYWORDS = [
    "적자전환", "당기순손실", "영업손실", "감자", "유상증자",
    "전환사채 발행", "관리종목 지정", "상장폐지 사유", "불성실공시",
    "감사의견 거절", "한정의견", "부도", "기한이익상실",
    "공정위 조사", "검찰 압수수색", "과징금", "행정처분",
    "임원 매도", "내부자 매도", "신용등급 하향", "공장 화재",
    "거래처 이탈", "수주 감소", "특허 침해 소송", "리콜",
]
POS_FILLERS = [
    "당사는", "회사는", "최근", "이번", "이로 인해",
    "결과적으로", "전망", "긍정적", "확대", "강화", "성장",
]
NEG_FILLERS = [
    "당사는", "회사는", "최근", "이번", "이로 인해",
    "결과적으로", "우려", "악재", "감소", "위험", "충격",
]


POS_TEMPLATES = [
    "{f} {kw} 등 호재로 작용할 것으로 보입니다.",
    "{f} {kw}을(를) 결정하였습니다.",
    "{kw} 발표로 주가가 강세를 보였습니다.",
    "{f} {kw} 등 긍정적 효과가 예상됩니다.",
    "{kw}에 따라 실적 개선이 기대됩니다.",
    "{kw} 공시 이후 시장 반응 긍정적.",
]
NEG_TEMPLATES = [
    "{f} {kw} 등 악재가 발생했습니다.",
    "{f} {kw}이(가) 우려됩니다.",
    "{kw} 발생으로 주가가 약세입니다.",
    "{f} {kw}에 따른 부정적 영향이 예상됩니다.",
    "{kw} 공시 이후 매도세 확대.",
    "{kw}로 인해 단기 리스크가 커졌습니다.",
]


def make_training_corpus(n_each: int = 100,
                         seed: int = 42) -> Tuple[List[str], List[int]]:
    """합성 라벨 데이터 생성. (text, label) — label 1=긍정, 0=부정.

    각 라벨 n_each건씩, 키워드 + 다중 템플릿 조합으로 다양성 확보.
    템플릿 어미보다 키워드 자체가 분류에 더 기여하도록 변형.
    """
    rng = random.Random(seed)
    rows: List[Tuple[str, int]] = []
    for _ in range(n_each):
        kw = rng.choice(POS_KEYWORDS)
        f = rng.choice(POS_FILLERS)
        tpl = rng.choice(POS_TEMPLATES)
        rows.append((tpl.format(f=f, kw=kw), 1))
        # 키워드 단독도 일부 추가 (강한 신호)
        if rng.random() < 0.2:
            rows.append((kw, 1))
        kw = rng.choice(NEG_KEYWORDS)
        f = rng.choice(NEG_FILLERS)
        tpl = rng.choice(NEG_TEMPLATES)
        rows.append((tpl.format(f=f, kw=kw), 0))
        if rng.random() < 0.2:
            rows.append((kw, 0))
    rng.shuffle(rows)
    texts = [r[0] for r in rows]
    labels = [r[1] for r in rows]
    return texts, labels


# ============================================================
# 백엔드 1: TF-IDF + Logistic Regression
# ============================================================
class TfidfSentiment:
    label = "TF-IDF + LogReg (베이스라인)"

    def __init__(self, n_each: int = 100):
        texts, labels = make_training_corpus(n_each=n_each)
        self._train_size = len(texts)
        self.pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                min_df=1, ngram_range=(1, 3), sublinear_tf=True,
                analyzer="char_wb",  # 한국어 형태소 미사용 — char n-gram이 강건
            )),
            ("clf", LogisticRegression(max_iter=1000, random_state=42, C=2.0)),
        ])
        self.pipe.fit(texts, labels)
        # 학습 정확도 (간단 검증)
        self.train_acc = float((self.pipe.predict(texts) == np.array(labels)).mean())

    def analyze(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        probs = self.pipe.predict_proba(texts)[:, 1]  # P(긍정)
        return probs * 2 - 1  # [-1, +1]

    def info(self) -> dict:
        return {
            "backend": "TF-IDF + LogReg",
            "train_size": self._train_size,
            "train_acc": round(self.train_acc, 3),
            "vocab_size": len(self.pipe.named_steps["tfidf"].vocabulary_),
        }


# ============================================================
# 백엔드 2: KR-FinBERT (HuggingFace)
# ============================================================
class KrFinBertSentiment:
    label = "KR-FinBERT (snunlp/KR-FinBert-SC)"

    def __init__(self):
        from transformers import pipeline as hf_pipeline  # lazy import
        self.pipe = hf_pipeline(
            "text-classification",
            model="snunlp/KR-FinBert-SC",
            top_k=None,
        )

    def analyze(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        out = self.pipe(texts)
        scores = []
        for item in out:
            # top_k=None → 각 입력당 라벨 리스트 반환
            if isinstance(item, list):
                pos_score = 0.0
                neg_score = 0.0
                for r in item:
                    lbl = r["label"].lower()
                    if "pos" in lbl or lbl == "1" or "긍정" in lbl:
                        pos_score = r["score"]
                    elif "neg" in lbl or lbl == "0" or "부정" in lbl:
                        neg_score = r["score"]
                scores.append(pos_score - neg_score)  # [-1, +1]
            else:
                # 단일 결과
                lbl = item["label"].lower()
                s = item["score"]
                if "pos" in lbl or "긍정" in lbl:
                    scores.append(s)
                elif "neg" in lbl or "부정" in lbl:
                    scores.append(-s)
                else:
                    scores.append(0.0)
        return np.array(scores)

    def info(self) -> dict:
        return {
            "backend": "KR-FinBERT (snunlp)",
            "model_size_mb": "~430 MB",
            "note": "transformers + torch 의존",
        }


# ============================================================
# Factory
# ============================================================
def get_sentiment_model(backend: str) -> Tuple[object, Optional[str]]:
    """backend ∈ {'tfidf', 'krfinbert'}. Returns (model, error_msg).

    KR-FinBERT 로드 실패 시 TF-IDF로 자동 fallback (error_msg에 사유 기록).
    """
    if backend == "krfinbert":
        try:
            return KrFinBertSentiment(), None
        except Exception as e:
            return TfidfSentiment(), f"KR-FinBERT 로드 실패: {e}. TF-IDF로 fallback."
    return TfidfSentiment(), None


# ============================================================
# CLI 데모
# ============================================================
if __name__ == "__main__":
    m = TfidfSentiment()
    print("=== TF-IDF 모델 정보 ===")
    print(m.info())
    samples = [
        "당사는 자사주 300만주를 매입하기로 결정하였습니다.",
        "당사는 영업손실 확대로 적자전환하였습니다.",
        "분기보고서 제출.",
        "공정거래위원회 조사 및 압수수색이 진행 중입니다.",
        "신약 임상 3상 성공 발표.",
    ]
    scores = m.analyze(samples)
    print("\n=== 샘플 분석 ===")
    for t, s in zip(samples, scores):
        print(f"  [{s:+.3f}] {t}")
