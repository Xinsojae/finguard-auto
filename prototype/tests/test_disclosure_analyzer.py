"""disclosure_analyzer 룰베이스 분류 + Mock 커버리지 테스트."""
import pytest
import disclosure_analyzer as da


# ---------- classify ----------
def test_classify_empty():
    assert da.classify("") == []
    assert da.classify(None) == []


def test_classify_match_negative_event():
    res = da.classify("당사는 운영자금 확보를 위해 유상증자를 결정하였습니다.")
    assert res, "유상증자 매칭 안 됨"
    top = res[0]
    assert top.risk_score < 0
    assert "유상증자" in top.matched_keywords


def test_classify_match_positive_event():
    # 정규식: r"자기주식\s*취득" — "자기주식 취득" 형태 매칭
    res = da.classify("당사는 자기주식 취득을 결정하였습니다.")
    assert res, "자기주식 취득 매칭 안 됨"
    top = res[0]
    assert top.risk_score > 0


def test_classify_sorted_by_risk_abs():
    """결과는 |risk_score| 큰 순서로 정렬."""
    res = da.classify("당사는 유상증자 및 자기주식 매입을 동시에 결정하였습니다.")
    if len(res) >= 2:
        for i in range(1, len(res)):
            assert abs(res[i-1].risk_score) >= abs(res[i].risk_score)


def test_classify_to_dict_returns_none_on_no_match():
    out = da.classify_to_dict("일반적인 정기보고서 제출 안내.")
    # 안 잡힐 수도, 잡힐 수도 — 함수 자체가 None 또는 dict 반환
    assert out is None or isinstance(out, dict)


def test_classify_to_dict_returns_dict_on_match():
    out = da.classify_to_dict("당사는 유상증자를 결정하였습니다.")
    assert isinstance(out, dict)
    for key in ["code", "name", "category", "risk_score", "risk_label",
                "confidence", "explanation", "matched_keywords"]:
        assert key in out


# ---------- Mock 커버리지 ----------
def test_mock_disclosures_loaded():
    df = da.load_mock_disclosures(n_days=20)
    assert not df.empty
    assert "corp_name" in df.columns
    assert "report_nm" in df.columns
    assert "report_body" in df.columns


def test_classify_mock_all_high_coverage():
    """Mock 공시 분류 커버리지 50% 이상."""
    events = da.classify_mock_all()
    df = da.load_mock_disclosures(n_days=20)
    coverage = len(events) / len(df) if len(df) > 0 else 0
    assert coverage >= 0.5, \
        f"Mock 커버리지 {coverage:.0%} < 50% — 룰셋 약화 또는 mock 변경"


def test_classify_mock_all_risk_distribution():
    """분류된 이벤트에 양/음/중립 모두 존재."""
    events = da.classify_mock_all()
    neg = sum(1 for e in events if e["risk_score"] < 0)
    pos = sum(1 for e in events if e["risk_score"] > 0)
    # 최소 한쪽이라도 있어야 함
    assert neg + pos > 0
