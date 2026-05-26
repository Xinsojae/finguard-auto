"""각 탭 모듈은 render(ctx: AppCtx) 함수를 export한다.

AppCtx는 streamlit_app.py가 만든 dataclass — 패널, 모델, 스냅샷,
사이드바 설정값(임계값·k_top·hold·risk_pct), 폰트 객체를 담는다.
"""
from dataclasses import dataclass
from typing import List, Optional, Any
import pandas as pd


@dataclass
class AppCtx:
    # 데이터
    panel: pd.DataFrame
    snap: pd.DataFrame
    latest_date: Any
    # 모델
    m_up: Any
    m_cr: Any
    metrics: dict
    # 사이드바 선택
    picks: List[str]
    use_real: bool
    # 사이드바 슬라이더
    up_th: int
    risk_th: int
    k_top: int
    hold_days: int
    risk_pct: float
    # 폰트
    kfont_fp: Optional[Any]
