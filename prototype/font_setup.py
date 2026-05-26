"""한글 폰트 설정 모듈.

matplotlib에 한글 폰트를 등록하고, FontProperties 객체를 제공한다.
import 시점에 자동 등록되며, 다른 모듈은 다음과 같이 사용:

    from font_setup import KFONT_FP, KFONT_PATH

    ax.set_title("제목", fontproperties=KFONT_FP)

폰트 우선순위:
    1. 번들 NanumGothic (_data/fonts/) — 항상 동작 보장
    2. Linux 시스템 (apt fonts-nanum / Noto CJK)
    3. Windows 맑은 고딕
    4. macOS Apple SD Gothic Neo
"""
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager

_BUNDLED = Path(__file__).resolve().parent / "_data" / "fonts" / "NanumGothic-Regular.ttf"

_KFONT_CANDIDATES = [
    str(_BUNDLED),  # 1순위: 저장소 번들 (Streamlit Cloud apt 실패 대비)
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/malgun.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]

KFONT_PATH: str | None = None
KFONT_FP: font_manager.FontProperties | None = None

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
