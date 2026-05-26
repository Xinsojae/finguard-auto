"""한글 폰트 설정 모듈.

matplotlib에 한글 폰트를 등록하고, FontProperties 객체를 제공한다.
import 시점에 자동 등록되며, 다른 모듈은 다음과 같이 사용:

    from font_setup import KFONT_FP, KFONT_PATH

    ax.set_title("제목", fontproperties=KFONT_FP)

폰트 우선순위:
    Linux (Streamlit Cloud): NanumGothic > NanumBarunGothic > Noto CJK
    Windows: 맑은 고딕
    macOS: Apple SD Gothic Neo
"""
import os
import matplotlib.pyplot as plt
from matplotlib import font_manager

_KFONT_CANDIDATES = [
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
