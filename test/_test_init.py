import sys
from pathlib import Path

WS_ROOT = Path(__file__).resolve().parents[1]
if str(WS_ROOT) not in sys.path:
    sys.path.insert(0, str(WS_ROOT))


import matplotlib.pyplot as plt

from GLOBAL import *

# * 在最一开始设置这个，保证后面的字体全部生效
plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]
plt.rcParams.update(
    {
        "axes.labelsize": 10.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "mathtext.fontset": "stix",  # 公式字体（有效）
    }
)
