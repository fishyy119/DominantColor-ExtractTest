import sys
from pathlib import Path

WS_ROOT = Path(__file__).resolve().parents[1]
if str(WS_ROOT) not in sys.path:
    sys.path.insert(0, str(WS_ROOT))

from GLOBAL import *
