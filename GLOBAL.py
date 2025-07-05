from pathlib import Path

WS_ROOT = Path(__file__).resolve().parent
DATA_ROOT = WS_ROOT / "data"


class Config:
    img_suffixes = {".png", ".jpg", ".jpeg"}
