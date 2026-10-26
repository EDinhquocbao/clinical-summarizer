"""Đọc file cấu hình YAML. Khóa `base` (đường dẫn tương đối) cho phép kế thừa một file khác."""
from pathlib import Path
import yaml


def load_config(path) -> dict:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        merged = load_config(path.parent / base)
        merged.update(cfg)
        return merged
    return cfg
