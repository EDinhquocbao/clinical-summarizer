"""Kiểm tra môi trường: phiên bản Python, thư viện chính và GPU.
Chạy:  python scripts/check_env.py
"""
import importlib
import platform
import sys

print("Python:", sys.version.split()[0], "|", platform.platform())

for name in ("torch", "transformers", "datasets", "evaluate", "seqeval", "gradio"):
    try:
        mod = importlib.import_module(name)
        print(f"[OK]   {name} {getattr(mod, '__version__', '')}")
    except ImportError:
        print(f"[THIẾU] {name}  ->  pip install {name}")

try:
    import torch
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, {props.total_memory / 1024**3:.1f} GB VRAM")
    else:
        print("GPU: không có CUDA (sẽ chạy CPU, rất chậm cho huấn luyện)")
except ImportError:
    pass
