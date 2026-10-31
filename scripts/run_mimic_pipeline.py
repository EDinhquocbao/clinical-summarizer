"""Chạy tuần tự toàn bộ thí nghiệm MIMIC: huấn luyện 3 mô hình → sinh tóm tắt → baseline → chấm điểm.

Mỗi bước ghi log vào outputs/logs/mimic_<bước>.log và một dòng tiến độ vào outputs/logs/mimic_pipeline.log
(chỉ thời gian + mã thoát, không có văn bản ghi chú). Bước huấn luyện lỗi thì dừng cả chuỗi.
Chạy:  python scripts/run_mimic_pipeline.py --steps 11000
       python scripts/run_mimic_pipeline.py --skip train_pg train_s2s   # bỏ qua bước đã xong
       python scripts/run_mimic_pipeline.py --resume --steps 25000 --cov_steps 3000 --min_len 120 --prefix mimic_r2_
"""
import argparse
import os
import subprocess
import sys
import time

from clinsum.paths import OUTPUTS_DIR, ROOT

DATA = "data/processed/mimic_bhc"


def plan(args):
    py = [sys.executable, "-m"]
    train = py + ["clinsum.summarization.train", "--config"]
    p = args.prefix
    dec = ["--block_ngram", str(args.block_ngram)] + (["--min_len", str(args.min_len)] if args.min_len else [])

    def pred(model):
        return py + ["clinsum.summarization.predict", "--limit", str(args.limit), *dec,
                     "--checkpoint", f"models/{model}/best.pt", "--out", f"outputs/predictions/{p}{model[6:]}_test.jsonl"]

    def resume(model):   # --resume: học tiếp từ last.pt (giữ optimizer, lr, best val) thay vì học lại từ đầu
        return [f"resume=models/{model}/last.pt"] if args.resume else []

    # max_minutes chỉ là chốt an toàn; Seq2seq và PG dừng ở cùng max_steps
    cov_stop = f"max_steps={args.cov_steps}" if args.cov_steps else f"max_minutes={args.cov_minutes}"
    return [
        ("train_pg", True, train + ["configs/mimic_pointer_gen.yaml", "--set", f"max_steps={args.steps}",
                                    "max_minutes=900", *resume("mimic_pointer_gen")]),
        ("train_s2s", True, train + ["configs/mimic_seq2seq_attn.yaml", "--set", f"max_steps={args.steps}",
                                     "max_minutes=900", *resume("mimic_seq2seq_attn")]),
        ("train_cov", True, train + ["configs/mimic_pointer_gen_cov.yaml", "--set", cov_stop, "max_minutes=300"]),
        ("pred_cov", False, pred("mimic_pointer_gen_cov")),
        ("pred_pg", False, pred("mimic_pointer_gen")),
        ("pred_s2s", False, pred("mimic_seq2seq_attn")),
        ("baselines", False, py + ["clinsum.summarization.baselines", "--data_dir", DATA, "--limit", str(args.limit),
                                   "--prefix", p]),
        ("score", False, [sys.executable, "scripts/score_predictions.py", "--data_dir", DATA, "--prefix", p]),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=11000, help="tổng số bước của PG và Seq2seq (tính cả bước đã học)")
    ap.add_argument("--resume", action="store_true", help="PG và Seq2seq học tiếp từ last.pt")
    ap.add_argument("--cov_minutes", type=int, default=40)
    ap.add_argument("--cov_steps", type=int, default=0, help=">0: coverage dừng theo số bước (See et al.: ~3000)")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--block_ngram", type=int, default=3, help="chặn lặp n-gram khi giải mã (0 = tắt)")
    ap.add_argument("--min_len", type=int, default=0, help="độ dài tối thiểu khi giải mã (0 = theo config)")
    ap.add_argument("--prefix", default="mimic_", help="tiền tố file dự đoán/kết quả, vd mimic_r2_")
    ap.add_argument("--skip", nargs="*", default=[])
    args = ap.parse_args()

    if sys.platform == "win32":
        # Báo Windows đừng ngủ khi rảnh trong lúc script chạy (ES_CONTINUOUS | ES_SYSTEM_REQUIRED);
        # không đổi cài đặt nguồn, tự hết khi tiến trình kết thúc. Gập nắp máy thì vẫn ngủ.
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

    log_dir = OUTPUTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    progress = open(log_dir / "mimic_pipeline.log", "a", encoding="utf-8")

    def note(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        progress.write(line + "\n")
        progress.flush()

    env = {"PYTHONIOENCODING": "utf-8"}
    for name, critical, cmd in plan(args):
        if name in args.skip:
            note(f"bỏ qua {name}")
            continue
        note(f"bắt đầu {name}")
        t = time.time()
        with open(log_dir / f"mimic_{name}.log", "w", encoding="utf-8") as f:
            code = subprocess.call(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                                   env={**os.environ, **env})
        note(f"xong {name}: mã thoát {code}, {(time.time() - t) / 60:.1f} phút")
        if code != 0 and critical:
            note("dừng chuỗi vì bước huấn luyện lỗi")
            sys.exit(code)
    note("hoàn tất")


if __name__ == "__main__":
    main()
