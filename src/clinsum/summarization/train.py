"""Huấn luyện Seq2seq-attention / Pointer-Generator / Pointer-Generator + coverage.

Chạy:
  python -m clinsum.summarization.train --config configs/pointer_gen.yaml
  python -m clinsum.summarization.train --config configs/pointer_gen.yaml --set max_minutes=5 train_limit=2000

Như See et al. (2017), coverage được bật ở pha huấn luyện thứ hai, khởi tạo từ checkpoint
Pointer-Generator (khóa init_from trong configs/pointer_gen_cov.yaml).
"""
import argparse
import json
import math
import random
import time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from clinsum.config import load_config
from clinsum.paths import ROOT
from clinsum.text import tokenize
from clinsum.summarization.data import SummDataset, collate, read_jsonl
from clinsum.summarization.model import PointerGenerator
from clinsum.summarization.vocab import Vocab


def parse_overrides(pairs):
    out = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def get_vocab(cfg, train_rows) -> Vocab:
    path = ROOT / cfg["vocab_path"]
    if path.exists():
        return Vocab.load(path)
    print("Tạo từ điển từ tập train...")
    vocab = Vocab.build(
        (tokenize(r["source"] + " " + r["target"]) for r in train_rows), max_size=cfg["vocab_size"]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    vocab.save(path)
    return vocab


def build_model(cfg, vocab: Vocab) -> PointerGenerator:
    return PointerGenerator(
        vocab_size=len(vocab),
        emb_dim=cfg["emb_dim"],
        hidden_dim=cfg["hidden_dim"],
        use_copy=cfg["use_copy"],
        use_coverage=cfg["use_coverage"],
        pad_id=vocab.pad,
        unk_id=vocab.unk,
        dropout=cfg.get("dropout", 0.0),
    )


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) and k != "src_len" else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate_loss(model, loader, device, max_batches=None, amp=False) -> float:
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            out = model(to_device(batch, device))
        total += out["nll"].item()
        n += 1
    model.train()
    return total / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", help="ghi đè cấu hình, vd max_minutes=10")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg.update(parse_overrides(args.set))

    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = ROOT / cfg["data_dir"]
    train_rows = read_jsonl(data_dir / "train.jsonl")
    val_rows = read_jsonl(data_dir / "val.jsonl")
    if cfg.get("train_limit"):
        train_rows = train_rows[: cfg["train_limit"]]
    val_rows = val_rows[: cfg.get("val_limit", 1000)]
    vocab = get_vocab(cfg, train_rows)

    mk = partial(SummDataset, vocab=vocab, max_source_len=cfg["max_source_len"],
                 max_target_len=cfg["max_target_len"])
    col = partial(collate, pad_id=vocab.pad)
    train_loader = DataLoader(mk(train_rows), batch_size=cfg["batch_size"], shuffle=True, collate_fn=col)
    val_loader = DataLoader(mk(val_rows), batch_size=cfg["batch_size"], collate_fn=col)

    model = build_model(cfg, vocab).to(device)
    if cfg.get("init_from"):
        state = torch.load(ROOT / cfg["init_from"], map_location=device)["model"]
        missing, _ = model.load_state_dict(state, strict=False)
        print("Khởi tạo từ", cfg["init_from"], "| tham số mới:", missing)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Mô hình {cfg['name']}: {n_params / 1e6:.1f}M tham số | {len(train_rows)} mẫu train | {device}")

    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    # amp: fp16 cho attention/LSTM (nhanh hơn, ít VRAM hơn); loss vẫn tính fp32 trong autocast
    amp = bool(cfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    # lr_patience > 0: giảm lr một nửa khi val NLL không giảm sau lr_patience lần đánh giá
    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=cfg["lr_patience"])
             if cfg.get("lr_patience") else None)
    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    best_val, step, start_epoch = math.inf, 0, 0
    if cfg.get("resume"):   # chạy tiếp từ last.pt (vd máy tắt giữa chừng); max_minutes tính lại từ đầu
        ck = torch.load(ROOT / cfg["resume"], map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        if sched and ck.get("sched"):
            sched.load_state_dict(ck["sched"])
        best_val, step, start_epoch = ck["best_val"], ck["step"], ck["epoch"] + 1
        print(f"Chạy tiếp từ {cfg['resume']}: bước {step}, best val NLL {best_val:.4f}")
    log_f = open(log_path, "a" if cfg.get("resume") else "w", encoding="utf-8")

    start = time.time()
    budget = cfg.get("max_minutes", 0) * 60
    stop = False
    for epoch in range(start_epoch, cfg["epochs"]):
        for batch in train_loader:
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                out = model(to_device(batch, device), cov_weight=cfg.get("cov_weight", 1.0))
            opt.zero_grad()
            scaler.scale(out["loss"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            step += 1

            if step % cfg["log_every"] == 0:
                rec = {"step": step, "epoch": epoch, "minutes": round((time.time() - start) / 60, 2),
                       **{k: round(v.item(), 4) for k, v in out.items()},
                       "lr": opt.param_groups[0]["lr"]}
                if device.type == "cuda":
                    rec["gpu_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
                if step % cfg["eval_every"] == 0:
                    rec["val_nll"] = round(evaluate_loss(model, val_loader, device, cfg.get("val_batches"), amp), 4)
                    if sched:
                        sched.step(rec["val_nll"])
                    if rec["val_nll"] < best_val:
                        best_val = rec["val_nll"]
                        torch.save({"model": model.state_dict(), "config": cfg, "step": step},
                                   out_dir / "best.pt")
                    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                                "sched": sched.state_dict() if sched else None, "config": cfg, "step": step,
                                "epoch": epoch, "best_val": best_val}, out_dir / "last.pt")
                print(json.dumps(rec, ensure_ascii=False), flush=True)
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
            # max_steps: cho Seq2seq và PG cùng số bước để so sánh công bằng
            if (budget and time.time() - start > budget) or (cfg.get("max_steps") and step >= cfg["max_steps"]):
                stop = True
                break
        if stop:
            break

    val_nll = evaluate_loss(model, val_loader, device, cfg.get("val_batches"), amp)
    if val_nll < best_val:
        best_val = val_nll
        torch.save({"model": model.state_dict(), "config": cfg, "step": step}, out_dir / "best.pt")
    log_f.write(json.dumps({"step": step, "final_val_nll": round(val_nll, 4)}) + "\n")
    log_f.close()
    print(f"Xong: {step} bước, {(time.time() - start) / 60:.1f} phút, best val NLL = {best_val:.4f}")


if __name__ == "__main__":
    main()
