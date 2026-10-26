"""Vẽ đường cong val NLL và biểu đồ ROUGE cho báo cáo / slide.

Đọc models/<tên>/train_log.jsonl và outputs/results/summary_results.json,
ghi outputs/figures/val_nll.png và outputs/figures/rouge.png.
Chạy:  python scripts/plot_results.py
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from clinsum.paths import MODELS_DIR, OUTPUTS_DIR  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f2328", "#5b6470", "#e6e6e3"
SERIES = {"seq2seq_attn": "#2a78d6", "pointer_gen": "#eb6834", "pointer_gen_cov": "#1baf7a"}
LABEL = {"request_only": "Chỉ câu yêu cầu", "lead3": "Lead-3", "oracle_ext": "Oracle trích chọn", "seq2seq_attn": "Seq2seq + attention",
         "pointer_gen": "Pointer-Generator", "pointer_gen_cov": "PG + coverage"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=12)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_val_nll(fig_dir):
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    _style(ax)
    offset = 0
    for name, color in SERIES.items():
        path = MODELS_DIR / name / "train_log.jsonl"
        if not path.exists():
            continue
        recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        pts = [(r["step"], r["val_nll"]) for r in recs if "val_nll" in r]
        if not pts:
            continue
        if name == "pointer_gen_cov":  # pha 2 tiếp nối sau bước cuối của pointer_gen
            pts = [(s + offset, v) for s, v in pts]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5)
        # nhãn PG + coverage đặt dưới để không đè nhãn Pointer-Generator
        dy = -18 if name == "pointer_gen_cov" else 10
        ax.annotate(f"{LABEL[name]} ({ys[-1]:.2f})", (xs[-1], ys[-1]), xytext=(-10, dy),
                    textcoords="offset points", color=INK, fontsize=12, ha="right", va="center")
        if name == "pointer_gen":
            offset = xs[-1]
    ax.set_xlabel("Bước huấn luyện (batch 32)", color=MUTED, fontsize=12)
    ax.set_ylabel("NLL trên tập val", color=MUTED, fontsize=12)
    ax.set_ylim(bottom=0.8)
    fig.tight_layout()
    fig.savefig(fig_dir / "val_nll.png", dpi=200)


def plot_safety(fig_dir):
    """Small multiples: lặp trigram, recall số liệu, số liệu bịa cho 3 mô hình học sâu."""
    path = OUTPUTS_DIR / "results" / "summary_results.json"
    if not path.exists():
        return
    res = json.loads(path.read_text(encoding="utf-8"))
    names = [n for n in SERIES if n in res][::-1]
    panels = [("rep3", "Lặp trigram (%) ↓"), ("num_recall", "Giữ số liệu (%) ↑"), ("num_halluc", "Số liệu bịa (%) ↓")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), facecolor=SURFACE, sharey=True)
    for ax, (key, title) in zip(axes, panels):
        _style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        vals = [res[n][key] or 0 for n in names]
        bars = ax.barh(range(len(names)), vals, color=[SERIES[n] for n in names], height=0.6)
        for b, v in zip(bars, vals):
            ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center", color=INK, fontsize=12)
        ax.set_title(title, color=INK, fontsize=13, loc="left")
        ax.set_xlim(0, max(max(vals) * 1.3, 10))
    axes[0].set_yticks(range(len(names)), [LABEL[n] for n in names], fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(fig_dir / "safety.png", dpi=200)


def plot_rouge(fig_dir):
    path = OUTPUTS_DIR / "results" / "summary_results.json"
    if not path.exists():
        return
    res = json.loads(path.read_text(encoding="utf-8"))
    names = [n for n in LABEL if n in res]
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    names = names[::-1]  # mô hình chính ở trên cùng
    vals = [res[n]["rougeL"] for n in names]
    # học sâu: xanh; baseline / cận trên: xám
    colors = ["#2a78d6" if n in SERIES else "#b8bcc2" for n in names]
    bars = ax.barh(range(len(names)), vals, color=colors, height=0.6)
    for b, v in zip(bars, vals):
        ax.text(v + 0.8, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center", color=INK, fontsize=12)
    ax.set_yticks(range(len(names)), [LABEL[n] for n in names], fontsize=12, color=INK)
    ax.set_xlabel("ROUGE-L (F1, %) trên 500 mẫu test", color=MUTED, fontsize=12)
    ax.set_xlim(0, max(vals) * 1.15)
    fig.tight_layout()
    fig.savefig(fig_dir / "rouge.png", dpi=200)


def main() -> None:
    fig_dir = OUTPUTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_val_nll(fig_dir)
    plot_rouge(fig_dir)
    plot_safety(fig_dir)
    print("Đã ghi hình vào", fig_dir)


if __name__ == "__main__":
    main()
