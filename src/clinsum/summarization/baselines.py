"""Baseline không học: Lead-k (lấy k câu đầu của đầu vào) và Oracle trích chọn (cận trên của trích chọn).

Chạy:  python -m clinsum.summarization.baselines --data_dir data/processed/asclepius_sum --limit 1000
"""
import argparse
import json

from rouge_score import rouge_scorer

from clinsum.paths import ROOT
from clinsum.text import split_sentences
from clinsum.summarization.data import read_jsonl


def note_sentences(source: str, min_words: int = 5):
    """Câu của ghi chú: bỏ dòng yêu cầu ("request: ...") và các dòng tiêu đề ngắn."""
    if source.startswith("request:"):
        source = source.split("\n", 1)[-1]
    return [s for s in split_sentences(source) if len(s.split()) >= min_words]


def lead_k(source: str, k: int = 3) -> str:
    return " ".join(note_sentences(source)[:k])


def greedy_oracle(source: str, target: str, max_sents: int = 5) -> str:
    """Chọn tham lam các câu của đầu vào làm tăng ROUGE-1+2 so với tham chiếu."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2"], use_stemmer=True)
    sents, chosen, best = note_sentences(source), [], 0.0
    for _ in range(max_sents):
        gains = []
        for i, s in enumerate(sents):
            if i in chosen:
                continue
            cand = " ".join(sents[j] for j in sorted(chosen + [i]))
            sc = scorer.score(target, cand)
            gains.append((sc["rouge1"].fmeasure + sc["rouge2"].fmeasure, i))
        if not gains:
            break
        val, i = max(gains)
        if val <= best:
            break
        best, chosen = val, chosen + [i]
    return " ".join(sents[j] for j in sorted(chosen))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed/asclepius_sum")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--prefix", default="", help="tiền tố tên file dự đoán, vd mimic_")
    args = ap.parse_args()
    rows = read_jsonl(ROOT / args.data_dir / f"{args.split}.jsonl")[: args.limit]
    out_dir = ROOT / "outputs/predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    baselines = [("lead3", lambda r: lead_k(r["source"], 3)),
                 ("oracle_ext", lambda r: greedy_oracle(r["source"], r["target"]))]
    # request_only: chỉ chép câu yêu cầu, đo phần ROUGE "miễn phí" do câu yêu cầu đã nêu chẩn đoán (chỉ Asclepius)
    if rows and rows[0]["source"].startswith("request:"):
        baselines.insert(0, ("request_only", lambda r: r["source"].split("\n", 1)[0].removeprefix("request: ")))
    for name, fn in baselines:
        path = out_dir / f"{args.prefix}{name}_{args.split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"id": r["id"], "prediction": fn(r)}, ensure_ascii=False) + "\n")
        print("Đã ghi", path)


if __name__ == "__main__":
    main()
