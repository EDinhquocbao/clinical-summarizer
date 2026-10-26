"""Chấm điểm các file dự đoán trong outputs/predictions so với tham chiếu.

Độ đo: ROUGE-1/2/L (F1, có stemming), độ dài, novel bigram, lặp trigram,
entity recall / hallucination trên tập con số (liều, tuổi, kết quả xét nghiệm).
Ghi bảng ra outputs/results/summary_results.md và .json.

Chạy:  python scripts/score_predictions.py --data_dir data/processed/asclepius_sum
"""
import argparse
import json
from statistics import mean

from rouge_score import rouge_scorer

from clinsum.evaluate import entity_recall, hallucination_rate, novel_ngram_rate, repetition_rate
from clinsum.paths import OUTPUTS_DIR, ROOT
from clinsum.text import numbers, tokenize
from clinsum.summarization.data import read_jsonl

ORDER = ["request_only", "lead3", "oracle_ext", "seq2seq_attn", "pointer_gen", "pointer_gen_cov"]


def score_file(path, refs, srcs, max_source_len):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    m = {k: [] for k in ("rouge1", "rouge2", "rougeL", "len", "novel2", "rep3", "num_recall", "num_halluc")}
    p_gens = []
    for row in read_jsonl(path):
        ref, pred = refs[row["id"]], row["prediction"]
        src_tok = tokenize(srcs[row["id"]])[:max_source_len]
        pred_tok, ref_tok = tokenize(pred), tokenize(ref)
        sc = scorer.score(ref, pred)
        for k in ("rouge1", "rouge2", "rougeL"):
            m[k].append(sc[k].fmeasure * 100)
        m["len"].append(len(pred_tok))
        m["novel2"].append(novel_ngram_rate(src_tok, pred_tok) * 100)
        m["rep3"].append(repetition_rate(pred_tok) * 100)
        ref_nums = [n for n in numbers(ref_tok) if n in set(src_tok)]
        if ref_nums:
            m["num_recall"].append(entity_recall(ref_nums, numbers(pred_tok)) * 100)
        if numbers(pred_tok):
            m["num_halluc"].append(hallucination_rate(numbers(src_tok), numbers(pred_tok)) * 100)
        if row.get("p_gen") is not None:
            p_gens.append(row["p_gen"])
    out = {k: round(mean(v), 2) if v else None for k, v in m.items()}
    out["n"] = len(m["rouge1"])
    out["p_gen"] = round(mean(p_gens), 3) if p_gens else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed/asclepius_sum")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_source_len", type=int, default=512)
    args = ap.parse_args()
    rows = read_jsonl(ROOT / args.data_dir / f"{args.split}.jsonl")
    refs = {r["id"]: r["target"] for r in rows}
    srcs = {r["id"]: r["source"] for r in rows}
    ref_len = mean(len(tokenize(r["target"])) for r in rows)

    results = {}
    for name in ORDER:
        path = OUTPUTS_DIR / "predictions" / f"{name}_{args.split}.jsonl"
        if path.exists():
            results[name] = score_file(path, refs, srcs, args.max_source_len)
            print(name, results[name])

    res_dir = OUTPUTS_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "summary_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    cols = ["rouge1", "rouge2", "rougeL", "len", "novel2", "rep3", "num_recall", "num_halluc", "p_gen"]
    lines = [f"Tập {args.split}, n = {next(iter(results.values()))['n']}, độ dài tham chiếu TB = {ref_len:.0f} token\n",
             "| Mô hình | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, r in results.items():
        lines.append(f"| {name} | " + " | ".join("-" if r[c] is None else str(r[c]) for c in cols) + " |")
    (res_dir / "summary_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
