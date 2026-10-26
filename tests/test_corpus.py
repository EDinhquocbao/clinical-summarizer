from clinsum.corpus import make_pair, split_by_group
from clinsum.evaluate import novel_ngram_rate, repetition_rate
from clinsum.text import detokenize, numbers, tokenize

NOTE = (
    "Discharge Summary:\n\n"
    "History of Present Illness:\nA 60-year-old man presented with fever and dry cough for five days "
    "and was found to have bilateral infiltrates on chest imaging with low oxygen saturation.\n\n"
    "Hospital Course:\nThe patient was admitted and treated with dexamethasone 6 mg daily and oxygen. "
    "He improved gradually and was weaned to room air before discharge home.\n\n"
    "Discharge Diagnosis:\nCOVID-19 pneumonia\n"
)


def test_make_pair_uses_hospital_course_as_target_and_hides_it():
    source, target = make_pair(NOTE)
    assert target.startswith("The patient was admitted")
    assert "dexamethasone" not in source
    assert "COVID-19 pneumonia" not in source  # chẩn đoán xuất viện bị bỏ khỏi đầu vào
    assert "fever and dry cough" in source


def test_make_pair_without_target_section():
    assert make_pair("History of Present Illness:\nfever\n") is None


def test_split_by_group_keeps_patient_in_one_split():
    rows = [{"pid": i // 2, "x": i} for i in range(100)]
    parts = split_by_group(rows, "pid")
    seen = {}
    for name, rs in parts.items():
        for r in rs:
            assert seen.setdefault(r["pid"], name) == name
    assert sum(len(v) for v in parts.values()) == 100


def test_tokenize_keeps_doses():
    toks = tokenize("Aspirin 81 mg, INR 2.5 [DEID].")
    assert toks == ["aspirin", "81", "mg", ",", "inr", "2.5", "[deid]", "."]
    assert numbers(toks) == ["81", "2.5"]
    assert detokenize(["dose", "is", "5", "mg", "."]) == "dose is 5 mg."


def test_abstractiveness_metrics():
    assert novel_ngram_rate("a b c".split(), "a b c".split()) == 0.0
    assert repetition_rate("a b c a b c".split()) == 0.25
