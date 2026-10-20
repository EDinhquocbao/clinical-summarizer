from clinsum.evaluate import entity_recall, hallucination_rate


def test_entity_recall():
    src = ["Aspirin", "metoprolol", "myocardial infarction"]
    summ = ["aspirin", "myocardial  infarction"]
    assert abs(entity_recall(src, summ) - 2 / 3) < 1e-9


def test_hallucination_rate():
    src = ["aspirin", "metoprolol"]
    summ = ["aspirin", "warfarin"]  # warfarin không có trong bản gốc
    assert hallucination_rate(src, summ) == 0.5


def test_empty_cases():
    assert entity_recall([], ["a"]) == 1.0
    assert hallucination_rate(["a"], []) == 0.0
