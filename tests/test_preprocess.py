from clinsum.preprocess import normalize_placeholders, split_sections


def test_normalize_mimic3_and_mimic4():
    text = "Patient [**Name (NI) 123**] was seen on [**2101-3-4**]. Seen by Dr. ___ today."
    out = normalize_placeholders(text)
    assert "[**" not in out and "___" not in out
    assert out.count("[DEID]") == 3


def test_split_sections():
    text = (
        "Admission Date: unknown\n\n"
        "Chief Complaint:\nchest pain\n\n"
        "Brief Hospital Course:\nAdmitted with chest pain. Treated with aspirin.\n\n"
        "Discharge Medications:\n1. Aspirin 81 mg daily\n"
    )
    secs = split_sections(text)
    assert "chief complaint" in secs
    assert secs["chief complaint"] == "chest pain"
    assert secs["brief hospital course"].startswith("Admitted with chest pain")
    assert "aspirin 81 mg" in secs["discharge medications"].lower()


def test_no_headers():
    assert split_sections("just text") == {"preamble": "just text"}
