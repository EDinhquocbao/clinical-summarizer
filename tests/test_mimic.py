"""Kiểm thử phần xử lý MIMIC-IV-Note bằng ghi chú GIẢ LẬP tự soạn theo định dạng mẫu (không phải dữ liệu thật)."""
import gzip
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from clinsum.corpus import make_mimic_pair, split_mimic_sections, subject_split

ROOT = Path(__file__).resolve().parents[1]

FAKE_NOTE = """ \nName:  ___                     Unit No:   ___\n \nAdmission Date:  ___              Discharge Date:   ___\n \nDate of Birth:  ___             Sex:   M\n \nService: MEDICINE\n \nAllergies: \nPenicillins\n \nAttending: ___.\n \nChief Complaint:\nshortness of breath\n \nMajor Surgical or Invasive Procedure:\nNone\n \nHistory of Present Illness:\nMr. ___ is a 67 year old man with heart failure who presents with two days of worsening dyspnea, \northopnea and leg swelling after running out of furosemide. In the ED he was given furosemide 40 mg IV \nwith good urine output and was admitted for further diuresis.\n \nPast Medical History:\nHFrEF (EF 30%), hypertension, type 2 diabetes\n \nSocial History:\n___\nFamily History:\nNo early cardiac death.\n \nPhysical Exam:\nADMISSION PHYSICAL EXAM:\nVS: 98.1 150/90 102 22 91% RA\nLungs: bibasilar crackles\nDISCHARGE PHYSICAL EXAM:\nLungs: clear\n \nPertinent Results:\nADMISSION LABS:\n___ 07:10AM BLOOD WBC-7.1 Hgb-12.0 proBNP-5400\n \nBrief Hospital Course:\nMr. ___ is a 67 year old man with HFrEF admitted with acute decompensated heart failure \nin the setting of medication nonadherence.\n \nACUTE ISSUES:\n# Acute on chronic systolic heart failure: diuresed with IV furosemide 40 mg twice daily with net negative \n4 liters. Transitioned to oral furosemide 40 mg daily. Weight on discharge 82 kg.\n \nTRANSITIONAL ISSUES:\n- Repeat chemistry panel in one week\n- Discharge weight 82 kg\n \nMedications on Admission:\n1. Furosemide 40 mg PO DAILY\n \nDischarge Medications:\n1. Furosemide 40 mg PO DAILY\n \nDischarge Disposition:\nHome\n \nDischarge Diagnosis:\nAcute on chronic systolic heart failure\n \nDischarge Condition:\nMental Status: Clear and coherent.\n \nDischarge Instructions:\nDear Mr. ___, you were admitted for heart failure. We gave you medicine to remove fluid.\n \nFollowup Instructions:\n___\n"""


def test_split_keeps_subheaders_inside_parent_section():
    secs = split_mimic_sections(FAKE_NOTE)
    bhc = secs["brief hospital course"]
    assert "ACUTE ISSUES" in bhc and "TRANSITIONAL ISSUES" in bhc  # không bị cắt ở tiêu đề con
    assert "Medications on Admission" not in bhc
    assert "DISCHARGE PHYSICAL EXAM" in secs["physical exam"]
    assert secs["service"] == "MEDICINE"  # nội dung cùng dòng với tiêu đề
    assert secs["chief complaint"] == "shortness of breath"


def test_make_mimic_pair_hides_target_and_discharge_text():
    pair, _ = make_mimic_pair(FAKE_NOTE, min_target_words=10, min_source_words=10)
    source, target = pair
    assert target.startswith("Mr. [DEID] is a 67 year old man")
    assert "TRANSITIONAL ISSUES" in target
    assert "net negative" not in source             # không lộ diễn tiến điều trị
    assert "you were admitted for heart failure" not in source  # bỏ hướng dẫn xuất viện
    assert source.startswith("chief complaint: shortness of breath")
    assert "history of present illness:" in source and "pertinent results:" in source


def test_section_cap_and_missing_target():
    pair, _ = make_mimic_pair(FAKE_NOTE, min_target_words=10, min_source_words=10, section_max_words=5)
    assert all(len(line.split(": ", 1)[1].split()) <= 5 for line in pair[0].split("\n"))
    no_bhc = FAKE_NOTE.split("Brief Hospital Course:")[0]
    assert make_mimic_pair(no_bhc)[0] is None


def test_subject_split_is_stable_and_balanced():
    assert subject_split("10000032") == subject_split("10000032")
    counts = {"train": 0, "val": 0, "test": 0}
    for i in range(5000):
        counts[subject_split(str(10_000_000 + i))] += 1
    assert 0.77 < counts["train"] / 5000 < 0.83 and counts["val"] > 350 and counts["test"] > 350


def test_prepare_mimic_end_to_end(tmp_path):
    rows = []
    for i in range(40):
        note = FAKE_NOTE if i % 4 else FAKE_NOTE.split("Brief Hospital Course:")[0]  # 1/4 không có BHC
        rows.append({"note_id": f"{1000 + i}-DS-1", "subject_id": str(1000 + i // 2), "hadm_id": str(2000 + i),
                     "note_type": "DS", "note_seq": 1, "charttime": "", "storetime": "", "text": note})
    src = tmp_path / "discharge.csv.gz"
    with gzip.open(src, "wt", encoding="utf-8", newline="") as f:
        pd.DataFrame(rows).to_csv(f, index=False)
    out_dir, stats = tmp_path / "out", tmp_path / "stats.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "prepare_mimic.py"), "--input", str(src),
                        "--out_dir", str(out_dir), "--stats", str(stats), "--chunksize", "7"],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    s = json.loads(stats.read_text(encoding="utf-8"))
    assert s["notes"] == 40 and s["pairs"] == 30
    assert sum(s["split_pairs"].values()) == 30
    assert s["dropped"] == {"không có mục Brief Hospital Course": 10}
    groups = {}
    for split in ("train", "val", "test"):
        for line in (out_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            groups.setdefault(json.loads(line)["group"], set()).add(split)
    assert all(len(v) == 1 for v in groups.values())  # một bệnh nhân chỉ ở một tập
    assert "FAKE" not in r.stdout and "furosemide" not in r.stdout.lower()  # không in nội dung ghi chú
