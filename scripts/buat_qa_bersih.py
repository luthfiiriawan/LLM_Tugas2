"""Versi 2 - Langkah 1: Gabungkan semua QA yang sudah direview (status=ok) jadi satu file fakta.

Pemakaian:
    python scripts/buat_qa_bersih.py data/raw/qa_draft.csv data/raw/qa_manual.csv

Output: data/processed/qa_bersih.jsonl
        satu baris = satu fakta {"id", "pertanyaan", "jawaban", "kategori", "sumber"}
File ini di-upload ke Drive lalu diparafrase di notebooks/parafrase_v2_colab.ipynb.
"""
import argparse
import csv
import json
import re
from pathlib import Path


def rapikan(s):
    return re.sub(r"\s+", " ", (s or "")).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", default="data/processed/qa_bersih.jsonl")
    args = ap.parse_args()

    rows, seen = [], set()
    for path in args.csv:
        for r in csv.DictReader(open(path, encoding="utf-8-sig", newline="")):
            if (r.get("status") or "").strip().lower() != "ok":
                continue
            q, a = rapikan(r["pertanyaan"]), rapikan(r["jawaban"])
            k = re.sub(r"[^a-z0-9 ]", "", q.lower())
            if len(q) < 8 or len(a) < 10 or k in seen:
                continue
            seen.add(k)
            rows.append({"id": len(rows), "pertanyaan": q, "jawaban": a,
                         "kategori": rapikan(r.get("kategori")).lower() or "lainnya",
                         "sumber": rapikan(r.get("sumber"))})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} fakta -> {args.out}")


if __name__ == "__main__":
    main()
