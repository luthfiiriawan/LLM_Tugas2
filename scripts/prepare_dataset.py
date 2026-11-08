"""Langkah 3: Bersihkan, deduplikasi, split, dan konversi dataset Q&A ke JSONL (format chat).

Pemakaian:
    python scripts/prepare_dataset.py data/raw/qa_draft.csv data/raw/qa_manual.csv

Input : satu/lebih CSV dengan kolom pertanyaan, jawaban, kategori, sumber, status.
        Hanya baris dengan status "ok" yang dipakai.
Output: data/processed/train.jsonl, val.jsonl, test.jsonl
        docs/analisis_dataset.md (+ docs/img/distribusi_dataset.png jika matplotlib ada)
Tidak butuh library tambahan (matplotlib opsional untuk grafik).
"""
import argparse
import csv
import json
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import SYSTEM_PROMPT  # noqa: E402


def rapikan(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    return s.strip('"').strip()


def kunci(s: str) -> str:
    """Normalisasi untuk deteksi duplikat: huruf kecil, tanpa tanda baca."""
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def ke_chat(row):
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["pertanyaan"]},
        {"role": "assistant", "content": row["jawaban"]},
    ], "kategori": row["kategori"], "sumber": row["sumber"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    stat = Counter()
    rows, seen = [], set()
    for path in args.csv:
        with open(path, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                stat["total_baris"] += 1
                if (r.get("status") or "").strip().lower() != "ok":
                    stat["dibuang_status_bukan_ok"] += 1
                    continue
                q, a = rapikan(r.get("pertanyaan")), rapikan(r.get("jawaban"))
                if len(q) < 8 or len(a) < 10:
                    stat["dibuang_terlalu_pendek"] += 1
                    continue
                k = kunci(q)
                if k in seen:
                    stat["dibuang_duplikat"] += 1
                    continue
                seen.add(k)
                rows.append({"pertanyaan": q, "jawaban": a,
                             "kategori": rapikan(r.get("kategori")).lower() or "lainnya",
                             "sumber": rapikan(r.get("sumber"))})
    if not rows:
        sys.exit("Tidak ada baris berstatus 'ok'. Review dulu CSV-nya.")

    random.Random(args.seed).shuffle(rows)
    n = len(rows)
    n_test, n_val = max(1, int(n * args.test)), max(1, int(n * args.val))
    split = {"test": rows[:n_test], "val": rows[n_test:n_test + n_val], "train": rows[n_test + n_val:]}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for nama, data in split.items():
        with open(out / f"{nama}.jsonl", "w", encoding="utf-8") as fo:
            for r in data:
                fo.write(json.dumps(ke_chat(r), ensure_ascii=False) + "\n")

    # ---- analisis singkat dataset (untuk laporan, kriteria rubrik 20%) ----
    kat = Counter(r["kategori"] for r in rows)
    pq = [len(r["pertanyaan"].split()) for r in rows]
    pa = [len(r["jawaban"].split()) for r in rows]
    docs = Path("docs")
    (docs / "img").mkdir(parents=True, exist_ok=True)
    lines = [
        "# Analisis Dataset", "",
        "## Ringkasan preprocessing", "",
        "| Tahap | Jumlah |", "|---|---|",
        *[f"| {k.replace('_', ' ')} | {v} |" for k, v in stat.items()],
        f"| **data bersih** | **{n}** |", "",
        "## Pembagian data", "",
        "| Split | Jumlah |", "|---|---|",
        *[f"| {k} | {len(v)} |" for k, v in split.items()], "",
        "## Panjang teks (jumlah kata)", "",
        "| | min | median | rata-rata | maks |", "|---|---|---|---|---|",
        f"| pertanyaan | {min(pq)} | {statistics.median(pq)} | {statistics.mean(pq):.1f} | {max(pq)} |",
        f"| jawaban | {min(pa)} | {statistics.median(pa)} | {statistics.mean(pa):.1f} | {max(pa)} |", "",
        "## Distribusi kategori", "",
        "| Kategori | Jumlah | % |", "|---|---|---|",
        *[f"| {k} | {v} | {100 * v / n:.1f}% |" for k, v in kat.most_common()],
    ]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        top = kat.most_common(12)[::-1]
        ax[0].barh([k for k, _ in top], [v for _, v in top], color="#4C72B0")
        ax[0].set_title("Jumlah data per kategori")
        ax[1].hist(pa, bins=20, color="#55A868")
        ax[1].set_title("Distribusi panjang jawaban (kata)")
        fig.tight_layout()
        fig.savefig(docs / "img" / "distribusi_dataset.png", dpi=120)
        lines += ["", "![Distribusi dataset](img/distribusi_dataset.png)"]
    except ImportError:
        pass
    (docs / "analisis_dataset.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(dict(stat), f"-> bersih: {n}")
    print({k: len(v) for k, v in split.items()})
    print("Analisis -> docs/analisis_dataset.md")
    if n < 300:
        print(f"PERINGATAN: baru {n} data. Targetkan minimal ~300-500 pasangan QA.")


if __name__ == "__main__":
    main()
