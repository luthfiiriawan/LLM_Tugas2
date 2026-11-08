"""Langkah 2: Buat DRAF pasangan tanya-jawab dari chunk peraturan memakai LLM lokal (Ollama).

Pemakaian:
    ollama pull qwen2.5:7b        # generator; 7b lebih akurat, 3b lebih cepat
    python scripts/generate_qa.py --model qwen2.5:7b --per-chunk 8

Output: data/raw/qa_draft.csv  (kolom: pertanyaan, jawaban, kategori, sumber, status)

WAJIB: hasilnya adalah DRAF. Data Engineer harus mengecek setiap baris terhadap PDF asli,
memperbaiki jawaban yang salah, lalu mengisi kolom status = "ok" (atau hapus barisnya).
Hanya baris berstatus "ok" yang dipakai prepare_dataset.py.
Tidak butuh library tambahan (hanya standard library Python).
"""
import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import OLLAMA_URL  # noqa: E402

PROMPT = """Berikut potongan peraturan akademik kampus:

\"\"\"
{teks}
\"\"\"

Buat {n} pasangan tanya-jawab dalam bahasa Indonesia yang mungkin ditanyakan mahasiswa,
HANYA berdasarkan isi teks di atas (jangan mengarang informasi yang tidak ada).
- Variasikan gaya pertanyaan: formal, santai, singkat, ada yang memakai kata "gimana", "boleh gak", dll.
- Jawaban 1-4 kalimat, sopan, jelas, sebutkan angka/syarat persis seperti di teks.
- Isi "kategori" dengan topik singkat (misal: cuti akademik, KRS, kelulusan, nilai, beasiswa, sanksi).

Keluarkan HANYA JSON dengan format:
{{"qa": [{{"pertanyaan": "...", "jawaban": "...", "kategori": "..."}}]}}"""


def tanya_ollama(model, prompt):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.7},
    }).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/processed/chunks.jsonl")
    ap.add_argument("--out", default="data/raw/qa_draft.csv")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--per-chunk", type=int, default=8)
    ap.add_argument("--mulai", type=int, default=0, help="lanjutkan dari chunk ke-N")
    args = ap.parse_args()

    chunks = [json.loads(l) for l in open(args.chunks, encoding="utf-8")]
    out = Path(args.out)
    baru = not out.exists() or args.mulai == 0
    with open(out, "w" if baru else "a", newline="", encoding="utf-8-sig") as fo:
        w = csv.writer(fo)
        if baru:
            w.writerow(["pertanyaan", "jawaban", "kategori", "sumber", "status"])
        for i, c in enumerate(chunks[args.mulai:], start=args.mulai):
            try:
                hasil = json.loads(tanya_ollama(args.model, PROMPT.format(teks=c["teks"], n=args.per_chunk)))
                items = hasil.get("qa", [])
            except Exception as e:  # JSON rusak / timeout -> lewati chunk ini
                print(f"[{i}] gagal: {e}")
                continue
            for qa in items:
                if qa.get("pertanyaan") and qa.get("jawaban"):
                    w.writerow([qa["pertanyaan"].strip(), qa["jawaban"].strip(),
                                qa.get("kategori", "").strip().lower(), "; ".join(c["sumber"]), "draft"])
            fo.flush()
            print(f"[{i + 1}/{len(chunks)}] +{len(items)} QA")
    print(f"Selesai -> {out}. Sekarang review manual & isi status='ok'.")


if __name__ == "__main__":
    main()
