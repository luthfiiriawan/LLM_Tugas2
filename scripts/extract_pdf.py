"""Langkah 1: Ekstrak teks dari PDF peraturan akademik menjadi potongan (chunk).

Pemakaian:
    python scripts/extract_pdf.py data/raw/Pedoman_Akademik_Tel-U_2024.pdf --mulai-halaman 7
    python scripts/extract_pdf.py data/raw/            # semua PDF di folder

--mulai-halaman : lewati halaman awal (cover, SK, daftar isi). Untuk Pedoman Tel-U 2024 = 7.
                  Halaman ke-N PDF itu dihitung sebagai hal. 1 (sesuai nomor tercetak).
Mode potong:
  - per pasal (default): satu chunk = satu "Pasal N"; pasal yang panjang dipecah per ayat.
    Cocok untuk dokumen peraturan, sumber tercatat sebagai nomor pasal + halaman.
  - --per-karakter    : potong tiap ~maks karakter (untuk dokumen tanpa struktur pasal).

Output: data/processed/chunks.jsonl  (satu baris = {"teks", "sumber", "bab", "pasal"})
Butuh: pip install pypdf
"""
import argparse
import json
import re
from pathlib import Path

from pypdf import PdfReader

RE_BAB = re.compile(r"^BAB\s+([IVXLC]+)\s*$")
RE_PASAL = re.compile(r"^Pasal\s+(\d+)\s+(.*?)\s*$")


def bersihkan_halaman(teks: str) -> list[str]:
    teks = teks.replace("­", "")
    teks = re.sub(r"-\n(\w)", r"\1", teks)
    baris = []
    for b in teks.split("\n"):
        b = re.sub(r"[ \t]+", " ", b).strip()
        if not b or re.fullmatch(r"[ivxlc\d]+", b):   # baris kosong / nomor halaman
            continue
        baris.append(b)
    return baris


def baca_pdf(files, mulai):
    """Hasilkan (nama_file, no_halaman, baris) untuk setiap baris teks."""
    for f in files:
        pages = PdfReader(f).pages
        print(f"{f.name}: {len(pages)} halaman (mulai dari hal. {mulai})")
        for i, page in enumerate(pages, start=1):
            if i < mulai:
                continue
            for b in bersihkan_halaman(page.extract_text() or ""):
                yield f.name, i - mulai + 1, b   # nomor halaman tercetak di dokumen


def gabung_baris(baris):
    """Gabungkan baris yang terpotong; baris baru hanya untuk ayat (1), huruf a., dst."""
    out = ""
    for b in baris:
        if re.match(r"^(\(\d+\)|[a-z]\.|\d+\.|\d+\))\s", b) or not out:
            out += ("\n" if out else "") + b
        else:
            out += " " + b
    return out


def per_pasal(rows, maks):
    bab, judul_bab, pasal, judul, isi, hal = "", "", None, "", [], set()
    tunggu_judul_bab = False

    def keluarkan():
        if pasal is None or not isi:
            return
        teks = gabung_baris(isi)
        kepala = f"BAB {bab} {judul_bab} - Pasal {pasal} {judul}".strip()
        hal_str = f"hal. {min(hal)}" + (f"-{max(hal)}" if max(hal) != min(hal) else "")
        bagian, cur = [], ""
        for ayat in re.split(r"\n(?=\(\d+\)\s)", teks):   # pecah per ayat bila kepanjangan
            if cur and len(cur) + len(ayat) > maks:
                bagian.append(cur)
                cur = ""
            cur += ("\n" if cur else "") + ayat
        bagian.append(cur)
        for k, b in enumerate(bagian, 1):
            label = f" (bagian {k}/{len(bagian)})" if len(bagian) > 1 else ""
            yield {"teks": f"{kepala}{label}\n{b}", "bab": bab, "pasal": pasal,
                   "sumber": [f"{nama_file} Pasal {pasal} {hal_str}"]}

    nama_file, terakhir = "", 0
    for nama_file_, no, b in rows:
        if tunggu_judul_bab:
            judul_bab, tunggu_judul_bab = b, False
            continue
        if m := RE_BAB.match(b):
            yield from keluarkan()
            bab, pasal, isi, hal, tunggu_judul_bab = m.group(1), None, [], set(), True
            continue
        if nama_file_ != nama_file:
            terakhir = 0                      # file PDF baru -> nomor pasal mulai lagi
        # Judul pasal asli: nomornya tepat terakhir+1 dan bukan rujukan seperti
        # "Pasal 14 ayat (1)" yang kebetulan jatuh di awal baris.
        if ((m := RE_PASAL.match(b)) and "...." not in b
                and int(m.group(1)) == terakhir + 1
                and not re.match(r"(ayat|huruf|angka)\b", m.group(2))):
            yield from keluarkan()
            pasal, judul, isi, hal = int(m.group(1)), m.group(2), [], set()
            terakhir = pasal
            nama_file = nama_file_
            continue
        if pasal is not None:
            isi.append(b)
            hal.add(no)
    yield from keluarkan()


def per_karakter(rows, maks):
    chunk, sumber = "", []
    for nama_file, no, b in rows:
        if len(chunk) + len(b) > maks and chunk:
            yield {"teks": chunk.strip(), "sumber": sumber}
            chunk, sumber = "", []
        chunk += b + "\n"
        tag = f"{nama_file} hal. {no}"
        if tag not in sumber:
            sumber.append(tag)
    if chunk.strip():
        yield {"teks": chunk.strip(), "sumber": sumber}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="file PDF atau folder berisi PDF")
    ap.add_argument("--out", default="data/processed/chunks.jsonl")
    ap.add_argument("--maks", type=int, default=2500, help="maks karakter per chunk")
    ap.add_argument("--mulai-halaman", type=int, default=1)
    ap.add_argument("--per-karakter", action="store_true")
    args = ap.parse_args()

    p = Path(args.input)
    files = sorted(p.glob("*.pdf")) if p.is_dir() else [p]
    rows = list(baca_pdf(files, args.mulai_halaman))
    chunks = list((per_karakter if args.per_karakter else per_pasal)(rows, args.maks))
    if not chunks and not args.per_karakter:
        print("Tidak ditemukan struktur 'Pasal N'; beralih ke mode per karakter.")
        chunks = list(per_karakter(rows, args.maks))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fo:
        for c in chunks:
            fo.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Selesai: {len(chunks)} chunk -> {args.out}")
    if not chunks:
        print("PERINGATAN: tidak ada teks. PDF mungkin hasil scan (gambar) -> perlu OCR dulu.")


if __name__ == "__main__":
    main()
