# Panduan Review Draf Tanya-Jawab (`data/raw/qa_draft.csv`)

Draf dibuat otomatis oleh Qwen 2.5 14B dari Pedoman Akademik Tel-U 2024, jadi **setiap baris wajib dicek manusia** sebelum dipakai training. Hanya baris dengan `status = ok` yang masuk dataset.

## Cara kerja

1. Buka `qa_draft.csv` di Excel atau Google Sheets. Aktifkan filter di baris judul.
2. Bagi pekerjaan per rentang `chunk_id` supaya tidak bentrok, misalnya:

   | Reviewer | chunk_id | Isi pedoman |
   |---|---|---|
   | Anggota 1 | 0 – 40 | BAB I–III awal (pengertian, kompetensi, SKS, masa studi, TA) |
   | Anggota 2 | 41 – 75 | BAB III akhir – BAB IV (pembelajaran, penilaian, evaluasi studi) |
   | Anggota 3 | 76 – 112 | BAB V – IX (registrasi, cuti, nonaktif, dll.) |
   | Anggota 4 | cek silang 10% sampel acak | konsistensi & QA |

3. Untuk tiap baris, buka pedoman di halaman yang tertulis di kolom `sumber`, lalu:
   - **Benar & relevan** → `status` = `ok`
   - **Salah sedikit** → perbaiki `jawaban`, lalu `status` = `ok`
   - **Salah total / tidak berguna / mengulang** → `status` = `buang` (atau hapus barisnya)
4. **Jangan ubah** kolom `chunk_id`.

## Kolom `catatan_otomatis` (cek baris ini dulu)

| Catatan | Tindakan |
|---|---|
| `menyebut teks sumber` | Jawaban berbunyi "dalam teks ini...". Tulis ulang seperti petugas akademik menjawab langsung, atau buang. |
| `jawaban tidak tahu - cek/hapus` | Model tidak menemukan jawabannya di potongan itu. Cari jawabannya di pasal lain (misal masa studi → Pasal 12) dan perbaiki, atau buang. |
| `bukan jenjang sarjana` | Aturan untuk D3/S2/S3. Pertahankan hanya jika pertanyaannya jelas menyebut jenjang itu; kalau tidak, buang. |
| `jawaban terlalu pendek` | Lengkapi jawabannya. |
| `sumber sudah dikoreksi` | Label pasal sudah diperbaiki otomatis (Pasal 15 / Pasal 57). Review seperti biasa. |

## Kriteria jawaban yang baik

- Angka, syarat, dan istilah **persis** sesuai pedoman (misal: biaya status cuti **10% dari BPP paket**, cuti maksimal **2 semester tidak berturut-turut**).
- Lengkap. Contoh kurang tepat: "mahasiswa semester 1 tidak boleh cuti". Yang benar: cuti tidak boleh bagi mahasiswa yang **belum menyelesaikan tingkat-1** (dua semester pertama).
- Menjawab pertanyaannya. Pertanyaan "kapan" harus dijawab dengan waktu, bukan definisi.
- 1–4 kalimat, sopan, bahasa Indonesia.
- Utamakan topik yang benar-benar ditanyakan mahasiswa (SKS, masa studi, cuti, nilai/IPK, TA, kelulusan, registrasi). Pasal normatif (kebebasan akademik, suasana akademik) cukup sedikit saja.

## Tambahan di `qa_manual.csv`

- Hapus 2 baris CONTOH.
- Tambahkan ±20–30 **pertanyaan di luar topik** (resep, cuaca, politik, kampus lain) dengan jawaban sopan menolak, misalnya: "Maaf, saya hanya dapat membantu informasi seputar peraturan akademik Universitas Telkom. Silakan hubungi pihak terkait untuk pertanyaan tersebut." Isi `status` = `ok`.
- Tambahkan pertanyaan penting yang belum tercakup draf.

## Setelah review

```bash
python scripts/prepare_dataset.py data/raw/qa_draft.csv data/raw/qa_manual.csv
```

## Hasil review (7 November 2026)

Review pertama dilakukan dengan bantuan AI (Claude): setiap draf dicocokkan dengan teks pasal sumbernya.

| Hasil | Jumlah |
|---|---|
| Benar, langsung dipakai | 416 |
| Diperbaiki (pertanyaan/jawaban) | 151 |
| Dibuang (salah, mengarang, duplikat, terlalu normatif) | 106 |
| **Dipakai dari draf** | **567** |
| Tambahan `qa_manual.csv` (fakta penting S1, di luar pedoman, di luar topik) | 40 |
| **Total dataset** | **607** (train 487 / val 60 / test 60) |

Kolom `catatan_review` di `qa_draft.csv` menandai baris `diperbaiki` / `dibuang`. Masalah yang paling sering ditemukan: angka dari tabel yang rusak saat ekstraksi PDF (skor EPrT per jenjang), salah baca tabel komponen evaluasi (Pasal 39), jawaban "tidak disebutkan dalam teks", dan pertanyaan yang tidak menyebut jenjang (magister/doktor).

**Tetap wajib:** anggota kelompok melakukan cek silang minimal 10% sampel acak (terutama baris `diperbaiki`) sebelum training, dan menyebutkan penggunaan AI ini di laporan.
