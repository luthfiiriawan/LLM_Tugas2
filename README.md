# Chatbot Regulasi Kampus: Fine-Tuning Local LLM (Qwen 2.5 3B + QLoRA + Ollama)

Proyek 2 CAK4KBB3 (Generative AI). Kami melakukan fine-tuning **Qwen 2.5 3B Instruct** dengan **QLoRA (Unsloth)** di **Google Colab** memakai dataset tanya-jawab peraturan akademik kampus. Model hasilnya dikonversi ke **GGUF** lalu di-deploy **lokal** dengan **Ollama + Streamlit**.

> **Hasil akhir (versi 2):** token F1 naik dari 0,208 (model dasar) menjadi **0,798**, dan ketepatan angka dari 0/18 menjadi **15/18**, pada 68 soal uji. Langkah 0–4 di bawah adalah alur dasar (eksperimen 1). Perbaikan dan hasil final ada di bagian [Versi 2](#versi-2-perbaikan-akurasi-fakta). Laporan tutorial lengkap dikumpulkan terpisah lewat LMS.

## File yang tidak ada di repo

| File | Alasan | Cara mendapatkan |
|---|---|---|
| `deploy/*.gguf` (1,9–3,3 GB) | Terlalu besar untuk GitHub | Jalankan notebook fine-tuning, lalu ekspor (Versi 2 langkah 4–5) |
| `outputs_v2/lora_adapter/` (±250 MB) | Terlalu besar | Output `notebooks/finetune_v2_colab.ipynb` di Google Drive |
| `data/raw/Pedoman_Akademik_Tel-U_2024.pdf` | Dokumen milik kampus | Unduh dari situs resmi kampus, simpan di `data/raw/` |

## Arsitektur

```
 PDF Peraturan Akademik
        │  scripts/extract_pdf.py
        ▼
 chunks.jsonl ──► scripts/generate_qa.py (draf QA via Ollama) ──► review manual (status=ok)
                                                                        │
                                       scripts/prepare_dataset.py  ◄────┘
                                                │  bersihkan, dedup, split 80/10/10
                                                ▼
                         train.jsonl / val.jsonl / test.jsonl  (format chat)
                                                │  upload ke Google Drive
 ┌──────────────── GOOGLE COLAB (GPU T4) ───────▼──────────────────────────┐
 │ notebooks/finetune_qwen2.5_3b_colab.ipynb                               │
 │ Qwen2.5-3B-Instruct (4-bit) + LoRA r=16 ─► SFT ─► kurva loss + log      │
 │                                          └► export GGUF Q4_K_M          │
 └──────────────────────────────────────────────┬──────────────────────────┘
                                                │  download .gguf
 ┌──────────────── LAPTOP LOKAL (CPU, RAM 16GB) ▼──────────────────────────┐
 │ deploy/Modelfile ─► ollama create chatbot-kampus                        │
 │ deploy/app.py (Streamlit): mode Chat + mode Bandingkan sebelum/sesudah  │
 │ eval/eval_versi.py: token F1, ketepatan angka, waktu respons            │
 └─────────────────────────────────────────────────────────────────────────┘
```

Kenapa hybrid: laptop tidak punya GPU NVIDIA (Intel Iris Xe), padahal QLoRA butuh CUDA. Training dilakukan di Colab, sedangkan **deployment dan demo sepenuhnya lokal**, sesuai ketentuan tugas.

## Struktur folder

| Path | Isi | PIC |
|---|---|---|
| `data/raw/` | PDF peraturan, `qa_draft.csv`, `qa_manual.csv` | Anggota 1 |
| `data/processed/` | `chunks.jsonl`, `train/val/test.jsonl` | Anggota 1 |
| `scripts/` | `config.py` (system prompt bersama), ekstraksi, generate QA, preprocessing | Anggota 1 |
| `notebooks/` | Notebook fine-tuning Colab | Anggota 2 |
| `deploy/` | `Modelfile`, `app.py` (Streamlit), file `.gguf` | Anggota 3 |
| `eval/` | `eval_versi.py`, hasil evaluasi | Anggota 4 |
| `docs/` | Analisis dataset, kurva loss, screenshot, laporan | Anggota 4 |

---

## Langkah 0: Persiapan lingkungan (laptop)

1. Install **Python 3.10+** dan **Git**.
2. Install **Ollama** dari https://ollama.com/download, lalu simpan model di D: supaya C: tidak penuh:
   ```bash
   setx OLLAMA_MODELS "D:\ollama\models"
   ```
   Tutup dan buka lagi Ollama (ikon tray) agar pengaturan terbaca.
3. Siapkan virtual env dan library:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install pypdf streamlit requests matplotlib
   ```
4. Ganti `NAMA_KAMPUS` di `scripts/config.py`, di sel konfigurasi notebook, dan di baris `SYSTEM` pada `deploy/Modelfile`. **Ketiganya harus sama persis.**

## Langkah 1: Dataset (Anggota 1, bobot 20%)

1. Taruh PDF peraturan/pedoman akademik di `data/raw/`.
2. Ekstrak teks:
   ```bash
   python scripts/extract_pdf.py data/raw/Pedoman_Akademik_Tel-U_2024.pdf --mulai-halaman 7
   ```
3. Buat draf QA dengan LLM. Pilih salah satu:
   - **Colab (disarankan, ~25–50 menit, Qwen 2.5 14B):** upload `data/processed/chunks.jsonl` ke `MyDrive/proyek-llm/data/processed/`, jalankan `notebooks/generate_qa_colab.ipynb` (runtime T4), hasilnya tersimpan di `MyDrive/proyek-llm/data/raw/qa_draft.csv`; download ke `data/raw/`. Jumlah QA per chunk menyesuaikan panjang pasal (3–10), total sekitar 690 draf.
   - **Laptop (CPU, 2–4 jam, Qwen 2.5 7B):**
     ```bash
     ollama pull qwen2.5:7b
     python scripts/generate_qa.py --model qwen2.5:7b --per-chunk 8
     ```
4. **Review manual** `data/raw/qa_draft.csv` di Excel/Google Sheets: cocokkan setiap jawaban dengan PDF, perbaiki yang salah, lalu isi `status` = `ok`. Tambahkan juga pertanyaan buatan sendiri di `qa_manual.csv`, misalnya variasi bahasa santai dan pertanyaan di luar topik dengan jawaban "tidak tahu, silakan hubungi bagian akademik". Hapus 2 baris CONTOH.
5. Proses dataset:
   ```bash
   python scripts/prepare_dataset.py data/raw/qa_draft.csv data/raw/qa_manual.csv
   ```
   Hasilnya `train/val/test.jsonl` dan `docs/analisis_dataset.md` (tabel preprocessing, distribusi kategori, panjang teks, grafik).

Target: **≥ 300–500 pasangan QA** yang sudah direview, tersebar di banyak kategori (KRS, cuti, nilai/IPK, kelulusan, TA/skripsi, sanksi, beasiswa, dll.).

Format satu baris JSONL:
```json
{"messages": [{"role": "system", "content": "Kamu adalah asisten..."}, {"role": "user", "content": "Gimana cara cuti?"}, {"role": "assistant", "content": "Cuti akademik diajukan..."}], "kategori": "cuti akademik", "sumber": "peraturan.pdf hal. 12"}
```

## Langkah 2: Fine-tuning di Colab (Anggota 2, bobot 25%)

1. Upload `train.jsonl`, `val.jsonl`, `test.jsonl` ke `Google Drive/MyDrive/proyek-llm/data/processed/`.
2. Buka `notebooks/finetune_qwen2.5_3b_colab.ipynb` di Colab, pilih runtime **T4 GPU**, lalu jalankan semua sel.
3. Hyperparameter awal:

   | Parameter | Nilai | Alasan |
   |---|---|---|
   | Kuantisasi | 4-bit (QLoRA) | muat di VRAM T4 15 GB |
   | LoRA r / alpha | 16 / 16 | kapasitas cukup untuk domain sempit, risiko overfit rendah |
   | Target modules | q,k,v,o,gate,up,down | semua proyeksi attention + MLP |
   | Learning rate | 2e-4, linear, warmup 5% | standar QLoRA |
   | Effective batch | 2 × 4 = 8 | stabil di T4 |
   | Epoch | 3 | dataset kecil; hentikan/turunkan jika val loss naik |
   | Loss | hanya pada jawaban asisten | `train_on_responses_only` |

4. Output di Drive `proyek-llm/outputs/`: `training_log.csv`, `loss_curve.png`, `perbandingan_sebelum_sesudah.csv`, `lora_adapter/`, `chatbot-kampus-q4_k_m.gguf`.
5. Analisis kurva: train loss dan val loss harus turun. Kalau val loss mulai naik sementara train loss terus turun, itu overfitting; kurangi epoch atau rank.

## Langkah 3: Deploy lokal (Anggota 3, bobot 20%)

```bash
copy chatbot-kampus-q4_k_m.gguf deploy\
cd deploy
ollama create chatbot-kampus -f Modelfile
ollama pull qwen2.5:3b
ollama run chatbot-kampus "Bagaimana cara mengajukan cuti akademik?"
cd ..
streamlit run deploy/app.py
```

`qwen2.5:3b` adalah model dasar untuk pembanding "sebelum". Di UI ada dua mode: **Chat** dan **Bandingkan** (jawaban berdampingan + waktu respons) untuk demo. Tema tampilan ada di `.streamlit/config.toml`, sengaja netral tanpa logo/warna kampus.

## Langkah 4: Evaluasi & laporan (Anggota 4, bobot 25% + 10%)

```bash
python eval/eval_versi.py --test data/processed/test_v2.jsonl --model qwen2.5:3b chatbot-kampus-v2
```
Hasilnya `eval/hasil_versi.csv` (jawaban per soal) dan `eval/ringkasan_versi.md` (token F1, ketepatan angka, waktu respons). Masukkan ke laporan bersama screenshot setiap langkah.

## Error handling (catat error yang kalian temui juga)

| Masalah | Solusi |
|---|---|
| `extract_pdf.py` menghasilkan 0 chunk | PDF hasil scan, jadi perlu OCR (misal `ocrmypdf`) dulu |
| `generate_qa.py` error JSON / timeout | Chunk itu otomatis dilewati; lanjutkan dengan `--mulai N` |
| Colab `CUDA out of memory` | Turunkan `BATCH_SIZE` ke 1 dan naikkan `GRAD_ACCUM` ke 8, atau `MAX_SEQ_LENGTH` 1024 |
| Sesi Colab terputus | Output tersimpan di Drive; jalankan ulang dari atas |
| `ollama create` gagal "no such file" | Pastikan nama file `.gguf` sama dengan baris `FROM` di Modelfile |
| Jawaban model berulang / tidak berhenti | Pastikan `PARAMETER stop "<|im_end|>"` ada di Modelfile |
| Streamlit: "Ollama tidak berjalan" | Buka aplikasi Ollama, cek `http://localhost:11434` |
| Disk C: penuh | Cek `OLLAMA_MODELS` sudah diarahkan ke D: |

## Pembagian tugas

| Anggota | Peran | Deliverable |
|---|---|---|
| 1 | Data Engineer & Prep | PDF sumber, QA tervalidasi, JSONL, `analisis_dataset.md` |
| 2 | ML / Fine-Tuning Engineer | Notebook berjalan, log + kurva loss, justifikasi hyperparameter, GGUF |
| 3 | Deployment & Integration | Modelfile, Ollama, UI Streamlit, video/screenshot demo |
| 4 | Technical Writer & QA | Evaluasi sebelum/sesudah, laporan tutorial, error handling |

---

## Versi 2: perbaikan akurasi fakta

**Temuan evaluasi versi 1:** model sudah meniru gaya jawaban dan menolak pertanyaan di luar topik, tetapi sering salah fakta. Contohnya "IPS 2,8 boleh ambil berapa SKS?" dijawab 15 SKS, padahal yang benar 20 SKS (Pasal 12 ayat 4), walaupun contoh itu ada di data training. Penyebabnya: (1) setiap fakta hanya muncul satu kali sehingga model underfit (train loss akhir ±0,72, val loss terendah 1,34 di step 120 lalu naik ke 1,39), dan (2) test set berisi fakta yang tidak pernah diajarkan.

**Perbaikan:**

| | Versi 1 | Versi 2 |
|---|---|---|
| Data training | 487 QA, 1 kalimat per fakta | ±2.300 contoh: 607 fakta × (asli + 3 parafrase) |
| Test set | fakta baru (tidak diajarkan) | parafrase baru dari fakta yang sudah diajarkan |
| LoRA rank / alpha | 16 / 16 | 32 / 32 |
| Batch efektif | 8 | 16 |
| Paparan tiap fakta | ±3 kali | ±12 kali |

**Langkah:**
1. Buat file fakta: `python scripts/buat_qa_bersih.py data/raw/qa_draft.csv data/raw/qa_manual.csv` lalu upload `data/processed/qa_bersih.jsonl` ke `MyDrive/proyek-llm/data/processed/`.
2. Jalankan `notebooks/parafrase_v2_colab.ipynb` (T4, ±30–40 menit) untuk membuat `train_v2/val_v2/test_v2.jsonl` di Drive. Setelah selesai, putuskan runtime.
3. Jalankan `notebooks/finetune_v2_colab.ipynb` (T4, ±40–60 menit). Output ada di `MyDrive/proyek-llm/outputs_v2/`.
4. Download folder `outputs_v2/lora_adapter` ke `outputs_v2/lora_adapter/` di laptop. **Jangan pakai GGUF Q4_K_M dari Colab** (lihat temuan di bawah).
5. Ekspor ulang di laptop (CPU, ±5 menit, tanpa GPU). Butuh venv dengan `torch` (CPU), `transformers`, `peft`, `gguf`, `safetensors`:
   ```bash
   ollama pull qwen2.5:3b-instruct-fp16
   # base FP16 = blob GGUF model tersebut (lihat: ollama show qwen2.5:3b-instruct-fp16 --modelfile)
   python scripts/merge_lora_lokal.py --base <folder GGUF FP16> --gguf <nama file .gguf> \
       --config <folder config.json Qwen2.5-3B-Instruct> --nf4 \
       --adapter outputs_v2/lora_adapter --out D:/llm-export/merged_v2_nf4
   python scripts/tulis_gguf.py --base-gguf <file GGUF FP16> --merged D:/llm-export/merged_v2_nf4 \
       --out deploy/chatbot-kampus-v2-q8_0.gguf --tipe q8_0
   cd deploy && ollama create chatbot-kampus-v2 -f Modelfile.v2
   ```
6. `FINETUNED_MODEL = "chatbot-kampus-v2"` di `scripts/config.py`, lalu evaluasi semua versi:
   ```bash
   python eval/eval_versi.py --test data/processed/test_v2.jsonl --model qwen2.5:3b chatbot-kampus chatbot-kampus-v2-q4 chatbot-kampus-v2
   ```

### Temuan penting: akurasi hilang saat ekspor (detail: `docs/v2/temuan_kuantisasi.md`)

Di Colab, model v2 sudah hafal fakta (eval loss 0,109, token F1 0,80). Namun GGUF **Q4_K_M** hasil ekspor Unsloth hanya mendapat F1 ±0,53 di Ollama. Penyebabnya ada dua:
1. Kuantisasi 4-bit membulatkan koreksi kecil dari LoRA.
2. LoRA dilatih di atas base **NF4** (QLoRA), tetapi saat ekspor di-merge ke base 16-bit asli yang sedikit berbeda.

Perbaikannya adalah membulatkan base ke NF4 dulu (`--nf4`), lalu merge, lalu simpan sebagai **Q8_0**.

**Hasil akhir** (`eval/ringkasan_versi.md`, 68 soal `test_v2`, greedy, dijalankan di Ollama laptop):

| Model | Token F1 | Angka benar | Waktu/jawaban |
|---|---|---|---|
| `qwen2.5:3b` (sebelum fine-tuning) | 0,208 | 0/18 (0%) | 6,0 s |
| `chatbot-kampus` (v1, Q4_K_M) | 0,372 | 3/18 (17%) | 2,8 s |
| `chatbot-kampus-v2-q4` (v2, ekspor Q4_K_M Colab) | 0,526 | 4/18 (22%) | 3,1 s |
| **`chatbot-kampus-v2`** (v2, base NF4 + Q8_0) | **0,798** | **15/18 (83%)** | 4,6 s |

Catatan: `test_v2` berisi parafrase dari fakta yang diajarkan di v2. Sebagian fakta itu tidak ada di data training v1, jadi skor v1 sedikit dirugikan. Pertanyaan dengan kalimat yang sangat berbeda dari data training masih bisa dijawab salah, misalnya "Berapa persen maksimal pengakuan SKS lewat RPL?" dijawab 50%, padahal yang benar 70%.

Hasil versi 1 (`docs/loss_curve.png`, `docs/training_log.csv`, `docs/perbandingan_sebelum_sesudah.csv`) tetap disimpan sebagai eksperimen pertama untuk laporan.
