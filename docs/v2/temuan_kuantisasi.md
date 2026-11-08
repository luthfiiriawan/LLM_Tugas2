# Temuan: akurasi v2 hilang saat ekspor GGUF Q4_K_M (9 November 2026)

## Hasil training v2 (di Colab, model 4-bit + LoRA)

| Metrik | v1 | v2 |
|---|---|---|
| Eval loss terbaik | 1,342 | **0,109** |
| Train loss akhir | ±0,72 | ±0,04 |
| Token F1 test (sebelum → sesudah fine-tuning) | – | 0,202 → **0,798** |
| Angka benar di jawaban (test) | – | 14/18 |

## Setelah diekspor ke GGUF Q4_K_M dan dijalankan di Ollama

12 pertanyaan test pertama, greedy (temperature 0):

| | Token F1 rata-rata |
|---|---|
| Model di Colab | 0,824 |
| GGUF Q4_K_M di Ollama | **0,547** |

Contoh pertanyaan yang **ada persis di data training**:

| Pertanyaan | Kunci | v1 (Q4) | v2 (Q4) |
|---|---|---|---|
| IPS 2,8, boleh ambil berapa SKS? | 20 SKS | 15 SKS | 24 SKS |
| Masa studi maksimal S-1? | 12 semester (14 paruh waktu) | 12 semester | 8 / 10 semester |
| Pengakuan SKS maksimal lewat RPL? | 70% | 50% | 50% |
| Resep nasi goreng (di luar topik) | menolak | menolak ✓ | menolak ✓ |

## Diagnosis

1. **Template prompt bukan penyebab.** Prompt ChatML mentah (`raw: true`) memberi jawaban yang sama persis.
2. **File berbeda dari v1.** Hash SHA-256 GGUF v1 dan v2 berbeda.
3. **Penyebab paling mungkin:** LoRA dilatih di atas base 4-bit (QLoRA). Saat ekspor, adapter di-merge ke base 16-bit lalu dikompres lagi ke 4-bit (Q4_K_M). Perubahan bobot hasil LoRA yang kecil, yaitu "hafalan fakta", ikut hilang terbulatkan. Gaya bahasa dan penolakan topik lain tetap bertahan karena polanya lebih kuat.

## Perbaikan (dikerjakan di laptop, CPU)

Hugging Face CDN dan GitHub tidak bisa diakses dari jaringan laptop, jadi base FP16 diambil lewat `ollama pull qwen2.5:3b-instruct-fp16`.

| Percobaan | IPS 2,8 (kunci 20 SKS) | Masa studi S-1 (kunci 12/14) | F1 (12 soal) |
|---|---|---|---|
| Model Colab (base 4-bit + LoRA) | – | – | 0,824 |
| Ekspor Unsloth Q4_K_M | 24 SKS ❌ | 8/10 ❌ | 0,547 |
| Merge ke base 16-bit, Q8_0 | 16 SKS ❌ | 8/10 ❌ | 0,758 |
| **Base dibulatkan ke NF4 dulu, merge, lalu Q8_0** | **20 SKS ✓** | **12/14 ✓** (Python) | **0,857** |

Kesimpulannya, ada dua penyebab yang saling menumpuk:
1. **Kuantisasi Q4** membulatkan koreksi kecil dari LoRA.
2. **Ketidakcocokan base.** LoRA belajar mengoreksi base NF4 (QLoRA), tetapi saat ekspor dipasang ke base 16-bit asli.

Pipeline akhir:
- `scripts/merge_lora_lokal.py --nf4` (base FP16 → dibulatkan ke NF4 → merge LoRA → safetensors)
- `scripts/tulis_gguf.py --tipe q8_0` (metadata dari GGUF FP16 resmi + bobot hasil merge → GGUF Q8_0, 3,29 GB)
- `ollama create chatbot-kampus-v2 -f deploy/Modelfile.v2`

Model Q4 lama tetap tersimpan sebagai `chatbot-kampus-v2-q4` untuk perbandingan. Ekspor ulang Q8_0 di Colab tanpa langkah NF4 tidak dipakai, karena hasilnya setara baris ketiga tabel.

## Hasil evaluasi akhir (68 soal test_v2, Ollama, greedy)

| Model | Token F1 | Angka benar | Waktu/jawaban |
|---|---|---|---|
| `qwen2.5:3b` (base) | 0,208 | 0/18 | 6,0 s |
| `chatbot-kampus` (v1) | 0,372 | 3/18 | 2,8 s |
| `chatbot-kampus-v2-q4` | 0,526 | 4/18 | 3,1 s |
| **`chatbot-kampus-v2`** | **0,798** | **15/18** | 4,6 s |

Detail per soal: `eval/hasil_versi.csv`.

**Pelajaran untuk laporan:** evaluasi harus dilakukan pada **model yang benar-benar di-deploy** (GGUF di Ollama), bukan hanya di notebook training.
