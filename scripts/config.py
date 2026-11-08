"""Konfigurasi bersama untuk seluruh script proyek.

PENTING: SYSTEM_PROMPT harus SAMA persis dengan yang dipakai saat training
(notebook Colab) dan di deploy/Modelfile. Kalau diubah di sini, ubah juga di sana.
"""

NAMA_KAMPUS = "Universitas Telkom"

SYSTEM_PROMPT = (
    f"Kamu adalah asisten layanan informasi akademik {NAMA_KAMPUS}. "
    "Jawab pertanyaan mahasiswa tentang peraturan dan regulasi akademik kampus "
    "dengan bahasa Indonesia yang sopan, singkat, dan jelas. "
    "Jika informasinya tidak ada dalam peraturan, katakan bahwa kamu tidak tahu "
    "dan sarankan menghubungi bagian akademik."
)

OLLAMA_URL = "http://127.0.0.1:11434"
BASE_MODEL = "qwen2.5:3b"            # model sebelum fine-tuning (dari library Ollama)
FINETUNED_MODEL = "chatbot-kampus-v2"   # v2 (Q8_0, merge di base NF4). Model v1 tetap ada: "chatbot-kampus"
