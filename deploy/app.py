"""UI Chatbot Regulasi Akademik (Streamlit + Ollama).

Jalankan dari folder proyek (Ollama harus aktif, model sudah dibuat dengan `ollama create`):
    pip install streamlit requests
    streamlit run deploy/app.py

Dua mode:
  - Tanya jawab : ngobrol dengan model hasil fine-tuning
  - Bandingkan  : satu pertanyaan dijawab model dasar vs model fine-tuned (untuk demo)
Tema warna ada di .streamlit/config.toml. Sengaja tanpa logo/warna institusi
(hindari isu hak cipta merek).
"""
import json
import sys
import time
from pathlib import Path

import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from config import BASE_MODEL, FINETUNED_MODEL, NAMA_KAMPUS, OLLAMA_URL, SYSTEM_PROMPT  # noqa: E402

# Pertanyaan contoh: kalimat PERSIS ini sudah diuji dan dijawab benar oleh model v2 (temperature 0).
# Kalau diubah sedikit saja, uji ulang dulu: model sensitif terhadap susunan kalimat.
CONTOH = [
    (":material/menu_book:", "Kalau IPS saya 2,8, semester depan boleh ambil berapa SKS?"),
    (":material/grade:", "Kalau nilai skor 76, nilai hurufnya apa?"),
    (":material/schedule:", "Masa studi maksimal S1 berapa semester?"),
    (":material/event_busy:", "Cuti maksimal berapa semester?"),
]
MODE_CHAT, MODE_BANDING = "Chat", "Bandingkan"


def stream_chat(model, messages, system=None):
    """Generator token dari Ollama /api/chat."""
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    with requests.post(f"{OLLAMA_URL}/api/chat", stream=True, timeout=600,
                       json={"model": model, "messages": msgs, "stream": True,
                             # temperature 0: jawaban konsisten (sama dengan setelan evaluasi)
                             "options": {"temperature": 0}}) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                data = json.loads(line)
                yield data.get("message", {}).get("content", "")
                if data.get("done"):
                    break


def cek_ollama():
    try:
        return [m["name"] for m in requests.get(f"{OLLAMA_URL}/api/tags", timeout=3).json()["models"]]
    except requests.RequestException:
        return None


st.set_page_config(page_title="Asisten Regulasi Akademik", page_icon=":material/auto_stories:",
                   layout="centered")

st.html("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  html, body, .stApp, .stMarkdown, p, li, button, input, textarea, label { font-family: 'Inter', system-ui, sans-serif; }
  [data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded' !important; }

  :root { --aksen: #4F46E5; --aksen-muda: #EEF0FF; --abu: #71717A; --garis: #E4E4E7; }

  /* rapikan bawaan Streamlit */
  [data-testid="stAppDeployButton"], #MainMenu, footer, header [data-testid="stDecoration"] { display: none; }
  .block-container { padding-top: 1.6rem; padding-bottom: 7rem; max-width: 780px; }
  [data-testid="stSidebar"] { background: #FAFAFA; border-right: 1px solid var(--garis); }

  /* bar atas */
  .brand { display: flex; align-items: center; gap: .6rem; font-weight: 600; font-size: 1rem; color: #18181B; }
  .brand .logo { width: 30px; height: 30px; border-radius: 9px; display: grid; place-items: center;
                 background: linear-gradient(135deg, #6366F1, #4F46E5); color: #fff; font-size: 16px; }
  .brand .versi { font-size: .7rem; font-weight: 600; color: var(--aksen); background: var(--aksen-muda);
                  padding: .12rem .45rem; border-radius: 999px; }

  /* hero (layar kosong) */
  .hero { text-align: center; padding: 3.2rem 0 1.6rem; }
  .hero h1 { font-size: 2.1rem !important; font-weight: 700 !important; letter-spacing: -.02em;
             margin: 0 0 .5rem !important; padding: 0 !important; color: #18181B !important;
             font-family: 'Inter', system-ui, sans-serif !important; }
  .hero .sorot { background: linear-gradient(90deg, #4F46E5, #9333EA); -webkit-background-clip: text;
                  background-clip: text; color: transparent; }
  .hero p { color: var(--abu); font-size: 1rem; margin: 0 auto; max-width: 520px; line-height: 1.6; }

  /* kartu contoh pertanyaan */
  div[data-testid="stButton"] button[kind="secondary"] {
      justify-content: flex-start; text-align: left; border-radius: 14px; border: 1px solid var(--garis);
      background: #fff; color: #27272A; min-height: 4.2rem; height: 100%; padding: .8rem 1rem;
      box-shadow: 0 1px 2px rgba(0,0,0,.04); transition: all .15s ease; }
  div[data-testid="stButton"] button[kind="secondary"]:hover {
      border-color: #C7D2FE; background: #FAFAFF; color: #18181B; transform: translateY(-1px);
      box-shadow: 0 6px 16px rgba(79,70,229,.08); }
  div[data-testid="stButton"] button[kind="secondary"] p,
  div[data-testid="stButton"] button[kind="secondary"] div {
      white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
      text-align: left; line-height: 1.4; font-size: .9rem; }
  div[data-testid="stButton"] button[kind="secondary"] span[data-testid="stIconMaterial"] { color: var(--aksen); }

  /* chat: pengguna di kanan (gelembung), asisten di kiri (polos) */
  [data-testid="stChatMessage"] { background: transparent !important; padding: .35rem 0; gap: .75rem; }
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { flex-direction: row-reverse; }
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageAvatarUser"] { display: none; }
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
      background: var(--aksen); color: #fff; border-radius: 18px 18px 4px 18px; padding: .6rem 1rem;
      max-width: 78%; margin-left: auto !important; margin-right: 0 !important; flex-grow: 0; }
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] p { color: #fff; }
  [data-testid="stChatMessageAvatarCustom"] { background: var(--aksen-muda) !important; color: var(--aksen) !important;
      border-radius: 10px !important; }
  [data-testid="stChatMessageContent"] p { line-height: 1.65; }

  /* input kapsul */
  [data-testid="stChatInput"] > div { border-radius: 999px !important; border: 1px solid var(--garis) !important;
      box-shadow: 0 8px 24px rgba(24,24,27,.06); background: #fff !important; }
  [data-testid="stChatInput"] textarea { padding-left: .6rem; }

  /* kartu perbandingan */
  .kartu-kepala { margin-bottom: .6rem; }
  [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMarkdownContainer"] p { font-size: .93rem; line-height: 1.6; }
  [data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px !important; }
  .lencana { font-size: .72rem; font-weight: 600; padding: .2rem .6rem; border-radius: 999px; }
  .lencana.abu { background: #F4F4F5; color: #52525B; }
  .lencana.ungu { background: var(--aksen-muda); color: var(--aksen); }
  .kartu-model { display: block; margin-top: .35rem; font-family: ui-monospace, Consolas, monospace;
                 font-size: .72rem; color: #A1A1AA; }
  .waktu { display: inline-block; margin-top: .6rem; font-size: .75rem; color: var(--abu);
           background: #F4F4F5; padding: .15rem .55rem; border-radius: 999px; }

  /* sidebar */
  .judul-sb { font-size: .72rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
              color: #A1A1AA; margin: 1.2rem 0 .5rem; }
  .status { display: flex; align-items: center; gap: .5rem; font-size: .85rem; color: #3F3F46; margin: .3rem 0; }
  .titik { width: 8px; height: 8px; border-radius: 50%; }
  .catatan { color: #A1A1AA; font-size: .75rem; text-align: center; margin-top: 1rem; }
</style>
""")

# ---------- sidebar ----------
models = cek_ollama()
with st.sidebar:
    if st.button("Percakapan baru", icon=":material/edit_square:", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.rerun()

    st.markdown('<div class="judul-sb">Tips bertanya</div>', unsafe_allow_html=True)
    st.markdown(
        "- Sebutkan **jenjangnya** (S1, D3, magister).\n"
        "- Satu pertanyaan, satu topik.\n"
        "- Tulis pertanyaan lengkap. Chatbot belum bisa menyambung ke pertanyaan sebelumnya.")

    st.markdown('<div class="judul-sb">Status model</div>', unsafe_allow_html=True)
    if models is None:
        st.error("Ollama belum berjalan. Buka aplikasi Ollama, lalu muat ulang halaman ini.")
        st.stop()
    for m, ket in ((FINETUNED_MODEL, "fine-tuned"), (BASE_MODEL, "dasar")):
        ada = (m if ":" in m else f"{m}:latest") in models
        warna = "#22C55E" if ada else "#EF4444"
        st.markdown(f'<div class="status"><span class="titik" style="background:{warna}"></span>'
                    f'<code>{m}</code><span style="color:#A1A1AA">· {ket}</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="judul-sb">Tentang</div>', unsafe_allow_html=True)
    st.caption(f"Sumber data: Pedoman Akademik {NAMA_KAMPUS} 2024. Qwen 2.5 3B Instruct, "
               "fine-tuning QLoRA, berjalan lokal dengan Ollama.")

# ---------- bar atas ----------
kiri, kanan = st.columns([5, 3], vertical_alignment="center")
with kiri:
    st.markdown('<div class="brand"><div class="logo">&#10022;</div>Asisten Regulasi Akademik'
                '<span class="versi">v2</span></div>', unsafe_allow_html=True)
with kanan:
    mode = st.segmented_control("Mode", [MODE_CHAT, MODE_BANDING], default=MODE_CHAT,
                                label_visibility="collapsed", key="mode") or MODE_CHAT

AVATAR = {"user": None, "assistant": ":material/auto_awesome:"}  # user: avatar bawaan (disembunyikan lewat CSS)

# ---------- mode tanya jawab ----------
if mode == MODE_CHAT:
    st.session_state.setdefault("messages", [])
    pilihan = None

    if not st.session_state.messages:
        st.markdown('<div class="hero"><h1>Ada yang ingin <span class="sorot">ditanyakan?</span></h1>'
                    '<p>Tanya apa saja seputar peraturan akademik: beban SKS, masa studi, cuti, nilai, '
                    'sampai syarat kelulusan.</p></div>', unsafe_allow_html=True)
        k1, k2 = st.columns(2, gap="small")
        for i, (ikon, q) in enumerate(CONTOH):
            if (k1 if i % 2 == 0 else k2).button(q, icon=ikon, key=f"contoh{i}", use_container_width=True):
                pilihan = q

    for m in st.session_state.messages:
        st.chat_message(m["role"], avatar=AVATAR[m["role"]]).markdown(m["content"])

    prompt = st.chat_input("Tanyakan sesuatu tentang peraturan akademik...") or pilihan
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        if pilihan:  # sembunyikan layar sapaan setelah contoh diklik
            st.rerun()
        st.chat_message("user", avatar=AVATAR["user"]).markdown(prompt)

    # jawab pesan pengguna terakhir yang belum dijawab
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant", avatar=AVATAR["assistant"]):
            # SYSTEM sudah tertanam di Modelfile, jadi tidak perlu dikirim lagi
            jawaban = st.write_stream(stream_chat(FINETUNED_MODEL, st.session_state.messages))
        st.session_state.messages.append({"role": "assistant", "content": jawaban})

    st.markdown('<p class="catatan">Jawaban dibuat oleh AI dan bisa keliru. Untuk keputusan penting, '
                'cek kembali pedoman akademik resmi atau bagian akademik.</p>', unsafe_allow_html=True)

# ---------- mode bandingkan ----------
else:
    st.markdown('<div class="hero" style="padding:2rem 0 1rem"><h1>Sebelum vs <span class="sorot">sesudah</span></h1>'
                '<p>Pertanyaan yang sama dikirim ke model dasar dan model hasil fine-tuning, '
                'dengan system prompt yang sama.</p></div>', unsafe_allow_html=True)
    with st.form("bandingkan", border=False):
        q = st.text_input("Pertanyaan", value=CONTOH[0][1], label_visibility="collapsed",
                          placeholder="Tulis pertanyaan...")
        kirim = st.form_submit_button("Bandingkan jawaban", type="primary", icon=":material/compare_arrows:",
                                      use_container_width=True)

    if kirim and q:
        kol_kiri, kol_kanan = st.columns(2, gap="medium")
        for kol, model, judul, kelas in ((kol_kiri, BASE_MODEL, "Sebelum", "abu"),
                                         (kol_kanan, FINETUNED_MODEL, "Sesudah", "ungu")):
            with kol, st.container(border=True):
                st.markdown(f'<div class="kartu-kepala"><span class="lencana {kelas}">{judul}</span>'
                            f'<span class="kartu-model">{model}</span></div>', unsafe_allow_html=True)
                t0 = time.time()
                st.write_stream(stream_chat(model, [{"role": "user", "content": q}], system=SYSTEM_PROMPT))
                st.markdown(f'<span class="waktu">{time.time() - t0:.1f} detik</span>', unsafe_allow_html=True)
