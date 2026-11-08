"""Versi 2: gabungkan adapter LoRA ke base model 16-bit di laptop (CPU), tanpa Colab.

Alasan: ekspor GGUF Q4_K_M dari Colab kehilangan banyak akurasi fakta
(lihat docs/v2/temuan_kuantisasi.md). Di sini adapter di-merge ke base bf16,
disimpan sebagai safetensors, lalu Ollama mengkuantisasinya sendiri ke Q8_0:

    python scripts/merge_lora_lokal.py --adapter outputs_v2/lora_adapter --out D:/llm-export/merged_v2
    ollama create chatbot-kampus-v2 -q q8_0 -f deploy/Modelfile.v2-safetensors

Butuh: torch (CPU), transformers, peft, safetensors. RAM ±8 GB, disk ±7 GB.
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from config import SYSTEM_PROMPT  # noqa: E402

UJI = [
    ("Kalau IPS saya 2,8, semester depan boleh ambil berapa SKS?", "20 SKS"),
    ("Masa studi maksimal untuk S-1 itu berapa sih?", "12 semester"),
]


GGUF_KE_HF = {  # nama tensor GGUF (llama.cpp) -> nama parameter Qwen2 di transformers
    "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj", "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm",
}


def muat_base_dari_gguf(path_gguf, config_dir):
    """Baca bobot FP16 dari GGUF (tanpa dekuantisasi/kemasan) ke Qwen2ForCausalLM biasa (bf16)."""
    from gguf import GGUFReader
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(config_dir)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    sd = {}
    for t in GGUFReader(str(path_gguf)).tensors:
        if t.name == "token_embd.weight":
            nama = "model.embed_tokens.weight"
        elif t.name == "output_norm.weight":
            nama = "model.norm.weight"
        elif t.name == "output.weight":
            nama = "lm_head.weight"
        else:
            _, i, jenis, akhir = t.name.split(".")      # blk.<i>.<jenis>.<weight|bias>
            nama = f"model.layers.{i}.{GGUF_KE_HF[jenis]}.{akhir}"
        sd[nama] = torch.from_numpy(t.data.copy()).to(torch.bfloat16)
    if "lm_head.weight" not in sd:                          # Qwen2.5-3B: embedding di-tie
        sd["lm_head.weight"] = sd["model.embed_tokens.weight"]
    model.load_state_dict(sd, strict=True, assign=True)
    model.tie_weights()
    # buffer non-bobot (RoPE inv_freq) masih di device "meta": buat ulang di CPU
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
    model.model.rotary_emb = Qwen2RotaryEmbedding(config=config)
    sisa = [n for n, b in list(model.named_parameters()) + list(model.named_buffers()) if b.is_meta]
    assert not sisa, f"masih meta: {sisa[:5]}"
    print(f"   {len(sd)} tensor dimuat dari GGUF")
    return model


# Kode NF4 bitsandbytes (QLoRA, Dettmers dkk. 2023)
NF4 = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
                    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
                    0.07958029955625534, 0.16093020129203796, 0.24611230194568634, 0.33791524171829224,
                    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0])


@torch.no_grad()
def bulatkan_nf4(model, blok=64):
    """Ganti bobot Linear di tiap layer dengan versi NF4 yang sudah di-dequantize.

    LoRA v2 dilatih di atas base 4-bit (QLoRA), jadi koreksinya cocok untuk base NF4,
    bukan base 16-bit asli. Dengan membulatkan base ke NF4 dulu, hasil merge meniru
    model persis seperti saat training/evaluasi di Colab.
    """
    batas = (NF4[1:] + NF4[:-1]) / 2
    n = 0
    for layer in model.model.layers:
        for lin in [m for m in layer.modules() if isinstance(m, torch.nn.Linear)]:
            w = lin.weight.data
            x = w.float().reshape(-1, blok)
            absmax = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
            idx = torch.bucketize((x / absmax).contiguous(), batas)
            lin.weight.data = (NF4[idx] * absmax).reshape(w.shape).to(w.dtype)
            n += 1
    print(f"   {n} matriks Linear dibulatkan ke NF4 (blok {blok})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="unsloth/Qwen2.5-3B-Instruct",
                    help="repo Hugging Face, atau folder berisi file GGUF FP16 (pakai --gguf)")
    ap.add_argument("--gguf", default=None,
                    help="nama file GGUF FP16 di folder --base, misal dari `ollama pull qwen2.5:3b-instruct-fp16` "
                         "(dipakai bila CDN Hugging Face tidak bisa diakses)")
    ap.add_argument("--config", default=None, help="folder berisi config.json Qwen2.5-3B-Instruct (wajib bila --gguf)")
    ap.add_argument("--adapter", default="outputs_v2/lora_adapter")
    ap.add_argument("--out", default="D:/llm-export/merged_v2")
    ap.add_argument("--tanpa-uji", action="store_true")
    ap.add_argument("--nf4", action="store_true",
                    help="bulatkan base ke NF4 sebelum merge (meniru base 4-bit saat QLoRA training)")
    args = ap.parse_args()

    t0 = time.time()
    print("Memuat base model 16-bit:", args.base)
    if args.gguf:
        model = muat_base_dari_gguf(Path(args.base) / args.gguf, args.config)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if args.nf4:
        bulatkan_nf4(model)
    print("Memasang adapter:", args.adapter)
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    tok = AutoTokenizer.from_pretrained(args.adapter)

    if not args.tanpa_uji:
        model.eval()
        for q, kunci in UJI:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}]
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_dict=True, return_tensors="pt")
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=70, do_sample=False)
            ans = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            print(f"[uji] {q}\n      kunci: {kunci}\n      jawab: {ans}")

    print("Menyimpan ke", args.out)
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    print(f"Selesai dalam {time.time() - t0:.0f} detik")


if __name__ == "__main__":
    main()
