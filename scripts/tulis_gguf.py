"""Versi 2: tulis GGUF (Q8_0 / F16) dari model hasil merge (scripts/merge_lora_lokal.py).

Metadata (tokenizer, chat template, hyperparameter) disalin dari GGUF FP16 resmi
(`ollama pull qwen2.5:3b-instruct-fp16`), sedangkan bobotnya diganti dengan bobot hasil merge.
Dipakai karena llama.cpp (GitHub) tidak bisa diunduh dan impor safetensors Ollama
belum mendukung Qwen2.

    python scripts/tulis_gguf.py --base-gguf D:/llm-export/base_fp16/qwen2.5-3b-instruct-f16.gguf \
        --merged D:/llm-export/merged_v2 --out deploy/chatbot-kampus-v2-q8_0.gguf --tipe q8_0
"""
import argparse
import time
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter, LlamaFileType
from gguf.quants import quantize
from safetensors import safe_open

KE_HF = {
    "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj", "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm",
}


def nama_hf(nama_gguf):
    if nama_gguf == "token_embd.weight":
        return "model.embed_tokens.weight"
    if nama_gguf == "output_norm.weight":
        return "model.norm.weight"
    if nama_gguf == "output.weight":
        return "lm_head.weight"
    _, i, jenis, akhir = nama_gguf.split(".")
    return f"model.layers.{i}.{KE_HF[jenis]}.{akhir}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-gguf", required=True)
    ap.add_argument("--merged", required=True, help="folder berisi model.safetensors hasil merge")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tipe", choices=["q8_0", "f16"], default="q8_0")
    ap.add_argument("--nama", default="chatbot-kampus-v2")
    args = ap.parse_args()
    t0 = time.time()

    base = GGUFReader(args.base_gguf)
    arch = base.fields["general.architecture"].contents()
    w = GGUFWriter(args.out, arch)

    # 1. salin metadata
    for f in base.fields.values():
        if f.name.startswith("GGUF.") or f.name in ("general.architecture", "general.file_type", "general.name"):
            continue
        vtype = GGUFValueType(f.types[0])
        sub = GGUFValueType(f.types[-1]) if vtype == GGUFValueType.ARRAY else None
        w.add_key_value(f.name, f.contents(), vtype, sub_type=sub)
    w.add_name(args.nama)
    w.add_file_type(LlamaFileType.MOSTLY_Q8_0 if args.tipe == "q8_0" else LlamaFileType.MOSTLY_F16)

    # 2. bobot hasil merge, urutan & tipe mengikuti GGUF asli (norm/bias tetap F32)
    st = safe_open(str(Path(args.merged) / "model.safetensors"), framework="pt")
    for t in base.tensors:
        x = st.get_tensor(nama_hf(t.name)).float().numpy()
        assert x.shape == tuple(t.data.shape), (t.name, x.shape, t.data.shape)
        if t.tensor_type == GGMLQuantizationType.F32:
            w.add_tensor(t.name, x.astype(np.float32))
        elif args.tipe == "f16":
            w.add_tensor(t.name, x.astype(np.float16))
        else:
            w.add_tensor(t.name, quantize(x, GGMLQuantizationType.Q8_0), raw_dtype=GGMLQuantizationType.Q8_0)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()
    print(f"Selesai: {args.out} ({Path(args.out).stat().st_size / 1e9:.2f} GB) dalam {time.time() - t0:.0f} detik")


if __name__ == "__main__":
    main()
