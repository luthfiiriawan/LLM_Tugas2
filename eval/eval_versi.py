"""Evaluasi beberapa model Ollama pada test set yang sama (token F1 + ketepatan angka).

    python eval/eval_versi.py --test data/processed/test_v2.jsonl \
        --model qwen2.5:3b chatbot-kampus chatbot-kampus-v2-q4 chatbot-kampus-v2

Output: eval/hasil_versi.csv (per soal) dan eval/ringkasan_versi.md (tabel ringkas).
Semua model dijalankan greedy (temperature 0) dengan SYSTEM_PROMPT yang sama.
"""
import argparse
import csv
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from config import OLLAMA_URL, SYSTEM_PROMPT  # noqa: E402


def tok(s):
    return re.findall(r"\w+", s.lower())


def f1(pred, ref):
    a, b = tok(pred), tok(ref)
    c = sum(min(a.count(w), b.count(w)) for w in set(a))
    return 0.0 if not c else 2 * c / (len(a) + len(b))


def angka(s):
    return set(re.findall(r"\d+(?:[.,]\d+)?", s))


def tanya(model, q):
    body = json.dumps({"model": model, "stream": False, "keep_alive": "10m",
                       "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}],
                       "options": {"temperature": 0, "num_predict": 200}}).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", body, {"Content-Type": "application/json"})
    t0 = time.time()
    out = json.loads(urllib.request.urlopen(req, timeout=600).read())["message"]["content"].strip()
    return out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/processed/test_v2.jsonl")
    ap.add_argument("--model", nargs="+", required=True)
    ap.add_argument("--out", default="eval/hasil_versi.csv")
    ap.add_argument("--ringkasan", default="eval/ringkasan_versi.md")
    args = ap.parse_args()

    soal = [json.loads(l) for l in open(args.test, encoding="utf-8")]
    baris = [{"pertanyaan": d["messages"][1]["content"], "referensi": d["messages"][2]["content"]} for d in soal]
    ring = []
    for m in args.model:                       # model per model agar tidak bolak-balik load
        s_f1 = s_t = 0.0
        n_ang = ok_ang = 0
        for i, b in enumerate(baris, 1):
            jwb, dt = tanya(m, b["pertanyaan"])
            sk = f1(jwb, b["referensi"])
            b[f"{m} | jawaban"], b[f"{m} | f1"] = jwb, round(sk, 3)
            ref_ang = angka(b["referensi"])
            if ref_ang:
                n_ang += 1
                ok = ref_ang <= angka(jwb)
                ok_ang += ok
                b[f"{m} | angka_benar"] = ok
            s_f1 += sk
            s_t += dt
            print(f"[{m}] {i}/{len(baris)} f1={sk:.2f} {dt:.1f}s", flush=True)
        ring.append((m, s_f1 / len(baris), ok_ang, n_ang, s_t / len(baris)))

    kolom = list(baris[0].keys())
    for b in baris:
        kolom += [k for k in b if k not in kolom]
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=kolom)
        w.writeheader()
        w.writerows(baris)

    teks = [f"# Ringkasan evaluasi ({args.test}, {len(baris)} soal, greedy)", "",
            "| Model | Token F1 rata-rata | Angka benar | Waktu/jawaban |", "|---|---|---|---|"]
    teks += [f"| `{m}` | {f:.3f} | {a}/{n} ({a / max(n, 1):.0%}) | {t:.1f} s |" for m, f, a, n, t in ring]
    Path(args.ringkasan).write_text("\n".join(teks) + "\n", encoding="utf-8")
    print("\n".join(teks))


if __name__ == "__main__":
    main()
