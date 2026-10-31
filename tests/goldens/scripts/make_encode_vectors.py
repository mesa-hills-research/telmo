"""Write encode-vectors.json for both models. It holds words and phrases with the token ids Hugging Face's
tokenizers library splits them into, for checking the crate's BPE encoder.

Usage:
  DATA=<data folder> python -I make_encode_vectors.py <goldens dir>

DATA holds models/parakeet-tdt-0.6b-v3/ and models/granite-speech-5.0-470m-turboctc/.
"""

import json
import os
import sys
import unicodedata

import tokenizers

DATA = os.environ.get("DATA") or sys.exit("set DATA to the data folder")
MODELS = ["parakeet-tdt-0.6b-v3", "granite-speech-5.0-470m-turboctc"]

# Technical terms, names, common words, mixed case, digits, punctuation, accents, phrases and
# characters the vocabularies lack (which fall back to bytes).
TEXTS = """
kubectl Kubectl nginx Nginx mcp MCP serde Serde tokio Tokio rustc clippy rustfmt cargo Cargo
PyTorch pytorch TensorFlow numpy NumPy pandas matplotlib FastAPI fastapi Django django Flask
GitHub github GitLab gitlab Kubernetes kubernetes k8s K8s Docker docker dockerfile Dockerfile
PostgreSQL postgres Postgres MySQL sqlite SQLite Redis redis MongoDB mongodb GraphQL graphql
WebAssembly wasm Wasm WebGPU wgpu WGSL Vulkan vulkan OpenGL CUDA cuda ROCm Metal
JSON json YAML yaml TOML toml XML HTML CSS JavaScript javascript TypeScript typescript
async await impl struct enum trait mut fn usize u32 i64 f32 Vec HashMap BTreeMap Arc Mutex
stdout stderr stdin argv env PATH HOME sudo chmod chown grep sed awk xargs tmux zsh bash
Granite Parakeet Whisper gpui ratatui crossterm anyhow thiserror Ollama llama vLLM
Raskolnikov Bartley Tiberius Hilda Dorcas Gwynplaine Leocadia Rodolfo Cuthbert Mrs Mr Dr St
the The THE a A and of to in is it you that he was for on are with as I his they be at one
hello world Hello World don't Don't it's I'm you're O'Brien rock'n'roll self-driving e-mail
3D 2nd 42 2026 v2 x86_64 aarch64 utf-8 UTF-8 i18n a11y l10n
Zürich zürich naïve café Café résumé Ångström Øresund Łódź São Paulo Straße straße Müller
Привет мир Αθήνα 東京 日本語 こんにちは 한국어 🙂 ✓ — … § ½
new york New York Visual Studio Code hugging face Hugging Face
""".split("\n")


def main():
    out_dir = sys.argv[1]
    texts = [w for line in TEXTS for w in line.split()]
    phrases = ["New York", "Visual Studio Code", "Hugging Face", "São Paulo", "rock'n'roll self-driving",
               "the quick brown fox", "Привет мир"]
    for model in MODELS:
        tok = tokenizers.Tokenizer.from_file(os.path.join(DATA, "models", model, "tokenizer.json"))
        # The crate doesn't apply the files' NFKC-like normalization, so texts it changes are left out.
        chosen = [t for t in texts + phrases if unicodedata.normalize("NFKC", t) == t]
        if model.startswith("parakeet"):
            # Parakeet's file lists the digits as added tokens, and the library then starts a new
            # word after each one ("k8s" becomes "k8 s"). The model was trained on SentencePiece's
            # split, which goes on with the word, as the crate does, so texts with digits are left out.
            chosen = [t for t in chosen if not any(c.isdigit() for c in t)]
        vectors = [{"text": t, "ids": tok.encode(t, add_special_tokens=False).ids} for t in chosen]
        path = os.path.join(out_dir, model, "encode-vectors.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"tokenizers": tokenizers.__version__, "vectors": vectors}, f, ensure_ascii=False, indent=0)
            f.write("\n")
        print(path, len(vectors))


main()
