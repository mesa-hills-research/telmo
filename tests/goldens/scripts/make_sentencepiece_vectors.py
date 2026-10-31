"""Write sentencepiece-vectors.json for Parakeet. It holds words with the token ids the model's own SentencePiece model
(tokenizer.model inside NVIDIA's parakeet-tdt-0.6b-v3.nemo) splits them into. Words with digits are where
Hugging Face's converted tokenizer.json differs from it.

Usage:
  python -I make_sentencepiece_vectors.py <tokenizer.model> <goldens dir>
"""

import json
import os
import sys

import sentencepiece

TEXTS = [
    "k8s", "K8s", "x86", "x86_64", "aarch64", "web3", "i18n", "a11y", "l10n", "3D", "2nd", "v2", "42", "2026",
    "utf-8", "ipv6", "mp3", "h264", "s3", "ec2", "py3", "gpt4", "1password",
    "kubectl", "nginx", "serde", "tokio", "Fauchelevent", "PostgreSQL", "café", "rock'n'roll",
]


def main():
    model, out_dir = sys.argv[1], sys.argv[2]
    sp = sentencepiece.SentencePieceProcessor(model_file=model)
    vectors = [{"text": t, "ids": sp.encode(t)} for t in TEXTS]
    path = os.path.join(out_dir, "parakeet-tdt-0.6b-v3", "sentencepiece-vectors.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"sentencepiece": sentencepiece.__version__, "vectors": vectors}, f, ensure_ascii=False, indent=0)
        f.write("\n")
    print(path, len(vectors))


main()
