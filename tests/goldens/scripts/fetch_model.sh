#!/bin/sh
# Fetch the Parakeet TDT 0.6B v3 files at a pinned revision into $DATA/models/parakeet-tdt-0.6b-v3 and check them.
set -eu
REV=541d1f99c6b0c3cd0b11a95167540bb8edefd82b
BASE=https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/$REV
OUT=${DATA:?set DATA to the data folder}/models/parakeet-tdt-0.6b-v3
mkdir -p "$OUT"
for f in tokenizer.json config.json generation_config.json processor_config.json tokenizer_config.json model.safetensors; do
  echo "fetching $f"
  curl -fL --retry 5 --retry-delay 5 -C - -o "$OUT/$f" "$BASE/$f"
done
cd "$OUT"
sha256sum -c <<EOF
3a2026366188c8c68598edbbff92f8d11590a08e0ae2e6775544e7b07d6a5e11  model.safetensors
bd321b096832a3f270bd3b2a88823957920f1a5c5ada71114a26ea729d0cbe91  tokenizer.json
e747b85e1bdfd300c8b8ac63bac8dd5221f8fe9bc275b48d06c735fcd6971b6e  config.json
b141de6ec6d7f982ece13f98f604e3fe1807ea9c0e839185d0ab7064604209d0  generation_config.json
8346a93a3b987fa1dec57a78f045cd0817d21786589a5a096b41a57a446fd1d7  processor_config.json
0b2fe0037599ee335f0b972fa682bf0ece74e4ccfec755cb7daa3405d3d3e874  tokenizer_config.json
EOF
echo done
