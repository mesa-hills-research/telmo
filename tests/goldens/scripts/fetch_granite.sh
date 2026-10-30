#!/bin/sh
# Fetch the Granite Speech 5.0 470M TurboCTC files at a pinned revision into $DATA/models/granite-speech-5.0-470m-turboctc and check them.
set -eu
REV=1dd23f46f6d4e6eae2531982ba2e33220086dd97
BASE=https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc/resolve/$REV
OUT=${DATA:?set DATA to the data folder}/models/granite-speech-5.0-470m-turboctc
mkdir -p "$OUT"
for f in config.json tokenizer.json tokenizer_config.json processor_config.json generation_config.json model.safetensors; do
  echo "fetching $f"
  curl -fsSL --retry 5 --retry-delay 5 -C - -o "$OUT/$f" "$BASE/$f"
done
cd "$OUT"
sha256sum -c <<EOF
5b569c3744c42a9172ac976881539926b66176d03fbaaa3c51b5d2960067f6fc  model.safetensors
a1c3ca763ce0690faf0df3456365358547acfdf766122c9e4a69be8e8708042e  tokenizer.json
aabd64a2da736a1a1c21237c896b71628f6dd7364f1f5ec8a77f5e04b3ef1011  config.json
2f61ece3468bf7ca7a1b6dd3fd5b809814315bea9c298feea035fda4931b9f02  tokenizer_config.json
3d7c14d3ed7a5218b2fd6797eaa5b3aa615c681938bd1734a510d7a6f8c1b9fb  processor_config.json
ad32e8b87734e621ede4a52e64072a90b42b1599973a7c008efc1268541ea034  generation_config.json
EOF
sha256sum model.safetensors tokenizer.json config.json tokenizer_config.json processor_config.json generation_config.json > ../granite-speech-5.0-470m-turboctc.sha256
stat -c '%s %n' model.safetensors tokenizer.json config.json tokenizer_config.json processor_config.json generation_config.json >> ../granite-speech-5.0-470m-turboctc.sha256
echo done
