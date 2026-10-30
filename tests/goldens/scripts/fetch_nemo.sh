#!/bin/sh
# Fetch the .nemo archive of Parakeet TDT 0.6B v3 at the pinned revision into its own directory and check it.
set -eu
REV=541d1f99c6b0c3cd0b11a95167540bb8edefd82b
OUT=${DATA:?set DATA to the data folder}/models/parakeet-tdt-0.6b-v3-nemo
mkdir -p "$OUT"
curl -fL --retry 5 --retry-delay 5 -C - -o "$OUT/parakeet-tdt-0.6b-v3.nemo" \
  "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/$REV/parakeet-tdt-0.6b-v3.nemo"
echo "3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d  $OUT/parakeet-tdt-0.6b-v3.nemo" | sha256sum -c
stat -c '%s %n' "$OUT/parakeet-tdt-0.6b-v3.nemo"
echo done
