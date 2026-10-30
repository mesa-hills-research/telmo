#!/bin/sh
# Fetch LibriSpeech test-clean and test-other (OpenSLR 12, CC BY 4.0), each into its own new directory,
# check the md5 that OpenSLR publishes, and extract in place.
set -eu
DL=${DATA:?set DATA to the data folder}/dl
for set in test-clean:32fa31d27d2e1cad72775fee3f4849a9 test-other:fb5a50374b501bb3bac4815ee91d3135; do
  name=${set%%:*}; md5=${set##*:}
  dir=$DL/librispeech-$name
  mkdir -p "$dir"
  curl -fsSL --retry 5 --retry-delay 5 -C - -o "$dir/$name.tar.gz" "https://www.openslr.org/resources/12/$name.tar.gz"
  echo "$md5  $dir/$name.tar.gz" | md5sum -c
  tar xzf "$dir/$name.tar.gz" -C "$dir" --no-same-owner --no-same-permissions
  echo "extracted $name"
done
echo done
