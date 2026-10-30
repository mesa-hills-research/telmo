# Parakeet TDT 0.6B v3 goldens

Reference outputs of NVIDIA's Parakeet TDT 0.6B v3 for six LibriSpeech clips, saved from NVIDIA NeMo 3.0.0 on the CPU in float32. Each clip has its audio as a 16 kHz mono 16-bit WAV and a safetensors file with the features, the subsampling output, the outputs of Conformer layers 0, 1, 12 and 23, the joint's projections, the first 20 joint outputs and the tokens of greedy decoding. `manifest.json` lists the clips, the model revision, the package versions, the commands and a comparison with the transformers port. The scripts that made the files are in `../scripts/`.

## Licences

The audio and the tensors are both under the Creative Commons Attribution 4.0 International licence (https://creativecommons.org/licenses/by/4.0/).

- Audio from the LibriSpeech ASR corpus, (c) 2014 Vassil Panayotov, https://www.openslr.org/12, CC BY 4.0. The clips are converted from FLAC to WAV, and `clean-joined-40s` joins six consecutive utterances.
- Tensors derived from the model nvidia/parakeet-tdt-0.6b-v3 by NVIDIA, https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3 (revision 541d1f99c6b0c3cd0b11a95167540bb8edefd82b), CC BY 4.0. They are the model's outputs on these clips.
