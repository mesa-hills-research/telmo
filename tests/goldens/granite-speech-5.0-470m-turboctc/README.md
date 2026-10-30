# Granite Speech 5.0 470M TurboCTC goldens

Reference outputs of IBM's Granite Speech 5.0 470M TurboCTC for five LibriSpeech clips and one Japanese FLEURS clip, saved from transformers' `GraniteSpeech5ForCTC` on the CPU in float32. Each clip has a safetensors file with the features, the outputs of encoder blocks 0, 1, 7 and 15, the self-conditioning injection after block 7, the CTC argmax of every frame with its logit, the full logits of two frames and the tokens. The 40 s clip and the FLEURS clip keep fewer tensors, as `manifest.json` lists. `tokenizer-vectors.json` holds decoded text for every token id and for 400 random id sequences.

The LibriSpeech audio is the WAVs of `../parakeet-tdt-0.6b-v3/`. `manifest.json` lists the clips, the model revision, the package versions and the commands, and the scripts that made the files are in `../scripts/`.

## Licences

- Audio from the LibriSpeech ASR corpus, (c) 2014 Vassil Panayotov, https://www.openslr.org/12, CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
- Audio from FLEURS (google/fleurs, https://huggingface.co/datasets/google/fleurs, revision 70bb2e84b976b7e960aa89f1c648e09c59f894dd), CC BY 4.0. The clip is the dataset's float WAV converted to 16-bit PCM.
- Tensors and tokenizer vectors derived from the model ibm-granite/granite-speech-5.0-470m-turboctc by IBM, https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc (revision 1dd23f46f6d4e6eae2531982ba2e33220086dd97), Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0). They are the model's outputs on these clips and its tokenizer's decoded text.
