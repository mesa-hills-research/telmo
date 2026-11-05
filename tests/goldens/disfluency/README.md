# Disfluency tagger goldens

`tokenizer-vectors.json` holds 202 made-up strings and the token ids that the tokenizer of Ettin encoder 68M (jhu-clsp/ettin-encoder-68m, revision ac19ae4bc51093b31c475665ac872a936d056cc2) gives them, from Hugging Face tokenizers 0.23.3. `tests/disfluency_goldens.rs` checks Telmo's tokenizer against them.

## Licence

The ids come from Ettin's tokenizer, under the MIT License, Copyright (c) 2025 Johns Hopkins University Center for Language and Speech Processing. NOTICE at the repository's root has the licence's full text.
