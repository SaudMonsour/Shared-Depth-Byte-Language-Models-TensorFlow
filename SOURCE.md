# Sources and provenance

Corpus: Tiny Shakespeare, distributed by Andrej Karpathy in char-rnn.
Pinned raw URL: https://raw.githubusercontent.com/karpathy/char-rnn/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt
Git blob SHA-1: 7dcb3a2d4cc3b48b6283dd46870bfeb78f88aac9
SHA-256: 86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed

The complete downloaded byte stream was checked against the source Git blob, decoded strictly as UTF-8, and counted before splitting. No data snapshot is included.

The underlying Shakespeare works are public-domain historical texts. The upstream repository tree has no standalone license file, and the Hugging Face dataset card leaves licensing metadata unspecified; no MIT license is asserted for the corpus compilation. Reference: https://huggingface.co/datasets/karpathy/tiny_shakespeare/blob/main/README.md .

Architecture references:
- Dehghani et al., Universal Transformers (ICLR 2019): https://arxiv.org/abs/1807.03819
- Xue et al., ByT5: Towards a token-free future with pre-trained byte-to-byte models (TACL 2022): https://arxiv.org/abs/2105.13626

The implementation is independently written TensorFlow code. It uses fixed-depth causal weight sharing, not adaptive computation or the ByT5 encoder-decoder architecture. Training uses no pretrained weights or paid language-model API.
