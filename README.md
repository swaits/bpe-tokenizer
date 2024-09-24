# bpe-tokenizer

A Rust implementation of Byte Pair Encoding (BPE) tokenization. This crate
provides functionality to tokenize text into subword units using pre-trained
vocabularies. BPE is widely used in natural language processing (NLP) tasks,
where it breaks down words into subword tokens using a vocabulary of the most
frequent token pairs.

It supports **Unicode-aware** text segmentation for sentence and word splitting,
making it suitable for processing a variety of languages and scripts.

## Features

- **Bring your own BPE token vocabularies**, or use ...
- **Pre-trained multilingual vocabularies** sourced from the [BPEmb](https://github.com/bheinzerling/bpemb) project, with support for tokenizing text in **275 languages**.
- **Unicode-aware sentence and word segmentation**: Leveraging the [`unicode-segmentation`](https://docs.rs/unicode-segmentation) crate for proper text splitting.

## Installation

To add this crate to your project, run:

```bash
cargo add bpe-tokenizer
```

Or manually include it in your `Cargo.toml`:

```toml
[dependencies]
bpe-tokenizer = "<version>"
```

## Full Example

Here is an example of how to create a `BytePairEncoder` from a string and use it
to tokenize text:

```rust
use bpe_tokenizer::{BytePairEncoder, BytePairEncoderError};

let vocab = BytePairEncoder::new_from_str("hello\t1\nworld\t2").unwrap();
let tokenized = vocab.tokenize("Hello, world!");
println!("{:?}", tokenized);
```

The output will be a vector of tokens:

```text
["<s>", "▁hello", "▁world", "</s>"]
```

Or load a vocabulary from a file:

```rust
use bpe_tokenizer::{BytePairEncoder, BytePairEncoderError};
let vocab = BytePairEncoder::new_from_file("path/to/file.vocab").unwrap();
```

## Cargo Features

The crate also includes several sizes of default pre-trained vocabularies, which
are **optional** and can be enabled via Cargo features. They are sourced from
Wikipedia data, pre-trained as part of the
[BPEmb](https://github.com/bheinzerling/bpemb) project. These MIT-licensed
vocabularies support 275 languages and provide different sizes depending on
usage needs:

### Available Optional Features

- **`default-small` (100,000 tokens)**: Suitable for memory-constrained environments.
- **`default-medium` (320,000 tokens)**: Balances between token coverage and memory efficiency.
- **`default-large` (1,000,000 tokens)**: Provides the most detailed token representations for high granularity tasks.

### Enabling Optional Features

To use these default vocabularies, specify the feature in your `Cargo.toml`:

```toml
[dependencies]
bpe-tokenizer = { version = "<version>", features = ["default-medium"] }
```

### Example with `default-medium` Vocabulary

An example of using the **medium** vocabulary (320,000 tokens):

```rust
# #[cfg(feature = "default-medium")] {
use bpe_tokenizer::{BytePairEncoder, BytePairEncoderError};

let encoder = BytePairEncoder::new_default_medium().unwrap();
let tokenized = encoder.tokenize("This is a test sentence.");
println!("{:?}", tokenized);
// Output: ["<s>", "▁this", "▁is", "▁a", "▁test", "▁sentence", "</s>"]
# }
```

## Tokenization Functions

The crate provides various ways to interact with the tokenizer:

- **Tokenize into a flat `Vec<String>`**:

  - `BytePairEncoder::tokenize`

  Splits and flattens the text into tokens.

  ```rust
  let tokenized = vocab.tokenize("Example sentence.");
  // Output: ["<s>", "▁example", "▁sentence", "</s>"]
  ```

- **Tokenize into nested sentence vectors `Vec<Vec<String>>`**:

  - `BytePairEncoder::tokenize_sentences`

  Useful for processing multiple sentences separately.

  ```rust
  let tokenized = vocab.tokenize_sentences("This is sentence one. And this is sentence two.");
  // Output: [["<s>", "▁this", "▁is", "▁sentence", "▁one", "</s>"], ["<s>", "▁and", "▁this", "▁is", "▁sentence", "▁two", "</s>"]]
  ```

- **Iterative tokenization**:

  - `BytePairEncoder::tokenize_iter` and `BytePairEncoder::tokenize_sentences_iter`

  Provides an iterator over generated tokens for better memory efficiency in
  large-scale text.

  ```rust
  let tokens_iter: Vec<String> = vocab.tokenize_iter("Example sentence").collect();
  // Output: ["<s>", "▁example", "▁sentence", "</s>"]
  ```

## Licensing

This crate is licensed under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please open an issue, submit a pull request, or reach
out if you'd like to contribute awesome new features or fixes to this crate.
