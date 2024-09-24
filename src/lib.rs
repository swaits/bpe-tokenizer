//! # A Byte Pair Encoding (BPE) tokenizer implementation.
//!
//! This module provides functionality for [BPE
//! tokenization](https://en.wikipedia.org/wiki/Byte_pair_encoding), a text tokenization technique
//! that iteratively replaces the most frequent pair of bytes in a sequence with a single, unused
//! byte. In natural language processing, it's used to break down words into subword
//! tokens.
//!
//! This implementation does not start with bytes and iteratively replace them with pairs as
//! described above. Instead, it uses a pre-trained token vocabulary to identify the most frequent
//! pairs.
//!
//! Text input for tokenization is first split into sentences, which are then split into words.
//! All sentence and word splitting is Unicode-aware through the functionality provided by the
//! [`unicode-segmentation`](https://docs.rs/unicode-segmentation) crate. Next, each word (`&str`)
//! is tokenized into a vector of tokens (`Vec<String>`) as follows:
//!
//! 1. Iterate through possible substrings of the word, from longest to shortest.
//! 1. For each substring length, find any matching token in the vocabulary.
//! 1. Choose the matching token with the highest score in the vocabulary.
//! 1. Split the word at the chosen token and recursively tokenize the parts before and after it.
//!
//! ##  Main Components
//!
//! ### Initialization
//!
//! A `BytePairEncoder` is created from a pre-trained token vocabulary file. You can find
//! MIT-licensed vocabulary files at the [BPEmb](https://github.com/bheinzerling/bpemb) project.
//!
//! Initialization can be done in two ways:
//!
//! - [`BytePairEncoder::new_from_file`]: Create a `BytePairEncoder` from a file.
//! - [`BytePairEncoder::new_from_str`]: Create a `BytePairEncoder` from a string.
//!
//! The crate also includes default token vocabularies which support 275 languages. These are
//! disabled by default and can be enabled with the "default-{small,medium,large}" features.
//!
//! - [`BytePairEncoder::new_default_small`]: Create a `BytePairEncoder` for the default small
//!   model (100k vocabulary).
//! - [`BytePairEncoder::new_default_medium`]: Create a `BytePairEncoder` for the default medium
//!   model (320k vocabulary).
//! - [`BytePairEncoder::new_default_large`]: Create a `BytePairEncoder` for the default large
//!   model (1M vocabulary).
//!
//! For more information on these, see the **Features** section below.
//!
//! ### Tokenization into `Vec<String>` or `Vec<Vec<String>>`
//!
//! Once you have a `BytePairEncoder`, you can use the following associated functions to tokenize
//! text into vectors of tokens:
//!
//! - [`BytePairEncoder::tokenize`]: Tokenize text into a flat vector of BPE tokens.
//! - [`BytePairEncoder::tokenize_sentences`]: Tokenize text into nested vectors of sentences and tokens.
//!
//! ### Tokenization via Iterators
//!
//! Alternatively, you can use the following associated functions to tokenize text into iterators:
//!
//! - [`BytePairEncoder::tokenize_iter`]: Tokenize text into a flat sequence of BPE tokens.
//! - [`BytePairEncoder::tokenize_sentences_iter`]: Tokenize text into nested sentences and tokens.
//!
//! ##  Example
//!
//! ```
//! use bpe_tokenizer::{BytePairEncoder, BytePairEncoderError};
//!
//! let vocab = BytePairEncoder::new_from_str("hello\t1\nworld\t2").unwrap();
//! let tokenized = vocab.tokenize("Hello, world!");
//! ```
//!
//! ## Features
//!
//! This crate offers the following optional features that can be enabled via Cargo features in
//! your `Cargo.toml`. Depending on your application, you can choose a default vocabulary size for
//! the `BytePairEncoder` to work with multilingual tokens. The default vocabularies are
//! pre-trained on wikipedia data by the [BPEmb](https://github.com/bheinzerling/bpemb) project,
//! providing multilingual tokenization support for 275 languages.
//!
//! ### `default-small` (100,000 tokens):
//! - Enables construction of `BytePairEncoder` with a smaller vocabulary size of 100,000 tokens.
//! - Suitable for memory-constrained environments and simpler tasks where fine-grained
//!   tokenization is less necessary.
//!
//!   Example of enabling this in your `Cargo.toml`:
//!   ```toml
//!   [dependencies]
//!   bpe-tokenizer = { version = "<version", features = ["default-small"] }
//!   ```
//!
//! ### `default-medium` (320,000 tokens):
//! - Enables construction of `BytePairEncoder` with a vocabulary size of 320,000 tokens.
//! - Provides a balance between vocabulary size and memory usage, making it suitable for a
//!   broader range of tasks.
//!
//!   Example of enabling this in your `Cargo.toml`:
//!   ```toml
//!   [dependencies]
//!   bpe-tokenizer = { version = "<version", features = ["default-medium"] }
//!   ```
//!
//! ### `default-large` (1,000,000 tokens):
//! - Enables construction of `BytePairEncoder` with a vocabulary size of 1,000,000 tokens.
//! - Ideal for tasks that require high token coverage, providing the most detailed token
//!   representations at the expense of additional memory usage.
//!
//!   Example of enabling this in your `Cargo.toml`:
//!   ```toml
//!   [dependencies]
//!   bpe-tokenizer = { version = "<version>", features = ["default-large"] }
//!   ```
//!
//! The vocabulary size directly impacts the granularity of the tokenization and memory
//! consumption, so choose based on your application's needs.
//!
//! ### Example with Default Vocabularies
//!
//! ```rust
//! # #[cfg(feature = "default-medium")] {
//! use bpe_tokenizer::{BytePairEncoder, BytePairEncoderError};
//!
//! let encoder = BytePairEncoder::new_default_medium().unwrap();
//! let tokenized = encoder.tokenize("This is a test sentence.");
//! assert_eq!(tokenized[0], "<s>".to_string());
//! # }
//! ```
//!
//! Note that when multiple features are enabled, the respective `new_default_*` functions (e.g.,
//! [`BytePairEncoder::new_default_small`], [`BytePairEncoder::new_default_medium`],
//! [`BytePairEncoder::new_default_large`]) become available for constructing a `BytePairEncoder`.
//! Only enable the features that you need to ensure minimized memory and binary size.

mod constants;
mod default_vocabs;
mod errors;
mod tokenizer;

// tests
#[cfg(test)]
mod tests;

// re-exports
pub use errors::BytePairEncoderError;
pub use tokenizer::BytePairEncoder;
