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

use std::{collections::HashMap, fs, iter};

use thiserror::Error;
use unicode_segmentation::UnicodeSegmentation;

#[cfg(any(
    feature = "default-small",
    feature = "default-medium",
    feature = "default-large"
))]
use lz4_flex::decompress_size_prepended;

/// Represents errors that can occur during BPE tokenization operations.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum BytePairEncoderError {
    /// Indicates an error occurred while reading the vocabulary file.
    #[error("Error reading file: {0}")]
    InvalidFile(String),

    /// Indicates that the vocabulary input was invalid or could not be parsed correctly.
    #[error("Invalid vocabulary input: Could not parse vocabulary file.")]
    InvalidVocabularyInput,

    /// Indicates an error occurred during decompression of the vocabulary data.
    #[error("Error decompressing vocabulary data: {0}")]
    DecompressionError(String),

    /// Indicates an error occurred during deserialization of the vocabulary data.
    #[error("Error deserializing vocabulary data: {0}")]
    DeserializationError(String),
}

/// The character used to denote word breaks in the tokenized output.
const WORD_BREAK_CHAR: &str = "▁";

/// The token used to mark the start of a sentence.
const SENTENCE_START_TOKEN: &str = "<s>";

/// The token used to mark the end of a sentence.
const SENTENCE_END_TOKEN: &str = "</s>";

/// The token used to represent unknown words or subwords.
const UNKNOWN_TOKEN: &str = "<unk>";

/// Token vocabulary data for the default small model.
#[cfg(feature = "default-small")]
const DEFAULT_SMALL_DATA: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/",
    "multi.wiki.bpe.vs100000.vocab.hashmap.bincode.lz4"
));

/// Token vocabulary data for the default small model.
#[cfg(feature = "default-medium")]
const DEFAULT_MEDIUM_DATA: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/",
    "multi.wiki.bpe.vs320000.vocab.hashmap.bincode.lz4"
));

/// Token vocabulary data for the default small model.
#[cfg(feature = "default-large")]
const DEFAULT_LARGE_DATA: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/",
    "multi.wiki.bpe.vs1000000.vocab.hashmap.bincode.lz4"
));

/// # Represents a Byte Pair Encoding (BPE) vocabulary used for tokenization.
///
/// This struct holds the mapping of tokens to their respective scores (or IDs)
/// and provides methods for tokenizing text using the BPE algorithm.
///
/// The vocabulary is typically loaded from a file or string where each line
/// contains a token and its score, separated by a tab character.
///
/// ## Example
///
/// ```
/// use bpe_tokenizer::BytePairEncoder;
///
/// let vocab = BytePairEncoder::new_from_str("hello\t1\nworld\t2").unwrap();
/// let tokenized = vocab.tokenize("Hello, world!");
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BytePairEncoder {
    /// # A mapping of tokens to their respective scores (or IDs).
    ///
    /// In BPE, tokens with lower scores (or IDs) are typically more common
    /// and are preferred during the tokenization process.
    tokens: HashMap<String, isize>,
}

impl BytePairEncoder {
    /// # Creates a new `BytePairEncoder` from a file containing token-score pairs.
    ///
    /// This function reads the contents of the file specified by `file_path` and constructs
    /// a `BytePairEncoder` from it. The file should contain token-score pairs, with each pair
    /// on a separate line and the token and score separated by a tab character (`\t`).
    ///
    /// ## Input Format
    ///
    /// The file is expected to follow this format:
    ///
    /// ```text
    /// <token>\t<score>\n
    /// ```
    ///
    /// Each line should consist of:
    /// * A token (a string) followed by a tab character (`\t`)
    /// * A score (an integer) as either a positive or negative value.
    ///
    /// Example lines from the file:
    ///
    /// ```text
    /// <unk>    0
    /// ▁t       -0
    /// ▁the     -4
    /// ```
    ///
    /// ## Arguments
    ///
    /// * `file_path` - A string slice that holds the path to the file containing token-score pairs.
    ///
    /// ## Returns
    ///
    /// * `Result<Self, BytePairEncoderError>` - A Result containing the created `BytePairEncoder` if successful,
    ///   or a `BytePairEncoderError` if there was an error reading the file or parsing its contents.
    ///
    /// ## Errors
    ///
    /// This function will return an error if:
    /// * The file cannot be read (returns `BytePairEncoderError::InvalidFile`)
    /// * The file contents are not in the expected format (returns `BytePairEncoderError::InvalidVocabularyInput`)
    ///
    /// ## Example
    ///
    /// ```
    /// use bpe_tokenizer::BytePairEncoder;
    ///
    /// let vocab = BytePairEncoder::new_from_file("path/to/vocabulary/file.txt");
    /// ```
    pub fn new_from_file(file_path: &str) -> Result<Self, BytePairEncoderError> {
        Self::new_from_str(
            fs::read_to_string(file_path)
                .map_err(|_| BytePairEncoderError::InvalidFile(file_path.to_string()))?
                .as_ref(),
        )
    }

    /// # Creates a new `BytePairEncoder` from a string containing token-score pairs.
    ///
    /// This function parses the input string to construct a `BytePairEncoder`. The input should
    /// contain token-score pairs, with each pair on a separate line and the token and score
    /// separated by a tab character (`\t`).
    ///
    /// ## Input Format
    ///
    /// The string must follow this format:
    ///
    /// ```text
    /// <token>\t<score>\n
    /// ```
    ///
    /// Each line in the string should consist of:
    /// * A token (a string) followed by a tab character (`\t`)
    /// * A score (an integer) as either a positive or negative value.
    ///
    /// For example:
    ///
    /// ```text
    /// hello   1
    /// world   2
    /// ▁the    -4
    /// ```
    ///
    /// ## Arguments
    ///
    /// * `input` - A string slice that holds the token-score pairs.
    ///
    /// ## Returns
    ///
    /// * `Result<Self, BytePairEncoderError>` - A Result containing the created `BytePairEncoder` if successful,
    ///   or a `BytePairEncoderError` if there was an error parsing the input.
    ///
    /// ## Errors
    ///
    /// This function will return `BytePairEncoderError::InvalidVocabularyInput` if:
    /// * A line doesn't contain a tab character to separate token and score.
    /// * The score cannot be parsed as an `isize`.
    ///
    /// ## Example
    ///
    /// ```
    /// use bpe_tokenizer::BytePairEncoder;
    ///
    /// let input = "hello\t1\nworld\t2";
    /// let vocab = BytePairEncoder::new_from_str(input).unwrap();
    /// ```
    pub fn new_from_str(input: &str) -> Result<Self, BytePairEncoderError> {
        let mut tokens = HashMap::new();

        for line in input.lines() {
            let (token, score_str) = match line.split_once('\t') {
                Some(pair) => pair,
                None => return Err(BytePairEncoderError::InvalidVocabularyInput),
            };
            let score = match score_str.parse::<isize>() {
                Ok(score) => score,
                Err(_) => return Err(BytePairEncoderError::InvalidVocabularyInput),
            };
            tokens.insert(token.to_string(), score);
        }

        Ok(BytePairEncoder { tokens })
    }

    /// # Initializes a new `BytePairEncoder` from given compressed vocabulary data.
    ///
    /// This private function decompresses and deserializes a pre-trained multilingual vocabulary that supports
    /// 275 languages. The compressed vocabulary is expected to be in LZ4 format. It is then decompressed and
    /// deserialized into a `HashMap<String, isize>`, which is used to tokenize text.
    ///
    /// The function handles any decompression or deserialization failures by returning a `BytePairEncoderError`.
    ///
    /// ## Arguments
    ///
    /// * `data` - A reference to a binary slice holding compressed LZ4 vocabulary data.
    ///
    /// ## Returns
    ///
    /// A `Result<Self, BytePairEncoderError>` where on success, a `BytePairEncoder` instance is returned, and
    /// on failure, a `BytePairEncoderError` indicates what went wrong.
    ///
    /// ## Errors
    ///
    /// * `DecompressionError`: Returned if decompression of the LZ4-based vocabulary data fails.
    /// * `DeserializationError`: Returned if deserialization of the decompressed data fails.
    #[cfg(any(
        feature = "default-small",
        feature = "default-medium",
        feature = "default-large"
    ))]
    fn new_default(data: &'static [u8]) -> Result<Self, BytePairEncoderError> {
        // Decompress the binary data
        let uncompressed = decompress_size_prepended(data)
            .map_err(|e| BytePairEncoderError::DecompressionError(e.to_string()))?;

        // Deserialize the uncompressed data into a HashMap
        let tokens: HashMap<String, isize> = bincode::deserialize(&uncompressed)
            .map_err(|e| BytePairEncoderError::DeserializationError(e.to_string()))?;

        // Successfully create a BytePairEncoder
        Ok(Self { tokens })
    }

    /// # Creates a new `BytePairEncoder` with a default small vocabulary size (100,000 tokens).
    ///
    /// This function constructs a `BytePairEncoder` using a pre-trained multilingual vocabulary
    /// that supports 275 languages. The vocabulary is sourced from the
    /// [BPEmb](https://github.com/bheinzerling/bpemb) project, licensed under MIT. The small-sized
    /// vocabulary file consists of 100,000 tokens, allowing for highly compressed tokenization
    /// suitable for tasks with limited memory constraints.
    ///
    /// ## Returns
    ///
    /// A `Result<Self, BytePairEncoderError>`, constructing the `BytePairEncoder` on successful
    /// vocabulary loading, or a corresponding error if initialization fails.
    ///
    /// ## Example
    ///
    /// ```
    /// use bpe_tokenizer::BytePairEncoder;
    ///
    /// let encoder = BytePairEncoder::new_default_small().unwrap();
    /// ```
    ///
    /// ## Note
    ///
    /// This is only enabled when the `default-small` feature is enabled in Cargo.toml.
    ///
    ///   ```toml
    ///   [dependencies]
    ///   bpe-tokenizer = { version = "<version", features = ["default-small"] }
    ///   ```

    #[cfg(feature = "default-small")]
    pub fn new_default_small() -> Result<Self, BytePairEncoderError> {
        Self::new_default(DEFAULT_SMALL_DATA)
    }

    /// # Creates a new `BytePairEncoder` with a default medium vocabulary size (320,000 tokens).
    ///
    /// This function constructs a `BytePairEncoder` using a pre-trained multilingual vocabulary
    /// that supports 275 languages. The vocabulary is sourced from the
    /// [BPEmb](https://github.com/bheinzerling/bpemb) project, licensed under MIT. The
    /// medium-sized vocabulary file consists of 320,000 tokens, offering a balance between token
    /// coverage and memory efficiency, making it suitable for a wide variety of NLP tasks.
    ///
    /// ## Returns
    ///
    /// A `Result<Self, BytePairEncoderError>`, constructing the `BytePairEncoder` on successful
    /// vocabulary loading, or a corresponding error if initialization fails.
    ///
    /// ## Example
    ///
    /// ```
    /// use bpe_tokenizer::BytePairEncoder;
    ///
    /// let encoder = BytePairEncoder::new_default_medium().unwrap();
    /// ```
    ///
    /// ## Note
    ///
    /// This is only enabled when the `default-medium` feature is enabled in Cargo.toml.
    ///
    ///   ```toml
    ///   [dependencies]
    ///   bpe-tokenizer = { version = "<version", features = ["default-medium"] }
    ///   ```
    #[cfg(feature = "default-medium")]
    pub fn new_default_medium() -> Result<Self, BytePairEncoderError> {
        Self::new_default(DEFAULT_MEDIUM_DATA)
    }

    /// # Creates a new `BytePairEncoder` with a default large vocabulary size (1,000,000 tokens).
    ///
    /// This function constructs a `BytePairEncoder` using a pre-trained multilingual vocabulary
    /// that supports 275 languages. The vocabulary is sourced from the
    /// [BPEmb](https://github.com/bheinzerling/bpemb) project, licensed under MIT. The large-sized
    /// vocabulary consists of 1,000,000 tokens, providing maximum coverage for detailed language
    /// representation, especially useful in applications requiring high granularity.
    ///
    /// ## Returns
    ///
    /// A `Result<Self, BytePairEncoderError>`, constructing the `BytePairEncoder` on successful
    /// vocabulary loading, or a corresponding error if initialization fails.
    ///
    /// ## Example
    ///
    /// ```
    /// use bpe_tokenizer::BytePairEncoder;
    ///
    /// let encoder = BytePairEncoder::new_default_large().unwrap();
    /// ```
    ///
    /// ## Note
    ///
    /// This is only enabled when the `default-large` feature is enabled in Cargo.toml.
    ///
    ///   ```toml
    ///   [dependencies]
    ///   bpe-tokenizer = { version = "<version", features = ["default-large"] }
    ///   ```
    #[cfg(feature = "default-large")]
    pub fn new_default_large() -> Result<Self, BytePairEncoderError> {
        Self::new_default(DEFAULT_LARGE_DATA)
    }

    /// # Tokenizes a text into sentences, then words, and finally into BPE tokens.
    ///
    /// This function takes a string of text and returns an iterator that yields
    /// vectors of tokens, where each vector represents a tokenized sentence.
    ///
    /// ## Arguments
    ///
    /// * `text` - A string slice containing the text to be tokenized.
    ///
    /// ## Returns
    ///
    /// An iterator that yields `Vec<String>`, where each `Vec<String>` represents
    /// a tokenized sentence.
    ///
    /// ## Example
    ///
    /// ```
    /// use bpe_tokenizer::BytePairEncoder;
    ///
    /// let vocab = BytePairEncoder::new_from_str("hello\t1\nworld\t2").unwrap();
    /// let text = "Hello, world! How are you?";
    /// let tokenized: Vec<Vec<String>> = vocab
    ///     .tokenize_sentences_iter(text)
    ///     .map(|sentence_iter| sentence_iter.collect())  // Collect each inner iterator into a Vec<String>
    ///     .collect();  // Then collect everything into Vec<Vec<String>>
    /// ```
    ///
    /// ## Notes
    ///
    /// - This function uses Unicode-aware sentence and word segmentation.
    /// - Each sentence is wrapped with sentence start (`<s>`) and end (`</s>`) tokens.
    /// - Words are prefixed with the word break character (`▁`).
    /// - Unknown tokens are replaced with the `<unk>` token.
    pub fn tokenize_sentences_iter<'a>(
        &'a self,
        text: &'a str,
    ) -> impl Iterator<Item = impl Iterator<Item = String> + 'a> + 'a {
        UnicodeSegmentation::unicode_sentences(text)
            .map(move |sentence| self.tokenize_with_sentence_markers_iter(sentence))
    }

    /// # Tokenizes a text into a flat sequence of BPE tokens.
    ///
    /// This function takes a string of text and returns an iterator that yields
    /// individual tokens. It first tokenizes the text into sentences, then words,
    /// and finally into BPE tokens, flattening the result into a single sequence.
    ///
    /// ## Arguments
    ///
    /// * `text` - A string slice containing the text to be tokenized.
    ///
    /// ## Returns
    ///
    /// An iterator that yields `String`, where each `String` represents a token.
    ///
    /// ## Example
    ///
    /// ```
    /// use bpe_tokenizer::BytePairEncoder;
    ///
    /// let vocab = BytePairEncoder::new_from_str("hello\t1\nworld\t2").unwrap();
    /// let text = "Hello, world! How are you?";
    /// let tokenized: Vec<String> = vocab.tokenize_iter(text).collect();
    /// ```
    ///
    /// ## Notes
    ///
    /// - This function uses Unicode-aware sentence and word segmentation.
    /// - Each sentence is wrapped with sentence start (`<s>`) and end (`</s>`) tokens.
    /// - Words are prefixed with the word break character (`▁`).
    /// - Unknown tokens are replaced with the `<unk>` token.
    pub fn tokenize_iter<'a>(&'a self, text: &'a str) -> impl Iterator<Item = String> + 'a {
        self.tokenize_sentences_iter(text).flatten()
    }

    /// # Tokenizes a text into sentences, then words, and finally into BPE tokens.
    ///
    /// This function takes a string of text and returns a vector of tokenized sentences,
    /// where each sentence is represented as a vector of tokens.
    ///
    /// ## Arguments
    ///
    /// * `text` - A string slice containing the text to be tokenized.
    ///
    /// ## Returns
    ///
    /// A `Vec<Vec<String>>`, where each inner `Vec<String>` represents a tokenized sentence.
    ///
    /// ## Example
    ///
    /// ```
    /// use bpe_tokenizer::BytePairEncoder;
    ///
    /// let vocab = BytePairEncoder::new_from_str("hello\t1\nworld\t2").unwrap();
    /// let text = "Hello, world! How are you?";
    /// let tokenized = vocab.tokenize_sentences(text);
    /// ```
    ///
    /// ## Notes
    ///
    /// - This function uses Unicode-aware sentence and word segmentation.
    /// - Each sentence is wrapped with sentence start (`<s>`) and end (`</s>`) tokens.
    /// - Words are prefixed with the word break character (`▁`).
    /// - Unknown tokens are replaced with the `<unk>` token.
    pub fn tokenize_sentences(&self, text: &str) -> Vec<Vec<String>> {
        self.tokenize_sentences_iter(text)
            .map(|sentence_iter| sentence_iter.collect())
            .collect()
    }

    /// # Tokenizes a text into a flat sequence of BPE tokens.
    ///
    /// This function takes a string of text and returns a vector of tokens.
    /// It first tokenizes the text into sentences, then words, and finally into BPE tokens,
    /// flattening the result into a single sequence.
    ///
    /// ## Arguments
    ///
    /// * `text` - A string slice containing the text to be tokenized.
    ///
    /// ## Returns
    ///
    /// A `Vec<String>`, where each `String` represents a token.
    ///
    /// ## Example
    ///
    /// ```
    /// use bpe_tokenizer::BytePairEncoder;
    ///
    /// let vocab = BytePairEncoder::new_from_str("hello\t1\nworld\t2").unwrap();
    /// let text = "Hello, world! How are you?";
    /// let tokenized = vocab.tokenize(text);
    /// ```
    ///
    /// ## Notes
    ///
    /// - This function uses Unicode-aware sentence and word segmentation.
    /// - Each sentence is wrapped with sentence start (`<s>`) and end (`</s>`) tokens.
    /// - Words are prefixed with the word break character (`▁`).
    /// - Unknown tokens are replaced with the `<unk>` token.
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        self.tokenize_iter(text).collect()
    }

    /// # Tokenizes a single sentence, adding sentence start and end markers.
    ///
    /// This function breaks down the tokenization process for a single sentence:
    /// 1. Adds a sentence start token.
    /// 2. Splits the sentence into words using Unicode-aware word segmentation.
    /// 3. Prepends each word with the word break character.
    /// 4. Tokenizes each word using the BPE vocabulary.
    /// 5. Adds a sentence end token.
    ///
    /// ## Arguments
    ///
    /// * `sentence` - A string slice containing a single sentence to be tokenized.
    ///
    /// ## Returns
    ///
    /// An iterator that yields `String`s representing the tokenized sentence,
    /// including start and end markers.
    ///
    /// ## Implementation Notes
    ///
    /// - Uses `unicode_words` for word segmentation to handle various Unicode scripts correctly.
    /// - Converts words to lowercase before tokenization to match the vocabulary.
    /// - Returns an iterator instead of a fully collected `Vec<String>` to allow for
    ///   more efficient tokenization and processing.
    fn tokenize_with_sentence_markers_iter<'a>(
        &'a self,
        sentence: &'a str,
    ) -> impl Iterator<Item = String> + 'a {
        iter::once(SENTENCE_START_TOKEN.to_string())
            .chain(sentence.unicode_words().flat_map(move |word| {
                self.tokenize_word(&format!("{}{}", WORD_BREAK_CHAR, word.to_lowercase()))
            }))
            .chain(iter::once(SENTENCE_END_TOKEN.to_string()))
    }

    /// # Tokenizes a single word using the Byte Pair Encoding (BPE) algorithm.
    ///
    /// This function implements the core BPE tokenization logic:
    /// 1. If the word is empty, return an empty vector.
    /// 2. Convert the word to a vector of Unicode characters.
    /// 3. Iterate through possible substrings of the word, from longest to shortest.
    /// 4. For each substring length, find all matching tokens in the vocabulary.
    /// 5. Choose the matching token with the highest score in the vocabulary.
    /// 6. Split the word at the chosen token and recursively tokenize the parts before and after.
    /// 7. If no match is found, return the unknown token.
    ///
    /// ## Arguments
    ///
    /// * `text` - A string slice containing a single word to be tokenized.
    ///
    /// ## Returns
    ///
    /// A `Vec<String>` containing the BPE tokens for the input word.
    ///
    /// ## Implementation Notes
    ///
    /// - The algorithm prioritizes longer matches over shorter ones.
    /// - In case of multiple matches of the same length, it chooses the one with the highest score.
    /// - The function is recursive, handling subwords created by splitting at a matched token.
    /// - If no match is found in the vocabulary, it returns the unknown token.
    fn tokenize_word(&self, text: &str) -> Vec<String> {
        // Base case: If the input is empty, return an empty vector
        if text.is_empty() {
            return vec![];
        }

        // Convert the `text` to a Vec of `char`s to index by character rather than byte
        let word: Vec<char> = text.chars().collect();

        // Look for the longest matching token in the vocabulary
        for len in (1..=word.len()).rev() {
            let mut matches = vec![];
            // Iterate over each possible start position for substrings of length `len`
            for start in 0..=(word.len() - len) {
                let end = start + len;

                // Extract candidate substring (convert chars[start..end] back to a &str)
                let candidate = &word[start..end].iter().collect::<String>();

                // If we have an exact match, just store it for now
                if self.tokens.contains_key(candidate) {
                    matches.push((candidate.to_string(), start, end));
                }
            }

            // If we got matches, choose the one with the highest score
            if !matches.is_empty() {
                let (candidate, start, end) = matches
                    .into_iter()
                    .max_by_key(|(candidate, _, _)| {
                        self.tokens.get(candidate).copied().unwrap_or(isize::MIN)
                    })
                    .unwrap();

                // Recursively process the left part (before the match)
                let left: String = word[..start].iter().collect();
                let left_tokens = self.tokenize_word(&left);

                // The middle part is the matched token
                let middle = vec![candidate];

                // Recursively process the right part (after the match)
                let right: String = word[end..].iter().collect();
                let right_tokens = self.tokenize_word(&right);

                // Concatenate the result of left, middle, and right
                return [left_tokens, middle, right_tokens].concat();
            }
        }

        // If no match is found, return <unk> for the whole text
        vec![UNKNOWN_TOKEN.to_string()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;

    #[test]
    fn test_new_valid_file() {
        // Create a temporary file with valid content
        let file_path = "test_vocab.txt";
        let mut file = File::create(file_path).unwrap();
        file.write_all(b"hello\t1\nworld\t2").unwrap();

        // Test the new function
        let result = BytePairEncoder::new_from_file(file_path);
        assert!(result.is_ok());

        let vocab = result.unwrap();
        assert_eq!(vocab.tokens.len(), 2);
        assert_eq!(vocab.tokens.get("hello"), Some(&1));
        assert_eq!(vocab.tokens.get("world"), Some(&2));

        // Clean up the temporary file
        std::fs::remove_file(file_path).unwrap();
    }

    #[test]
    fn test_new_invalid_file() {
        // Test with a non-existent file
        let result = BytePairEncoder::new_from_file("non_existent_file.txt");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            BytePairEncoderError::InvalidFile(_)
        ));
    }

    #[test]
    fn test_new_from_str_valid_input() {
        let input = "hello\t1\nworld\t2\ntest\t3";
        let result = BytePairEncoder::new_from_str(input);

        assert!(result.is_ok());
        let vocab = result.unwrap();

        assert_eq!(vocab.tokens.len(), 3);
        assert_eq!(vocab.tokens.get("hello"), Some(&1));
        assert_eq!(vocab.tokens.get("world"), Some(&2));
        assert_eq!(vocab.tokens.get("test"), Some(&3));
    }

    #[test]
    fn test_new_from_str_empty_input() {
        let input = "";
        let result = BytePairEncoder::new_from_str(input);

        assert!(result.is_ok());
        let vocab = result.unwrap();

        assert_eq!(vocab.tokens.len(), 0);
    }

    #[test]
    fn test_new_from_str_invalid_format() {
        let input = "hello 1\nworld\t2";
        let result = BytePairEncoder::new_from_str(input);

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            BytePairEncoderError::InvalidVocabularyInput
        );
    }

    #[test]
    fn test_new_from_str_invalid_score() {
        let input = "hello\t1\nworld\tabc";
        let result = BytePairEncoder::new_from_str(input);

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            BytePairEncoderError::InvalidVocabularyInput
        );
    }

    #[test]
    #[cfg(feature = "default-small")]
    fn test_new_default_small_with_tokenization() {
        // Initialize the BytePairEncoder with the default small vocabulary
        let result = BytePairEncoder::new_default_small();
        assert!(result.is_ok());

        let vocab = result.unwrap();
        assert!(!vocab.tokens.is_empty());

        // Test tokenizing a phrase
        let text = "This is a test sentence.";
        let tokenized = vocab.tokenize(text);

        // Ensure we get the correct tokens. Since the vocabulary is pre-trained, ensure it returns sensible results.
        let expected_tokens = vec![
            "<s>".to_string(),   // Sentence start
            "▁this".to_string(), // Word break for 'This'
            "▁is".to_string(),   // Word break for 'This'
            "▁a".to_string(),    // Word break for 'This'
            "▁test".to_string(), // Word break for 'This'
            "▁sent".to_string(), // Word break for 'This'
            "ence".to_string(),  // Word break for 'This'
            "</s>".to_string(),  // Sentence end
        ];

        assert_eq!(tokenized, expected_tokens);
    }

    #[test]
    #[cfg(feature = "default-medium")]
    fn test_new_default_medium_with_tokenization() {
        // Initialize the BytePairEncoder with the default medium vocabulary
        let result = BytePairEncoder::new_default_medium();
        assert!(result.is_ok());

        let vocab = result.unwrap();
        assert!(!vocab.tokens.is_empty());

        // Test tokenizing a phrase
        let text = "This is a test sentence.";
        let tokenized = vocab.tokenize(text);

        // Ensure we get the correct tokens. Since the vocabulary is pre-trained, ensure it returns sensible results.
        let expected_tokens = vec![
            "<s>".to_string(),       // Sentence start
            "▁this".to_string(),     // Word break for 'This'
            "▁is".to_string(),       // Word break for 'This'
            "▁a".to_string(),        // Word break for 'This'
            "▁test".to_string(),     // Word break for 'This'
            "▁sentence".to_string(), // Word break for 'This'
            "</s>".to_string(),      // Sentence end
        ];

        assert_eq!(tokenized, expected_tokens);
    }

    #[test]
    #[cfg(feature = "default-large")]
    fn test_new_default_large_with_tokenization() {
        // Initialize the BytePairEncoder with the default large vocabulary
        let result = BytePairEncoder::new_default_large();
        assert!(result.is_ok());

        let vocab = result.unwrap();
        assert!(!vocab.tokens.is_empty());

        // Test tokenizing a phrase
        let text = "This is a test sentence.";
        let tokenized = vocab.tokenize(text);

        // Ensure we get the correct tokens. Since the vocabulary is pre-trained, ensure it returns sensible results.
        let expected_tokens = vec![
            "<s>".to_string(),       // Sentence start
            "▁this".to_string(),     // Word break for 'This'
            "▁is".to_string(),       // Word break for 'This'
            "▁a".to_string(),        // Word break for 'This'
            "▁test".to_string(),     // Word break for 'This'
            "▁sentence".to_string(), // Word break for 'This'
            "</s>".to_string(),      // Sentence end
        ];

        assert_eq!(tokenized, expected_tokens);
    }

    #[test]
    fn test_tokenize_sentences_iter() {
        let vocab_str = "hello\t1\nworld\t2\n▁\t3";
        let vocab = BytePairEncoder::new_from_str(vocab_str).unwrap();

        let text = "Hello, world! How are you?";
        let tokenized: Vec<Vec<String>> = vocab
            .tokenize_sentences_iter(text)
            .map(|sentence_iter| sentence_iter.collect())
            .collect();

        assert_eq!(tokenized.len(), 2);

        assert_eq!(
            tokenized[0],
            vec![
                "<s>".to_string(),
                "▁".to_string(),
                "hello".to_string(),
                "▁".to_string(),
                "world".to_string(),
                "</s>".to_string(),
            ]
        );

        assert_eq!(
            tokenized[1],
            vec![
                "<s>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "</s>".to_string(),
            ]
        );
    }

    #[test]
    fn test_tokenize_sentences_iter_empty_input() {
        let vocab = BytePairEncoder::new_from_str("test\t1").unwrap();
        let text = "";
        let tokenized: Vec<Vec<String>> = vocab
            .tokenize_sentences_iter(text)
            .map(|sentence_iter| sentence_iter.collect())
            .collect();

        assert_eq!(tokenized.len(), 0);
    }

    #[test]
    fn test_tokenize_sentences_iter_unicode() {
        let vocab_str = "こんにちは\t1\n世界\t2\n▁\t3";
        let vocab = BytePairEncoder::new_from_str(vocab_str).unwrap();

        let text = "こんにちは、世界！お元気ですか？";
        let tokenized: Vec<Vec<String>> = vocab
            .tokenize_sentences_iter(text)
            .map(|sentence_iter| sentence_iter.collect())
            .collect();

        assert_eq!(tokenized.len(), 2);

        assert_eq!(
            tokenized[0],
            vec![
                "<s>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "</s>".to_string(),
            ]
        );

        assert_eq!(
            tokenized[1],
            vec![
                "<s>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "</s>".to_string(),
            ]
        );
    }

    #[test]
    fn test_tokenize_iter() {
        let vocab_str = "hello\t1\nworld\t2\n▁\t3";
        let vocab = BytePairEncoder::new_from_str(vocab_str).unwrap();

        let text = "Hello, world! How are you?";
        let tokenized: Vec<String> = vocab.tokenize_iter(text).collect();

        assert_eq!(
            tokenized,
            vec![
                "<s>".to_string(),
                "▁".to_string(),
                "hello".to_string(),
                "▁".to_string(),
                "world".to_string(),
                "</s>".to_string(),
                "<s>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "</s>".to_string(),
            ]
        );
    }

    #[test]
    fn test_tokenize_iter_empty_input() {
        let vocab = BytePairEncoder::new_from_str("test\t1").unwrap();
        let text = "";
        let tokenized: Vec<String> = vocab.tokenize_iter(text).collect();

        assert_eq!(tokenized.len(), 0);
    }

    #[test]
    fn test_tokenize_iter_unicode() {
        let vocab_str = "こんにちは\t1\n世界\t2\n▁\t3";
        let vocab = BytePairEncoder::new_from_str(vocab_str).unwrap();

        let text = "こんにちは、世界！お元気ですか？";
        let tokenized: Vec<String> = vocab.tokenize_iter(text).collect();

        assert_eq!(
            tokenized,
            vec![
                "<s>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "</s>".to_string(),
                "<s>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "</s>".to_string(),
            ]
        );
    }

    #[test]
    fn test_tokenize_sentences() {
        let vocab_str = "hello\t1\nworld\t2\n▁\t3";
        let vocab = BytePairEncoder::new_from_str(vocab_str).unwrap();

        let text = "Hello, world! How are you?";
        let tokenized = vocab.tokenize_sentences(text);

        assert_eq!(tokenized.len(), 2);
        assert_eq!(
            tokenized[0],
            vec![
                "<s>".to_string(),
                "▁".to_string(),
                "hello".to_string(),
                "▁".to_string(),
                "world".to_string(),
                "</s>".to_string(),
            ]
        );
        assert_eq!(
            tokenized[1],
            vec![
                "<s>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "</s>".to_string(),
            ]
        );
    }

    #[test]
    fn test_tokenize() {
        let vocab_str = "hello\t1\nworld\t2\n▁\t3";
        let vocab = BytePairEncoder::new_from_str(vocab_str).unwrap();

        let text = "Hello, world! How are you?";
        let tokenized = vocab.tokenize(text);

        assert_eq!(
            tokenized,
            vec![
                "<s>".to_string(),
                "▁".to_string(),
                "hello".to_string(),
                "▁".to_string(),
                "world".to_string(),
                "</s>".to_string(),
                "<s>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "</s>".to_string(),
            ]
        );
    }

    #[test]
    fn test_tokenize_empty_input() {
        let vocab = BytePairEncoder::new_from_str("test\t1").unwrap();
        let text = "";

        assert_eq!(vocab.tokenize_sentences(text), Vec::<Vec<String>>::new());
        assert_eq!(vocab.tokenize(text), Vec::<String>::new());
    }

    #[test]
    fn test_tokenize_with_sentence_markers() {
        let vocab_str = "hello\t1\nworld\t2\n▁\t3";
        let vocab = BytePairEncoder::new_from_str(vocab_str).unwrap();

        let sentence = "Hello, World!";
        let tokenized: Vec<String> = vocab
            .tokenize_with_sentence_markers_iter(sentence)
            .collect();

        assert_eq!(
            tokenized,
            vec![
                "<s>".to_string(),
                "▁".to_string(),
                "hello".to_string(),
                "▁".to_string(),
                "world".to_string(),
                "</s>".to_string(),
            ]
        );
    }

    #[test]
    fn test_tokenize_with_sentence_markers_unicode() {
        let vocab_str = "こんにちは\t1\n世界\t2\n▁\t3";
        let vocab = BytePairEncoder::new_from_str(vocab_str).unwrap();

        let sentence = "こんにちは、世界！";
        let tokenized: Vec<String> = vocab
            .tokenize_with_sentence_markers_iter(sentence)
            .collect();

        assert_eq!(
            tokenized,
            vec![
                "<s>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "▁".to_string(),
                "<unk>".to_string(),
                "</s>".to_string(),
            ]
        );
    }

    #[test]
    fn test_tokenize_word() {
        let vocab_str = "hell\t1\no\t2\nwo\t3\nrld\t4\n▁\t5";
        let vocab = BytePairEncoder::new_from_str(vocab_str).unwrap();

        assert_eq!(
            vocab.tokenize_word("▁hello"),
            vec!["▁".to_string(), "hell".to_string(), "o".to_string()]
        );
        assert_eq!(
            vocab.tokenize_word("▁world"),
            vec!["▁".to_string(), "wo".to_string(), "rld".to_string()]
        );
        assert_eq!(
            vocab.tokenize_word("▁unknown"),
            vec![
                "▁".to_string(),
                "<unk>".to_string(),
                "o".to_string(),
                "<unk>".to_string()
            ]
        );
    }

    #[test]
    fn test_tokenize_word_empty() {
        let vocab = BytePairEncoder::new_from_str("test\t1").unwrap();
        assert_eq!(vocab.tokenize_word(""), Vec::<String>::new());
    }

    #[test]
    fn test_tokenize_word_partial_match() {
        let vocab_str = "partial\t1\npar\t2\ntial\t3\n▁\t4";
        let vocab = BytePairEncoder::new_from_str(vocab_str).unwrap();

        assert_eq!(
            vocab.tokenize_word("▁partial"),
            vec!["▁".to_string(), "partial".to_string()]
        );
        assert_eq!(
            vocab.tokenize_word("▁partially"),
            vec!["▁".to_string(), "partial".to_string(), "<unk>".to_string()]
        );
    }
}
