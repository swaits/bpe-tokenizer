use std::{collections::HashMap, fs, iter};

use unicode_segmentation::UnicodeSegmentation;

use crate::{
    constants::*,
    default_vocabs::{new_default, DefaultVocab},
    BytePairEncoderError,
};

/// # Represents a Byte Pair Encoding (BPE) vocabulary used for tokenization.
///
/// This struct holds the mapping of tokens to their respective scores and provides methods for
/// tokenizing text using the BPE algorithm.
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
    /// # A mapping of tokens to their respective scores.
    ///
    /// In BPE, tokens with higher scores are typically more common and are preferred during the
    /// tokenization process.
    pub(crate) tokens: HashMap<String, isize>,
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
    /// # #[cfg(feature = "default-small")] {
    /// use bpe_tokenizer::BytePairEncoder;
    ///
    /// let encoder = BytePairEncoder::new_default_small().unwrap();
    /// # }
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
    pub fn new_default_small() -> Result<Self, BytePairEncoderError> {
        new_default(DefaultVocab::Small)
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
    /// # #[cfg(feature = "default-medium")] {
    /// use bpe_tokenizer::BytePairEncoder;
    ///
    /// let encoder = BytePairEncoder::new_default_medium().unwrap();
    /// # }
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
    pub fn new_default_medium() -> Result<Self, BytePairEncoderError> {
        new_default(DefaultVocab::Medium)
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
    /// # #[cfg(feature = "default-large")] {
    /// use bpe_tokenizer::BytePairEncoder;
    ///
    /// let encoder = BytePairEncoder::new_default_large().unwrap();
    /// # }
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
    pub fn new_default_large() -> Result<Self, BytePairEncoderError> {
        new_default(DefaultVocab::Large)
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
    pub(crate) fn tokenize_with_sentence_markers_iter<'a>(
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
    pub(crate) fn tokenize_word(&self, text: &str) -> Vec<String> {
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
