use thiserror::Error;

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

    // Indicates attempt to use a default vocabulary without enabling its Cargo feature.
    #[error("Error, must enable defualt-small, default-medium, and/or default-large feature(s) to use default vocabulary.")]
    NoDefaultVocabFeature,
}
