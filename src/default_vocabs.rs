use crate::{BytePairEncoder, BytePairEncoderError};

#[cfg(any(
    feature = "default-small",
    feature = "default-medium",
    feature = "default-large"
))]
use {lz4_flex::decompress_size_prepended, std::collections::HashMap};

#[allow(dead_code)]
#[derive(Debug)]
pub(crate) enum DefaultVocab {
    Small,
    Medium,
    Large,
}

#[cfg(feature = "default-small")]
const DEFAULT_SMALL_DATA: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/",
    "multi.wiki.bpe.vs100000.vocab.hashmap.bincode.lz4"
));

#[allow(dead_code)]
#[cfg(not(feature = "default-small"))]
const DEFAULT_SMALL_DATA: &[u8] = &[];

#[cfg(feature = "default-medium")]
const DEFAULT_MEDIUM_DATA: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/",
    "multi.wiki.bpe.vs320000.vocab.hashmap.bincode.lz4"
));

#[allow(dead_code)]
#[cfg(not(feature = "default-medium"))]
const DEFAULT_MEDIUM_DATA: &[u8] = &[];

#[cfg(feature = "default-large")]
const DEFAULT_LARGE_DATA: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/",
    "multi.wiki.bpe.vs1000000.vocab.hashmap.bincode.lz4"
));

#[allow(dead_code)]
#[cfg(not(feature = "default-large"))]
const DEFAULT_LARGE_DATA: &[u8] = &[];

// The helper function to initialize a BytePairEncoder from a compressed vocabulary dataset.
// This doesn't need to be public, so we use `pub(crate)` or leave it private entirely.
#[cfg(any(
    feature = "default-small",
    feature = "default-medium",
    feature = "default-large"
))]
pub(crate) fn new_default(vocab: DefaultVocab) -> Result<BytePairEncoder, BytePairEncoderError> {
    let data = match vocab {
        DefaultVocab::Small => DEFAULT_SMALL_DATA,
        DefaultVocab::Medium => DEFAULT_MEDIUM_DATA,
        DefaultVocab::Large => DEFAULT_LARGE_DATA,
    };

    // Decompress the LZ4 binary data.
    let uncompressed = decompress_size_prepended(data)
        .map_err(|e| BytePairEncoderError::DecompressionError(e.to_string()))?;

    // Deserialize the uncompressed data into a HashMap.
    let tokens: HashMap<String, isize> = bincode::deserialize(&uncompressed)
        .map_err(|e| BytePairEncoderError::DeserializationError(e.to_string()))?;

    // Return the BytePairEncoder.
    Ok(BytePairEncoder { tokens })
}

#[cfg(not(any(
    feature = "default-small",
    feature = "default-medium",
    feature = "default-large"
)))]
pub(crate) fn new_default(_vocab: DefaultVocab) -> Result<BytePairEncoder, BytePairEncoderError> {
    Err(BytePairEncoderError::NoDefaultVocabFeature)
}
