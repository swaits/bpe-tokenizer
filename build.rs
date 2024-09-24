#[cfg(any(
    feature = "default-small",
    feature = "default-medium",
    feature = "default-large"
))]
use {
    lz4_flex::block::compress_prepend_size,
    std::{collections::HashMap, env, fs, path::PathBuf},
};

fn main() {
    #[cfg(feature = "default-small")]
    process_vocab("multi.wiki.bpe.vs100000.vocab");

    #[cfg(feature = "default-medium")]
    process_vocab("multi.wiki.bpe.vs320000.vocab");

    #[cfg(feature = "default-large")]
    process_vocab("multi.wiki.bpe.vs1000000.vocab");
}

#[cfg(any(
    feature = "default-small",
    feature = "default-medium",
    feature = "default-large"
))]
fn process_vocab(name: &str) {
    // Path to the vocabulary file (ensure this path is correct)
    let vocab_path = PathBuf::from(format!("vocab/{}", name));

    // Load and parse the vocabulary into a HashMap.
    let tokens: HashMap<String, isize> = load_vocab_hashmap(&vocab_path);

    // Serialize the HashMap using bincode
    let serialized = bincode::serialize(&tokens).unwrap();

    // Compress the serialized data using zstd (with ultra compression level)
    let compressed = compress_prepend_size(&serialized);

    // Write the compressed data to a file in the build output directory
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let output_path = out_dir.join(format!("{}.hashmap.bincode.lz4", name));
    fs::write(&output_path, compressed).unwrap();
}

#[cfg(any(
    feature = "default-small",
    feature = "default-medium",
    feature = "default-large"
))]
fn load_vocab_hashmap(path: &PathBuf) -> HashMap<String, isize> {
    // Read file's contents
    let content = fs::read_to_string(path).unwrap();

    let mut tokens = HashMap::new();

    // Process each line in the file, each being a token-score pair
    for line in content.lines() {
        let (token, score_str) = match line.split_once('\t') {
            Some(pair) => pair,
            None => panic!("Invalid line in vocabulary file: {}", line),
        };
        let score = match score_str.parse::<isize>() {
            Ok(score) => score,
            Err(_) => panic!("Invalid score in vocabulary file: {}", line),
        };
        tokens.insert(token.to_string(), score);
    }

    tokens
}
