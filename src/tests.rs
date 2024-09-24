use std::{fs::File, io::Write};

use crate::{BytePairEncoder, BytePairEncoderError};

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
