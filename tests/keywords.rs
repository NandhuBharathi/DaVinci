use davinci::keywords::Keyword;

#[test]
fn valid_keyword() {
    let result = Keyword::parse("DAVINCI::CUDA::ENABLE");
    assert!(result.is_ok());
}

#[test]
fn valid_memory_keyword() {
    let result = Keyword::parse("DAVINCI::GPU::MEMORY_LIMIT::80");

    assert!(result.is_ok());
}

#[test]
fn invalid_namespace() {
    let result = Keyword::parse("CUDA::ENABLE");
    assert!(result.is_err());
}

#[test]
fn empty_keyword() {
    let result = Keyword::parse("");
    assert!(result.is_err());
}

#[test]
fn invalid_segment() {
    let result = Keyword::parse("DAVINCI::cuda::ENABLE");
    assert!(result.is_err());
}

#[test]
fn empty_segment() {
    let result = Keyword::parse("DAVINCI::::ENABLE");
    assert!(result.is_err());
}
