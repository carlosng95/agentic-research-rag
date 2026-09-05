from agentic_research_rag.tokenizer import RegexTokenizer, WhitespaceTokenizer


def test_regex_tokenizer_extracts_words():
    tokenizer = RegexTokenizer()

    tokens = tokenizer.tokenize(
        "Semantic search retrieves documents."
    )

    assert tokens == [
        "semantic",
        "search",
        "retrieves",
        "documents",
    ]


def test_regex_tokenizer_removes_punctuation():
    tokenizer = RegexTokenizer()

    tokens = tokenizer.tokenize(
        "semantic, retrieval! ranking?"
    )

    assert tokens == [
        "semantic",
        "retrieval",
        "ranking",
    ]


def test_regex_tokenizer_converts_text_to_lowercase():
    tokenizer = RegexTokenizer()

    tokens = tokenizer.tokenize(
        "RETRIEVAL Retrieval retrieval"
    )

    assert tokens == [
        "retrieval",
        "retrieval",
        "retrieval",
    ]


def test_regex_tokenizer_handles_numbers():
    tokenizer = RegexTokenizer()

    tokens = tokenizer.tokenize(
        "Model1 and Model2 embeddings"
    )

    assert tokens == [
        "model1",
        "and",
        "model2",
        "embeddings",
    ]


def test_regex_tokenizer_returns_empty_for_empty_text():
    tokenizer = RegexTokenizer()

    assert tokenizer.tokenize("") == []
    assert tokenizer.tokenize("   ") == []


def test_whitespace_tokenizer_splits_on_whitespace():
    tokenizer = WhitespaceTokenizer()

    tokens = tokenizer.tokenize(
        "semantic   retrieval\nranking"
    )

    assert tokens == [
        "semantic",
        "retrieval",
        "ranking",
    ]


def test_whitespace_tokenizer_converts_text_to_lowercase():
    tokenizer = WhitespaceTokenizer()

    tokens = tokenizer.tokenize(
        "Semantic RETRIEVAL"
    )

    assert tokens == [
        "semantic",
        "retrieval",
    ]


def test_whitespace_tokenizer_preserves_punctuation():
    tokenizer = WhitespaceTokenizer()

    tokens = tokenizer.tokenize(
        "semantic, retrieval!"
    )

    assert tokens == [
        "semantic,",
        "retrieval!",
    ]
