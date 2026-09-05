from agentic_research_rag.indexing.bm25_index import BM25Index
from agentic_research_rag.tokenizer import Tokenizer
from agentic_research_rag.types import Chunk


class FakeTokenizer(Tokenizer):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def tokenize(self, text: str) -> list[str]:
        self.calls.append(text)
        return text.lower().split()


def make_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id = 1,
            document_name = "paper.pdf",
            page_number = 1,
            text = "semantic retrieval ranking",
        ),
        Chunk(
            chunk_id = 2,
            document_name = "paper.pdf",
            page_number = 2,
            text = "vector database indexing",
        ),
        Chunk(
            chunk_id = 3,
            document_name = "paper.pdf",
            page_number = 3,
            text = "weather forecast temperature",
        ),
    ]


def test_bm25_index_tokenizes_corpus_on_creation():
    chunks = make_chunks()
    tokenizer = FakeTokenizer()

    BM25Index(
        chunks = chunks,
        tokenizer = tokenizer,
    )

    assert tokenizer.calls == [
        "semantic retrieval ranking",
        "vector database indexing",
        "weather forecast temperature",
    ]


def test_bm25_search_returns_lexically_relevant_chunk():
    chunks = make_chunks()
    tokenizer = FakeTokenizer()

    index = BM25Index(
        chunks = chunks,
        tokenizer = tokenizer,
    )

    results = index.search(
        query = "semantic retrieval",
        k = 2,
    )

    assert len(results) == 2
    assert results[0].chunk_id == 1


def test_bm25_search_assigns_score():
    chunks = make_chunks()
    tokenizer = FakeTokenizer()

    index = BM25Index(
        chunks = chunks,
        tokenizer = tokenizer,
    )

    results = index.search(
        query = "semantic retrieval",
        k = 1,
    )

    assert results[0].score is not None
    assert results[0].score > 0


def test_bm25_search_tokenizes_query():
    chunks = make_chunks()
    tokenizer = FakeTokenizer()

    index = BM25Index(
        chunks = chunks,
        tokenizer = tokenizer,
    )

    tokenizer.calls.clear()

    index.search(
        query = "semantic retrieval",
        k = 2,
    )

    assert tokenizer.calls == [
        "semantic retrieval",
    ]


def test_bm25_search_returns_empty_for_empty_query():
    chunks = make_chunks()
    tokenizer = FakeTokenizer()

    index = BM25Index(
        chunks = chunks,
        tokenizer = tokenizer,
    )

    tokenizer.calls.clear()

    results = index.search(
        query = "   ",
        k = 5,
    )

    assert results == []
    assert tokenizer.calls == []


def test_bm25_search_returns_empty_for_zero_k():
    chunks = make_chunks()
    tokenizer = FakeTokenizer()

    index = BM25Index(
        chunks = chunks,
        tokenizer = tokenizer,
    )

    tokenizer.calls.clear()

    results = index.search(
        query = "retrieval",
        k = 0,
    )

    assert results == []
    assert tokenizer.calls == []


def test_bm25_search_limits_results_to_k():
    chunks = make_chunks()
    tokenizer = FakeTokenizer()

    index = BM25Index(
        chunks = chunks,
        tokenizer = tokenizer,
    )

    results = index.search(
        query = "retrieval",
        k = 1,
    )

    assert len(results) == 1


def test_bm25_search_does_not_modify_original_chunks():
    chunks = make_chunks()
    tokenizer = FakeTokenizer()

    index = BM25Index(
        chunks = chunks,
        tokenizer = tokenizer,
    )

    results = index.search(
        query = "semantic",
        k = 1,
    )

    assert chunks[0].score is None
    assert results[0].score is not None
