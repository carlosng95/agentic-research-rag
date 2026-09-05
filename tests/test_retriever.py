from agentic_research_rag.retriever import Retriever
from agentic_research_rag.types import Chunk


class FakeSemanticIndex:
    def __init__(self, results: list[Chunk]) -> None:
        self._results = results
        self.calls: list[dict] = []

    def search(self, query: str, k: int) -> list[Chunk]:
        self.calls.append(
            {
                "query": query,
                "k": k,
            }
        )

        return self._results[:k]


class FakeBM25Index:
    def __init__(self, results: list[Chunk]) -> None:
        self._results = results
        self.calls: list[dict] = []

    def search(self, query: str, k: int) -> list[Chunk]:
        self.calls.append(
            {
                "query": query,
                "k": k,
            }
        )

        return self._results[:k]


def make_chunk(chunk_id: int) -> Chunk:
    return Chunk(
        chunk_id = chunk_id,
        document_name = f"paper_{chunk_id}.pdf",
        page_number = 1,
        text = f"Chunk {chunk_id}",
    )


def test_semantic_search_delegates_to_semantic_index():
    semantic_results = [
        make_chunk(1),
        make_chunk(2),
    ]

    semantic_index = FakeSemanticIndex(
        results = semantic_results,
    )

    bm25_index = FakeBM25Index(
        results = [],
    )

    retriever = Retriever(
        semantic_index = semantic_index,
        bm25_index = bm25_index,
        rrf_k = 60,
    )

    results = retriever.semantic_search(
        query = "semantic retrieval",
        k = 2,
    )

    assert results == semantic_results

    assert semantic_index.calls == [
        {
            "query": "semantic retrieval",
            "k": 2,
        }
    ]


def test_bm25_search_delegates_to_bm25_index():
    bm25_results = [
        make_chunk(1),
        make_chunk(2),
    ]

    semantic_index = FakeSemanticIndex(
        results = [],
    )

    bm25_index = FakeBM25Index(
        results = bm25_results,
    )

    retriever = Retriever(
        semantic_index = semantic_index,
        bm25_index = bm25_index,
        rrf_k = 60,
    )

    results = retriever.bm25_search(
        query = "semantic retrieval",
        k = 2,
    )

    assert results == bm25_results

    assert bm25_index.calls == [
        {
            "query": "semantic retrieval",
            "k": 2,
        }
    ]


def test_retrieve_queries_both_indexes_with_candidate_k():
    semantic_index = FakeSemanticIndex(
        results = [
            make_chunk(1),
            make_chunk(2),
        ],
    )

    bm25_index = FakeBM25Index(
        results = [
            make_chunk(2),
            make_chunk(3),
        ],
    )

    retriever = Retriever(
        semantic_index = semantic_index,
        bm25_index = bm25_index,
        rrf_k = 60,
    )

    retriever.retrieve(
        query = "hybrid retrieval",
        k = 2,
        candidate_k = 10,
    )

    assert semantic_index.calls == [
        {
            "query": "hybrid retrieval",
            "k": 10,
        }
    ]

    assert bm25_index.calls == [
        {
            "query": "hybrid retrieval",
            "k": 10,
        }
    ]


def test_retrieve_fuses_semantic_and_bm25_results():
    chunk_1 = make_chunk(1)
    chunk_2 = make_chunk(2)
    chunk_3 = make_chunk(3)
    chunk_4 = make_chunk(4)

    semantic_index = FakeSemanticIndex(
        results = [
            chunk_1,
            chunk_2,
            chunk_3,
        ],
    )

    bm25_index = FakeBM25Index(
        results = [
            chunk_4,
            chunk_1,
            chunk_3,
        ],
    )

    retriever = Retriever(
        semantic_index = semantic_index,
        bm25_index = bm25_index,
        rrf_k = 60,
    )

    results = retriever.retrieve(
        query = "hybrid retrieval",
        k = 4,
        candidate_k = 3,
    )

    assert len(results) == 4
    assert results[0].chunk_id == 1


def test_retrieve_limits_results_to_k():
    semantic_index = FakeSemanticIndex(
        results = [
            make_chunk(1),
            make_chunk(2),
            make_chunk(3),
        ],
    )

    bm25_index = FakeBM25Index(
        results = [
            make_chunk(4),
            make_chunk(5),
            make_chunk(6),
        ],
    )

    retriever = Retriever(
        semantic_index = semantic_index,
        bm25_index = bm25_index,
        rrf_k = 60,
    )

    results = retriever.retrieve(
        query = "retrieval",
        k = 2,
        candidate_k = 3,
    )

    assert len(results) == 2


def test_retrieve_returns_empty_for_zero_k():
    semantic_index = FakeSemanticIndex(
        results = [
            make_chunk(1),
        ],
    )

    bm25_index = FakeBM25Index(
        results = [
            make_chunk(2),
        ],
    )

    retriever = Retriever(
        semantic_index = semantic_index,
        bm25_index = bm25_index,
        rrf_k = 60,
    )

    results = retriever.retrieve(
        query = "retrieval",
        k = 0,
        candidate_k = 10,
    )

    assert results == []
    assert semantic_index.calls == []
    assert bm25_index.calls == []


def test_retrieve_returns_empty_for_zero_candidate_k():
    semantic_index = FakeSemanticIndex(
        results = [
            make_chunk(1),
        ],
    )

    bm25_index = FakeBM25Index(
        results = [
            make_chunk(2),
        ],
    )

    retriever = Retriever(
        semantic_index = semantic_index,
        bm25_index = bm25_index,
        rrf_k = 60,
    )

    results = retriever.retrieve(
        query = "retrieval",
        k = 5,
        candidate_k = 0,
    )

    assert results == []
    assert semantic_index.calls == []
    assert bm25_index.calls == []
