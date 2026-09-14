import re
from collections import defaultdict
from typing import Hashable

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ..config import Settings
from ..observability.operations import observe_operation


def _tokenize(text: str) -> list[str]:
    return re.findall(
        r"\b\w+\b",
        text.lower(),
    )


def _document_key(
    document: Document,
) -> Hashable:
    chunk_id = document.metadata.get("chunk_id")

    if chunk_id is not None:
        return (
            "chunk_id",
            chunk_id,
        )

    return (
        document.metadata.get("source"),
        document.metadata.get("page_number"),
        document.page_content,
    )


class HybridRetriever(BaseRetriever):
    semantic_retriever: BaseRetriever
    bm25_retriever: BaseRetriever

    semantic_weight: float = 0.5
    bm25_weight: float = 0.5

    rrf_k: int = 60
    top_k: int = 20

    def _get_relevant_documents(
        self,
        query: str,
    ) -> list[Document]:
        with observe_operation("semantic_retrieval"):
            semantic_documents = self.semantic_retriever.invoke(
                query
            )

        with observe_operation("bm25_retrieval"):
            bm25_documents = self.bm25_retriever.invoke(
                query
            )

        with observe_operation("rrf_fusion"):
            rankings = [
                (
                    semantic_documents,
                    self.semantic_weight,
                ),
                (
                    bm25_documents,
                    self.bm25_weight,
                ),
            ]

            scores: dict[Hashable, float] = defaultdict(float)
            documents: dict[Hashable, Document] = {}
            first_seen: dict[Hashable, int] = {}

            order = 0

            for ranking, weight in rankings:
                seen_in_ranking: set[Hashable] = set()

                for rank, document in enumerate(
                    ranking,
                    start = 1,
                ):
                    key = _document_key(
                        document = document
                    )

                    if key in seen_in_ranking:
                        continue

                    seen_in_ranking.add(key)

                    if key not in documents:
                        documents[key] = document
                        first_seen[key] = order
                        order += 1

                    scores[key] += (
                        weight
                        / (self.rrf_k + rank)
                    )

            ranked_keys = sorted(
                scores,
                key = lambda key: (
                    -scores[key],
                    first_seen[key],
                ),
            )

            ranked_documents = [
                documents[key]
                for key in ranked_keys[:self.top_k]
            ]

        return ranked_documents


def build_hybrid_retriever(
    documents: list[Document],
    vector_store: FAISS,
    settings: Settings,
) -> HybridRetriever:
    if not documents:
        raise ValueError(
            "Cannot build a hybrid retriever from an empty corpus."
        )

    semantic_retriever = vector_store.as_retriever(
        search_type = "similarity",
        search_kwargs = {
            "k": settings.candidate_k,
        },
    )

    bm25_retriever = BM25Retriever.from_documents(
        documents = documents,
        preprocess_func = _tokenize,
    )

    bm25_retriever.k = settings.candidate_k

    return HybridRetriever(
        semantic_retriever = semantic_retriever,
        bm25_retriever = bm25_retriever,
        semantic_weight = settings.semantic_weight,
        bm25_weight = settings.bm25_weight,
        rrf_k = settings.rrf_k,
        top_k = settings.rerank_k,
    )