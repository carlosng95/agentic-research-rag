from pydantic import ConfigDict

from langchain_community.cross_encoders import (
    HuggingFaceCrossEncoder,
)
from langchain_core.cross_encoders import BaseCrossEncoder
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ..config import Settings


class RerankingRetriever(BaseRetriever):
    model_config = ConfigDict(
        arbitrary_types_allowed = True
    )

    base_retriever: BaseRetriever
    cross_encoder: BaseCrossEncoder
    top_n: int = 5

    def _get_relevant_documents(
        self,
        query: str,
    ) -> list[Document]:
        documents = self.base_retriever.invoke(
            query
        )

        if not documents:
            return []

        text_pairs = [
            (
                query,
                document.page_content,
            )
            for document in documents
        ]

        scores = self.cross_encoder.score(
            text_pairs
        )

        ranked = sorted(
            zip(
                documents,
                scores,
            ),
            key = lambda item: item[1],
            reverse = True,
        )

        return [
            document
            for document, _ in ranked[:self.top_n]
        ]


def build_reranking_retriever(
    base_retriever: BaseRetriever,
    settings: Settings,
) -> RerankingRetriever:
    cross_encoder = HuggingFaceCrossEncoder(
        model_name = settings.cross_encoder_model,
    )

    return RerankingRetriever(
        base_retriever = base_retriever,
        cross_encoder = cross_encoder,
        top_n = settings.final_k,
    )