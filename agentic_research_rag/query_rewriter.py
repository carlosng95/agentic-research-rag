from .memory import ConversationMemory
from .providers.llm import LLMProvider


class QueryRewriter:
    """
    Rewrite conversational queries into standalone search queries.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def rewrite(self, query: str, memory: ConversationMemory) -> str:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        history = memory.build_context()

        if not history:
            return query

        prompt = (
            "Rewrite the user's current question as a standalone research query.\n\n"
            "Use the conversation history only to resolve references, omitted context, "
            "pronouns, or ambiguous terms.\n\n"
            "Instructions:\n"
            "- Preserve the original meaning of the current question.\n"
            "- Do not answer the question.\n"
            "- Do not add facts that are not present in the conversation.\n"
            "- If the current question is already standalone, return it unchanged.\n"
            "- Return only the rewritten query.\n\n"
            f"CONVERSATION HISTORY:\n{history}\n\n"
            f"CURRENT QUESTION:\n{query}\n\n"
            "STANDALONE QUERY:"
        )

        rewritten_query = self._llm.generate(
            prompt = prompt,
        ).strip()

        if not rewritten_query:
            return query

        return rewritten_query