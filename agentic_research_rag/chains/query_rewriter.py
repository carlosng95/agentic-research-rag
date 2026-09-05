from typing import TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
)


class QueryRewriteInput(TypedDict):
    question: str
    history: list[BaseMessage]


_QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Rewrite the user's latest question as a standalone "
                "research query using the conversation history.\n\n"
                "Rules:\n"
                "- Preserve the user's original intent.\n"
                "- Resolve pronouns and implicit references when possible.\n"
                "- Include only context necessary to make the query "
                "self-contained.\n"
                "- Do not answer the question.\n"
                "- Do not add facts that are not present in the conversation.\n"
                "- Return only the rewritten query."
            ),
        ),
        MessagesPlaceholder(
            variable_name = "history"
        ),
        (
            "human",
            "{question}",
        ),
    ]
)


def _original_question(
    state: QueryRewriteInput,
) -> str:
    return state["question"]


def build_query_rewriter(
    model: BaseChatModel,
) -> Runnable[QueryRewriteInput, str]:
    rewrite_chain = (
        _QUERY_REWRITE_PROMPT
        | model
        | StrOutputParser()
        | RunnableLambda(
            lambda query: query.strip()
        )
    )

    return RunnableBranch(
        (
            lambda state: bool(
                state["history"]
            ),
            rewrite_chain,
        ),
        RunnableLambda(
            _original_question
        ),
    )