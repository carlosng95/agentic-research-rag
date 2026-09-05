def build_rag_prompt(query: str, context: str) -> str:
    """
    Build the prompt used to answer a question from retrieved context.
    """

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if not context.strip():
        raise ValueError("Context cannot be empty.")

    return (
        "You are a research assistant answering questions using scientific papers.\n\n"
        "Instructions:\n"
        "- Answer only using the information provided in the context.\n"
        "- Do not use external knowledge.\n"
        "- If the context does not contain enough information, say so clearly.\n"
        "- Cite supporting information using [SOURCE N].\n"
        "- Do not invent sources or citations.\n"
        "- Prefer a clear and concise scientific explanation.\n\n"
        f"QUESTION:\n{query}\n\n"
        f"CONTEXT:\n{context}\n\n"
        "ANSWER:"
    )