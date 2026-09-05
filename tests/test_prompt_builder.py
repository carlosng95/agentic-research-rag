from agentic_research_rag.prompt_builder import build_rag_prompt


def test_rag_prompt_contains_query():
    query = "How does hybrid retrieval work?"

    prompt = build_rag_prompt(
        query = query,
        context = "Some scientific evidence.",
    )

    assert query in prompt


def test_rag_prompt_contains_context():
    context = (
        "[SOURCE 1]\n"
        "Document: retrieval.pdf\n"
        "Page: 3\n"
        "Hybrid retrieval combines lexical and semantic ranking."
    )

    prompt = build_rag_prompt(
        query = "What is hybrid retrieval?",
        context = context,
    )

    assert context in prompt


def test_rag_prompt_instructs_model_to_use_sources():
    prompt = build_rag_prompt(
        query = "What is semantic search?",
        context = "[SOURCE 1]\nEvidence",
    )

    assert "[SOURCE" in prompt


def test_rag_prompt_mentions_insufficient_information():
    prompt = build_rag_prompt(
        query = "What is semantic search?",
        context = "[SOURCE 1]\nEvidence",
    )

    normalized = prompt.lower()

    assert (
        "insufficient" in normalized
        or "not enough" in normalized
        or "does not contain enough information" in normalized
    )


def test_rag_prompt_discourages_external_knowledge():
    prompt = build_rag_prompt(
        query = "What is semantic search?",
        context = "[SOURCE 1]\nEvidence",
    )

    normalized = prompt.lower()

    assert "context" in normalized


def test_rag_prompt_contains_source_citation_format():
    prompt = build_rag_prompt(
        query = "What is semantic search?",
        context = "[SOURCE 1]\nEvidence",
    )

    assert "[SOURCE N]" in prompt or "[SOURCE 1]" in prompt
