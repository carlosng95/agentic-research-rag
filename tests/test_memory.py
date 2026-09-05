import pytest

from agentic_research_rag.memory import ConversationMemory


def test_memory_starts_empty():
    memory = ConversationMemory(
        max_turns = 5,
    )

    assert memory.turns == []
    assert memory.build_context() == ""


def test_memory_adds_user_message():
    memory = ConversationMemory(
        max_turns = 5,
    )

    memory.add_user(
        "What is semantic search?"
    )

    assert len(memory.turns) == 1
    assert memory.turns[0].role == "user"
    assert memory.turns[0].content == "What is semantic search?"


def test_memory_adds_assistant_message():
    memory = ConversationMemory(
        max_turns = 5,
    )

    memory.add_assistant(
        "Semantic search retrieves documents using vector similarity."
    )

    assert len(memory.turns) == 1
    assert memory.turns[0].role == "assistant"
    assert memory.turns[0].content == (
        "Semantic search retrieves documents using vector similarity."
    )


def test_memory_preserves_message_order():
    memory = ConversationMemory(
        max_turns = 5,
    )

    memory.add_user(
        "What is semantic search?"
    )

    memory.add_assistant(
        "Semantic search retrieves documents using vector similarity."
    )

    memory.add_user(
        "What about lexical search?"
    )

    assert len(memory.turns) == 3
    assert memory.turns[0].role == "user"
    assert memory.turns[1].role == "assistant"
    assert memory.turns[2].role == "user"


def test_memory_builds_context():
    memory = ConversationMemory(
        max_turns = 5,
    )

    memory.add_user(
        "What is semantic search?"
    )

    memory.add_assistant(
        "Semantic search retrieves documents using vector similarity."
    )

    context = memory.build_context()

    expected = (
        "USER:\n"
        "What is semantic search?\n\n"
        "ASSISTANT:\n"
        "Semantic search retrieves documents using vector similarity."
    )

    assert context == expected


def test_memory_ignores_empty_messages():
    memory = ConversationMemory(
        max_turns = 5,
    )

    memory.add_user("")
    memory.add_user("   ")
    memory.add_assistant("")
    memory.add_assistant("   ")

    assert memory.turns == []


def test_memory_trims_old_messages():
    memory = ConversationMemory(
        max_turns = 2,
    )

    memory.add_user("Question 1")
    memory.add_assistant("Answer 1")

    memory.add_user("Question 2")
    memory.add_assistant("Answer 2")

    memory.add_user("Question 3")
    memory.add_assistant("Answer 3")

    assert len(memory.turns) == 4

    assert memory.turns[0].content == "Question 2"
    assert memory.turns[1].content == "Answer 2"
    assert memory.turns[2].content == "Question 3"
    assert memory.turns[3].content == "Answer 3"


def test_memory_clear_removes_all_messages():
    memory = ConversationMemory(
        max_turns = 5,
    )

    memory.add_user(
        "What is semantic search?"
    )

    memory.add_assistant(
        "Semantic search retrieves documents using vector similarity."
    )

    memory.clear()

    assert memory.turns == []
    assert memory.build_context() == ""


def test_memory_rejects_zero_max_turns():
    with pytest.raises(
        ValueError,
        match = "max_turns must be greater than 0",
    ):
        ConversationMemory(
            max_turns = 0,
        )


def test_memory_rejects_negative_max_turns():
    with pytest.raises(
        ValueError,
        match = "max_turns must be greater than 0",
    ):
        ConversationMemory(
            max_turns = -1,
        )
