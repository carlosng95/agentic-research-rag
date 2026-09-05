from .types import Turn


class ConversationMemory:
    """
    Store conversation turns for the current session.
    """

    def __init__(self, max_turns: int = 10) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns must be greater than 0")

        self._max_turns = max_turns
        self._turns: list[Turn] = []

    @property
    def turns(self) -> list[Turn]:
        return list(self._turns)

    def add_user(self, content: str) -> None:
        if not content.strip():
            return

        self._turns.append(
            Turn(
                role = "user",
                content = content,
            )
        )

        self._trim()

    def add_assistant(self, content: str) -> None:
        if not content.strip():
            return

        self._turns.append(
            Turn(
                role = "assistant",
                content = content,
            )
        )

        self._trim()

    def _trim(self) -> None:
        max_messages = self._max_turns * 2

        if len(self._turns) > max_messages:
            self._turns = self._turns[-max_messages:]

    def build_context(self) -> str:
        if not self._turns:
            return ""

        lines: list[str] = []

        for turn in self._turns:
            role = turn.role.upper()
            lines.append(f"{role}:\n{turn.content}")

        return "\n\n".join(lines)

    def clear(self) -> None:
        self._turns.clear()