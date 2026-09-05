from abc import ABC, abstractmethod


class Tool(ABC):
    """
    Interface implemented by tools available to agents.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def run(self, query: str):
        raise NotImplementedError