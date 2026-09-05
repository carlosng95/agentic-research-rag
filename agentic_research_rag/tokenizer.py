from abc import ABC, abstractmethod
import re


class Tokenizer(ABC):
    """
    Interface implemented by text tokenizers.
    """

    @abstractmethod
    def tokenize(
        self,
        text: str,
    ) -> list[str]:
        """
        Convert text into a sequence of tokens.
        """

        raise NotImplementedError


class RegexTokenizer(Tokenizer):
    """
    Simple word tokenizer based on a regular expression.

    Text is converted to lowercase before tokenization.
    """

    def tokenize(
        self,
        text: str,
    ) -> list[str]:
        return re.findall(
            r"\b\w+\b",
            text.lower(),
        )


class WhitespaceTokenizer(Tokenizer):
    """
    Simple tokenizer that splits text on whitespace.
    """

    def tokenize(
        self,
        text: str,
    ) -> list[str]:
        return text.lower().split()