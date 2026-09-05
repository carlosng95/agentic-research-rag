from dataclasses import dataclass, field
from typing import Literal

@dataclass
class Page:
    """One extracted page from a PDF document"""
    document_name: str
    page_number: int
    text:str
    
@dataclass
class Chunk:
    """A Searchable fragment extracted from a document"""
    chunk_id: int
    document_name: str
    page_number: int
    text: str
    score: float | None = None
    
@dataclass
class Source:
    """A source supporting the final answer"""
    type: Literal['paper','web']
    ref: str
    locator: str | None = None
    snippet: str | None = None
    
@dataclass
class Turn:
    """One message in a conversation"""
    role: Literal['user','assistant']
    content: str
    
@dataclass
class ResearchResponse:
    """Structured response produced by the research agent"""
    answer: str
    sources: list[Source] = field(default_factory = list)
    reasoning_steps: list[str] = field(default_factory = list)
    tools_used: list[str] = field(default_factory = list)
    confidence: float = 0.0
    flags: list[str] = field(default_factory = list)

    