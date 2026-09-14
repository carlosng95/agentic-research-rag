from typing import Literal
from pydantic import BaseModel, Field, field_validator

class ResearchRequest(BaseModel):
    question: str = Field(min_length = 1)
    thread_id: str = Field(default = "default", min_length = 1)

    @field_validator("question", "thread_id")
    @classmethod
    def strip_non_empty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Value cannot be empty.")
        return value

class HealthResponse(BaseModel):
    status: Literal["healthy"] = "healthy"

class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    detail: str | None = None