from typing import List
from pydantic import BaseModel, Field, validator, root_validator


class SourceMatch(BaseModel):
    # Breaks in v2: 'regex' removed -> use 'pattern'
    kb_id: str = Field(..., regex=r"^KB-\d{2}$")
    fact: str

    # Breaks in v2: 'class Config' -> use model_config = ConfigDict(...)
    class Config:
        anystr_strip_whitespace = True
        extra = "forbid"


class AIQueryResult(BaseModel):
    query: str
    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    # Breaks in v2: 'min_items' removed -> use 'min_length'
    sources: List[SourceMatch] = Field(..., min_items=1)
    verified: bool = True

    class Config:
        anystr_strip_whitespace = True

    # Breaks in v2: @validator(pre=True) -> use @field_validator(mode="before") + @classmethod
    @validator("query", pre=True)
    def ensure_question(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip().endswith("?"):
            raise ValueError("Query must end with '?'")
        return v.strip()

    # Breaks in v2: @root_validator -> use @model_validator(mode="after")
    @root_validator
    def validate_verification_threshold(cls, values: dict) -> dict:
        if values.get("confidence", 0.0) < 0.6 and values.get("verified", False):
            raise ValueError("Low-confidence answers cannot be marked verified")
        return values
