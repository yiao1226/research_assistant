"""Search endpoint schemas."""

from pydantic import BaseModel, Field
from typing import Literal


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    sources: list[str] = Field(default=["arxiv"])
    sort_by: Literal["relevance", "recency", "citations"] = "relevance"
    username: str = Field(default="eval")


class PaperItem(BaseModel):
    title: str
    arxiv_id: str = ""
    year: str = ""
    composite_score: float = 0.0
    citation_count: int = 0
    core_contribution: str = ""
    abstract: str = ""


class SearchResponse(BaseModel):
    papers: list[dict]
    total_found: int
    search_focus: str = ""
    duration_sec: float = 0.0
