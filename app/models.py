"""Shared Pydantic models for OpenWebUI <-> proxy <-> Exa."""
from __future__ import annotations

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User search query from OpenWebUI")
    count: int = Field(default=5, ge=1, le=20, description="Max results requested")


class SearchResult(BaseModel):
    link: str
    title: str | None = None
    snippet: str | None = None
