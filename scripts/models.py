from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class Claim(BaseModel):
    """A single claim extracted from a paper."""

    claim_text: str
    intervention: Optional[str] = None
    capability: Optional[str] = None
    conditions: Optional[str] = None


class PaperClaims(BaseModel):
    """Collection of claims associated with a single paper."""

    paper_id: str
    title: str
    year: int
    claims: List[Claim]

