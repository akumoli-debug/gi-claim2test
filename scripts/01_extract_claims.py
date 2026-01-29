from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import List


def _find_project_root() -> Path:
    """Project root: directory containing requirements.txt and claims/."""
    start = Path(__file__).resolve().parent
    for candidate in [start] + list(start.parents):
        if (candidate / "requirements.txt").exists() and (candidate / "claims").is_dir():
            return candidate
    return start.parent


def _extract_arxiv_id(url: str) -> str | None:
    """Extract arXiv ID from abs/pdf URL, e.g. 2601.03192 from https://arxiv.org/abs/2601.03192"""
    if not url or not isinstance(url, str):
        return None
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+)", url.strip(), re.I)
    return m.group(1) if m else None


import fitz  # PyMuPDF
import pandas as pd
from anthropic import Anthropic

try:
    # When run as a module: python -m scripts.01_extract_claims
    from scripts.models import Claim, PaperClaims
except ModuleNotFoundError:  # pragma: no cover - fallback for direct script run
    from models import Claim, PaperClaims


KEYWORDS = [
    "memory",
    "recurrent",
    "lstm",
    "partial observability",
    "generalization",
    "out-of-distribution",
    "ood",
]


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    text_parts: List[str] = []

    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text_parts.append(page.get_text())
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Error reading PDF '{pdf_path}': {exc}")
        return ""

    return "\n".join(text_parts)


def _infer_capability(sentence_lower: str) -> str | None:
    if any(k in sentence_lower for k in ["memory", "recurrent", "lstm", "partial observability"]):
        return "memory"
    if any(k in sentence_lower for k in ["generalization", "out-of-distribution", "ood"]):
        return "generalization"
    return None


def _infer_intervention(sentence: str) -> str | None:
    sentence_lower = sentence.lower()
    if "lstm" in sentence_lower:
        return "LSTM"
    if "recurrent" in sentence_lower:
        return "recurrent policy"
    if "memory" in sentence_lower:
        return "memory mechanism"
    return None


def extract_claims_heuristic(text: str, paper_id: str, title: str, year: int) -> PaperClaims:
    """Very rough heuristic extraction of candidate claims from text."""
    claims: List[Claim] = []

    if text:
        # Naive sentence split – good enough for a first pass.
        raw_sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]

        for sent in raw_sentences:
            sent_lower = sent.lower()
            if any(keyword in sent_lower for keyword in KEYWORDS):
                capability = _infer_capability(sent_lower)
                intervention = _infer_intervention(sent)
                claims.append(
                    Claim(
                        claim_text=sent + ".",
                        intervention=intervention,
                        capability=capability,
                        conditions=None,
                    )
                )

    if not claims:
        claims.append(
            Claim(
                claim_text="Requires manual review",
                intervention=None,
                capability=None,
                conditions=None,
            )
        )

    return PaperClaims(
        paper_id=str(paper_id),
        title=title,
        year=int(year),
        claims=claims,
    )


def _extract_claude_text(message) -> str:
    """
    Anthropic SDK typically returns `message.content` as a list of blocks.
    We defensively normalize to a single string.
    """
    try:
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    parts.append(str(block["text"]))
                else:
                    parts.append(str(getattr(block, "text", "")))
            return "".join(parts).strip()
    except Exception:
        pass
    return str(message)


def extract_claims_with_llm(text: str, paper_id: str, title: str, year: int) -> PaperClaims:
    """
    Extract 3-5 testable empirical claims using Claude.
    Falls back to `extract_claims_heuristic` on any failure.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("LLM extraction requested but ANTHROPIC_API_KEY is not set; falling back to heuristic.")
        return extract_claims_heuristic(text, paper_id, title, year)

    try:
        client = Anthropic(api_key=api_key)
    except Exception as exc:
        print(f"Failed to initialize Anthropic client ({exc}); falling back to heuristic.")
        return extract_claims_heuristic(text, paper_id, title, year)

    prompt = f"""You are a research assistant analyzing reinforcement learning papers.
      
Extract 3-5 testable empirical claims from this paper excerpt.
      
For each claim, identify:
- claim_text: The specific empirical claim (1-2 sentences)
- intervention: What technique/method is being claimed to help (e.g., "LSTM memory", "recurrent networks", "data augmentation")
- capability: What improves (e.g., "partial observability", "generalization", "sample efficiency", "long-horizon tasks")
- conditions: Under what conditions this holds (e.g., "in sparse reward environments", "with limited observability")
      
Focus on claims that:
- Are empirically testable (not theoretical)
- Compare methods (X is better than Y)
- Have measurable outcomes
      
Paper title: {title}
      
Paper excerpt (first 8000 chars):
{text[:8000]}
      
Return ONLY valid JSON in this exact format:
{{
  "claims": [
    {{
      "claim_text": "...",
      "intervention": "...",
      "capability": "...",
      "conditions": "..."
    }}
  ]
}}
"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        print(f"Claude API call failed ({exc}); falling back to heuristic.")
        return extract_claims_heuristic(text, paper_id, title, year)

    raw = _extract_claude_text(message)
    try:
        payload = json.loads(raw)
        claim_items = payload.get("claims", [])
        if not isinstance(claim_items, list) or not claim_items:
            raise ValueError("JSON missing non-empty 'claims' list")

        claims: List[Claim] = []
        for item in claim_items:
            if not isinstance(item, dict):
                continue
            claims.append(
                Claim(
                    claim_text=str(item.get("claim_text", "")).strip() or "Requires manual review",
                    intervention=(str(item.get("intervention")).strip() if item.get("intervention") is not None else None),
                    capability=(str(item.get("capability")).strip() if item.get("capability") is not None else None),
                    conditions=(str(item.get("conditions")).strip() if item.get("conditions") is not None else None),
                )
            )

        if not claims:
            raise ValueError("Parsed 0 claims")

        return PaperClaims(paper_id=str(paper_id), title=title, year=int(year), claims=claims)
    except Exception as exc:
        print(f"Failed to parse Claude JSON response ({exc}); falling back to heuristic.")
        print(f"Claude raw response (first 300 chars): {raw[:300]!r}")
        return extract_claims_heuristic(text, paper_id, title, year)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["llm", "heuristic"], default="llm")
    args = parser.parse_args()

    root = _find_project_root()
    papers_csv = root / "claims" / "papers.csv"
    output_path = root / "claims" / "extracted_claims.jsonl"

    if not papers_csv.exists():
        print(f"papers.csv not found at {papers_csv}")
        return

    df = pd.read_csv(papers_csv)

    results: List[PaperClaims] = []

    for _, row in df.iterrows():
        paper_id = row.get("paper_id")
        title = row.get("title", "")
        year = int(row.get("year", 0)) if not pd.isna(row.get("year")) else 0
        pdf_path = row.get("pdf_path", "")
        
        # Handle NaN values from pandas
        if pd.isna(pdf_path):
            pdf_path = ""

        # Resolve PDF path: use pdf_path from CSV, or fallback to claims/pdfs/{arxiv_id}.pdf from source_url
        if isinstance(pdf_path, str) and pdf_path.strip():
            pdf_full = root / pdf_path.strip() if not os.path.isabs(pdf_path.strip()) else Path(pdf_path.strip())
        else:
            source_url = row.get("source_url", "") or ""
            arxiv_id = _extract_arxiv_id(str(source_url))
            if not arxiv_id:
                print(f"Skipping paper {paper_id}: no pdf_path or arXiv source_url")
                continue
            pdf_full = root / "claims" / "pdfs" / f"{arxiv_id}.pdf"
        if not pdf_full.exists():
            print(f"Skipping paper {paper_id}: PDF not found at '{pdf_full}'")
            continue

        print(f"Processing paper {paper_id}: {title}")
        text = extract_text_from_pdf(str(pdf_full))
        if args.method == "llm":
            paper_claims = extract_claims_with_llm(text, str(paper_id), title, year)
        else:
            paper_claims = extract_claims_heuristic(text, str(paper_id), title, year)
        results.append(paper_claims)

    if not results:
        print("No claims extracted; nothing to write.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pc in results:
            json.dump(pc.model_dump(), f)
            f.write("\n")

    print(f"Wrote {len(results)} papers' claims to {output_path}")


if __name__ == "__main__":
    main()

