#!/usr/bin/env python3
"""
Download paper PDFs from arXiv using claims/papers.csv.
Uses requests with retries. Run from project root or any subdirectory.
"""
import csv
import re
import sys
from pathlib import Path

import requests

# Find project root (directory containing requirements.txt or claims/)
def find_project_root() -> Path:
    start = Path(__file__).resolve().parent
    for candidate in [start] + list(start.parents):
        if (candidate / "requirements.txt").exists() and (candidate / "claims").is_dir():
            return candidate
    return start.parent  # fallback to parent of scripts/

def extract_arxiv_id(url: str) -> str | None:
    """Extract arXiv ID from abs or pdf URL. E.g. 2601.03192 from https://arxiv.org/abs/2601.03192"""
    if not url:
        return None
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+)", url, re.I)
    return m.group(1) if m else None

def main() -> None:
    root = find_project_root()
    csv_path = root / "claims" / "papers.csv"
    pdfs_dir = root / "claims" / "pdfs"
    pdfs_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"Error: {csv_path} not found.", file=sys.stderr)
        sys.exit(1)

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    success_count = 0
    for i, row in enumerate(rows):
        source_url = (row.get("source_url") or "").strip()
        if not source_url:
            continue
        arxiv_id = extract_arxiv_id(source_url)
        if not arxiv_id:
            print(f"Skipping paper (no arXiv ID): {row.get('title', '?')}")
            continue

        pdf_path_val = (row.get("pdf_path") or "").strip()
        if pdf_path_val:
            out_path = root / "claims" / "pdfs" / Path(pdf_path_val).name
            rel_pdf_path = f"claims/pdfs/{Path(pdf_path_val).name}"
        else:
            out_path = pdfs_dir / f"{arxiv_id}.pdf"
            rel_pdf_path = f"claims/pdfs/{arxiv_id}.pdf"

        title = row.get("title", "?")
        print(f"Downloading paper {i+1}: {title}...")

        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        last_error = None
        for attempt in range(3):
            try:
                r = requests.get(pdf_url, timeout=60)
                r.raise_for_status()
                out_path.write_bytes(r.content)
                row["pdf_path"] = rel_pdf_path
                success_count += 1
                break
            except Exception as e:
                last_error = e
                if attempt < 2:
                    print(f"  Retry {attempt+2}/3...")
        else:
            print(f"  Failed after 3 attempts: {last_error}")

    # Write back CSV so pdf_path points to claims/pdfs/ for extract script
    if success_count and fieldnames:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nDownloaded {success_count} papers successfully. Run: python scripts/01_extract_claims.py --method llm")

if __name__ == "__main__":
    main()
