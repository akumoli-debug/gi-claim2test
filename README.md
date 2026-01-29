# gi-claim2test

**Automated validation of empirical claims from reinforcement learning papers through standardized observational testing.**

## Overview

Scientific papers make empirical claims, but full replication takes months. This pipeline validates core claims in hours by:

1. **Extracting claims** from papers using Claude API
2. **Mapping claims** to standardized observational tests  
3. **Running experiments** with controlled comparisons
4. **Generating reports** with reproducibility verdicts

Instead of replicating entire studies, we probe specific claims with targeted tests.

## Quick Start
```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download papers
python scripts/download_papers.py

# Extract claims (using LLM)
python scripts/01_extract_claims.py --method llm

# Run observational tests
python scripts/02_run_experiments.py --test A --timesteps 200000

# Generate report
python scripts/03_make_report.py
```

## Example Result (V1)

**Paper:** MemRL: Self-Evolving Agents via Runtime RL on Episodic Memory (Jan 2025)

**Claim:** "Memory-based learning improves performance in partially observable environments"

**Test A Result:**
- Baseline (PPO): 55% success
- Memory (RecurrentPPO): 100% success
- **Verdict: REPRODUCED ✓** (+45% improvement)

Full report: [`results/report.md`](results/report.md)

## Observational Tests

### Test A: Memory & Partial Observability
- **Environment:** MiniGrid-DoorKey-5x5
- **Comparison:** Baseline PPO vs RecurrentPPO (LSTM)
- **Evaluates:** Whether memory mechanisms help in tasks requiring recall
- **Runtime:** ~1 hour on laptop CPU

### Test B: Out-of-Distribution Generalization  
- **Environment:** MiniGrid with varied layouts/sizes
- **Comparison:** Performance on novel configurations
- **Evaluates:** Whether techniques improve generalization
- **Runtime:** ~1 hour on laptop CPU

## Repository Structure

```
gi-claim2test/
├── claims/
│   ├── papers.csv              # Paper metadata & arXiv URLs
│   ├── pdfs/                   # Downloaded PDFs (gitignored)
│   └── extracted_claims.jsonl  # Extracted claims (gitignored)
├── scripts/
│   ├── download_papers.py     # Fetch PDFs from arXiv (requests + retries)
│   ├── 01_extract_claims.py   # Claim extraction (LLM or heuristic)
│   ├── 02_run_experiments.py  # Train & evaluate baseline vs intervention
│   ├── 03_make_report.py      # Build markdown report from runs + claims
│   └── models.py              # Pydantic models for claims
├── results/                    # Runs CSV, report, plots (gitignored)
├── requirements.txt
└── README.md
```

**Dependencies:** Python 3.10+, Gymnasium, MiniGrid, Stable-Baselines3, SB3-Contrib, PyMuPDF, Anthropic SDK (for LLM extraction), requests.
