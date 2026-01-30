# gi-claim2test: Paper Claim Validation Pipeline

## Summary

Semi-automated pipeline that extracts empirical claims from RL papers and validates them via standardized observational tests. The pipeline runs targeted tests that probe specific claims rather than full replications.

## V1 Components

1. **Claim extraction** — Extracts claims from papers (2025 focus) on memory and generalization in RL (heuristic or LLM-assisted).
2. **Observational tests** — Two standardized tests: Test A (memory/partial observability, MiniGrid-DoorKey); Test B (OOD generalization, varied env sizes).
3. **Report generation** — Produces reproducibility verdicts with effect sizes.

## Example result (V1)

**Paper:** MemRL: Self-Evolving Agents via Runtime RL on Episodic Memory (Jan 2025)

**Claim:** "Memory-based learning improves performance in partially observable environments"

**Test result:**
- Baseline (PPO, no memory): 55% success
- Memory (RecurrentPPO with LSTM): 100% success
- Delta: +45% improvement

**Verdict:** REPRODUCED ✓ (strong effect size)

Full results: `results/report.md`

## Design notes

- **Falsifiable criteria:** Pass/fail threshold (e.g. >10% improvement) for reproduced vs not.
- **Runtime:** On the order of hours per test for default timesteps on laptop CPU; depends on hardware and settings.
- **Scope:** V1 uses a small set of recent papers; throughput depends on compute budget and how many tests map to each paper.

## Technical stack

- **Environments:** MiniGrid
- **Algorithms:** Stable-Baselines3 (PPO, RecurrentPPO)
- **Extraction:** Heuristic or Claude API (V1)
- **Runtime:** ~1 hour per test on laptop CPU for default timesteps

## Repo structure

```
gi-claim2test/
  scripts/
    01_extract_claims.py    # PDF → structured claims
    02_run_experiments.py   # Run observational tests
    03_make_report.py       # Generate verdict report
  results/
    report.md
    plots/
  claims/
    papers.csv
    extracted_claims.jsonl
```

## Possible extensions (V2)

- LLM-powered claim extraction (partially in place)
- Additional test templates (e.g. sample efficiency, safety)
- Batch runs over larger paper sets
- CI for automated validation runs

## How to run

```bash
python scripts/01_extract_claims.py
python scripts/02_run_experiments.py --test A --timesteps 200000
python scripts/03_make_report.py
```
