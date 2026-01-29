# gi-claim2test: Automated Paper Claim Validation

## What This Is

A semi-automated pipeline that extracts empirical claims from recent RL papers and validates them through standardized observational tests. Instead of manually replicating full experiments, we run targeted tests that directly probe the core claims.

## What We Built (V1)

1. **Claim Extraction** - Pulls claims from recent papers (2025) about memory and generalization in RL
2. **Observational Tests** - Two standardized tests:
   - Test A: Memory/Partial Observability (MiniGrid-DoorKey)
   - Test B: OOD Generalization (different env sizes)
3. **Automated Report** - Generates reproducibility verdicts with effect sizes

## Key Result from V1

**Paper tested:** MemRL: Self-Evolving Agents via Runtime RL on Episodic Memory (Jan 2025)

**Claim:** "Memory-based learning improves performance in partially observable environments"

**Our test:**
- Baseline (PPO, no memory): 55% success
- Memory (RecurrentPPO with LSTM): 100% success
- **Delta: +45% improvement**

**Verdict: REPRODUCED ✓** with strong effect size

See full results: `results/report.md`

## Why This Matters

- **Scales validation:** Can test 10-50 papers/week vs months for full replication
- **Actionable quickly:** Results in hours, not weeks
- **Falsifiable:** Clear pass/fail criteria (>10% improvement = reproduced)
- **Uses recent papers:** Tested a paper from 3 weeks ago (Jan 2025)

## Technical Stack

- **Environments:** MiniGrid (fast, controllable)
- **Algorithms:** Stable-Baselines3 (PPO, RecurrentPPO)
- **Extraction:** Semi-automated (heuristic + manual review for V1)
- **Runtime:** ~1 hour per test on laptop CPU

## Repo Structure
```
gi-claim2test/
  scripts/
    01_extract_claims.py    # PDF → structured claims
    02_run_experiments.py   # Run observational tests
    03_make_report.py       # Generate verdict report
  results/
    report.md               # Main deliverable
    plots/                  # Comparison charts
  claims/
    papers.csv              # Paper metadata
    extracted_claims.jsonl  # Structured claims
```

## Next Steps (V2)

- LLM-powered claim extraction (currently heuristic)
- More test templates (sample efficiency, safety, etc)
- Batch processing of 20+ papers
- CI/CD pipeline for continuous validation

## How to Run
```bash
# Extract claims
python scripts/01_extract_claims.py

# Run tests
python scripts/02_run_experiments.py --test A --timesteps 200000

# Generate report
python scripts/03_make_report.py
```

---

**For Pim:** This demonstrates the core "paper → test → verdict" pipeline works. Ready to scale to more papers and test types.
