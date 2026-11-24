# Federated-MARTI: verified results (paper version, November 2026)

All numbers come from the final experiment bundle (run on a MacBook, arm64, 8 cores, PyTorch 2.13). They are produced by a single script from saved predictions; the two headline paired differences were recomputed independently from the bootstrap replicates.

**Benchmark:** synthetic multi-bank dataset v2 (this repository's generator, seed 42): 8 banks, trained January–August 2024, validated September–October, tested November–December 2024 (191,352 test transactions, 1.42% fraud).

**Model:** each bank keeps its own gradient-boosted model (GBDT). Three experts are trained across banks: a heterogeneous graph attention network over accounts, devices and transfers; a behavioural Transformer over session event sequences; and a drift expert. A context-aware gate fuses the four in log-odds space. Under differential privacy (DP) the fusion stays inside each bank, and a bank keeps its own GBDT if the fusion does not beat it on its validation months.

## Main comparison

PR-AUC on the test period; mean ± s.d. over seeds 42, 1 and 2 where given; 95% bootstrap CI for seed 42.

| Model | PR-AUC | 95% CI |
|---|---|---|
| Rule system | 0.184 | – |
| GBDT, each bank alone | 0.883 | [0.867, 0.898] |
| GBDT, pooled data (not deployable) | 0.950 | [0.941, 0.958] |
| Federated-MARTI, each bank alone | 0.886 ± 0.006 | [0.876, 0.905] |
| **Federated-MARTI, federated** | **0.924 ± 0.001** | [0.911, 0.935] |
| Federated-MARTI, pooled data (not deployable) | 0.962 ± 0.003 | [0.956, 0.973] |
| + DP experts, local fusion (ε ≤ 4.83, δ = 1e-6) | 0.882 ± 0.004 | [0.872, 0.902] |
| Three experts without GBDT, federated | 0.902 | [0.888, 0.916] |
| + DP, fully federated (ε ≤ 6.25) | 0.587 | [0.556, 0.613] |

- **Federated vs each bank's own GBDT:** +0.040 [+0.032, +0.052], significant.
- **Federated vs the same architecture trained by each bank alone:** +0.032 [+0.025, +0.039]. Federation recovers 50% of the gap between local-only and pooled training, and improves every bank in every seed.
- **Gate vs the bank's tabular expert alone (0.885):** +0.038 [+0.028, +0.048].
- **Against the rules:** at the rule system's recall (27.4%) the model raises 2 false alerts against the rules' 1,349. At the rules' alert volume it catches 73.5% of fraud.

## Differential privacy: an honest negative result

With DP experts and local fusion, the gain over each bank's own GBDT is +0.004 [−0.002, +0.011]: not distinguishable from zero. The adoption rule kept the bank's own model at 8 of 8, 8 of 8 and 7 of 8 banks (seeds 42, 1, 2). Across noise levels σ = 1, 2, 4, 8 (worst-bank ε 12.44, 4.83, 3.02, 2.52) PR-AUC stays at 0.886 because of that fallback. **At these privacy budgets the private experts are safe but add no measurable value.** Making DP useful is the main open problem.

## Robustness (federated parts not retrained)

| Population | Non-private | DP |
|---|---|---|
| In-distribution (seed 42) | 0.923 | 0.886 |
| New population, seed 7, sophistication x2 | 0.915 | 0.875 |
| New population, seed 11 | 0.916 | 0.886 |

## Real data: Elliptic (graph component only)

Bitcoin transaction graph, 3 seeds, threshold 0.5.

| Model | F1 | PR-AUC |
|---|---|---|
| GAT, all features | 0.498 | 0.506 |
| Random forest | 0.809 | – |
| Published baselines (Weber et al.): RF / EvolveGCN | 0.788 / 0.720 | – |

On a 4-institution split, federated training reaches F1 0.417 against 0.393 for local-only and 0.411 for centralised training without cross-institution edges. The graph expert on its own is clearly weaker than a tuned random forest on Elliptic, which is why the final design keeps each bank's GBDT as the tabular expert.

## Operations

| Measure | Value |
|---|---|
| End-to-end HTTP latency at 100 transactions/s | p50 12.7 ms, p95 18.3 ms (budget 80 ms) |
| Throughput of one process | saturates near 84 transactions/s |
| Calibration error after calibration (ECE) | 0.0005 |
| Network traffic per bank, full training run | 142.2 MB (non-private), 6.9 MB (DP) |

## Stated limitations

- The main benchmark is synthetic (this repository's generator); Elliptic covers only the graph component.
- First-party fraud stays hard (PR-AUC 0.077): the genuine customer uses their own device and behaviour.
- Honest-but-curious threat model; client dropout is handled by restarting rounds.
- DP hyperparameters were fixed during development on the same synthetic population.
