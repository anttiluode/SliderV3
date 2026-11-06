# Five-seed replication: blind waveform readout

Five independent runs of `blind_readout_benchmark.py`, seeds **20261106–20261110**, using **48 training pairs and 24 test pairs per seed**. Each split holds out entire suffix strings for lengths above 0 and draws fresh starting-edge perturbations (absolute SD 0.002). A single read means one isolated probe response; paired classification receives both matched responses.

## Mean accuracy across seeds

| Extra common suffix events | Wave single | Wave paired | ESN single | ESN paired | Wave single min–max |
|---:|---:|---:|---:|---:|---:|
| 0 | 91.7% | 100.0% | 100.0% | 100.0% | 87.5%–95.8% |
| 2 | 71.7% | 86.7% | 100.0% | 100.0% | 64.6%–81.2% |
| 4 | 69.6% | 88.3% | 100.0% | 100.0% | 62.5%–75.0% |
| 8 | 66.7% | 85.8% | 82.1% | 100.0% | 56.2%–72.9% |

## Noise sweep at suffix length 2

| Gaussian measurement noise (σ / clean RMS) | Wave single | Wave paired |
|---:|---:|---:|
| 0% | 71.7% | 86.7% |
| 1% | 62.5% | 81.7% |
| 3% | 61.3% | 71.7% |
| 10% | 53.3% | 58.3% |

## Non-writing probe placement, suffix length 2 (clean)

| Probe configuration | Wave single | Wave paired |
|---|---:|---:|
| 196 Hz, node 6 | 71.7% | 86.7% |
| 160 Hz, node 1 | 72.9% | 96.7% |
| 310 Hz, node 4 | 80.4% | 94.2% |

## Controls at suffix length 2 (clean)

| Control | Single | Paired |
|---|---:|---:|
| no_learning | 50.0% | 50.0% |
| erased | 50.0% | 50.0% |
| fixed_propagation | 50.0% | 50.0% |
| shuffled_train_labels | 55.4% | 50.8% |

**Interpretation.** The waveform carries order-related information after the fast state is reset, and some of it generalizes to unseen suffix strings and new initial matrices. But the linear decoder degrades under longer suffixes and weak additive measurement noise; it does **not** outperform the recurrent baseline under the tested conditions. We did not preregister an architecture search or examine physical media. Random initial weights are small synthetic perturbations, not manufacturing tolerances. The ESN sees its own final recurrent state and is not matched for cost or readout constraints.

**Statistics.** Means across five deterministic simulation seeds are descriptive. The 24 paired test observations per seed and the shared pair-level nuisances limit precision; do not interpret a few percentage points as significant. Paired results are an easier **two-choice task** than the independent single-read results. The balanced no-information controls are scored at 50% (chance) when the two waveforms match to numerical precision.

**Reproduce.** In PowerShell from the repository root:

```powershell
python3.13 -m pip install numpy matplotlib
python3.13 blind_readout_benchmark.py --selftest
foreach ($s in 20261106..20261110) { python3.13 blind_readout_benchmark.py --seed $s --out ("seed_" + $s) }
```

The optional Numba JIT is enabled if available and verifies its outputs against the original simulator before use; without Numba the computation is identical but much slower. To explicitly disable acceleration, use `--no-jit`.

See [single-seed details](BLIND_READOUT_RESULTS.md), [raw single-seed results](blind_readout_results.json), [multi-seed numerical summary](blind_readout_multiseed.json), and [benchmark script](blind_readout_benchmark.py).