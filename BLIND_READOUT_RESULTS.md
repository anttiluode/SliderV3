# Blind Readout Benchmark — Measured Results

Generated from `blind_readout_benchmark.py` using seed 20261106, 48 training pairs, 24 held-out test pairs per condition, initial-edge jitter σ=0.002.

**Question:** after writing `AB + suffix` or `BA + suffix`, can a decoder identify order from later **non-writing** signals without inspecting `W`? Both classes use the same suffix and the same starting graph within each matched pair. For suffix length > 0, suffix strings are disjoint between training and test.

**Two metrics:** single = classify each waveform independently; paired = choose which of two matched waveforms came from AB (two-alternative forced choice). Pairing removes shared nuisance graph variation and is an *easier task*; do not conflate it with single-trial recall. A test set has 24 pairs, so pair accuracy changes in steps of 4.17 percentage points.

## Sequence length / forgetting

| Extra suffix length | Wave single | Wave paired | ESN single | ESN paired | Median clean probe separation |
|---:|---:|---:|---:|---:|---:|
| 0 | 87.5% | 100.0% | 100.0% | 100.0% | 2.958% |
| 2 | 81.2% | 95.8% | 100.0% | 100.0% | 2.206% |
| 4 | 68.8% | 91.7% | 100.0% | 100.0% | 2.533% |
| 8 | 72.9% | 91.7% | 79.2% | 100.0% | 2.026% |

## Probe measurement noise (suffix length 2)

Noise σ is a specified percentage of clean training-probe RMS; Gaussian noise is added to both training and test probe traces. New decoder per noise level.

| Measurement noise / clean RMS | Single | Paired | 95% Wilson interval for paired |
|---:|---:|---:|---:|
| 0.0% | 81.2% | 95.8% | 79.8–99.3% |
| 1.0% | 70.8% | 91.7% | 74.2–97.7% |
| 3.0% | 60.4% | 70.8% | 50.8–85.1% |
| 10.0% | 54.2% | 62.5% | 42.7–78.8% |

## Probe placement and frequency (suffix length 2, clean)

| Probe | Frequency / input node | Single | Paired | Median read separation |
|---|---|---:|---:|---:|
| default | 196 Hz / 6 | 81.2% | 95.8% | 2.206% |
| low_other_port | 160 Hz / 1 | 81.2% | 100.0% | 0.279% |
| high_other_port | 310 Hz / 4 | 91.7% | 100.0% | 10.842% |

## Negative controls (suffix length 2, clean)

| Condition | Single | Paired | Maximum exact AB/BA waveform difference |
|---|---:|---:|---:|
| no_learning | 50.0% | 50.0% | 0 |
| erased | 50.0% | 50.0% | 0 |
| fixed_propagation | 50.0% | 50.0% | 4.72e-16 |
| shuffled_train_labels | 50.0% | 58.3% | n/a (labels only) |

**Important control interpretation:** if paired waveforms agree to numerical precision, the forced-choice result is credited as **50% (chance)** rather than letting rounding-level differences count as information. Single-trial accuracy is also 50% in the balanced no-information controls. A nonzero paired score in the shuffled-label control may happen by chance on a small test set.

## Interpretation / limitations

- An ESN is a conventional 96-unit tanh recurrent baseline with a fixed random reservoir and a linear readout trained on its final state. It is **not** an acoustically matched probe and can keep fast state; its scores are context, not a capacity-normalized hardware comparison.
- Classification labels are used for training, but the wave decoder only receives 128 fixed samples from each *non-writing* waveform, never a weight matrix or the input labels.
- Noise is synthetic i.i.d. measurement noise. The source model is deterministic and a deliberately designed write law. Suffix train/test splits and randomized initial graphs probe some generalization, not all forms of robustness.
- The paired metric has access to **two matched probe signals** at inference time. This can cancel shared nuisance variation; single-waveform accuracy is the stricter claim about blind recall.
- Results are one seeded run with small sample sizes and no claim of statistical significance against stronger baselines. Run alternative seeds before interpreting small differences.

Reproduce with `python blind_readout_benchmark.py`. Full machine-readable scores, 2×2 confusion matrices, settings, and Wilson intervals: [`blind_readout_results.json`](blind_readout_results.json).