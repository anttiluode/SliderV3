# Resource-matched memory comparison (measured)

This experiment replaces the earlier **96-state, directly observed ESN** with explicit **16 mutable-scalar** memory baselines and a restricted **128-sample one-sensor read**. All models receive the same sequence labels, suffix splits, training pairs, decoder and relative-noise convention. **Result is a comparison of algorithms and readout interfaces, not a fair hardware/energy benchmark**.

## Single-read classification, held-out suffix strings

5 seeds, 48 training AB/BA pairs and 24 held-out pairs per seed; means across seeds; 50% chance.

| Suffix events | Wave (16 edges) | Sparse ring (16) | ESN (8 states) | ESN (16 states) | Trace bank (16) | Additive bag (16) | ESN (32 states) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 90.0% | 100.0% | 100.0% | 100.0% | 100.0% | 50.0% | 100.0% |
| 2 | 70.4% | 60.0% | 90.0% | 100.0% | 56.7% | 50.0% | 90.0% |
| 4 | 71.7% | 61.3% | 79.6% | 90.8% | 57.9% | 50.0% | 96.2% |
| 8 | 61.7% | 52.9% | 55.4% | 59.6% | 52.9% | 50.0% | 57.9% |
| 16 | 52.9% | 50.8% | 50.4% | 50.4% | 50.4% | 50.0% | 50.8% |

## Noise sensitivity (suffix length 2)

| Noise σ / model clean RMS | Wave 16 | Sparse ring 16 | ESN 16 | Trace bank 16 |
|---:|---:|---:|---:|
| 0% | 70.4% | 60.0% | 100.0% | 56.7% |
| 1% | 68.3% | 60.0% | 100.0% | 56.7% |
| 3% | 68.8% | 61.7% | 100.0% | 56.7% |
| 10% | 57.9% | 66.7% | 94.6% | 54.2% |

## Cost accounting

These quantities are **not equal**. Only the count of mutable persistent scalars and number of scalar observations have been matched. Read transduction differs (fixed cosine bank vs graph-wave response); fixed weights, step count and arithmetic budget differ dramatically.

| Model | Mutable scalars | Write steps / event | Read samples | Example lower-bound dense MACs for AB + 2 suffix + probe |
|---|---:|---:|---:|---:|
| wave16 | 16 | 1536 | 128 | 458,752 |
| ring16 | 16 | 1 | 128 | 2,240 |
| esn8 | 8 | 1 | 128 | 1,280 |
| esn16 | 16 | 1 | 128 | 3,072 |
| trace16 | 16 | 1 | 128 | 2,240 |
| bag16 | 16 | 1 | 128 | 2,240 |
| esn32 | 32 | 1 | 128 | 8,192 |

### Implementation caveats

- Wave: 8 fast displacements + 8 previous displacements are reset at each write, while 16 edge weights persist. Each write requires 1,536 simulated time steps; a read requires 1,024 time steps, then 128 fixed samples are supplied to the decoder.
- Sparse ring comparator: 16 persistent units with local, shared recurrence coefficients and 3 input ports; **no dense 16x16 fixed-weight array**.
- ESN comparator: 8/16/32 persistent units, one nonlinear update per symbol; the frozen final state drives a static linear time-multiplexed scalar transducer. This sensor has good access to the state and is *not* a physically identical wave probe.
- Trace comparator: 16 decaying variables at different timescales; an ordinary explicit memory baseline, not a trained network. Bag comparator: 16 additive accumulators — must be at chance on AB/BA ordering (up to numerical precision).
- The wave has only 16 *mutable* edges, but the ESN uses additional **fixed** random recurrent/input matrices (e.g. 256 + 48 coefficients for 16 states). The transducer is analytically defined and shared for comparators, not included as persistent writable state.
- Each test pair shares the same nuisance initial weights and common suffix between AB and BA. Train/test suffix strings are disjoint except for zero suffix length. Tested seeds are deterministic; means are descriptive, with small per-seed test sets.
- A 128-sample transducer trace is *not the same sensor physics* as 128 waveform samples. No claim of equal energy, wall-time, hidden precision or silicon cost can be inferred. Dense MAC counts are rough lower bounds; wave exposure/Fourier computations add cost.
- No test-set tuning, supervised recurrent training, raw-matrix access or learned readout outside the common linear ridge decoder.

## Reproduction

```powershell
python3.13 -m pip install numpy matplotlib numba
python3.13 resource_matched_benchmark.py --selftest
python3.13 resource_matched_benchmark.py
```

Outputs: `resource_matched_results.json` (all seed results and cost proxies) and this report. Numba is optional; when available, its integrator is cross-checked against the unmodified wave simulator.

## Next fairer gate

An equal **wall-clock/operation-budget** comparison would need either longer recurrent updates or a radically cheaper wave-write implementation; an equal **sensor physics** comparison would require the same transducer model for both memories. Neither is solved by matching mutable state alone. A third objective is genuinely **multi-bit capacity** rather than binary AB/BA discrimination.
