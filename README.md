# Slider V3 — Wave-Written Graph Memory

**Can the history of waves become a lasting change in a network, and can a later wave read that history back?**

![Slider V3 interface](pic.png)

Slider V3 is a small, offline, reproducible experiment in **history-dependent wave propagation**. Eight damped oscillators are connected by 16 writable edges. A stimulus sends a sine-wave pulse into the graph; the motion measured *inside the graph* changes its edge weights. The next pulse therefore travels through a network changed by previous pulses.

The central result is modest but real: presenting the same two stimuli in opposite orders (**AB** versus **BA**) leaves different connection matrices and produces different responses to an identical, later, **non-writing** probe — even after the fast oscillator state has been reset.

> **The memory is in the altered connections, not in the beautiful 3D trace and not in a secret A/B lookup table.** This is a designed synthetic wave-and-plasticity model, not a demonstration of general intelligence, novel physics, or a useful large-capacity memory system.

## Run it

Requires Python 3.10+ with Tkinter (normally included on Windows) and NumPy/Matplotlib.

```powershell
python3.13 -m pip install numpy matplotlib
python3.13 SliderWaveMemoryV3.py
```

In the app:

1. Select **Mode C — wave-written graph memory**.
2. Click **COMPARE A–B vs B–A**, then **COMPARE A–B–X vs B–A–X**.
3. Use the **right-hand scrollbar** to view the result details and log.
4. Switch the **matrix write law** between **fourier** and **strain**.
5. Turn off **Learning enabled**, or **erase W**, to check that the order difference disappears.

The script is named **SliderWaveMemoryV3.py** in this repository (including capitalization).

For automated, headless checks:

```powershell
python3.13 SliderWaveMemoryV3.py --selftest
python3.13 SliderWaveMemoryV3.py --demo
```

No audio device, cloud service, GPU, or internet access is needed.

## The measured result

The repository's deterministic demo produces the following readings using the default Fourier write rule and default parameters:

| Mode | Histories | Matrix difference, $\lVert \Delta W\rVert_F$ | Probe RMS difference | Relative probe difference |
|:---|:---|---:|---:|---:|
| A — phase oscillator | AB / BA | 0.000000 | 0.000000 | 0.000% |
| A — phase oscillator | ABX / BAX | 0.000000 | 0.000000 | 0.000% |
| B — bounded feedback | AB / BA | 0.000000 | 0.000000 | 0.000% |
| B — bounded feedback | ABX / BAX | 0.000000 | 0.000000 | 0.000% |
| **C — writable wave graph** | **AB / BA** | **0.019438** | **0.002410** | **2.890%** |
| **C — writable wave graph** | **ABX / BAX** | **0.019905** | **0.002147** | **2.612%** |

The relative difference is the RMS of the difference between the two probe signals, divided by the root-mean-square of their two individual mean-square powers:

$$
d_{\mathrm{rel}} =
\frac{\mathrm{RMS}(y_1-y_2)}
{\sqrt{\bigl(\mathrm{RMS}(y_1)^2+\mathrm{RMS}(y_2)^2\bigr)/2}}.
$$

The reported percentage is $100d_{\mathrm{rel}}$. It is **not** an accuracy score or a measured probability of recalling a sequence.

In each paired trial, both histories begin with the **same initial graph**. Before their final probe, all fast oscillator variables, phases, sample clocks, and short-term output histories are reset. Only the altered edge weights remain. The probe itself does **not** update the edges.

The **ABX / BAX** result means an additional common **X** write does not immediately erase the earlier order distinction in this test. It does not establish indefinite retention.

Full recorded output: [slider_wave_memory_lab_v3_results.txt](slider_wave_memory_lab_v3_results.txt).

## How it works

### 1. A wave is an input, not an assigned memory pattern

Three stimulus labels choose only a drive frequency and injection node:

| Input | Frequency | Injection node |
|:---|---:|---:|
| A | 160 Hz | 0 |
| B | 230 Hz | 3 |
| X | 310 Hz | 5 |
| Common read probe | 196 Hz | 6 |

The graph has eight nodes, with nearest-neighbor ring links and next-nearest-neighbor links (16 undirected edges). Its dynamics are synthetic discrete-time coupled oscillators. For displacement vector $x_t$, the actual implemented step is:

$$
x_{t+1} =
1.975x_t - 0.975x_{t-1}
-0.15x_t
-0.87\bigl(D(W)-W\bigr)x_t
+u_t.
$$

Here $W$ is the symmetric connection matrix, $D(W)$ contains its row sums on the diagonal, and $u_t$ is the externally driven pulse at one node. The graph is integrated through time; it is not fed a precomputed weight-change stencil.

### 2. Measure what the wave actually did

For each pulse of length $T$, the default **fourier** rule measures a complex, Hann-windowed coefficient at the *drive frequency* $f$ for every node:

$$
F_i(f) =
\frac{2}{\sum_{t=0}^{T-1}h_t}
\sum_{t=0}^{T-1}
h_t\,x_i(t)\,
e^{-2\pi \mathrm{i}ft/f_s},
$$

where $h_t$ is the Hann window and the simulation sample rate is $f_s=2400$ steps per second.

Unlike a comparison of magnitudes alone, the complex difference $F_i-F_j$ depends on **relative phase as well as amplitude**. For each existing edge $(i,j)$, its measured exposure is:

$$
E_{ij}
=0.48\,\tanh\left(\frac{|F_i-F_j|^2}{0.25}\right).
$$

The optional **strain** rule instead uses the time-average of squared differences in node displacement:

$$
E_{ij}^{\mathrm{strain}}
=\frac1T\sum_{t=0}^{T-1}
\bigl(x_i(t)-x_j(t)\bigr)^2.
$$

Both rules are defined by the experimenter. **Phase sensitivity is present in the Fourier rule; this experiment does not show that phase is uniquely necessary for order memory.**

### 3. The measured response writes the graph

Connected weights are updated symmetrically using the measured exposure:

$$
W_{ij}^{\mathrm{new}}
=\operatorname{clip}\left(
W_{ij}^{\mathrm{old}}+
\eta s\,E_{ij},\ 0.01,\ 0.70
\right).
$$

Here $\eta=0.055$ is the default write rate and $s=1$ the default strength. Unconnected edges are not added. The fast wave state is reset between stimulus events, so the next event's response depends on the retained network $W$, not on a leftover oscillation from the previous event.

Let $T_A(W)$ mean *propagate A through the current graph, measure the response, then update that graph*. Then, for this implementation:

$$
T_B\bigl(T_A(W_0)\bigr)
\ne
T_A\bigl(T_B(W_0)\bigr).
$$

This is the precise sense in which the **write operations are order-dependent (noncommuting)**. It is **not** a claim that the matrices $W_A$ and $W_B$ themselves have a nonzero matrix commutator.

### 4. Ask the same question of two different histories

After AB and BA, reset the fast state and apply the **same 196 Hz non-writing probe**. The observed signal is a fixed, normalized linear mixture of node displacements:

$$
y_t=r^\mathsf{T}x_t.
$$

Because the graph is different, the probe's wave propagation — and therefore the readout — can be different. The experiment observes **a distinction between histories**, not a reconstructed transcript of the original events.

```mermaid
flowchart TD
    S["Identical starting network W0"]
    S --> A["Write A, then B"]
    S --> B["Write B, then A"]
    A --> WA["Stored structure W_AB"]
    B --> WB["Stored structure W_BA"]
    WA --> RA["Reset fast state; probe at 196 Hz"]
    WB --> RB["Reset fast state; probe at 196 Hz"]
    RA --> YA["Read signal y_AB"]
    RB --> YB["Read signal y_BA"]
    YA --> D["Compare responses"]
    YB --> D
```

## Controls: why this is not V2's lookup-table shortcut

**V2** could stamp a prescribed hardcoded pattern into a matrix when A or B was selected. **V3 does not do that.** Its labels choose a sine input and a location; edge changes come from the integrated network's measured motion.

The headless self-tests check these distinct explanations:

| Control | Outcome | What it tests |
|:---|:---|:---|
| Mode A or B | No persistent AB/BA read difference after reset | Phase continuity or bounded feedback alone is not the stored structural memory in this paired test |
| Learning disabled | No persistent order difference | Writing is necessary |
| Erase the matrix after writing | No order difference | The final read depends on retained $W$ |
| Force both writing wavefields through unchanged $W_0$ | No order difference | Exposure must be affected by previous writes; two fixed, additive exposures commute |
| Reset fast state before read | Read difference remains in C | It is not an unreset oscillator phase |
| Repeat the same read | Identical result, unchanged matrix | Reads do not secretly write |
| Fourier and strain self-tests | Pass | Both implemented wave-derived write laws are numerically exercised |

The **fixed-propagation** control is particularly informative: it still computes real simulated wave exposure, but computes both writes against the *same unchanged propagation graph*. Without response-to-previous-write feedback, order memory disappears. The causal loop is:

**wave → measured exposure → altered connections → changed next wave → changed next write.**

## What the plots are (and are not)

- **3D trace:** a visualization of samples from the last three output buffers. Its shape alone is **not proof of memory**.
- **Matrix heat map:** the current edges, or the difference between two final edge matrices in a comparison.
- **Paired probe traces:** the same read applied after different histories, with fast states reset.

**Display noise is only added to the live visualization.** The paired comparison is deterministic and noise-free. Its reported 2.890% and 2.612% differences therefore do **not** establish robust retrieval under noise.

## Connection to the wider experiments

This is one concrete step in a longer investigation:

- [FourierFlower](https://github.com/anttiluode/FourierFlower): structured periodic relationships and phase geometry.
- [Friday](https://github.com/anttiluode/Friday): exact hidden modes in a graph, the difference between *information existing* and *information being observable*, and how changing internal coupling may make hidden states accessible.
- [SelfIndexingBeach](https://github.com/anttiluode/SelfIndexingBeach) and [ComputingBeachClaude](https://github.com/anttiluode/ComputingBeachClaude): writable wave media whose altered material properties change later responses to weak probes.

**Friday asks: what information can a probe reach? Slider V3 asks: can the probe-relevant structure itself be written by prior wave histories?** They are related questions, but these are not yet one combined system. V3 has not demonstrated Friday's protected hidden modes or selective access to them.

The shared research question is:

> **Can a physical-style medium retain distinctions between histories in its internal relationships, and later reveal the right distinctions through controlled, non-destructive probing?**

## Next experiment: distinguish sequences without inspecting W

A useful follow-up is a **blind readout benchmark**, not another prettier visualization:

1. Write many different sequences into fresh networks, including held-out sequences and histories with a common suffix.
2. Give a classifier or predictor only the later **non-writing probe signals** — never the internal edge matrix.
3. Vary the read frequency, port, and measurement noise; measure accuracy, confusion, and margin to noise.
4. Include learning-disabled, erased, fixed-propagation, shuffled-label, and conventional recurrent/reservoir baselines.
5. Evaluate forgetting under additional writes and whether any useful distinctions survive at increasing sequence length.

This would measure **usable memory**, rather than merely finding two different matrices and two different deterministic readouts.

## Scope and limits

This is **not** an AI that discovered its own plasticity law, a quantum computer, a proof of a special property of $\pi$, or a demonstrated improvement over ordinary neural/recurrent architectures. Frequencies, topology, drive ports, coupling dynamics, and write law were deliberately chosen. Readout is a synthetic fixed mixture, not real microphone data. No storage capacity, uncertainty bounds, long-term robustness, task learning, or scaling advantage has been established.

**What is supported:** in this eight-node system, measured wave motion can modify the network so that opposite input orders leave different structural states and different later non-writing responses; the existing controls isolate this effect from fast phase carry-over and hardcoded label-to-matrix assignments.

## Repository files

- [SliderWaveMemoryV3.py](SliderWaveMemoryV3.py) — GUI, wave simulator, comparisons, and self-tests.
- [slider_wave_memory_lab_v3_results.txt](slider_wave_memory_lab_v3_results.txt) — saved numerical demo output.
- [pic.png](pic.png) — interface screenshot.

All math above uses **GitHub-supported Markdown math** and the process diagram uses **GitHub-supported Mermaid**. View this file on GitHub to see them rendered.
