"""Blind readout benchmark for Slider V3.

Run: python blind_readout_benchmark.py
     python blind_readout_benchmark.py --quick

This experiment never passes the edge-weight matrix to the learned decoder.
It reads ONLY a fixed non-writing probe waveform (or, for the ESN control,
the independently generated reservoir observation). Every pair AB+s / BA+s
is balanced and receives the SAME nuisance initial graph and suffix.

Requires numpy and the repository's SliderWaveMemoryV3.py; matplotlib is
optional for an output plot. No scikit-learn, network, audio, or GPU required.
"""
from __future__ import annotations
import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from SliderWaveMemoryV3 import MemoryEngine, EDGES, EDGE_BASE, READOUT, initial_matrix


# Optional JIT acceleration: same timestep law as MemoryEngine._wave_samples.
# Results are cross-checked against the unmodified simulator before activation.
def enable_checked_numba():
    try:
        from numba import njit
    except ImportError:
        print('  numba unavailable; using reference Python integrator',flush=True)
        return False

    @njit(cache=True)
    def stepping(W, rows, oldx, oldxp, length, omega, port, pulse, clock, edges, readout):
        y = np.empty(length)
        exposure = np.zeros(edges.shape[0])
        spectrum = np.zeros(len(oldx), dtype=np.complex128)
        weight_sum = 0.0
        x=oldx.copy()
        xp=oldxp.copy()
        denom=max(1,length-1)
        for t in range(length):
            if pulse:
                envelope=math.sin(math.pi*t/denom)**2
                drive=.17*envelope*math.sin(omega*t)
            else:
                drive=.08*math.sin(omega*(t+clock))
            xn=np.empty(len(x))
            for i in range(len(x)):
                coupled=0.0
                for j in range(len(x)):
                    coupled+=W[i,j]*x[j]
                xn[i]=1.975*x[i]-.975*xp[i]-.15*x[i]-.87*(rows[i]*x[i]-coupled)
            xn[port]+=drive
            xp=x
            x=xn
            v=0.0
            for i in range(len(x)):
                v+=readout[i]*x[i]
            y[t]=v
            for k in range(len(edges)):
                dv=x[edges[k,0]]-x[edges[k,1]]
                exposure[k]+=dv*dv
            # Spectral features matter only when pulse=True (event writes).
            if pulse:
                weight=.5-.5*math.cos(2*math.pi*t/denom)
                z=complex(math.cos(omega*t),-math.sin(omega*t))
                for i in range(len(x)):
                    spectrum[i]+=weight*x[i]*z
                weight_sum+=weight
        return y,exposure/length,spectrum,weight_sum,x,xp

    reference = MemoryEngine._wave_samples
    def accelerated(self,frequency,port,length,*,pulse=False,propagation_matrix=None,accumulate=False):
        W=self.w if propagation_matrix is None else propagation_matrix
        omega=2*np.pi*float(np.clip(frequency,10,1100))/2400
        y,exp,spec,denom,x,xp=stepping(W,W.sum(axis=1),self.x,self.xprev,length,omega,port,pulse,
            self.sample_clock,EDGES,READOUT)
        self.x=x;self.xprev=xp;self.sample_clock+=length
        if accumulate:
            if not pulse:
                raise AssertionError('The benchmark expects accumulated writes to be pulses')
            self.last_fourier=spec*(2/max(1e-12,denom))
        return y,exp if accumulate else np.zeros(len(EDGES))

    # Compare the actual multievent network and nonwriting probe, including
    # Fourier write measurements, before replacing the simulator method.
    original=MemoryEngine('C');fast=MemoryEngine('C')
    for symbol in 'ABX':
        original.event(symbol)
        temp=MemoryEngine._wave_samples
        MemoryEngine._wave_samples=accelerated
        try: fast.event(symbol)
        finally: MemoryEngine._wave_samples=temp
        np.testing.assert_allclose(fast.w,original.w,atol=2e-11,rtol=1e-10)
        np.testing.assert_allclose(fast.last_fourier,original.last_fourier,atol=2e-11,rtol=1e-10)
    original.reset_fast();fast.reset_fast()
    y0=original.read_probe(blocks=2)
    MemoryEngine._wave_samples=accelerated
    try: y1=fast.read_probe(blocks=2)
    finally: MemoryEngine._wave_samples=reference
    np.testing.assert_allclose(y1,y0,atol=2e-11,rtol=1e-10)
    MemoryEngine._wave_samples=accelerated
    print('  optional Numba integrator verified against reference (3 writes + probe)',flush=True)
    return True

ALPHABET = 'ABX'
DEFAULT_PROBES = {'default': (196.0, 6), 'low_other_port': (160.0, 1), 'high_other_port': (310.0, 4)}


@dataclass(frozen=True)
class Case:
    suffix: str
    jitter: tuple[float, ...]


def suffix_split(length: int, seed: int):
    """Disjoint suffix *strings*, not merely random train/test trials."""
    if length == 0:
        return [''], ['']  # Cannot have held-out strings for empty suffix.
    names = [''.join(v) for v in itertools.product(ALPHABET, repeat=length)]
    rng = np.random.default_rng(seed + 179 * length)
    order = rng.permutation(len(names))
    cut = max(1, min(len(names)-1, round(0.7 * len(names))))
    train = [names[i] for i in order[:cut]]
    test = [names[i] for i in order[cut:]]
    assert set(train).isdisjoint(test)
    return train, test


def make_cases(pool: list[str], n: int, *, seed: int, jitter_std: float):
    rng = np.random.default_rng(seed)
    # Balance suffix sampling before repetitions so each split contains
    # several distinct suffixes even with small sample sizes.
    choices = [pool[i % len(pool)] for i in range(n)]
    rng.shuffle(choices)
    return [Case(s, tuple(float(t) for t in rng.normal(0, jitter_std, len(EDGES))))
            for s in choices]


def initial_network(jitter: tuple[float, ...]):
    w = initial_matrix()
    weights = np.clip(EDGE_BASE + np.asarray(jitter), .01, .70)
    w[EDGES[:, 0], EDGES[:, 1]] = weights
    w[EDGES[:, 1], EDGES[:, 0]] = weights
    return w


def make_wave_trace(seq: str, case: Case, *, control: str, freq: float, port: int):
    """No W enters the decoder; only y(t) leaves this function."""
    m = MemoryEngine('C', write_rule='fourier')
    m.w = initial_network(case.jitter)
    if control == 'no_learning':
        m.learning = False
    for symbol in seq:
        m.event(symbol, clamp_propagation=(control == 'fixed_propagation'))
    if control == 'erased':
        m.erase_matrix()
    m.reset_fast()
    before = m.w.copy()
    signal, _ = m._wave_samples(freq, port, 1024, pulse=False, accumulate=False)
    if not np.array_equal(before, m.w):
        raise AssertionError('A probe wrote the network')
    return signal


def pair_waveforms(cases: list[Case], *, control: str, freq: float, port: int):
    waveforms = np.empty((len(cases), 2, 1024), np.float64)
    for i, case in enumerate(cases):
        for label, first in enumerate(('AB', 'BA')):
            waveforms[i, label] = make_wave_trace(first+case.suffix, case,
                control=control, freq=freq, port=port)
    return waveforms


def features(waveforms: np.ndarray):
    """Fixed time samples of the measured wave; no label-coded features."""
    ix = np.linspace(0, 1023, 128, dtype=np.int64)
    return waveforms[..., ix]


class RidgeDecoder:
    """Fixed-regularization linear ridge decoder. No test-dependent tuning."""
    def fit(self, samples: np.ndarray, labels: np.ndarray):
        mu = samples.mean(axis=0)
        spread = samples.std(axis=0)
        floor = max(float(np.max(spread)) * 0.1, 1e-12)
        scale = np.maximum(spread, floor)
        x = (samples-mu)/scale
        x = np.column_stack((x, np.ones(len(x))))
        lam = 0.5 * len(x)
        gram = x.T @ x + np.eye(x.shape[1])*lam
        gram[-1,-1] = x[:, -1] @ x[:, -1]  # effectively no bias penalty
        self.coef = np.linalg.solve(gram, x.T @ labels)
        self.mu, self.scale = mu, scale
        return self

    def score(self, samples: np.ndarray):
        x = (samples-self.mu)/self.scale
        return x @ self.coef[:-1] + self.coef[-1]


def decode(train: np.ndarray, test: np.ndarray, *, train_labels=None, seed=0):
    """train/test shapes (pairs, 2, features); class 0=AB, 1=BA."""
    y = np.tile(np.array([1., -1.]), train.shape[0])
    if train_labels == 'shuffled':
        y = np.random.default_rng(seed).permutation(y)
    model = RidgeDecoder().fit(train.reshape(-1, train.shape[-1]), y)
    scores = model.score(test.reshape(-1, test.shape[-1])).reshape(-1, 2)
    # Per-example confusion, with equal class counts and a deterministic tie-break.
    predictions = scores > 0
    truth = np.broadcast_to([True, False], predictions.shape)
    singles = predictions == truth
    confusion = [[int(np.sum((truth == a) & (predictions == b))) for b in (True, False)]
                 for a in (True, False)]
    # Each *pair* is a two-alternative forced choice from two subsequent waveforms.
    # When the input waveforms agree to numerical precision, the experiment has
    # zero information: score as chance (half-credit), not as a strict < / > tie.
    # This also prevents ~1e-16 summation-order errors from faking recall in
    # the fixed-propagation negative control.
    waveform_diff = np.max(np.abs(test[:,0]-test[:,1]), axis=-1)
    tol = 1e-12 * max(1.0, float(np.max(np.abs(test))))
    ties = waveform_diff <= tol
    paired_hits = (scores[:, 0] > scores[:, 1]) & ~ties
    effective_correct = float(np.sum(paired_hits) + 0.5*np.sum(ties))
    return {
        'single_accuracy': float(singles.mean()),
        'single_correct': int(singles.sum()),
        'single_total': int(singles.size),
        'pair_accuracy': float(effective_correct/len(paired_hits)),
        'pair_effective_correct': effective_correct,
        'pair_correct': effective_correct,
        'pair_ties': int(np.sum(ties)),
        'pair_total': int(len(paired_hits)),
        'confusion_rows_true_AB_BA_cols_pred_AB_BA': confusion,
        'median_pair_score_margin': float(np.median(scores[:,0]-scores[:,1])),
    }


def add_measurement_noise(train, test, fraction: float, seed: int):
    # Noise std specified as a fraction of the mean CLEAN probe RMS; the same
    # absolute noise scale is applied to every trace (not sample-normalized).
    ref = float(np.sqrt(np.mean(train**2)))
    rng = np.random.default_rng(seed)
    return (train + rng.normal(0, fraction*ref, train.shape),
            test + rng.normal(0, fraction*ref, test.shape), ref)


def separation(test):
    rms = lambda x: np.sqrt(np.mean(x*x, axis=-1))
    diff = rms(test[:, 0]-test[:, 1])
    ref = np.sqrt((rms(test[:,0])**2+rms(test[:,1])**2)/2)
    ratio = diff/np.maximum(ref, 1e-15)
    return {'median_percent': float(np.median(ratio)*100),
            'p10_percent': float(np.percentile(ratio, 10)*100),
            'p90_percent': float(np.percentile(ratio, 90)*100)}


def esn_readouts(cases, seed: int=41):
    """Classical echo-state network: 96 fixed random recurrent units.

    It has a final-state observation (no structural writing), which decays
    with additional events; it is NOT a physically matched acoustic probe.
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(96,96))
    W *= 0.92 / max(abs(np.linalg.eigvals(W)))
    V = rng.normal(size=(96,3))*0.4
    result = np.zeros((len(cases),2,96))
    for i,case in enumerate(cases):
        for k,prefix in enumerate(('AB','BA')):
            h = np.zeros(96)
            for sym in prefix + case.suffix:
                u = np.zeros(3);u[ALPHABET.index(sym)] = 1
                h = np.tanh(W @ h + V @ u)
            result[i,k] = h
    return result


def wilson(k: int, n: int, z=1.96):
    if not n: return [0.,1.]
    p=k/n
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(center-half),float(center+half)]


def metric(train, test, **kwargs):
    m = decode(train, test, **kwargs)
    m['pair_wilson95'] = wilson(m['pair_correct'],m['pair_total'])
    return m


def run(args):
    lengths = [int(x) for x in args.lengths.split(',')]
    if any(x < 0 or x > 10 for x in lengths):
        raise ValueError('lengths must lie between 0 and 10')
    if args.quick:
        args.train_pairs, args.test_pairs = 12, 8
        lengths = [0,2]
    result = {
        'protocol': {
            'seed': args.seed, 'train_pairs': args.train_pairs,
            'test_pairs': args.test_pairs, 'suffix_lengths': lengths,
            'initial_weight_jitter_std': args.jitter,
            'noise_fraction_of_clean_probe_rms': list(args.noise),
            'probe_samples': 1024, 'feature_samples': 128,
            'probe_settings_freq_hz_port': DEFAULT_PROBES,
            'decoder': 'fixed linear ridge alpha=0.5*n_train, train-only standardization',
            'class_balance': 'each suffix / starting network tested for both AB and BA',
            'held_out': 'disjoint suffix strings when suffix length > 0; unseen initial graph jitters in all cases',
            'esn': '96 fixed tanh recurrent units, spectral radius 0.92, final-state observation',
        },
        'length_sweep': {}, 'noise_sweep': {}, 'probe_sweep': {}, 'controls': {},
        'note': 'Pair accuracy is a 2-alternative comparison of two separately probed final states; single accuracy classifies each read independently.'
    }
    datasets = {}
    for L in lengths:
        tr_pool, te_pool = suffix_split(L, args.seed)
        train_cases = make_cases(tr_pool,args.train_pairs, seed=args.seed+11+1000*L, jitter_std=args.jitter)
        test_cases = make_cases(te_pool,args.test_pairs, seed=args.seed+41+1000*L, jitter_std=args.jitter)
        assert not L or set(x.suffix for x in train_cases).isdisjoint(x.suffix for x in test_cases)
        datasets[L]=(train_cases,test_cases)
        print(f'  suffix length {L}: active wave reads',flush=True)
        tr = pair_waveforms(train_cases,control='active',freq=196,port=6)
        te = pair_waveforms(test_cases,control='active',freq=196,port=6)
        ftr,fte = features(tr), features(te)
        esn_train,esn_test = esn_readouts(train_cases),esn_readouts(test_cases)
        result['length_sweep'][str(L)] = {
            'heldout_suffix_count': len(set(x.suffix for x in test_cases)),
            'train_suffix_count': len(set(x.suffix for x in train_cases)),
            'active': metric(ftr,fte),
            'esn': metric(esn_train,esn_test),
            'read_separation': separation(te),
        }
        if L==2:
            cached=(train_cases,test_cases,tr,te)
    # L=2 is a fixed controlled slice; generate it even if the length sweep excludes it.
    if 2 not in datasets:
        tr_pool,te_pool=suffix_split(2,args.seed)
        trc=make_cases(tr_pool,args.train_pairs,seed=args.seed+11+2000,jitter_std=args.jitter)
        tec=make_cases(te_pool,args.test_pairs,seed=args.seed+41+2000,jitter_std=args.jitter)
        cached=(trc,tec,pair_waveforms(trc,control='active',freq=196,port=6),pair_waveforms(tec,control='active',freq=196,port=6))
    trc,tec,cleantr,cleante=cached
    result['protocol']['noise_protocol']='Fresh independent additive Gaussian noise for training and testing at each noise level; classifier retrained for each level.'
    for j,noise in enumerate(args.noise):
        a,b,rms=add_measurement_noise(cleantr,cleante,noise,seed=args.seed+73+j)
        result['noise_sweep'][str(noise)]={'active':metric(features(a),features(b)), 'clean_train_probe_rms':rms}
    result['probe_sweep']['default']={'active':metric(features(cleantr),features(cleante)), 'separation':separation(cleante)}
    for name,(frequency,port) in DEFAULT_PROBES.items():
        if name=='default': continue
        print(f'  new probe {name}: {frequency} Hz, node {port}',flush=True)
        tr=pair_waveforms(trc,control='active',freq=frequency,port=port)
        te=pair_waveforms(tec,control='active',freq=frequency,port=port)
        result['probe_sweep'][name]={'active':metric(features(tr),features(te)), 'separation':separation(te)}
    print('  control experiments, suffix length 2',flush=True)
    for control in ('no_learning','erased','fixed_propagation'):
        tr=pair_waveforms(trc,control=control,freq=196,port=6)
        te=pair_waveforms(tec,control=control,freq=196,port=6)
        maxdiff = float(max(np.max(np.abs(tr[:,0]-tr[:,1])),np.max(np.abs(te[:,0]-te[:,1]))))
        result['controls'][control] = {'active':metric(features(tr),features(te)),
            'max_pair_waveform_difference':maxdiff}
    result['controls']['shuffled_train_labels']={'active': metric(features(cleantr),features(cleante),train_labels='shuffled',seed=args.seed+123)}
    return result


def summary_md(results):
    cfg=results['protocol']; rows=[]
    rows.append('# Blind Readout Benchmark — Measured Results\n')
    rows.append('Generated from `blind_readout_benchmark.py` using seed '+str(cfg['seed'])+
        ', '+str(cfg['train_pairs'])+' training pairs, '+str(cfg['test_pairs'])+
        ' held-out test pairs per condition, initial-edge jitter σ='+str(cfg['initial_weight_jitter_std'])+'.\n')
    rows.append('**Question:** after writing `AB + suffix` or `BA + suffix`, can a decoder identify order from later **non-writing** signals without inspecting `W`? Both classes use the same suffix and the same starting graph within each matched pair. For suffix length > 0, suffix strings are disjoint between training and test.\n')
    rows.append('**Two metrics:** single = classify each waveform independently; paired = choose which of two matched waveforms came from AB (two-alternative forced choice). Pairing removes shared nuisance graph variation and is an *easier task*; do not conflate it with single-trial recall. A test set has '+str(cfg['test_pairs'])+' pairs, so pair accuracy changes in steps of '+str(round(100/cfg['test_pairs'],2))+' percentage points.\n')
    rows.append('## Sequence length / forgetting\n')
    rows.append('| Extra suffix length | Wave single | Wave paired | ESN single | ESN paired | Median clean probe separation |')
    rows.append('|---:|---:|---:|---:|---:|---:|')
    p=lambda d,key: f'{100*d[key]:.1f}%'
    for L,d in results['length_sweep'].items():
        rows.append(f"| {L} | {p(d['active'],'single_accuracy')} | {p(d['active'],'pair_accuracy')} | {p(d['esn'],'single_accuracy')} | {p(d['esn'],'pair_accuracy')} | {d['read_separation']['median_percent']:.3f}% |")
    rows.append('\n## Probe measurement noise (suffix length 2)\n')
    rows.append('Noise σ is a specified percentage of clean training-probe RMS; Gaussian noise is added to both training and test probe traces. New decoder per noise level.\n')
    rows.append('| Measurement noise / clean RMS | Single | Paired | 95% Wilson interval for paired |')
    rows.append('|---:|---:|---:|---:|')
    for noise,d in results['noise_sweep'].items():
        a=d['active'];w=a['pair_wilson95']
        rows.append(f"| {100*float(noise):.1f}% | {p(a,'single_accuracy')} | {p(a,'pair_accuracy')} | {100*w[0]:.1f}–{100*w[1]:.1f}% |")
    rows.append('\n## Probe placement and frequency (suffix length 2, clean)\n')
    rows.append('| Probe | Frequency / input node | Single | Paired | Median read separation |')
    rows.append('|---|---|---:|---:|---:|')
    for name,d in results['probe_sweep'].items():
        hz,port=cfg['probe_settings_freq_hz_port'][name];a=d['active']
        rows.append(f"| {name} | {hz:g} Hz / {port} | {p(a,'single_accuracy')} | {p(a,'pair_accuracy')} | {d['separation']['median_percent']:.3f}% |")
    rows.append('\n## Negative controls (suffix length 2, clean)\n')
    rows.append('| Condition | Single | Paired | Maximum exact AB/BA waveform difference |')
    rows.append('|---|---:|---:|---:|')
    for name,d in results['controls'].items():
        a=d['active'];differ=d.get('max_pair_waveform_difference')
        diff='n/a (labels only)' if differ is None else f'{differ:.3g}'
        rows.append(f"| {name} | {p(a,'single_accuracy')} | {p(a,'pair_accuracy')} | {diff} |")
    rows.append('\n**Important control interpretation:** if paired waveforms agree to numerical precision, the forced-choice result is credited as **50% (chance)** rather than letting rounding-level differences count as information. Single-trial accuracy is also 50% in the balanced no-information controls. A nonzero paired score in the shuffled-label control may happen by chance on a small test set.\n')
    rows.append('## Interpretation / limitations\n')
    rows.append('- An ESN is a conventional 96-unit tanh recurrent baseline with a fixed random reservoir and a linear readout trained on its final state. It is **not** an acoustically matched probe and can keep fast state; its scores are context, not a capacity-normalized hardware comparison.')
    rows.append('- Classification labels are used for training, but the wave decoder only receives 128 fixed samples from each *non-writing* waveform, never a weight matrix or the input labels.')
    rows.append('- Noise is synthetic i.i.d. measurement noise. The source model is deterministic and a deliberately designed write law. Suffix train/test splits and randomized initial graphs probe some generalization, not all forms of robustness.')
    rows.append('- The paired metric has access to **two matched probe signals** at inference time. This can cancel shared nuisance variation; single-waveform accuracy is the stricter claim about blind recall.')
    rows.append('- Results are one seeded run with small sample sizes and no claim of statistical significance against stronger baselines. Run alternative seeds before interpreting small differences.')
    rows.append('\nReproduce with `python blind_readout_benchmark.py`. Full machine-readable scores, 2×2 confusion matrices, settings, and Wilson intervals: [`blind_readout_results.json`](blind_readout_results.json).\n')
    return '\n'.join(rows)


def make_plot(results, filename:Path):
    try:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    lens=sorted(int(L) for L in results['length_sweep'])
    fig,ax=plt.subplots(figsize=(8.5,4.5))
    ax.plot(lens,[results['length_sweep'][str(i)]['active']['single_accuracy']*100 for i in lens],marker='o',label='Wave: single read')
    ax.plot(lens,[results['length_sweep'][str(i)]['active']['pair_accuracy']*100 for i in lens],marker='s',label='Wave: matched-pair choice')
    ax.plot(lens,[results['length_sweep'][str(i)]['esn']['single_accuracy']*100 for i in lens],marker='^',label='ESN: single read')
    ax.axhline(50,linestyle='--',linewidth=1,label='Single-trial chance')
    ax.set(xlabel='Number of common suffix events',ylabel='Accuracy (%)',ylim=(-5,105),title='Held-out sequence memory, different readout tasks')
    ax.grid(alpha=0.25);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(filename,dpi=150);plt.close(fig)
    return True


def selftest():
    # 1: read itself is nonwriting; 2: symmetric paired nulls are exactly equal.
    case=Case('XA',tuple(np.zeros(len(EDGES))))
    a=make_wave_trace('ABXA',case,control='active',freq=196,port=6)
    b=make_wave_trace('BAXA',case,control='active',freq=196,port=6)
    assert np.sqrt(np.mean((a-b)**2))>0
    for ctrl in ('no_learning','erased','fixed_propagation'):
        a=make_wave_trace('ABXA',case,control=ctrl,freq=196,port=6)
        b=make_wave_trace('BAXA',case,control=ctrl,freq=196,port=6)
        np.testing.assert_allclose(a,b,atol=1e-13,rtol=0)
    tr,te=suffix_split(3,1);assert not (set(tr)&set(te))
    sample=np.zeros((4,2,8))
    null=decode(sample,sample)
    assert null['single_accuracy']==0.5 and null['pair_accuracy']==0.5 and null['pair_ties']==4
    print('PASS: blind probe, common suffix, exact nulls, disjoint held-out suffixes, chance-scored indistinguishable pairs')


def cli():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-pairs',type=int,default=48)
    parser.add_argument('--test-pairs',type=int,default=24)
    parser.add_argument('--jitter',type=float,default=.002,help='absolute sd per initial nonzero edge weight')
    parser.add_argument('--lengths',default='0,2,4,8')
    parser.add_argument('--noise',type=float,nargs='+',default=[0.,.01,.03,.10])
    parser.add_argument('--seed',type=int,default=20261106)
    parser.add_argument('--out',default='.',type=Path)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--no-jit',action='store_true',help='force unmodified reference Python integrator')
    parser.add_argument('--selftest',action='store_true')
    args=parser.parse_args()
    if not args.no_jit: enable_checked_numba()
    selftest()
    if args.selftest: return
    args.out.mkdir(parents=True,exist_ok=True)
    results=run(args)
    (args.out/'blind_readout_results.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
    (args.out/'BLIND_READOUT_RESULTS.md').write_text(summary_md(results),encoding='utf-8')
    plot=make_plot(results,args.out/'blind_readout_results.png')
    print('Saved results (JSON, Markdown'+(', PNG' if plot else '')+') in '+str(args.out),flush=True)
    print(summary_md(results),flush=True)

if __name__=='__main__': cli()