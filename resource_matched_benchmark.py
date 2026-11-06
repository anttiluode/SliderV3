"""SliderV3 resource-matched memory benchmark (auditable CPU-only experiment).

Compare 16 persistent wave-edge scalars with 16 persistent ESN/trace scalars,
using the *same* held-out AB/BA+common-suffix cases, 128 samples of a single
scalar non-writing trace, the same ridge decoder, and matched relative read
noise. A fixed spectral scan converts a frozen comparator state into a scalar
trace; this is an idealized, well-observed comparator readout, NOT the same
physical sensor as the wave probe. We report estimated work separately.

Run: python resource_matched_benchmark.py --selftest
     python resource_matched_benchmark.py --quick
     python resource_matched_benchmark.py

Dependencies: numpy + existing blind_readout_benchmark.py + SliderWaveMemoryV3.py.
Optional: numba (accelerates the checked, numerically matched wave integrator).
"""
from __future__ import annotations
import argparse
import json
from functools import lru_cache
import math
import time
from pathlib import Path
import numpy as np
from blind_readout_benchmark import (
    ALPHABET, Case, RidgeDecoder, make_cases, suffix_split,
    pair_waveforms, features, enable_checked_numba,
)

N=16
SAMPLES=128
WAVE_STEPS_PER_WRITE=1536
WAVE_STEPS_PER_READ=1024


def suffix_pools(length: int, seed: int):
    if length <= 10:
        return suffix_split(length, seed)
    rng=np.random.default_rng(seed + length * 177)
    found=set()
    while len(found)<180:
        found.add(''.join(rng.choice(list(ALPHABET), size=length)))
    names=sorted(found)
    rng.shuffle(names)
    return names[:120],names[120:]


def make_dataset(seed, length, train_n, test_n, jitter):
    train_pool,test_pool=suffix_pools(length,seed)
    train=make_cases(train_pool,train_n,seed=seed+5001+79*length,jitter_std=jitter)
    test=make_cases(test_pool,test_n,seed=seed+11031+79*length,jitter_std=jitter)
    if length:
        assert set(x.suffix for x in train).isdisjoint(x.suffix for x in test)
    return train,test


def fixed_scan(n, steps=128):
    """One scalar fixed read channel over time, no label/sequence fitted mask.

    Static frequency-multiplexed transducer: x_j*cos(omega_j*t+phi_j).
    It has high observability and is deliberately an optimistic readout for
    comparator memories. It does not expose the vector directly to the decoder.
    """
    t=np.arange(steps,dtype=float)
    index=np.arange(n,dtype=float)
    # A DCT-II-like fixed bank: orthogonal, balanced measurement directions.
    return np.cos(np.pi*(t[:,None]+0.5)*(index[None,:]+0.5)/steps)*math.sqrt(2/n)


@lru_cache(maxsize=64)
def fixed_esn(seed, n=16):
    rng=np.random.default_rng(seed+230813+n)
    recurrent=rng.normal(size=(n,n))
    radius=max(abs(np.linalg.eigvals(recurrent)))
    recurrent*=0.92/radius
    input_map=rng.normal(size=(n,len(ALPHABET)))*0.45
    return recurrent,input_map


@lru_cache(maxsize=64)
def trace_embedding(seed):
    rng=np.random.default_rng(seed+48211)
    em=rng.normal(size=(N,3))
    em/=np.maximum(np.linalg.norm(em,axis=0,keepdims=True),1e-9)
    return em


def state_for(seq, case, model, seed):
    if model=='ring16':
        h=np.asarray(case.jitter[:N],dtype=float).copy()
        # A sparse 16-state nonlinear recurrence: identical weights at every node,
        # a local ring with 3 stimulus ports (not a trained dense matrix).
        ports={'A':0,'B':5,'X':10}
        for letter in seq:
            drive=np.zeros(N)
            drive[ports[letter]]=.50
            h=np.tanh(.60*h+.22*np.roll(h,1)-.14*np.roll(h,-1)+drive)
        return h
    if model.startswith('esn'):
        n=int(model[3:])
        W,V=fixed_esn(seed,n)
        h=np.zeros(n)
        # All models receive the same labels. Use a small, seed-dependent
        # nuisance initial state shared between AB and BA within one pair.
        # Jitter across independent cases enters as known fixed mixing only.
        j=np.asarray(case.jitter,dtype=float)
        h[:min(n,len(j))]=j[:min(n,len(j))]
        for letter in seq:
            h=np.tanh(W@h+V[:,ALPHABET.index(letter)])
        return h
    if model=='trace16':
        n=N
        # Multiscale leaky state: explicitly simple conventional memory.
        decay=np.geomspace(.55,.995,n)
        embeddings=trace_embedding(seed)
        h=np.asarray(case.jitter[:n],dtype=float).copy()
        for letter in seq:
            h=decay*h+(1-decay)*embeddings[:,ALPHABET.index(letter)]
        return h
    if model=='bag16':
        embeddings=trace_embedding(seed)
        h=np.asarray(case.jitter[:N],dtype=float).copy()
        for letter in seq:
            h=h+embeddings[:,ALPHABET.index(letter)]
        return h
    raise ValueError(model)


def comparator_traces(cases,model,seed):
    n=N if model in ('trace16','bag16','ring16') else int(model[3:])
    scan=fixed_scan(n)
    # One-dimensional time trace, not a direct h-state decoder.
    traces=np.empty((len(cases),2,SAMPLES))
    for i,case in enumerate(cases):
        for k,prefix in enumerate(('AB','BA')):
            h=state_for(prefix+case.suffix,case,model,seed)
            traces[i,k]=scan@h
    return traces


def ridge_scores(train,test):
    yy=np.tile([1.,-1.],len(train))
    dec=RidgeDecoder().fit(train.reshape(-1,train.shape[-1]),yy)
    scores=dec.score(test.reshape(-1,test.shape[-1])).reshape(-1,2)
    pred=scores>0
    truth=np.broadcast_to([True,False],pred.shape)
    return scores,pred,truth


def evaluate(train,test):
    scores,pred,truth=ridge_scores(train,test)
    # Calculate independent single-read classification, paired two-choice,
    # and pair-level accuracy in a symmetric way for aggregating uncertainty.
    single=float((pred==truth).mean())
    pairs=float(np.mean(scores[:,0]>scores[:,1]))
    pairdiff=np.max(np.abs(test[:,0]-test[:,1]),axis=1)
    tol=1e-12*max(1.,float(np.max(np.abs(test))))
    n_ties=int(np.sum(pairdiff<=tol))
    pair=float(((scores[:,0]>scores[:,1])&(pairdiff>tol)).sum()+n_ties*.5)/len(test)
    return {'single':single,'paired':pair,'ties':n_ties,
            'single_correct':int((pred==truth).sum()),'single_total':int(truth.size),
            'confusion':[[int(((truth==a)&(pred==b)).sum()) for b in (True,False)]
                         for a in (True,False)]}


def add_noise(tr,te,level,seed):
    """Matched *relative* measurement noise, calibrated on training RMS.
    Shared noise seed, separate Gaussian draws, same number of scalar samples.
    """
    rms=float(np.sqrt(np.mean(tr*tr)))
    rng=np.random.default_rng(seed)
    return (tr+rng.normal(size=tr.shape)*rms*level,
            te+rng.normal(size=te.shape)*rms*level)


def cost(model,length):
    """Transparent lower-bound work proxy, not CPU benchmark/GPU energy.
    Counts dense multiply-add-equivalents and stored mutable scalars only.
    The simulator incurs additional exposure, Fourier and driving operations.
    """
    events=length+2
    if model=='wave16':
        return {'mutable_scalars':16,'write_steps':events*WAVE_STEPS_PER_WRITE,
                'read_steps':WAVE_STEPS_PER_READ,
                'dense_matvec_mac_lower_bound':
                    (events*WAVE_STEPS_PER_WRITE+WAVE_STEPS_PER_READ)*64,
                'fixed_parameters':'16 graph edges; fixed drive/read vectors; physical update coefficients'}
    n=N if model in ('trace16','bag16','ring16') else int(model[3:])
    rec= n*n if model.startswith('esn') else (3*n if model=='ring16' else n*3)
    return {'mutable_scalars':n,'write_steps':events,'read_steps':128,
            'dense_matvec_mac_lower_bound':events*rec+n*SAMPLES,
            'fixed_parameters':('3 shared recurrent coefficients + 3 drive ports + analytically fixed scan' if model=='ring16' else f'{n*n} recurrent + {n*3} input entries + {n*SAMPLES} '
                'readout-mask entries (fixed analytically)' if model.startswith('esn')
                else f'{n*3} symbol embeddings + {n*SAMPLES} analytic scan entries')}


def run(args):
    seeds=[args.seed+i for i in range(args.seeds)]
    lengths=[int(x) for x in args.lengths.split(',')]
    models=['wave16','ring16','esn8','esn16','trace16','bag16','esn32']
    details=[]
    timing={model:[] for model in models}
    for seed in seeds:
        for length in lengths:
            trcases,tecases=make_dataset(seed,length,args.train_pairs,args.test_pairs,args.jitter)
            print(f'  seed={seed} suffix={length}',flush=True)
            for model in models:
                t0=time.perf_counter()
                if model=='wave16':
                    train=features(pair_waveforms(trcases,control='active',freq=196,port=6))
                    test=features(pair_waveforms(tecases,control='active',freq=196,port=6))
                else:
                    train=comparator_traces(trcases,model,seed)
                    test=comparator_traces(tecases,model,seed)
                elapsed=time.perf_counter()-t0
                timing[model].append(elapsed/(2*(len(trcases)+len(tecases))))
                obs=evaluate(train,test)
                details.append({'seed':seed,'suffix':length,'model':model,'noise':0.,**obs})
                if length==2:
                    for k,noise in enumerate((.01,.03,.10)):
                        noisy_train,noisy_test=add_noise(train,test,noise,seed+13000+211*k)
                        details.append({'seed':seed,'suffix':length,'model':model,'noise':noise,
                                        **evaluate(noisy_train,noisy_test)})
            print('    complete',flush=True)
    aggs=[]
    for length in lengths:
        for noise in (0.,.01,.03,.10):
            if noise and length!=2:continue
            for model in models:
                rows=[r for r in details if r['suffix']==length and r['noise']==noise and r['model']==model]
                vals=np.array([r['single'] for r in rows])
                pairs=np.array([r['paired'] for r in rows])
                aggs.append({'suffix':length,'model':model,'noise':noise,
                    'single_mean':float(vals.mean()),'single_min':float(vals.min()),
                    'single_max':float(vals.max()),'single_sd':float(vals.std(ddof=1)) if len(vals)>1 else None,
                    'paired_mean':float(pairs.mean()),'n_seeds':len(rows)})
    return {'protocol':{
        'dataset':'balanced AB/BA + same common suffix, disjoint held-out suffix strings (length >0), new edge jitter per pair',
        'seeds':seeds,'train_pairs_per_seed':args.train_pairs,'test_pairs_per_seed':args.test_pairs,
        'jitter_sigma':args.jitter,'lengths':lengths,
        'wave':'unchanged SliderWaveMemoryV3 eight-node, sixteen-edge Fourier plasticity; 1536 integration steps/write; 1024-step nonwriting probe at 196 Hz, node 6, 128 scalar samples',
        'comparators':'16-slot sparse-ring recurrence, 8-,16-,32-state untrained random tanh ESNs, 16-slot exponential trace bank and additive bag; outputs from 128 fixed time-varying scalar measurements of frozen state; neither classifier nor output observes state vector directly',
        'sensor_caveat':'Static cosine bank is an idealized, potentially better conditioned transducer than the wave graph probe; it is not identical physical readout hardware. Both are restricted to 128 scalar nonwriting readings.',
        'training':'same fixed ridge method/regularization, same cases and labels for all models; no hyperparameter tuning on test',
        'noise':'Gaussian noise with SD as fraction of each model clean training-read RMS; same noise seeds, separate train and test draws; no retraining on noisy test',
        'comparison_scope':'16 mutable scalar memory slots matched, not fixed model parameter count, actual hardware physics, FLOP budget or energy',
        'work_count':'lower-bound per-sample dense MAC proxy; ignores transcendental functions, exposure updates, JIT speed and latency',
        },'aggregate':aggs,'individual':details,'costs':{m:{str(L):cost(m,L) for L in lengths} for m in models},
        'measured_runtime_per_sequence_seconds':{m:float(np.median(times)) for m,times in timing.items()}}


def fmt(r,name):
    return f'{r[name]*100:.1f}%'


def report(d):
    ag=d['aggregate'];models=['wave16','ring16','esn8','esn16','trace16','bag16','esn32']
    get=lambda L,m,n=0.:next(r for r in ag if r['suffix']==L and r['model']==m and r['noise']==n)
    txt=['# Resource-matched memory comparison (measured)','',
        'This experiment replaces the earlier **96-state, directly observed ESN** with explicit **16 mutable-scalar** memory baselines and a restricted **128-sample one-sensor read**. All models receive the same sequence labels, suffix splits, training pairs, decoder and relative-noise convention. **Result is a comparison of algorithms and readout interfaces, not a fair hardware/energy benchmark**.','',
        '## Single-read classification, held-out suffix strings','',
        f'{len(d["protocol"]["seeds"])} seeds, {d["protocol"]["train_pairs_per_seed"]} training AB/BA pairs and {d["protocol"]["test_pairs_per_seed"]} held-out pairs per seed; means across seeds; 50% chance.','',
        '| Suffix events | Wave (16 edges) | Sparse ring (16) | ESN (8 states) | ESN (16 states) | Trace bank (16) | Additive bag (16) | ESN (32 states) |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for L in d['protocol']['lengths']:
        txt.append('| '+str(L)+' | '+' | '.join(fmt(get(L,m),'single_mean') for m in models)+' |')
    txt += ['', '## Noise sensitivity (suffix length 2)', '',
        '| Noise σ / model clean RMS | Wave 16 | Sparse ring 16 | ESN 16 | Trace bank 16 |',
        '|---:|---:|---:|---:|']
    for n in (0.,.01,.03,.1):
        txt.append(f'| {n*100:g}% | '+' | '.join(fmt(get(2,m,n),'single_mean') for m in ('wave16','ring16','esn16','trace16'))+' |')
    txt += ['', '## Cost accounting','',
        'These quantities are **not equal**. Only the count of mutable persistent scalars and number of scalar observations have been matched. Read transduction differs (fixed cosine bank vs graph-wave response); fixed weights, step count and arithmetic budget differ dramatically.','',
        '| Model | Mutable scalars | Write steps / event | Read samples | Example lower-bound dense MACs for AB + 2 suffix + probe |',
        '|---|---:|---:|---:|---:|']
    for m in models:
        c=d['costs'][m]['2']
        txt.append(f'| {m} | {c["mutable_scalars"]} | {c["write_steps"]//4} | 128 | {c["dense_matvec_mac_lower_bound"]:,} |')
    txt += ['', '### Implementation caveats','',
        '- Wave: 8 fast displacements + 8 previous displacements are reset at each write, while 16 edge weights persist. Each write requires 1,536 simulated time steps; a read requires 1,024 time steps, then 128 fixed samples are supplied to the decoder.',
        '- Sparse ring comparator: 16 persistent units with local, shared recurrence coefficients and 3 input ports; **no dense 16x16 fixed-weight array**.',
        '- ESN comparator: 8/16/32 persistent units, one nonlinear update per symbol; the frozen final state drives a static linear time-multiplexed scalar transducer. This sensor has good access to the state and is *not* a physically identical wave probe.',
        '- Trace comparator: 16 decaying variables at different timescales; an ordinary explicit memory baseline, not a trained network. Bag comparator: 16 additive accumulators — must be at chance on AB/BA ordering (up to numerical precision).',
        '- The wave has only 16 *mutable* edges, but the ESN uses additional **fixed** random recurrent/input matrices (e.g. 256 + 48 coefficients for 16 states). The transducer is analytically defined and shared for comparators, not included as persistent writable state.',
        '- Each test pair shares the same nuisance initial weights and common suffix between AB and BA. Train/test suffix strings are disjoint except for zero suffix length. Tested seeds are deterministic; means are descriptive, with small per-seed test sets.',
        '- A 128-sample transducer trace is *not the same sensor physics* as 128 waveform samples. No claim of equal energy, wall-time, hidden precision or silicon cost can be inferred. Dense MAC counts are rough lower bounds; wave exposure/Fourier computations add cost.',
        '- No test-set tuning, supervised recurrent training, raw-matrix access or learned readout outside the common linear ridge decoder.',
        '', '## Reproduction','',
        '```powershell','python3.13 -m pip install numpy matplotlib numba',
        'python3.13 resource_matched_benchmark.py --selftest',
        'python3.13 resource_matched_benchmark.py', '```','',
        'Outputs: `resource_matched_results.json` (all seed results and cost proxies) and this report. Numba is optional; when available, its integrator is cross-checked against the unmodified wave simulator.','',
        '## Next fairer gate','',
        'An equal **wall-clock/operation-budget** comparison would need either longer recurrent updates or a radically cheaper wave-write implementation; an equal **sensor physics** comparison would require the same transducer model for both memories. Neither is solved by matching mutable state alone. A third objective is genuinely **multi-bit capacity** rather than binary AB/BA discrimination.']
    return '\n'.join(txt)+'\n'


def selftest():
    s=fixed_scan(16)
    assert s.shape==(128,16)
    assert np.linalg.matrix_rank(s)==16
    case=Case('XX',(0.,)*16)
    for m in ('ring16','esn8','esn16','trace16','bag16','esn32'):
        tr=comparator_traces([case],m,777)
        assert tr.shape==(1,2,128) and np.all(np.isfinite(tr))
    bag=comparator_traces([case], 'bag16', 777)
    assert np.allclose(bag[0,0],bag[0,1],atol=1e-14)
    for L in (2,8,16):
        a,b=suffix_pools(L,777)
        assert set(a).isdisjoint(b)
    example=np.zeros((2,2,128))
    assert evaluate(example,example)['paired']==.5
    print('PASS: single-scalar reader, shape, held-out suffix, commutative null, symmetric chance scoring',flush=True)


def cli():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed',type=int,default=20261106)
    ap.add_argument('--seeds',type=int,default=5)
    ap.add_argument('--train-pairs',type=int,default=48)
    ap.add_argument('--test-pairs',type=int,default=24)
    ap.add_argument('--jitter',type=float,default=.002)
    ap.add_argument('--lengths',default='0,2,4,8,16')
    ap.add_argument('--out',type=Path,default=Path('.'))
    ap.add_argument('--quick',action='store_true')
    ap.add_argument('--no-jit',action='store_true')
    ap.add_argument('--selftest',action='store_true')
    a=ap.parse_args()
    selftest()
    if a.selftest:return
    if a.quick:
        a.seeds=1;a.train_pairs=12;a.test_pairs=8;a.lengths='0,2'
    if not a.no_jit: enable_checked_numba()
    results=run(a)
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'resource_matched_results.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
    (a.out/'RESOURCE_MATCHED_RESULTS.md').write_text(report(results),encoding='utf-8')
    print(report(results),flush=True)

if __name__=='__main__':cli()
