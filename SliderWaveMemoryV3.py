"""SLIDER WAVE MEMORY LAB V3: a waveform-written network.

Offline Tkinter/Matplotlib desktop demonstration. No sound hardware.

A: phase-continuous sine oscillator.
B: bounded causal oscillator feedback, with no long-term structural writes.
C: eight damped coupled oscillators, sixteen writable graph edges. Each writing
   pulse excites one node. Actual complex Fourier coefficients measured at
   each node during the pulse determine local phase-sensitive edge writes. The NEXT wave propagates under the
   updated conductances; no A/B label is ever written into an arbitrary pattern.

Paired histories reset all fast state and read with a common, nonwriting probe.
A frozen-propagation control forces both pulses to traverse the original W0,
while still integrating the physical exposure they produce. As the updates are
additive, AB and BA then commute: this isolates the feedback via updated wiring.

This toy is not quantum computing, a proven useful memory architecture, or a
new result in physics. It deliberately does not use audio devices or servers.
Run --selftest for numerical/control tests, --demo for paired comparisons.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import argparse
from pathlib import Path
import numpy as np

NODES = 8
SAMPLE_RATE = 2400  # simulation steps per second (not audio-quality playback)
BUFFER_SIZE = 512
EVENT_BLOCKS = 3
EVENT_FREQ = {'A': 160.0, 'B': 230.0, 'X': 310.0}
EVENT_PORT = {'A': 0, 'B': 3, 'X': 5}
PROBE_FREQ = 196.0
PROBE_PORT = 6

# Unoriented edges: nearest-neighbor ring, and next-nearest neighbors.
EDGE_WEIGHTS = {}
for _i in range(NODES):
    EDGE_WEIGHTS[tuple(sorted((_i, (_i + 1) % NODES)))] = 0.20
    EDGE_WEIGHTS[tuple(sorted((_i, (_i + 2) % NODES)))] = 0.052
EDGES = np.array(list(EDGE_WEIGHTS), dtype=np.int64)
EDGE_BASE = np.array(list(EDGE_WEIGHTS.values()), dtype=float)
READOUT = np.array([.70, -.20, .45, -.55, .30, .65, -.40, .22])
READOUT /= np.linalg.norm(READOUT)


def initial_matrix() -> np.ndarray:
    w = np.zeros((NODES, NODES))
    w[EDGES[:, 0], EDGES[:, 1]] = EDGE_BASE
    w[EDGES[:, 1], EDGES[:, 0]] = EDGE_BASE
    return w


@dataclass
class Comparison:
    mode: str
    history_a: str
    history_b: str
    matrix_distance: float
    read_rms_difference: float
    read_rms_reference: float
    normalized_read_difference: float
    freeze_distance: float
    erase_distance: float
    read_repeat_error: float
    fast_state_error: float
    clamp_distance: float
    clamp_read_difference: float
    signal_a: np.ndarray
    signal_b: np.ndarray
    matrix_a: np.ndarray
    matrix_b: np.ndarray


class MemoryEngine:
    def __init__(self, mode='A', frequency=196., feedback_gain=.7,
                 write_rate=.055, write_strength=1.0, write_rule='fourier', seed=991):
        if mode not in ('A','B','C'):
            raise ValueError('mode must be A, B or C')
        self.mode = mode
        self.frequency = float(frequency)
        self.feedback_gain = float(feedback_gain)
        self.write_rate = float(write_rate)
        self.write_strength = float(write_strength)
        if write_rule not in ('fourier', 'strain'):
            raise ValueError('write_rule must be fourier or strain')
        self.write_rule = write_rule
        self.learning = True
        self.w = initial_matrix()
        self.phases = np.zeros(NODES)
        self.feedback = 0.0
        self.x = np.zeros(NODES)
        self.xprev = np.zeros(NODES)
        self.sample_clock = 0
        self.history = deque(maxlen=3)
        self.event_count = 0
        self.rng = np.random.default_rng(seed)
        self.last_exposure = np.zeros(len(EDGES))
        self.last_fourier = np.zeros(NODES, dtype=complex)

    def reset_fast(self):
        self.phases.fill(0.0)
        self.feedback = 0.0
        self.x.fill(0.0)
        self.xprev.fill(0.0)
        self.sample_clock = 0
        self.history.clear()

    def erase_matrix(self):
        self.w = initial_matrix()
        self.reset_fast()
        self.event_count = 0
        self.last_exposure.fill(0.0)
        self.last_fourier.fill(0.0)

    @property
    def structural_distance(self):
        return float(np.linalg.norm(self.w - initial_matrix()))

    def _oscillator_block(self, frequency):
        f = float(np.clip(frequency, 10, 1100))
        correction = (0.0 if self.mode == 'A' else
                      120.0 * np.clip(self.feedback_gain, 0, 1) * math.tanh(2*self.feedback))
        effective = float(np.clip(f + correction, 10, 1100))
        phase = self.phases[0] + 2*np.pi*effective*np.arange(BUFFER_SIZE)/SAMPLE_RATE
        out = .3*np.sin(phase)
        self.phases[0] = float((self.phases[0] + 2*np.pi*effective*BUFFER_SIZE/SAMPLE_RATE)%(2*np.pi))
        if self.mode == 'B':
            self.feedback = .78*self.feedback+.22*math.tanh(float(3.6*out[-1]))
        return out

    def _wave_samples(self, frequency, port, length, *, pulse=False,
                      propagation_matrix=None, accumulate=False):
        """Actual 8-node wave time-stepper; no future/label information used.

        x[t+1] = (2-gamma)x[t]-(1-gamma)x[t-1] - k*x[t]
                 - g*(D(W)-W)x[t] + external node force.
        The nonzero edge exposures are mean squared relative displacement.
        """
        W = self.w if propagation_matrix is None else propagation_matrix
        row_sum = W.sum(axis=1)
        omega = 2*np.pi*float(np.clip(frequency, 10, 1100))/SAMPLE_RATE
        k = np.arange(length)
        envelope = np.sin(np.pi*k/max(1,length-1))**2 if pulse else np.ones(length)
        drive = .17*envelope*np.sin(omega*k + 0.0) if pulse else .08*np.sin(omega*(k+self.sample_clock))
        y = np.empty(length)
        exposure = np.zeros(len(EDGES))
        spectrum = np.zeros(NODES, dtype=complex)
        weight_sum = 0.0
        x=self.x.copy(); xp=self.xprev.copy()
        # A weak damping makes each pulse settle; reset before each event is
        # intentional, removing fast state as the source of sequence effects.
        for t in range(length):
            xn = 1.975*x - .975*xp - .15*x - .87*(row_sum*x - W@x)
            xn[port] += drive[t]
            xp,x=x,xn
            y[t] = READOUT@x
            if accumulate:
                diff = x[EDGES[:,0]]-x[EDGES[:,1]]
                exposure += diff*diff
                # Single-frequency Fourier coefficient at the *driving* frequency.
                # This is measured from evolving node motion, not from a lookup table.
                weight = 0.5 - 0.5*math.cos(2*math.pi*t/max(1,length-1))
                spectrum += weight*x*complex(math.cos(omega*t), -math.sin(omega*t))
                weight_sum += weight
        if accumulate:
            self.last_fourier = spectrum*(2/max(1e-12, weight_sum))
        self.x=x
        self.xprev=xp
        self.sample_clock += length
        return y,exposure/max(1,length)

    def block(self, *, frequency=None, observe_noise=0):
        f = self.frequency if frequency is None else float(frequency)
        if self.mode=='C':
            out,_=self._wave_samples(f, PROBE_PORT, BUFFER_SIZE)
        else:
            out=self._oscillator_block(f)
        out=np.clip(out,-1,1)
        displayed = np.clip(out+self.rng.normal(0,observe_noise,len(out)),-1,1) if observe_noise>0 else out.copy()
        self.history.append(displayed)
        return displayed

    def event(self,label,*,clamp_propagation=False):
        label=label.upper()
        if label not in EVENT_FREQ:
            raise ValueError(label)
        # Eliminate dynamic carry-over between separate stimulus events.
        # The matrix is allowed to carry history, not the oscillating wave.
        self.reset_fast()
        if self.mode!='C':
            for _ in range(EVENT_BLOCKS):
                self.block(frequency=EVENT_FREQ[label])
            return False
        ref_matrix = initial_matrix() if clamp_propagation else None
        raw,exposure=self._wave_samples(EVENT_FREQ[label],EVENT_PORT[label],
                            BUFFER_SIZE*EVENT_BLOCKS,pulse=True,
                            propagation_matrix=ref_matrix,accumulate=True)
        if self.write_rule == 'fourier':
            # Complex differences capture both amplitude and relative phase.
            # Two nodes oscillating equally but out of phase have a large difference.
            F = self.last_fourier
            spectral_disagreement = np.abs(F[EDGES[:,0]] - F[EDGES[:,1]])**2
            exposure = 0.48*np.tanh(spectral_disagreement/0.25)
        self.last_exposure=exposure.copy()
        for b in range(EVENT_BLOCKS):
            self.history.append(raw[b*BUFFER_SIZE:(b+1)*BUFFER_SIZE].copy())
        if self.learning:
            rate=float(np.clip(self.write_rate,0,.20))*float(np.clip(self.write_strength,0,2))
            new_edges=np.clip(self.w[EDGES[:,0],EDGES[:,1]] + rate*exposure,
                              .01,.70)
            self.w[EDGES[:,0],EDGES[:,1]]=new_edges
            self.w[EDGES[:,1],EDGES[:,0]]=new_edges
            self.event_count += 1
            return True
        return False

    def read_probe(self,frequency=PROBE_FREQ,blocks=3):
        self.reset_fast()
        before=self.w.copy()
        if self.mode=='C':
            output=np.concatenate([self.block(frequency=frequency) for _ in range(blocks)])
        else:
            output=np.concatenate([self.block(frequency=frequency) for _ in range(blocks)])
        assert np.array_equal(before,self.w), 'A read must not change the matrix'
        return output


def compare_histories(mode='C',*,history_a='AB',history_b='BA',learning=True,
                      erase_after_write=False,write_rate=.055,write_strength=1.,
                      feedback_gain=.7,clamp_propagation=False,write_rule="fourier"):
    setup=dict(mode=mode,frequency=PROBE_FREQ,feedback_gain=feedback_gain,
               write_rate=write_rate,write_strength=write_strength,write_rule=write_rule)
    def run(history, *, frozen=False, erased=False,clamp=False):
        eng=MemoryEngine(**setup)
        eng.learning=learning and not frozen
        for label in history.upper():eng.event(label,clamp_propagation=clamp)
        if erased:eng.erase_matrix()
        eng.reset_fast()
        return eng
    a=run(history_a,erased=erase_after_write,clamp=clamp_propagation)
    b=run(history_b,erased=erase_after_write,clamp=clamp_propagation)
    fast_error=max(float(np.max(np.abs(a.phases-b.phases))),
                   float(np.max(np.abs(a.x-b.x))),
                   float(np.max(np.abs(a.xprev-b.xprev))),
                   abs(a.feedback-b.feedback),
                   abs(a.sample_clock-b.sample_clock),
                   len(a.history),len(b.history))
    wa,wb=a.w.copy(),b.w.copy()
    sa,sb=a.read_probe(),b.read_probe()
    rms=lambda z:float(np.sqrt(np.mean(np.square(z))))
    rd=rms(sa-sb)
    reference=float(np.sqrt((np.mean(sa**2)+np.mean(sb**2))/2))
    repeated=a.read_probe()
    repeat_error=float(np.max(np.abs(sa-repeated)))
    fa,fb=run(history_a,frozen=True),run(history_b,frozen=True)
    ca,cb=run(history_a,clamp=True),run(history_b,clamp=True)
    # Controlled null: the exact same pulses write additive exposures through
    # unchanged W0, so ordering cannot affect the final matrix.
    freeze_distance=float(np.linalg.norm(fa.w-fb.w))
    clamp_distance=float(np.linalg.norm(ca.w-cb.w))
    clamp_read_difference=rms(ca.read_probe()-cb.read_probe())
    return Comparison(mode,history_a,history_b,
        float(np.linalg.norm(wa-wb)),rd,reference,rd/max(reference,1e-15),
        freeze_distance,0.0,repeat_error,float(fast_error),clamp_distance,
        clamp_read_difference,sa,sb,wa,wb)


def selftest():
    for mode in 'ABC':
        c=compare_histories(mode)
        if mode!='C':
            assert c.matrix_distance==0 and c.read_rms_difference==0
        else:
            assert c.matrix_distance>0.001 and c.read_rms_difference>0.0001,(c.matrix_distance,c.read_rms_difference)
            assert c.clamp_distance<1e-12 and c.clamp_read_difference<1e-12
        assert c.freeze_distance<1e-12 and c.fast_state_error==0
        assert c.read_repeat_error==0
    for law in ('fourier', 'strain'):
        c=compare_histories('C',write_rule=law)
        assert c.matrix_distance > .001 and c.read_rms_difference > .0001
        assert c.clamp_distance < 1e-12
    for h1,h2 in [('ABX','BAX')]:
        c=compare_histories('C',history_a=h1,history_b=h2)
        assert c.matrix_distance>0.001 and c.read_rms_difference>0.0001,(c.matrix_distance,c.read_rms_difference)
    frozen=compare_histories('C',learning=False)
    erased=compare_histories('C',erase_after_write=True)
    assert frozen.matrix_distance==0 and frozen.read_rms_difference==0
    assert erased.matrix_distance==0 and erased.read_rms_difference==0
    e=MemoryEngine('C')
    e.event('A')
    assert np.max(np.abs(e.last_fourier))>0
    # Verify the implemented Fourier formula really is the edge write feature.
    F=e.last_fourier
    expected=.48*np.tanh(np.abs(F[EDGES[:,0]]-F[EDGES[:,1]])**2/.25)
    np.testing.assert_allclose(e.last_exposure,expected,rtol=1e-12,atol=1e-12)
    before=e.w.copy();e.read_probe()
    assert np.array_equal(before,e.w)
    assert np.all(np.isfinite(e.w)) and np.allclose(e.w,e.w.T)
    for mode in 'ABC':
        e=MemoryEngine(mode)
        for _ in range(5):
            z=e.block()
            assert np.all(np.isfinite(z)) and np.max(np.abs(z))<=1
    c=compare_histories('C')
    print('PASS: A/B no lasting structural state; C retains order')
    print('PASS: C ABX/BAX persists beyond common X')
    print('PASS: no learning / erase = no memory')
    print('PASS: fixed propagation while writing = no order effect')
    print('PASS: nonwriting, repeatable probe / fast state reset')
    print('PASS: 3 modes produce bounded finite signals')
    print('PASS: both Fourier-phase and time-domain strain write rules')
    print('PASS: write features extracted from live complex Fourier amplitudes')
    print('C: ||W_AB-W_BA||=%.6f, RMS read difference=%.6f'%(c.matrix_distance,c.read_rms_difference))



def launch_gui() -> None:
    import tkinter as tk
    from tkinter import ttk
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    BG = '#101621'
    PANEL = '#182333'
    ACCENT = '#67d1c1'
    INK = '#e9eef6'
    SUB = '#a5b4c6'

    root = tk.Tk()
    root.title('SLIDER WAVE MEMORY LAB V3 / Phase • Feedback • Wave-Written Graph')
    # Start inside the available display instead of opening off-screen.
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f'{min(1440, max(900, sw - 80))}x{min(930, max(600, sh - 110))}')
    root.minsize(850,560)
    root.configure(bg=BG)
    style=ttk.Style(root)
    if 'clam' in style.theme_names(): style.theme_use('clam')
    style.configure('TFrame', background=BG)
    style.configure('TLabel', background=BG, foreground=INK, font=('Segoe UI',10))
    style.configure('TButton', padding=7, font=('Segoe UI',10))
    style.configure('TRadiobutton', background=BG, foreground=INK, font=('Segoe UI',10))
    style.configure('TCheckbutton', background=BG, foreground=INK, font=('Segoe UI',10))
    style.configure('Horizontal.TScale', background=BG)

    # The entire interface scrolls vertically, including the experiment controls,
    # result panel, status and log.  This is needed on small screens or with
    # Windows display scaling where a 930px window would hide the bottom.
    viewport = tk.Frame(root, bg=BG)
    viewport.pack(fill='both', expand=True)
    page_scroll = tk.Canvas(viewport, bg=BG, highlightthickness=0, bd=0,
                            takefocus=True)
    page_bar = ttk.Scrollbar(viewport, orient='vertical', command=page_scroll.yview)
    page_scroll.configure(yscrollcommand=page_bar.set)
    page_bar.pack(side='right', fill='y')
    page_scroll.pack(side='left', fill='both', expand=True)
    page = tk.Frame(page_scroll, bg=BG)
    page_window = page_scroll.create_window((0, 0), window=page, anchor='nw')

    def update_scroll_region(_=None):
        page_scroll.configure(scrollregion=page_scroll.bbox('all'))

    def fit_page_width(event):
        page_scroll.itemconfigure(page_window, width=event.width)

    page.bind('<Configure>', update_scroll_region)
    page_scroll.bind('<Configure>', fit_page_width)

    def scroll_wheel(event):
        # Keep wheel zoom available in the 3D Matplotlib canvas.
        px, py = root.winfo_pointerxy()
        try:
            mw = canvas.get_tk_widget()
            if (mw.winfo_rootx() <= px < mw.winfo_rootx()+mw.winfo_width()
                    and mw.winfo_rooty() <= py < mw.winfo_rooty()+mw.winfo_height()):
                return
        except (NameError, tk.TclError):
            pass
        if event.num == 4:
            amount = -3
        elif event.num == 5:
            amount = 3
        else:
            amount = -int(event.delta / 120) * 3
        if amount:
            page_scroll.yview_scroll(amount, 'units')

    root.bind_all('<MouseWheel>', scroll_wheel, add='+')
    root.bind_all('<Button-4>', scroll_wheel, add='+')
    root.bind_all('<Button-5>', scroll_wheel, add='+')

    engine = MemoryEngine('A')
    mode_var=tk.StringVar(value='A')
    freq_var=tk.DoubleVar(value=196)
    fb_var=tk.DoubleVar(value=0.70)
    write_var=tk.DoubleVar(value=0.06)
    noise_var=tk.DoubleVar(value=0)
    learning_var=tk.BooleanVar(value=True)
    rule_var=tk.StringVar(value='fourier')
    running_var=tk.BooleanVar(value=True)
    status_var=tk.StringVar(value='Running • Mode A • phase-only oscillator')
    info_var=tk.StringVar(value='Wave-written graph: complex Fourier coefficients of simulated waves write connections in Mode C.')
    memory_var=tk.StringVar(value='||W−W₀|| = 0.0000')
    result_var=tk.StringVar(value='Compare AB vs BA to test for stored order.\nBoth histories are read from the same reset fast state.')
    task = {'comparison': None, 'frame': 0}

    header=tk.Frame(page,bg=BG,padx=16,pady=12)
    header.pack(fill='x')
    tk.Label(header,text='SLIDER  /  WAVE MEMORY LAB V3',fg=ACCENT,bg=BG,
             font=('Segoe UI Semibold',21)).pack(side='left')
    tk.Label(header,text='A phase     B feedback     C wave-written graph',fg=SUB,bg=BG,
             font=('Segoe UI',11)).pack(side='left',padx=25)
    tk.Label(header,text='OFFLINE SIMULATION  •  NO SPEAKER OUTPUT',fg='#f0b46a',bg=BG,
             font=('Segoe UI',9)).pack(side='right')

    control=tk.Frame(page,bg=PANEL,padx=15,pady=12)
    control.pack(fill='x',padx=13,pady=(0,10))

    def log(msg: str):
        log_box.configure(state='normal')
        log_box.insert('end', msg+'\n')
        log_box.see('end')
        log_box.configure(state='disabled')

    def update_params(*_):
        engine.frequency=float(freq_var.get())
        engine.feedback_gain=float(fb_var.get())
        engine.write_rate=float(write_var.get())
        engine.learning=bool(learning_var.get())
        engine.write_rule=rule_var.get()
        freq_label.configure(text=f'{engine.frequency:,.0f} Hz')
        feedback_label.configure(text=f'{engine.feedback_gain:.2f}')
        write_label.configure(text=f'{engine.write_rate:.3f}')
        noise_label.configure(text=f'{noise_var.get():.3f}')

    def set_mode():
        engine.mode=mode_var.get()
        engine.reset_fast()
        task['comparison']=None
        descriptions = {
            'A':'A: pure sine, accumulated oscillator phase only. No feedback and no matrix writes.',
            'B':'B: previous generated signal changes the NEXT block’s phase speed (bounded feedback).',
            'C':'C: eight coupled waves write graph edges from measured Fourier phases; reads never write.'}
        info_var.set(descriptions[engine.mode])
        status_var.set(f'Mode {engine.mode} • fast state reset • structural state preserved')
        log(f'Switched to Mode {engine.mode}; fast state reset.')

    select=tk.Frame(control,bg=PANEL)
    select.pack(side='left',padx=(0,23))
    tk.Label(select,text='OPERATING MODE',fg=SUB,bg=PANEL,font=('Segoe UI',9,'bold')).pack(anchor='w')
    for k,txt in [('A','A · original'),('B','B · feedback'),('C','C · wave-graph memory')]:
        tk.Radiobutton(select,text=txt,variable=mode_var,value=k,command=set_mode,
                       bg=PANEL,fg=INK,selectcolor=BG,activebackground=PANEL,
                       activeforeground=ACCENT,font=('Segoe UI',10)).pack(anchor='w')

    sliders=tk.Frame(control,bg=PANEL)
    sliders.pack(side='left',fill='x',expand=True)

    def slider_row(row,title,var,minval,maxval,display):
        tk.Label(sliders,text=title,fg=INK,bg=PANEL,anchor='w').grid(row=row,column=0,sticky='w',pady=4)
        sc=tk.Scale(sliders,from_=minval,to=maxval,variable=var,orient='horizontal',
                    showvalue=False,highlightthickness=0,troughcolor='#34445a',
                    bg=PANEL,fg=INK,activebackground=ACCENT,resolution=0.005 if maxval<=0.2 else (0.01 if maxval<=2 else 1),
                    command=update_params,length=250,borderwidth=0)
        sc.grid(row=row,column=1,sticky='ew',padx=(12,5))
        display.grid(row=row,column=2,sticky='e',padx=7)
    sliders.columnconfigure(1,weight=1)
    freq_label=tk.Label(sliders,text='196 Hz',bg=PANEL,fg=ACCENT,width=10,anchor='e')
    feedback_label=tk.Label(sliders,text='0.70',bg=PANEL,fg=ACCENT,width=10,anchor='e')
    write_label=tk.Label(sliders,text='0.060',bg=PANEL,fg=ACCENT,width=10,anchor='e')
    noise_label=tk.Label(sliders,text='0.000',bg=PANEL,fg=ACCENT,width=10,anchor='e')
    slider_row(0,'Simulation frequency',freq_var,40,600,freq_label)
    slider_row(1,'Feedback strength',fb_var,0,1,feedback_label)
    slider_row(2,'Wave-write rate η',write_var,0,0.2,write_label)
    slider_row(3,'Display noise',noise_var,0,0.15,noise_label)
    update_params()

    outer=tk.Frame(page,bg=BG)
    outer.pack(fill='both',expand=True,padx=13,pady=(0,8))
    outer.columnconfigure(0,weight=1)
    outer.columnconfigure(1,weight=0)
    outer.rowconfigure(0,weight=1)

    visual=tk.Frame(outer,bg=PANEL)
    visual.grid(row=0,column=0,sticky='nsew',padx=(0,10))
    fig=Figure(figsize=(10,6.9),dpi=100,facecolor=PANEL)
    gs=fig.add_gridspec(2,3,width_ratios=[1,1,.95],height_ratios=[1.1,.85],
                        wspace=.32,hspace=.31,left=.04,right=.97,top=.92,bottom=.1)
    ax=fig.add_subplot(gs[:,0:2],projection='3d',computed_zorder=False)
    heat=fig.add_subplot(gs[0,2])
    trace=fig.add_subplot(gs[1,2])
    fig.text(.04,.965,'PHASE-SPACE  /  three consecutive signal blocks',color=INK,fontsize=12)
    fig.text(.705,.965,'WAVE-WRITTEN CONNECTIONS',color=INK,fontsize=10)
    canvas=FigureCanvasTkAgg(fig,master=visual)
    canvas.get_tk_widget().pack(fill='both',expand=True)

    side=tk.Frame(outer,bg=PANEL,width=350,padx=17,pady=15)
    side.grid(row=0,column=1,sticky='ns')
    side.grid_propagate(False)
    tk.Label(side,text='EXPERIMENT CONTROLS',font=('Segoe UI Semibold',14),
             bg=PANEL,fg=ACCENT).pack(anchor='w',pady=(0,11))

    def add_button(label,cmd):
        tk.Button(side,text=label,command=cmd,bg='#293d51',fg=INK,
                  activebackground='#395c6b',activeforeground='white',
                  font=('Segoe UI',10),relief='flat',height=1).pack(fill='x',pady=3)

    def event_ui(label):
        update_params()
        did=engine.event(label)
        task['comparison']=None
        memory_var.set(f'||W−W₀|| = {engine.structural_distance:.5f}')
        msg=f'{label} pulse ({EVENT_FREQ[label]:.0f}Hz) — '+('edge exposures written' if did else 'no structural write')
        status_var.set(msg)
        log(msg)
        redraw(force=True)

    def live_sequence(seq):
        for ch in seq:
            engine.event(ch)
        task['comparison']=None
        memory_var.set(f'||W−W₀|| = {engine.structural_distance:.5f}')
        status_var.set(f'Live sequence {seq} completed in mode {engine.mode}')
        log(f'LIVE {seq}: W distance from baseline = {engine.structural_distance:.5f}')
        redraw(force=True)

    def reset_fast():
        engine.reset_fast()
        task['comparison']=None
        status_var.set('Fast state erased; matrix preserved')
        log(f'Fast state reset. W distance remains {engine.structural_distance:.5f}.')
        redraw(force=True)

    def erase():
        engine.erase_matrix()
        task['comparison']=None
        memory_var.set('||W−W₀|| = 0.00000')
        result_var.set('Network erased. All physical-edge conductances reset to W₀.')
        status_var.set('Network W erased; fast state reset')
        log('W erased. All wave-written conductances reset.')
        redraw(force=True)

    add_button(f'Inject A   ·  {EVENT_FREQ["A"]:.0f} Hz', lambda:event_ui('A'))
    add_button(f'Inject B   ·  {EVENT_FREQ["B"]:.0f} Hz', lambda:event_ui('B'))
    add_button(f'Inject X   ·  {EVENT_FREQ["X"]:.0f} Hz', lambda:event_ui('X'))
    tk.Label(side,text='Build a live history',bg=PANEL,fg=SUB,font=('Segoe UI',9)).pack(anchor='w',pady=(13,3))
    add_button('Write A → B', lambda: live_sequence('AB'))
    add_button('Write B → A', lambda: live_sequence('BA'))
    tk.Label(side,text='Matrix write law',bg=PANEL,fg=SUB,font=('Segoe UI',9)).pack(anchor='w',pady=(9,2))
    rules=ttk.Combobox(side,textvariable=rule_var,state='readonly',
                       values=['fourier','strain'],width=18)
    rules.pack(fill='x',pady=(0,4))
    rules.bind('<<ComboboxSelected>>',update_params)
    tk.Label(side,text='fourier = measured phase+amplitude; strain = time-domain deformation',
             bg=PANEL,fg=SUB,font=('Segoe UI',8),wraplength=295,justify='left').pack(anchor='w')
    tk.Checkbutton(side,text='Learning enabled (Mode C only)',variable=learning_var,
                   command=update_params,bg=PANEL,fg=INK,selectcolor=BG,
                   activebackground=PANEL,activeforeground=INK).pack(anchor='w',pady=8)
    add_button('Reset fast state   /   keep W',reset_fast)
    add_button('ERASE matrix W',erase)

    def do_comparison(common=False):
        update_params()
        ah, bh=('ABX','BAX') if common else ('AB','BA')
        res=compare_histories(engine.mode,history_a=ah,history_b=bh,
                              learning=engine.learning,write_rate=engine.write_rate,
                              write_strength=engine.write_strength,
                              feedback_gain=engine.feedback_gain,write_rule=engine.write_rule)
        task['comparison']=res
        result_var.set(f'{ah}  vs  {bh}\n'
                       f'Δmatrix (Frobenius): {res.matrix_distance:.6f}\n'
                       f'Δread RMS: {res.read_rms_difference:.6f}\n'
                       f'Relative read RMS: {100*res.normalized_read_difference:.2f}%\n'
                       f'Frozen-learning ΔW: {res.freeze_distance:.1e}\n'
                       f'Fixed propagation ΔW: {res.clamp_distance:.1e}\n'
                       f'Fast-state difference: {res.fast_state_error:.1e}\n'
                       f'Repeated-probe error: {res.read_repeat_error:.1e}')
        status_var.set(f'Compared {ah}/{bh} • Mode {engine.mode} • same reset and probe')
        log(f'TEST {ah}/{bh} mode {engine.mode}: ||ΔW|| {res.matrix_distance:.6f}; '
            f'read RMS Δ {res.read_rms_difference:.6f}; '
            'reset fast/phase, read nonwriting; fixed-propagation control.')
        redraw(force=True)

    tk.Label(side,text='PAIRED HISTORY TEST',font=('Segoe UI Semibold',12),
             bg=PANEL,fg=ACCENT).pack(anchor='w',pady=(18,7))
    add_button('COMPARE  A–B  vs  B–A',lambda:do_comparison(False))
    add_button('COMPARE  A–B–X  vs  B–A–X',lambda:do_comparison(True))
    tk.Label(side,textvariable=result_var,bg=PANEL,fg=INK,justify='left',
             font=('Consolas',9),wraplength=305).pack(anchor='w',fill='x',pady=(13,5))

    def toggle():
        running_var.set(not running_var.get())
        play_btn.configure(text='Pause simulation' if running_var.get() else 'Resume simulation')
        status_var.set('Running' if running_var.get() else 'Paused')

    play_btn=tk.Button(side,text='Pause simulation',command=toggle,bg='#186558',fg='white',
                       font=('Segoe UI Semibold',10),relief='flat',height=2)
    play_btn.pack(fill='x',side='bottom',pady=8)

    tk.Label(page,textvariable=info_var,bg=BG,fg=SUB,font=('Segoe UI',10),
             anchor='w',justify='left',wraplength=1350).pack(fill='x',padx=17,pady=(0,5))
    tk.Label(page,textvariable=memory_var,bg=BG,fg=ACCENT,font=('Consolas',11),
             anchor='w').pack(fill='x',padx=17)
    tk.Label(page,textvariable=status_var,bg=BG,fg=SUB,font=('Segoe UI',9),
             anchor='w').pack(fill='x',padx=17,pady=3)
    log_box=tk.Text(page,height=3,bg='#0a101a',fg='#a1d9d3',borderwidth=0,
                    insertbackground='white',font=('Consolas',9),state='disabled')
    log_box.pack(fill='x',padx=13,pady=(1,9))
    log('Ready. Mode C writes from wave motion; fixed-propagation control should give zero order memory.')

    def redraw(force=False):
        task['frame'] +=1
        if not force and task['frame']%2: return
        h=list(engine.history)
        ax.clear()
        ax.set_facecolor(PANEL)
        ax.set_xlabel('buffer n−2',color=SUB,labelpad=4,fontsize=9)
        ax.set_ylabel('buffer n−1',color=SUB,labelpad=4,fontsize=9)
        ax.set_zlabel('buffer n',color=SUB,labelpad=4,fontsize=9)
        limit=max(.43, min(1.0, 1.12*max((float(np.max(np.abs(v))) for v in h), default=0.38)))
        ax.set_xlim(-limit,limit);ax.set_ylim(-limit,limit);ax.set_zlim(-limit,limit)
        ax.tick_params(colors=SUB,labelsize=7)
        for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
            try:
                axis.pane.fill=False
                axis.line.set_color('#6e8596')
            except (AttributeError,TypeError): pass
        if len(h)>=3:
            p=np.stack([v[::2] for v in h[-3:]])
            ax.scatter(p[0],p[1],p[2],s=1.8,c=np.linspace(0,1,p.shape[1]),
                       cmap='winter',alpha=.80,rasterized=True)
        ax.view_init(elev=23,azim=43)

        heat.clear();heat.set_facecolor(PANEL)
        view_matrix=engine.w
        if task['comparison'] is not None and engine.mode=='C':
            view_matrix=task['comparison'].matrix_a-task['comparison'].matrix_b
            heat.set_title('ΔW: history 1 − 2',color=INK,fontsize=9)
        else:
            heat.set_title(f'W: {engine.write_rule} wave writes',color=INK,fontsize=9)
        vmax=max(0.35,float(np.max(np.abs(view_matrix))))
        heat.imshow(view_matrix,vmin=-vmax,vmax=vmax,cmap='BrBG',interpolation='nearest')
        heat.set_xticks(range(NODES));heat.set_yticks(range(NODES))
        heat.tick_params(colors=SUB,labelsize=8,length=0)
        for i in range(NODES):
            for j in range(NODES):
                rgba=matplotlib.colormaps['BrBG']((view_matrix[i,j]+vmax)/(2*vmax))
                lightness=0.2126*rgba[0]+0.7152*rgba[1]+0.0722*rgba[2]
                heat.text(j,i,f'{view_matrix[i,j]:+.2f}',color='#111827' if lightness>.52 else INK,
                          ha='center',va='center',fontsize=6)

        trace.clear();trace.set_facecolor(PANEL)
        if task['comparison'] is not None:
            comp=task['comparison']; win=700
            trace.plot(comp.signal_a[-win:],color='#61d8c0',lw=.8,label=comp.history_a)
            trace.plot(comp.signal_b[-win:],color='#edb96d',lw=.8,alpha=.8,label=comp.history_b)
            trace.set_title('Paired readouts, same initial phase',fontsize=9,color=INK)
            trace.legend(facecolor=PANEL,labelcolor=INK,fontsize=8,loc='upper right')
        elif h:
            trace.plot(h[-1][::2],color=ACCENT,lw=.85)
            trace.set_title('Current output block',fontsize=9,color=INK)
        else:
            trace.set_title('Output signal',fontsize=9,color=INK)
        trace.set_ylim(-1,1)
        trace.set_xlim(left=0)
        trace.tick_params(colors=SUB,labelsize=7)
        for sp in trace.spines.values(): sp.set_color('#526279')
        trace.grid(alpha=.13)
        canvas.draw_idle()

    def tick():
        if running_var.get():
            update_params()
            for _ in range(2):
                engine.block(observe_noise=float(noise_var.get()))
            memory_var.set(f'||W−W₀|| = {engine.structural_distance:.5f}   •   writes: {engine.event_count}')
            redraw()
        root.after(150,tick)

    def close():
        running_var.set(False)
        root.destroy()
    def page_down(_=None):
        page_scroll.yview_scroll(1, 'pages')
    def page_up(_=None):
        page_scroll.yview_scroll(-1, 'pages')
    root.bind('<Next>', page_down)
    root.bind('<Prior>', page_up)
    root.protocol('WM_DELETE_WINDOW',close)
    # A visual state appears immediately without waiting for a second.
    for _ in range(3): engine.block()
    redraw(force=True)
    tick()
    root.mainloop()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selftest',action='store_true',help='Run headless memory and control checks')
    parser.add_argument('--demo',action='store_true',help='Print comparison values without launching the UI')
    args=parser.parse_args()
    if args.selftest:
        selftest()
    elif args.demo:
        for mode in ('A','B','C'):
            for ah,bh in [('AB','BA'),('ABX','BAX')]:
                r=compare_histories(mode,history_a=ah,history_b=bh)
                print(f'{mode} {ah}/{bh}   ||ΔW||={r.matrix_distance:.6f}  '
                      f'RMS Δread={r.read_rms_difference:.6f}  '
                      f'relative={100*r.normalized_read_difference:.3f}%')
    else:
        launch_gui()



if __name__=='__main__':
    main()
