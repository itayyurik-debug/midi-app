# -*- coding: utf-8 -*-
"""
notation.py -- score engine (what Songscription / Piano2Notes / MuseScore-style tools do after the neural net):
  beats + downbeats -> bars / time signatures / tempo map -> beat-synchronous quantization (binary or triplet feel)
  -> staves + voices + ties + rests + key signatures + pitch spelling -> MusicXML (+ MIDI) with chord symbols.
No heavy dependencies: numpy only (librosa / beat_this are imported lazily where needed).
All score positions are integers in "divisions" (24 per quarter note).
"""
import bisect, os, struct
from collections import Counter
from xml.sax.saxutils import escape
import numpy as np

DIV = 24
TICK_PER_DIV = 20          # MIDI ppq 480 = 24 * 20
MIDI_PPQ = 480

# kind -> (name, staves, clefs[(sign,line,octave_change)], monophonic, GM program, channel-less)
KINDS = {
    "piano":     dict(name="Piano",              staves=2, clefs=[("G", 2, 0), ("F", 4, 0)], mono=False, prog=0),
    "strings":   dict(name="Strings",            staves=2, clefs=[("G", 2, 0), ("F", 4, 0)], mono=False, prog=48),
    "other":     dict(name="Other",              staves=2, clefs=[("G", 2, 0), ("F", 4, 0)], mono=False, prog=50),
    "guitar":    dict(name="Guitar",             staves=1, clefs=[("G", 2, -1)],             mono=False, prog=26),
    "bass":      dict(name="Bass",               staves=1, clefs=[("F", 4, -1)],             mono=True,  prog=33),
    "vocals":    dict(name="Vocals",             staves=1, clefs=[("G", 2, 0)],              mono=True,  prog=53),
    "sax_sop":   dict(name="Soprano Sax",        staves=1, clefs=[("G", 2, 0)],              mono=True,  prog=64, tr=(-2, -1, 0)),
    "sax_alto":  dict(name="Alto Sax",           staves=1, clefs=[("G", 2, 0)],              mono=True,  prog=65, tr=(-9, -5, 0)),
    "sax_tenor": dict(name="Tenor Sax",          staves=1, clefs=[("G", 2, 0)],              mono=True,  prog=66, tr=(-2, -1, -1)),
    "sax_bari":  dict(name="Baritone Sax",       staves=1, clefs=[("G", 2, 0)],              mono=True,  prog=67, tr=(-9, -5, -1)),
    "brass":     dict(name="Trumpet / Brass",    staves=1, clefs=[("G", 2, 0)],              mono=True,  prog=56, tr=(-2, -1, 0)),
    "violin":    dict(name="Violin",             staves=1, clefs=[("G", 2, 0)],              mono=True,  prog=40),
    "cello":     dict(name="Cello",              staves=1, clefs=[("F", 4, 0)],              mono=True,  prog=42),
    "drums":     dict(name="Drums",              staves=1, clefs=[("percussion", 2, 0)],     mono=False, prog=0),
}
DRUM_INFO = {36: ("Kick", "F", 4, None), 38: ("Snare", "C", 5, None), 42: ("Hi-Hat", "G", 5, "x")}

# =============================================================================
# 1) Beats -> timeline (bars, meter, tempo)
# =============================================================================
_BEAT_MODEL = None


def track_beats(wav_path, device="cpu"):
    """Beat + downbeat times. Beat This! (ISMIR 2024 transformer) if installed, else librosa."""
    global _BEAT_MODEL
    try:
        from beat_this.inference import File2Beats
        if _BEAT_MODEL is None:
            _BEAT_MODEL = File2Beats(checkpoint_path="final0", device=device, dbn=False)
        b, d = _BEAT_MODEL(wav_path)
        b, d = np.asarray(b, float), np.asarray(d, float)
        if len(b) >= 8 and len(d) >= 2:
            return b, d, "beat_this"
    except Exception as e:
        print("Beat This! unavailable -> librosa fallback:", e)
    import librosa
    y, sr = librosa.load(wav_path, sr=22050, mono=True)
    oenv = librosa.onset.onset_strength(y=y, sr=sr)
    _, bf = librosa.beat.beat_track(onset_envelope=oenv, sr=sr)
    b = librosa.frames_to_time(bf, sr=sr)
    if len(b) < 8:
        raise RuntimeError("beat tracking failed")
    st = oenv[np.clip(bf, 0, len(oenv) - 1)]
    best = max(range(4), key=lambda ph: st[ph::4].sum())
    return np.asarray(b, float), np.asarray(b[best::4], float), "librosa"


class Timeline:
    """pulse_t: P+1 pulse boundary times (sec); pulse_div: P pulse lengths (divs); bars: list of dicts."""

    def __init__(self, pulse_t, pulse_div, bars, tempos):
        pt = np.asarray(pulse_t, float)
        for i in range(1, len(pt)):                     # strictly increasing
            if pt[i] <= pt[i - 1]:
                pt[i] = pt[i - 1] + 1e-3
        self.pt = pt
        self.pdiv = [int(x) for x in pulse_div]
        self.ppos = np.concatenate([[0], np.cumsum(self.pdiv)]).astype(float)
        self.bars = bars
        self.pbar = []
        for bi, b in enumerate(bars):
            b["d0"], b["d1"] = int(self.ppos[b["i0"]]), int(self.ppos[b["i1"]])
            b["d0o"] = b["d0"]; b["lead"] = 0
            self.pbar += [bi] * (b["i1"] - b["i0"])
        self.tempos = tempos                            # [(div_pos, quarter_bpm)]
        self.bar_d0 = [b["d0"] for b in bars]
        self.tri = [False] * len(bars)

    @property
    def total_div(self):
        return int(self.ppos[-1])

    def t2d(self, t):
        return np.interp(t, self.pt, self.ppos)

    def d2t(self, d):
        return float(np.interp(d, self.ppos, self.pt))

    def pulse_at(self, d):
        i = int(np.searchsorted(self.ppos, d, side="right")) - 1
        return max(0, min(len(self.pdiv) - 1, i))


def _density(pt, i0, i1, onsets):
    tern = binc = 0
    for i in range(i0, i1):
        a, b = bisect.bisect_left(onsets, pt[i]), bisect.bisect_left(onsets, pt[i + 1])
        L = pt[i + 1] - pt[i]
        for t in onsets[a:b]:
            f = (t - pt[i]) / L
            if min(f, 1 - f) < 0.06:
                continue
            d3 = min(abs(f - 1 / 3), abs(f - 2 / 3))
            d2 = min(abs(f - .5), abs(f - .25), abs(f - .75))
            if d3 < 0.07 and d3 < d2:
                tern += 1
            elif d2 < 0.07:
                binc += 1
    return tern, binc


def timeline_from_beats(beats, downbeats, t_min, t_max, onsets, force_n=0):
    beats = np.unique(np.round(np.asarray(beats, float), 4))
    med = float(np.median(np.diff(beats)))
    keep = [0]
    for i in range(1, len(beats)):
        if beats[i] - beats[keep[-1]] >= 0.45 * med:
            keep.append(i)
    beats = beats[keep]
    di = sorted({int(np.argmin(np.abs(beats - d))) for d in downbeats}) or [0]
    pre = 0
    while beats[0] > t_min + 0.04 and pre < 24:
        step = float(np.median(np.diff(beats[:9])))
        beats = np.insert(beats, 0, beats[0] - step); pre += 1
    di = [i + pre for i in di]
    post = 0
    while beats[-1] < t_max - 0.01 and post < 24:
        step = float(np.median(np.diff(beats[-9:])))
        beats = np.append(beats, beats[-1] + step); post += 1
    P = len(beats) - 1
    diffs = np.diff(di)
    typ = int(Counter(diffs.tolist()).most_common(1)[0][0]) if len(diffs) else 4
    if typ < 2 or typ > 16:
        typ = 4
    on_sorted = np.sort(np.asarray(onsets, float))
    tol = 0.12 * med
    acc = np.array([np.searchsorted(on_sorted, b + tol) - np.searchsorted(on_sorted, b - tol) for b in beats], float)
    phase_score = lambda ty: [acc[p::ty].sum() for p in range(ty)]
    if 2 <= force_n <= 16:                                # user fixed the meter: bar phase from onset accents
        typ = int(force_n)
        di = list(range(int(np.argmax(phase_score(typ))), P + 1, typ))
    elif len(di) >= 3 and float((np.diff(di) == typ).mean()) > 0.8:   # tracker phase sanity check
        sc = phase_score(typ)
        cur, best = di[0] % typ, int(np.argmax(sc))
        if best != cur and sc[best] > 1.3 * max(sc[cur], 1.0):
            di = list(range(best, P + 1, typ))
    bounds = [di[0]]
    while bounds[0] - typ >= 0:
        bounds.insert(0, bounds[0] - typ)
    if bounds[0] > 0:
        bounds.insert(0, 0)
    for d in di[1:]:
        if d > bounds[-1]:
            bounds.append(d)
    while bounds[-1] + typ <= P:
        bounds.append(bounds[-1] + typ)
    if bounds[-1] != P:
        bounds.append(P)
    ranges = list(zip(bounds[:-1], bounds[1:]))
    if ranges and ranges[0][1] - ranges[0][0] > typ:     # first "bar" too long = pickup + a full bar (not 5/4!)
        a0, b0 = ranges[0]
        ranges[:1] = [(a0, b0 - typ), (b0 - typ, b0)]
    out = []
    for a, b in ranges:                                  # merge 1-pulse bars
        if b - a == 1 and out:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    res = []
    for a, b in out:                                     # split overly long bars (tracker errors, not real big meters)
        if b - a > max(12, typ):
            k = max(2, int(round((b - a) / typ)))
            e = np.linspace(a, b, k + 1).round().astype(int)
            res += [(int(x), int(y)) for x, y in zip(e[:-1], e[1:]) if y > x]
        else:
            res.append((a, b))
    if len(res) > 1 and res[-1][1] - res[-1][0] < typ and res[-1][1] == len(beats) - 1:   # pad the last bar to full length
        miss = typ - (res[-1][1] - res[-1][0])
        step = float(np.median(np.diff(beats[-9:])))
        beats = np.concatenate([beats, beats[-1] + step * np.arange(1, miss + 1)])
        res[-1] = (res[-1][0], res[-1][1] + miss)
    # isolated odd bars (tracker errors) -> normalised to the typical size, pulses re-spaced evenly
    pts, bar_n = [beats[res[0][0]]], []
    for k, (a, b) in enumerate(res):
        n = b - a
        prev_n = res[k - 1][1] - res[k - 1][0] if k > 0 else None
        next_n = res[k + 1][1] - res[k + 1][0] if k + 1 < len(res) else None
        if n != typ and k > 0 and prev_n == typ and next_n == typ:
            seg = np.linspace(beats[a], beats[b], typ + 1)
            n = typ
        else:
            seg = beats[a:b + 1]
        pts += list(seg[1:])
        bar_n.append(n)
    pt = np.array(pts)
    onsets = sorted(onsets)
    # per-bar meter decision
    dens, i0 = [], 0
    for n in bar_n:
        dens.append(_density(pt, i0, i0 + n, onsets)); i0 += n
    bars, pulse_div, i0 = [], [], 0
    for bi, n in enumerate(bar_n):
        lo, hi = max(0, bi - 2), min(len(bar_n), bi + 3)
        tern = sum(d[0] for d in dens[lo:hi]); binc = sum(d[1] for d in dens[lo:hi])
        pbpm = 60.0 / float(np.median(np.diff(pt[i0:i0 + n + 1])))
        if n >= 5 and pbpm >= 100:                       # fast pulses -> the bar is counted in eighths (5/8, 7/8, 11/8, 13/8, ...)
            kind, pd, num, den = "eighth", 12, n, 8
        elif n in (2, 3) and tern > 1.6 * binc and tern + binc >= 6:
            kind, pd, num, den = "compound", 36, 3 * n, 8
        else:
            kind, pd, num, den = "simple", 24, n, 4
        bars.append(dict(i0=i0, i1=i0 + n, n=n, kind=kind, pd=pd, num=num, den=den, pickup=False))
        pulse_div += [pd] * n
        i0 += n
    if len(bars) > 1 and bars[0]["n"] < typ:
        bars[0]["pickup"] = True
        bars[0]["num"], bars[0]["den"] = bars[1]["num"], bars[1]["den"]
    tempos, last = [], None
    for b in bars:
        dts = np.diff(pt[b["i0"]:b["i1"] + 1])
        bpm = round(60.0 / float(np.median(dts)) * b["pd"] / DIV, 2)
        if last is None or abs(bpm - last) / last >= 0.04:
            tempos.append((int(sum(pulse_div[:b["i0"]])), bpm)); last = bpm
    return Timeline(pt, pulse_div, bars, tempos)


def timeline_from_anchor(tempo_map, timesig_map, t_end):
    """Bars/pulses/tempo straight from an anchor MIDI's tempo + time-signature maps (seconds, quarter BPM)."""
    segs, tick = [], 0.0
    for i, (t, bpm) in enumerate(tempo_map):
        if i > 0:
            pt_, pb = tempo_map[i - 1]
            tick += (t - pt_) * (pb / 60.0) * MIDI_PPQ
        segs.append((t, bpm, tick))

    def s2t(sec):
        for t, bpm, tk in reversed(segs):
            if sec >= t:
                return tk + (sec - t) * (bpm / 60.0) * MIDI_PPQ
        return 0.0

    def t2s(tk_):
        for t, bpm, tk in reversed(segs):
            if tk_ >= tk:
                return t + (tk_ - tk) / ((bpm / 60.0) * MIDI_PPQ)
        return 0.0

    ts = sorted((s2t(t), n, d) for t, n, d in timesig_map) or [(0, 4, 4)]
    end_tick = s2t(t_end)
    tick, pts, pdiv, bars, tempos = 0.0, [0.0], [], [], []
    while tick < end_tick:
        cur = next((x for x in reversed(ts) if x[0] <= tick + 1), ts[0])
        n, d = cur[1], cur[2]
        if d == 4:
            cnt, pd = n, 24
        elif d == 8:
            cnt, pd = (n // 3, 36) if (n % 3 == 0 and n >= 6) else (n, 12)
        elif d == 2:
            cnt, pd, n, d = 2 * n, 24, n, 2
        elif d == 16 and n % 2 == 0:
            cnt, pd = n // 2, 12
        else:
            cnt, pd, n, d = 4, 24, 4, 4
        i0 = len(pdiv)
        start_tick = tick
        for k in range(cnt):
            tick += pd * TICK_PER_DIV
            pts.append(t2s(tick)); pdiv.append(pd)
        bars.append(dict(i0=i0, i1=i0 + cnt, n=cnt, kind="anchor", pd=pd, num=n, den=d, pickup=False))
        b_bpm = round(next((bp for t, bp, tk in reversed(segs) if tk <= start_tick + 1), segs[0][1]), 2)
        if not tempos or abs(b_bpm - tempos[-1][1]) > 0.01:
            tempos.append((int(sum(pdiv[:i0])), b_bpm))
    return Timeline(pts, pdiv, bars, tempos)


# =============================================================================
# 2) Quantization (beat-synchronous: rubato disappears, swing -> triplets)
# =============================================================================
def choose_feels(tl, onset_secs):
    ds = np.sort(tl.t2d(np.asarray(onset_secs, float))) if len(onset_secs) else np.array([])
    tri = []
    for b in tl.bars:
        if b["pd"] != 24:
            tri.append(False); continue
        a, c = np.searchsorted(ds, b["d0"]), np.searchsorted(ds, b["d1"])
        rel = [(d - b["d0"]) % 24 for d in ds[a:c]]
        rel = [r for r in rel if 2 < r < 22]
        if len(rel) < 3:
            tri.append(False); continue
        eb = np.mean([abs(r - round(r / 6) * 6) for r in rel])
        et = np.mean([abs(r - round(r / 8) * 8) for r in rel])
        tri.append(bool(eb > 1.2 and et < 0.6 * eb))
    sm = [sum(tri[max(0, i - 1):i + 2]) >= 2 for i in range(len(tri))]
    tl.tri = sm
    return sm


def _snap(tl, d):
    i = tl.pulse_at(d)
    tri = tl.tri[tl.pbar[i]] and tl.pdiv[i] == 24
    g = 8 if tri else 6
    rel = d - tl.ppos[i]
    return int(tl.ppos[i] + round(rel / g) * g), g


def quantize_notes(notes, tl):
    """notes: (pitch, start_sec, end_sec, vel) -> dicts in divs, clean (no same-pitch overlaps)."""
    out = []
    for p, s, e, v in notes:
        ds, de = float(tl.t2d(s)), float(tl.t2d(e))
        qs, g = _snap(tl, ds)
        qe, _ = _snap(tl, de)
        qe = max(qe, qs + g)
        out.append(dict(p=int(p), s=qs, e=qe, v=int(v)))
    by = {}
    for n in sorted(out, key=lambda n: (n["p"], n["s"])):
        lst = by.setdefault(n["p"], [])
        if lst and n["s"] == lst[-1]["s"]:
            lst[-1]["e"] = max(lst[-1]["e"], n["e"]); lst[-1]["v"] = max(lst[-1]["v"], n["v"]); continue
        if lst and lst[-1]["e"] > n["s"]:
            lst[-1]["e"] = n["s"]
        if lst and lst[-1]["e"] <= lst[-1]["s"]:
            lst.pop()
        lst.append(n)
    return sorted([n for l in by.values() for n in l if n["e"] > n["s"]], key=lambda n: (n["s"], n["p"]))


def split_staves(notes, window=192, lo=56, hi=66):
    """Dynamic hand split on quantized notes (window in divs ~ +-8 beats)."""
    ns = sorted(notes, key=lambda n: n["s"])
    st = [n["s"] for n in ns]
    up, low = [], []
    for n in ns:
        a, b = bisect.bisect_left(st, n["s"] - window), bisect.bisect_right(st, n["s"] + window)
        ps = [m["p"] for m in ns[a:b]]
        split = float(np.clip(np.percentile(ps, 35), lo, hi)) if len(ps) >= 3 else 60
        (up if n["p"] >= split else low).append(n)
    return up, low


def assign_voices(notes, max_voices=4, mono=False):
    """One staff -> list of voices; each voice = list of objects dict(s,e,p=[...],v=[...])."""
    by_s = {}
    for n in notes:
        by_s.setdefault(n["s"], []).append(dict(n))
    objs = []
    for s, grp in by_s.items():
        grp.sort(key=lambda n: n["e"])
        cl_end, cluster = None, []
        for n in grp:
            if cl_end is not None and n["e"] - cl_end <= 6:
                cluster.append(n)
            else:
                if cluster:
                    m = max(x["e"] for x in cluster)
                    for x in cluster:
                        x["e"] = m
                cluster = [n]
            cl_end = n["e"]
        if cluster:
            m = max(x["e"] for x in cluster)
            for x in cluster:
                x["e"] = m
        groups = {}
        for n in grp:
            groups.setdefault(n["e"], []).append(n)
        for e, g in groups.items():
            objs.append(dict(s=s, e=e, p=sorted(x["p"] for x in g), v=[x["v"] for x in sorted(g, key=lambda x: x["p"])]))
    objs.sort(key=lambda o: (o["s"], -max(o["p"])))
    voices = [[] for _ in range(1 if mono else max_voices)]
    for o in objs:
        for vo in voices:
            if not vo or vo[-1]["e"] <= o["s"]:
                vo.append(o); break
        else:
            vo = voices[0]
            if vo[-1]["s"] == o["s"]:
                if not mono:
                    vo[-1]["p"] = sorted(set(vo[-1]["p"]) | set(o["p"]))
                    vo[-1]["v"] = (vo[-1]["v"] + o["v"])[:len(vo[-1]["p"])]
                    vo[-1]["e"] = min(vo[-1]["e"], o["e"])
            else:
                vo[-1]["e"] = o["s"]
                vo.append(o)
    for vi, vo in enumerate(voices):                    # tidy: close tiny gaps (legato), drop empty
        for a, b in zip(vo[:-1], vo[1:]):
            if 0 < b["s"] - a["e"] <= 6:
                a["e"] = b["s"]
    return [v for v in voices if v]


# =============================================================================
# 3) Keys, spelling, chords
# =============================================================================
_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _fifths(major_tonic_pc):
    f = (major_tonic_pc * 7) % 12
    return f - 12 if f > 6 else f


def key_per_bar(all_notes, tl):
    """(fifths, minor) per bar: 8-bar sliding histogram, hysteresis, single-bar blips removed."""
    nb = len(tl.bars)
    H = np.zeros((nb, 12))
    for n in all_notes:
        b = bisect.bisect_right(tl.bar_d0, n["s"]) - 1
        if 0 <= b < nb:
            H[b, n["p"] % 12] += (n["e"] - n["s"]) * n["v"]
    keys, prev, prev_c = [], None, 0
    for b in range(nb):
        h = H[max(0, b - 3):b + 5].sum(axis=0)
        if h.sum() <= 0 or np.std(h) == 0:
            keys.append(prev); continue
        best = (-2, 0, 0)
        sc = {}
        for t in range(12):
            for mi, prof in ((0, _MAJ), (1, _MIN)):
                c = float(np.corrcoef(h, np.roll(prof, t))[0, 1])
                sc[(t, mi)] = c
                if c > best[0]:
                    best = (c, t, mi)
        cur = (best[1], best[2])
        if prev is not None and sc[prev] >= best[0] - 0.04:
            cur = prev
        prev = cur
        keys.append(cur)
    first = next((k for k in keys if k), (0, 0))
    last = first
    for i in range(nb):
        if keys[i] is None:
            keys[i] = last
        else:
            last = keys[i]
    for i in range(1, nb - 1):
        if keys[i] != keys[i - 1] and keys[i] != keys[i + 1] and keys[i - 1] == keys[i + 1]:
            keys[i] = keys[i - 1]
    return [((_fifths(t if mi == 0 else (t + 3) % 12)), bool(mi), t) for t, mi in keys]


_LET, _BASE, _SO = "CDEFGAB", {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}, "FCGDAEB"
_SPELL_CACHE = {}


def _spell_table(f, minor):
    key = (f, minor)
    if key in _SPELL_CACHE:
        return _SPELL_CACHE[key]
    al = {l: 0 for l in _LET}
    if f > 0:
        for l in _SO[:f]:
            al[l] = 1
    elif f < 0:
        for l in _SO[::-1][:-f]:
            al[l] = -1
    pc = {}
    for l in _LET:
        pc.setdefault((_BASE[l] + al[l]) % 12, (l, al[l]))
    if minor:
        mt = ((f * 4) % 7 + 5) % 7
        lt = _LET[(mt - 1) % 7]
        pc.setdefault((_BASE[lt] + al[lt] + 1) % 12, (lt, al[lt] + 1))
    sharp = {1: ("C", 1), 3: ("D", 1), 6: ("F", 1), 8: ("G", 1), 10: ("A", 1)}
    flat = {1: ("D", -1), 3: ("E", -1), 6: ("G", -1), 8: ("A", -1), 10: ("B", -1)}
    white = {0: ("C", 0), 2: ("D", 0), 4: ("E", 0), 5: ("F", 0), 7: ("G", 0), 9: ("A", 0), 11: ("B", 0)}
    for p in range(12):
        if p not in pc:
            pc[p] = (sharp if f >= 0 else flat).get(p) or white[p]
    _SPELL_CACHE[key] = pc
    return pc


def spell(midi, f, minor):
    step, alter = _spell_table(f, minor)[midi % 12]
    octv = (midi - alter - _BASE[step]) // 12 - 1
    return step, alter, octv


_SHARP_N = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT_N = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
_QUAL = {  # name -> (intervals, symbol, musicxml kind)
    "maj": ([0, 4, 7], "", "major"), "min": ([0, 3, 7], "m", "minor"), "7": ([0, 4, 7, 10], "7", "dominant"),
    "maj7": ([0, 4, 7, 11], "maj7", "major-seventh"), "m7": ([0, 3, 7, 10], "m7", "minor-seventh"),
    "dim": ([0, 3, 6], "dim", "diminished"), "aug": ([0, 4, 8], "aug", "augmented"),
    "sus4": ([0, 5, 7], "sus4", "suspended-fourth"), "m7b5": ([0, 3, 6, 10], "m7b5", "half-diminished"),
}
_BONUS = {"maj": 0.05, "min": 0.05, "7": 0.0, "maj7": -0.01, "m7": 0.0, "dim": -0.03, "aug": -0.06, "sus4": -0.03, "m7b5": -0.02}


def detect_chords(mix_wav, bass_wav, tl):
    """Pulse-synchronous chroma template matching + Viterbi smoothing (+ slash chords from the bass stem).
    Returns [(div_pos, root_pc, quality, bass_pc|None)] (changes only)."""
    import librosa
    y, sr = librosa.load(mix_wav, sr=22050, mono=True)
    yh = librosa.effects.harmonic(y, margin=2)
    hop = 2048
    ch = librosa.feature.chroma_cqt(y=yh, sr=sr, hop_length=hop)
    bch = None
    if bass_wav and os.path.exists(bass_wav):
        yb, _ = librosa.load(bass_wav, sr=sr, mono=True)
        bch = librosa.feature.chroma_cqt(y=yb, sr=sr, hop_length=hop)
    ft = hop / sr
    P = len(tl.pdiv)
    C = np.zeros((P, 12)); B = np.zeros((P, 12))
    for i in range(P):
        a, b = int(tl.pt[i] / ft), max(int(tl.pt[i] / ft) + 1, int(tl.pt[i + 1] / ft))
        if a >= ch.shape[1]:
            break
        C[i] = np.median(ch[:, a:min(b, ch.shape[1])], axis=1)
        if bch is not None:
            B[i] = np.mean(bch[:, a:min(b, bch.shape[1])], axis=1)
    en = C.sum(axis=1)
    med = np.median(en[en > 0]) if (en > 0).any() else 1.0
    states = []
    for r in range(12):
        for q, (iv, _, _) in _QUAL.items():
            v = np.zeros(12)
            for k, x in enumerate(iv):
                v[(r + x) % 12] = 1.2 if k == 0 else (0.8 if k == 2 and len(iv) > 3 else 1.0)
            states.append((r, q, v / np.linalg.norm(v)))
    S = np.zeros((P, len(states) + 1))
    for i in range(P):
        c = C[i] / (np.linalg.norm(C[i]) + 1e-9)
        for k, (r, q, v) in enumerate(states):
            S[i, k] = float(c @ v) + _BONUS[q]
        S[i, -1] = 0.30 if en[i] < 0.25 * med else 0.0
    K = S.shape[1]
    cost, back = np.zeros((P, K)), np.zeros((P, K), int)
    cost[0] = -S[0]
    pen = 0.20
    for i in range(1, P):
        stay = cost[i - 1]
        best_prev = int(np.argmin(stay))
        for k in range(K):
            alt = stay[best_prev] + pen
            if stay[k] <= alt:
                back[i, k], c0 = k, stay[k]
            else:
                back[i, k], c0 = best_prev, alt
            cost[i, k] = c0 - S[i, k]
    path = [int(np.argmin(cost[-1]))]
    for i in range(P - 1, 0, -1):
        path.append(int(back[i, path[-1]]))
    path = path[::-1]
    out, last = [], None
    for i, k in enumerate(path):
        if k == K - 1:
            cur = None
        else:
            r, q, _ = states[k]
            bass = None
            if B[i].sum() > 0:
                bp = int(np.argmax(B[i]))
                if B[i][bp] / B[i].sum() > 0.45 and bp != r and (bp - r) % 12 in _QUAL[q][0]:
                    bass = bp
            cur = (r, q, bass)
        if cur != last:
            out.append((int(tl.ppos[i]), *(cur if cur else (None, None, None))))
            last = cur
    return [c for c in out if c[1] is not None]


# =============================================================================
# 4) Drums (basic: onset + band-energy classification)
# =============================================================================
def transcribe_drums(wav_path):
    import librosa
    y, sr = librosa.load(wav_path, sr=22050, mono=True)
    on = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=False, delta=0.12)
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    fr = librosa.fft_frequencies(sr=sr, n_fft=1024)
    peak = float(S.max()) or 1.0
    notes = []
    for t in on:
        f0 = int(t * sr / 256)
        seg = S[:, f0:f0 + 3]
        if seg.size == 0:
            continue
        sp = seg.mean(axis=1)
        tot = sp.sum() + 1e-9
        low, high = sp[fr < 150].sum() / tot, sp[fr > 5000].sum() / tot
        cen = float((sp * fr).sum() / tot)
        if low > 0.45:
            p = 36
        elif high > 0.35 and cen > 4500:
            p = 42
        else:
            p = 38
        v = int(np.clip(40 + 87 * float(seg.max()) / peak, 30, 120))
        notes.append((p, float(t), float(t) + 0.08, v))
    return notes


# =============================================================================
# 5) MusicXML + MIDI writers
# =============================================================================
NORM = {96: ("whole", 0), 72: ("half", 1), 48: ("half", 0), 36: ("quarter", 1), 24: ("quarter", 0),
        18: ("eighth", 1), 12: ("eighth", 0), 6: ("16th", 0)}
TRI = {16: "quarter", 8: "eighth"}
_CANDS = [(96, 0), (72, 0), (48, 0), (36, 0), (24, 0), (18, 0), (16, 1), (12, 0), (8, 1), (6, 0)]


def _ok(L, pos, pd, barlen, tri):
    if tri:
        return pd == 24 and pos % 8 == 0 and pos % 24 + L <= 24
    if L <= pd:
        return pos % 6 == 0 and pos % pd + L <= pd
    if pd == 24 and L in (36, 48):
        return pos % 48 == 0 or (barlen < 96 and pos % 24 == 0)
    if L % pd:
        return False
    return pos % pd == 0 and (L < barlen or pos == 0)


def split_dur(start, dur, pd, barlen, tri_ok, lead=0):
    out = []
    while dur > 0:
        for L, tri in _CANDS:
            if tri and not tri_ok:
                continue
            if tri_ok and not tri and (start + lead + L) % 8:
                continue                      # triplet bars: stay on the 1/3-beat grid
            if L <= dur and _ok(L, start + lead, pd, barlen + lead, tri):
                break
        else:
            L, tri = min(dur, 6), 0
        out.append((start, L, tri))
        start += L; dur -= L
    return out


def _note_xml(pitchxml, L, tri, voice, staff, chord=False, tie=(False, False), rest=False, hidden=False,
              tuplet=None, unpitched=None):
    s = ["<note%s>" % (' print-object="no"' if hidden else "")]
    if chord:
        s.append("<chord/>")
    if rest:
        s.append("<rest/>")
    elif unpitched:
        s.append(unpitched)
    else:
        s.append(pitchxml)
    s.append("<duration>%d</duration>" % L)
    if tie[1]:
        s.append('<tie type="stop"/>')
    if tie[0]:
        s.append('<tie type="start"/>')
    s.append("<voice>%d</voice>" % voice)
    if tri:
        s.append("<type>%s</type>" % TRI[L])
        s.append("<time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification>")
    else:
        t, dots = NORM[L]
        s.append("<type>%s</type>" % t + "<dot/>" * dots)
    s.append("<staff>%d</staff>" % staff)
    nt = []
    if tie[1]:
        nt.append('<tied type="stop"/>')
    if tie[0]:
        nt.append('<tied type="start"/>')
    if tuplet:
        nt.append('<tuplet type="%s" bracket="yes"/>' % tuplet)
    if nt:
        s.append("<notations>%s</notations>" % "".join(nt))
    s.append("</note>")
    return "".join(s)


def _pitch_xml(midi, f, minor):
    st, al, oc = spell(midi, f, minor)
    return "<pitch><step>%s</step>%s<octave>%d</octave></pitch>" % (st, "<alter>%d</alter>" % al if al else "", oc)


def _harmony_xml(root, qual, bass, f, offset):
    names = _FLAT_N if f < 0 else _SHARP_N

    def sa(pc):
        n = names[pc]
        return n[0], (1 if "#" in n else (-1 if "b" in n else 0))

    rs, ra = sa(root)
    x = "<harmony><root><root-step>%s</root-step>%s</root><kind text=\"%s\">%s</kind>" % (
        rs, "<root-alter>%d</root-alter>" % ra if ra else "", _QUAL[qual][1], _QUAL[qual][2])
    if bass is not None:
        bs, ba = sa(bass)
        x += "<bass><bass-step>%s</bass-step>%s</bass>" % (bs, "<bass-alter>%d</bass-alter>" % ba if ba else "")
    if offset:
        x += "<offset>%d</offset>" % offset
    return x + "</harmony>"


def _clef_xml(c, staff=None):
    sign, line, octc = c
    x = "<clef%s><sign>%s</sign>" % (' number="%d"' % staff if staff else "", sign)
    if sign != "percussion":
        x += "<line>%d</line>" % line
    if octc:
        x += "<clef-octave-change>%d</clef-octave-change>" % octc
    return x + "</clef>"


def _build_part_objs(kind, notes_div):
    cfg = KINDS[kind]
    if kind == "drums":
        return {1: assign_voices(notes_div, 1, mono=False)}
    if cfg["staves"] == 2:
        up, low = split_staves(notes_div)
        return {1: assign_voices(up, 4, cfg["mono"]), 2: assign_voices(low, 4, cfg["mono"])}
    return {1: assign_voices(notes_div, 4, cfg["mono"])}


def _pieces_by_bar(objs, tl):
    res = {}
    for o in objs:
        b0 = max(0, bisect.bisect_right(tl.bar_d0, o["s"]) - 1)
        for bi in range(b0, len(tl.bars)):
            bar = tl.bars[bi]
            if bar["d0"] >= o["e"]:
                break
            s, e = max(o["s"], bar["d0"]), min(o["e"], bar["d1"])
            if e > s:
                res.setdefault(bi, []).append((s - bar["d0"], e - bar["d0"], o["p"], s > o["s"], e < o["e"]))
    return res


def write_musicxml(path, parts, tl, chords, keys, title, written=False):
    """parts: list of dict(kind, objs={staff:[voices]}, notes_div=[...])."""
    X = ['<?xml version="1.0" encoding="UTF-8"?>',
         '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">',
         '<score-partwise version="3.1"><work><work-title>%s</work-title></work>' % escape(title),
         '<identification><encoding><software>Audio2Score (Colab)</software></encoding></identification><part-list>']
    for pi, p in enumerate(parts):
        cfg = KINDS[p["kind"]]
        pid = "P%d" % (pi + 1)
        X.append('<score-part id="%s"><part-name>%s</part-name>' % (pid, escape(cfg["name"])))
        if p["kind"] == "drums":
            for mp, (nm, _, _, _) in DRUM_INFO.items():
                X.append('<score-instrument id="%s-I%d"><instrument-name>%s</instrument-name></score-instrument>' % (pid, mp, nm))
            for mp in DRUM_INFO:
                X.append('<midi-instrument id="%s-I%d"><midi-channel>10</midi-channel><midi-unpitched>%d</midi-unpitched></midi-instrument>' % (pid, mp, mp + 1))
        else:
            X.append('<score-instrument id="%s-I1"><instrument-name>%s</instrument-name></score-instrument>' % (pid, escape(cfg["name"])))
            X.append('<midi-instrument id="%s-I1"><midi-channel>%d</midi-channel><midi-program>%d</midi-program></midi-instrument>' % (pid, min(pi + 1, 16) if pi < 9 else pi + 2, cfg["prog"] + 1))
        X.append("</score-part>")
    X.append("</part-list>")
    tempo_at = dict(tl.tempos)
    ch_by_bar = {}
    for pos, r, q, bs in chords:
        b = max(0, bisect.bisect_right(tl.bar_d0, pos) - 1)
        ch_by_bar.setdefault(b, []).append((max(0, pos - tl.bars[b]["d0"]), r, q, bs))
    for pi, p in enumerate(parts):
        kind, cfg = p["kind"], KINDS[p["kind"]]
        pid = "P%d" % (pi + 1)
        tr = cfg.get("tr") if written else None
        pcs = {st: [_pieces_by_bar(v, tl) for v in voices] for st, voices in p["objs"].items()}
        X.append('<part id="%s">' % pid)
        last_key, last_ts = None, None
        for bi, bar in enumerate(tl.bars):
            blen = bar["d1"] - bar["d0"]
            lead = bar.get("lead", 0)
            f, minor, tonic = keys[bi]
            if tr:
                f = _fifths(((tonic if not minor else (tonic + 3)) - tr[0] - 12 * tr[2]) % 12)
            X.append('<measure number="%d"%s>' % (bi if tl.bars[0]["pickup"] else bi + 1,
                                                  ' implicit="yes"' if bar["pickup"] else ""))
            attrs = []
            if bi == 0:
                attrs.append("<divisions>%d</divisions>" % DIV)
            if (f, minor) != last_key:
                attrs.append("<key><fifths>%d</fifths><mode>%s</mode></key>" % (f, "minor" if minor else "major"))
                last_key = (f, minor)
            if (bar["num"], bar["den"]) != last_ts:
                attrs.append("<time><beats>%d</beats><beat-type>%d</beat-type></time>" % (bar["num"], bar["den"]))
                last_ts = (bar["num"], bar["den"])
            if bi == 0:
                if cfg["staves"] > 1:
                    attrs.append("<staves>%d</staves>" % cfg["staves"])
                for si, c in enumerate(cfg["clefs"]):
                    attrs.append(_clef_xml(c, si + 1 if cfg["staves"] > 1 else None))
                if tr:
                    attrs.append("<transpose><diatonic>%d</diatonic><chromatic>%d</chromatic>%s</transpose>" % (
                        tr[1], tr[0], "<octave-change>%d</octave-change>" % tr[2] if tr[2] else ""))
            if attrs:
                X.append("<attributes>%s</attributes>" % "".join(attrs))
            if bar["d0o"] in tempo_at and pi == 0:
                bpm = tempo_at[bar["d0o"]]
                X.append('<direction placement="above"><direction-type><metronome><beat-unit>quarter</beat-unit>'
                         '<per-minute>%d</per-minute></metronome></direction-type><sound tempo="%.2f"/></direction>' % (round(bpm), bpm))
            if pi == 0:
                for off, r, q, bs in ch_by_bar.get(bi, []):
                    X.append(_harmony_xml(r, q, bs, f, off))
            tri_ok = tl.tri[bi]
            first_stream = True
            for st in sorted(p["objs"]):
                for vi, pc in enumerate(pcs[st]):
                    voice_no = vi + 1 + (st - 1) * 4
                    pieces = sorted(pc.get(bi, []), key=lambda x: x[0])
                    if not pieces and vi > 0:
                        continue
                    if not first_stream:
                        X.append("<backup><duration>%d</duration></backup>" % blen)
                    first_stream = False
                    if not pieces:
                        X.append('<note><rest measure="yes"/><duration>%d</duration><voice>%d</voice><staff>%d</staff></note>' % (blen, voice_no, st))
                        continue
                    cur, acc = 0, 0
                    seq = []
                    for (s, e, pits, t_start, t_stop) in pieces:
                        if s < cur:
                            s = cur
                        if e <= s:
                            continue
                        if s > cur:
                            for (ps, L, tri) in split_dur(cur, s - cur, bar["pd"], blen, tri_ok, lead):
                                seq.append(("rest", L, tri, None, (False, False)))
                        sp = split_dur(s, e - s, bar["pd"], blen, tri_ok, lead)
                        for k, (ps, L, tri) in enumerate(sp):
                            tie_stop = (k > 0) or t_start
                            tie_start = (k < len(sp) - 1) or t_stop
                            seq.append(("note", L, tri, pits, (tie_start, tie_stop)))
                        cur = e
                    if cur < blen:
                        for (ps, L, tri) in split_dur(cur, blen - cur, bar["pd"], blen, tri_ok, lead):
                            seq.append(("rest", L, tri, None, (False, False)))
                    for kind_, L, tri, pits, tie in seq:
                        tup = None
                        if tri:
                            if acc == 0:
                                tup = "start"
                            acc += L
                            if acc >= 24:
                                tup = "stop" if tup is None else None
                                acc = 0
                        if kind_ == "rest":
                            X.append(_note_xml("", L, tri, voice_no, st, rest=True, hidden=(vi > 0), tuplet=tup))
                        elif kind == "drums":
                            for k, pp in enumerate(pits):
                                nm, dstep, doct, nh = DRUM_INFO.get(pp, DRUM_INFO[38])
                                unp = "<unpitched><display-step>%s</display-step><display-octave>%d</display-octave></unpitched>" % (dstep, doct)
                                x = _note_xml("", L, tri, voice_no, st, chord=k > 0, tuplet=tup if k == 0 else None, unpitched=unp)
                                x = x.replace("<voice>", '<instrument id="%s-I%d"/><voice>' % (pid, pp if pp in DRUM_INFO else 38), 1)
                                if nh:
                                    x = x.replace("<staff>", "<notehead>%s</notehead><staff>" % nh, 1)
                                X.append(x)
                        else:
                            for k, pp in enumerate(pits):
                                wp = pp - tr[0] - 12 * tr[2] if tr else pp
                                X.append(_note_xml(_pitch_xml(wp, f, minor), L, tri, voice_no, st, chord=k > 0,
                                                   tie=tie, tuplet=tup if k == 0 else None))
            X.append("</measure>")
        X.append("</part>")
    X.append("</score-partwise>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(X))


def _vlq(n):
    b = [n & 0x7F]
    n >>= 7
    while n:
        b.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(b[::-1])


def _track(events, name):
    events.sort(key=lambda e: (e[0], e[1]))
    data, last = bytearray(), 0
    nb = name.encode("utf-8")
    data += b"\x00\xff\x03" + _vlq(len(nb)) + nb
    for tick, _, msg in events:
        data += _vlq(max(0, tick - last)) + msg
        last = max(last, tick)
    data += b"\x00\xff\x2f\x00"
    return b"MTrk" + struct.pack(">I", len(data)) + bytes(data)


def write_midi(path, parts, tl, chords, pedal_by_part=None, names=None):
    trk = []
    meta = []
    for d, bpm in tl.tempos:
        us = int(round(60e6 / bpm))
        meta.append((d * TICK_PER_DIV, 0, b"\xff\x51\x03" + us.to_bytes(3, "big")))
    last = None
    for bar in tl.bars:
        if (bar["num"], bar["den"]) != last:
            dd = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}.get(bar["den"], 2)
            meta.append((bar["d0"] * TICK_PER_DIV, 0, b"\xff\x58\x04" + bytes([bar["num"], dd, 24, 8])))
            last = (bar["num"], bar["den"])
    names_ = _SHARP_N
    for pos, r, q, bs in chords:
        txt = (names_[r] + _QUAL[q][1] + ("/" + names_[bs] if bs is not None else "")).encode("utf-8")
        meta.append((pos * TICK_PER_DIV, 1, b"\xff\x06" + _vlq(len(txt)) + txt))
    trk.append(_track(meta, "Tempo / Chords"))
    ch = 0
    for pi, p in enumerate(parts):
        cfg = KINDS[p["kind"]]
        is_dr = p["kind"] == "drums"
        for st in sorted(p["objs"]):
            channel = 9 if is_dr else (ch if ch < 9 else ch + 1)
            ch += 0 if is_dr else 1
            ev = []
            if not is_dr:
                ev.append((0, 0, bytes([0xC0 | channel, cfg["prog"]])))
            for vo in p["objs"][st]:
                for o in vo:
                    for pp, vv in zip(o["p"], o["v"]):
                        ev.append((o["s"] * TICK_PER_DIV, 3, bytes([0x90 | channel, pp, int(np.clip(vv, 1, 127))])))
                        ev.append((o["e"] * TICK_PER_DIV, 1, bytes([0x80 | channel, pp, 0])))
            if st == 1 and pedal_by_part and p["kind"] == "piano":
                for t, val in pedal_by_part:
                    ev.append((int(round(float(tl.t2d(t)) * TICK_PER_DIV)), 2, bytes([0xB0 | channel, 64, 127 if val else 0])))
            label = cfg["name"] + (" RH" if (st == 1 and cfg["staves"] == 2) else " LH" if st == 2 else "")
            trk.append(_track(ev, label))
    with open(path, "wb") as fh:
        fh.write(b"MThd" + struct.pack(">IHHH", 6, 1, len(trk), MIDI_PPQ) + b"".join(trk))


def write_raw_midi(path, raw_parts, chords_sec=None):
    """Un-quantized MIDI exactly as transcribed (real timing, durations, velocities, pedal) at a fixed 120 bpm
    (1 sec = 960 ticks), so it lines up with the original audio in any DAW."""
    sec = lambda t: int(round(max(t, 0.0) * 960))
    meta = [(0, 0, b"\xff\x51\x03" + (500000).to_bytes(3, "big")), (0, 0, b"\xff\x58\x04" + bytes([4, 2, 24, 8]))]
    for t, r, q, bs in (chords_sec or []):
        txt = (_SHARP_N[r] + _QUAL[q][1] + ("/" + _SHARP_N[bs] if bs is not None else "")).encode("utf-8")
        meta.append((sec(t), 1, b"\xff\x06" + _vlq(len(txt)) + txt))
    trk, ch = [_track(meta, "Tempo / Chords")], 0
    for p in raw_parts:
        cfg = KINDS[p["kind"]]
        is_dr = p["kind"] == "drums"
        channel = 9 if is_dr else (ch if ch < 9 else ch + 1)
        ch += 0 if is_dr else 1
        ev = [] if is_dr else [(0, 0, bytes([0xC0 | channel, cfg["prog"]]))]
        for pp, s, e, v in p["notes"]:
            ev.append((sec(s), 3, bytes([0x90 | channel, int(pp), int(np.clip(v, 1, 127))])))
            ev.append((max(sec(e), sec(s) + 1), 1, bytes([0x80 | channel, int(pp), 0])))
        for t, val in (p.get("pedal") or []):
            ev.append((sec(t), 2, bytes([0xB0 | channel, 64, 127 if val else 0])))
        trk.append(_track(ev, cfg["name"]))
    with open(path, "wb") as fh:
        fh.write(b"MThd" + struct.pack(">IHHH", 6, 1, len(trk), MIDI_PPQ) + b"".join(trk))


# =============================================================================
# 6) Orchestrator
# =============================================================================
def engrave(parts_in, tl, chords, out_base, title="Transcription", written_pitch=False, key_override=None):
    """parts_in: [dict(kind, notes=[(p,s_sec,e_sec,v)], pedal=[(t,val)])] ; tl already built.
    Returns {'midi': path, 'xml': path}."""
    onsets = [n[1] for p in parts_in if p["kind"] != "drums" for n in p["notes"]]
    choose_feels(tl, onsets)
    parts, all_notes, pedal, qs = [], [], None, []
    for p in parts_in:
        nd = quantize_notes(p["notes"], tl)
        if p["kind"] == "drums":                                  # fixed short hits
            for n in nd:
                n["e"] = n["s"] + (8 if tl.tri[tl.pbar[tl.pulse_at(n["s"])]] else 6)
        if nd:
            qs.append((p, nd))
    if qs and tl.bars[0]["pickup"]:                               # pickup: start the bar at the first note
        s0 = min(n["s"] for _, nd in qs for n in nd)
        b0 = tl.bars[0]
        lead = ((s0 - b0["d0"]) // 6) * 6
        if lead >= 6 and s0 < b0["d1"]:
            b0["d0"] += lead; b0["lead"] = lead
            tl.bar_d0[0] = b0["d0"]
    for p, nd in qs:
        objs = _build_part_objs(p["kind"], nd)
        parts.append(dict(kind=p["kind"], objs=objs))
        if p["kind"] != "drums":
            all_notes += nd
        if p.get("pedal"):
            pedal = p["pedal"]
    if not parts:
        raise RuntimeError("no notes to engrave")
    if key_override:
        keys = [tuple(key_override)] * len(tl.bars)
    else:
        keys = key_per_bar(all_notes, tl) if all_notes else [(0, False, 0)] * len(tl.bars)
    xml_path, mid_path = out_base + ".musicxml", out_base + ".mid"
    write_musicxml(xml_path, parts, tl, chords or [], keys, title, written_pitch)
    write_midi(mid_path, parts, tl, chords or [], pedal)
    return {"midi": mid_path, "xml": xml_path}
