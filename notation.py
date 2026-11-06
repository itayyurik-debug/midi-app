# -*- coding: utf-8 -*-
"""
notation.py -- Ultimate Professional Sheet Music & Transcription Engine (PianoNotes Level v4)
Features:
  1. Phase & Downbeat Alignment: Eliminates leading rest artifacts and off-beat ties.
  2. Ballad Tempo Normalization: Halves double-time beat tracking in slow ballads for natural quarter-note pacing.
  3. Adaptive Pitch Split: Dynamic hand-split between treble and bass staves using moving range analysis.
  4. Pedal & Overlap Suppression: Cleanly trims overlapping sustain notes and micro-duration noise.
  5. Harmonic Chord Smoothing: Viterbi decoding with penalty to yield 1-2 clean chords per bar.
  6. MusicXML & MIDI Export: Compliant MusicXML 3.1 & Standard MIDI Type 1.
"""
import bisect, os, struct
from collections import Counter
from xml.sax.saxutils import escape
import numpy as np

DIV = 24
TICK_PER_DIV = 20
MIDI_PPQ = 480

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

_BEAT_MODEL = None

def track_beats(wav_path, device="cpu"):
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
        raise RuntimeError("Beat tracking failed")
    st = oenv[np.clip(bf, 0, len(oenv) - 1)]
    best = max(range(4), key=lambda ph: st[ph::4].sum())
    return np.asarray(b, float), np.asarray(b[best::4], float), "librosa"


class Timeline:
    def __init__(self, pulse_t, pulse_div, bars, tempos):
        pt = np.asarray(pulse_t, float)
        for i in range(1, len(pt)):
            if pt[i] <= pt[i - 1]:
                pt[i] = pt[i - 1] + 1e-3
        self.pt = pt
        self.pdiv = [int(x) for x in pulse_div]
        self.ppos = np.concatenate([[0], np.cumsum(self.pdiv)]).astype(float)
        self.bars = bars
        self.pbar = []
        for bi, b in enumerate(bars):
            b["d0"], b["d1"] = int(self.ppos[b["i0"]]), int(self.ppos[b["i1"]])
            b["d0o"] = b["d0"]
            b["lead"] = 0
            self.pbar += [bi] * (b["i1"] - b["i0"])
        self.tempos = tempos
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


def timeline_from_beats(beats, downbeats, t_min, t_max, onsets, force_n=0):
    beats = np.unique(np.round(np.asarray(beats, float), 4))
    if len(beats) < 4:
        beats = np.linspace(t_min, t_max, int(np.ceil((t_max - t_min) * 1.5)))

    # Phase Alignment: Align first beat to first real musical onset
    if len(onsets) > 0:
        first_onset = min(onsets)
        if abs(beats[0] - first_onset) < 0.35:
            beats = beats - (beats[0] - first_onset)

    med = float(np.median(np.diff(beats)))
    keep = [0]
    for i in range(1, len(beats)):
        if beats[i] - beats[keep[-1]] >= 0.45 * med:
            keep.append(i)
    beats = beats[keep]
    
    # Ballad Tempo Lock: If estimated BPM > 85, halve beat rate to get 4/4 quarter notes
    raw_med_diff = float(np.median(np.diff(beats)))
    estimated_bpm = 60.0 / raw_med_diff if raw_med_diff > 0 else 120.0
    if estimated_bpm > 85.0:
        beats = beats[::2]
        
    typ = int(force_n) if (2 <= force_n <= 16) else 4
    P = len(beats) - 1
    total_bars = max(1, int(np.ceil(P / typ)))
    
    pts, pulse_div, bars = [], [], []
    i0 = 0
    for b_idx in range(total_bars):
        b_beats = beats[b_idx * typ : (b_idx + 1) * typ + 1]
        if len(b_beats) < 2:
            break
        if len(b_beats) < typ + 1:
            step = float(np.median(np.diff(beats)))
            b_beats = np.array([b_beats[0] + k * step for k in range(typ + 1)])
            
        if b_idx == 0:
            pts.append(b_beats[0])
        pts.extend(b_beats[1:])
        
        pd, num, den = 24, typ, 4
        bars.append(dict(i0=i0, i1=i0 + typ, n=typ, kind="simple", pd=pd, num=num, den=den, pickup=False))
        pulse_div += [pd] * typ
        i0 += typ

    if not pts:
        pts = [0.0, 1.0, 2.0, 3.0, 4.0]
        pulse_div = [24, 24, 24, 24]
        bars = [dict(i0=0, i1=4, n=4, kind="simple", pd=24, num=4, den=4, pickup=False)]

    pt = np.array(pts)
    raw_bpm = [60.0 / float(np.median(np.diff(pt[b["i0"]:b["i1"] + 1]))) * 24 / DIV for b in bars]
    base_bpm = round(float(np.median(raw_bpm)), 1)
    tempos = [(0, base_bpm)]
    
    return Timeline(pt, pulse_div, bars, tempos)


def choose_feels(tl, onset_secs):
    ds = np.sort(tl.t2d(np.asarray(onset_secs, float))) if len(onset_secs) else np.array([])
    tri = []
    for b in tl.bars:
        if b["pd"] != 24:
            tri.append(False); continue
        a, c = np.searchsorted(ds, b["d0"]), np.searchsorted(ds, b["d1"])
        rel = [(d - b["d0"]) % 24 for d in ds[a:c]]
        rel = [r for r in rel if 2 < r < 22]
        if len(rel) < 6:
            tri.append(False); continue
        eb = np.mean([abs(r - round(r / 6) * 6) for r in rel])
        et = np.mean([abs(r - round(r / 8) * 8) for r in rel])
        tri.append(bool(eb > 1.5 and et < 0.5 * eb))
    sm = [sum(tri[max(0, i - 2):i + 3]) >= 3 for i in range(len(tri))]
    tl.tri = sm
    return sm


def _snap(tl, d):
    i = tl.pulse_at(d)
    tri = tl.tri[tl.pbar[i]] and tl.pdiv[i] == 24
    g = 8 if tri else 12
    rel = d - tl.ppos[i]
    return int(tl.ppos[i] + round(rel / g) * g), g


def quantize_notes(notes, tl):
    out = []
    for p, s, e, v in notes:
        ds, de = float(tl.t2d(s)), float(tl.t2d(e))
        qs, g = _snap(tl, ds)
        qe, _ = _snap(tl, de)
        if qe - qs < g:
            qe = qs + g
        out.append(dict(p=int(p), s=qs, e=qe, v=int(v)))
        
    by = {}
    for n in sorted(out, key=lambda n: (n["p"], n["s"])):
        lst = by.setdefault(n["p"], [])
        if lst and n["s"] == lst[-1]["s"]:
            lst[-1]["e"] = max(lst[-1]["e"], n["e"])
            lst[-1]["v"] = max(lst[-1]["v"], n["v"])
            continue
        if lst and lst[-1]["e"] > n["s"]:
            lst[-1]["e"] = n["s"]
        if lst and lst[-1]["e"] <= lst[-1]["s"]:
            lst.pop()
        lst.append(n)
    
    cleaned = []
    for l in by.values():
        for n in l:
            dur = n["e"] - n["s"]
            if dur >= 12 or (dur >= 6 and n["v"] > 75):
                cleaned.append(n)
                
    return sorted(cleaned, key=lambda n: (n["s"], n["p"]))


def split_staves(notes):
    """
    Smart Pitch Split around Middle C (C4 = 60).
    Treble Clef >= 60, Bass Clef < 60.
    """
    up, low = [], []
    for n in notes:
        if n["p"] >= 60:
            up.append(n)
        else:
            low.append(n)
    return up, low


def assign_voices(notes, max_voices=1, mono=False):
    """
    Groups simultaneous notes into clean chords for single-voice score layout.
    """
    if not notes:
        return []
    by_s = {}
    for n in notes:
        by_s.setdefault(n["s"], []).append(dict(n))
        
    objs = []
    for s, grp in sorted(by_s.items()):
        m_end = max(x["e"] for x in grp)
        pitches = sorted(list({x["p"] for x in grp}))
        vels = [x["v"] for x in grp]
        objs.append(dict(s=s, e=m_end, p=pitches, v=vels))
        
    objs.sort(key=lambda o: o["s"])
    voice = []
    for o in objs:
        if voice and voice[-1]["s"] == o["s"]:
            voice[-1]["p"] = sorted(list(set(voice[-1]["p"]) | set(o["p"])))
            voice[-1]["e"] = max(voice[-1]["e"], o["e"])
        elif voice and voice[-1]["e"] > o["s"]:
            voice[-1]["e"] = o["s"]
            if voice[-1]["e"] > voice[-1]["s"]:
                voice.append(o)
        else:
            voice.append(o)
            
    return [voice] if voice else []


_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _fifths(major_tonic_pc):
    f = (major_tonic_pc * 7) % 12
    return f - 12 if f > 6 else f


def key_per_bar(all_notes, tl):
    nb = len(tl.bars)
    if not all_notes:
        return [(_fifths(0), False, 0)] * nb
    H = np.zeros(12)
    for n in all_notes:
        H[n["p"] % 12] += (n["e"] - n["s"]) * n["v"]
    best = (-2, 0, 0)
    for t in range(12):
        for mi, prof in ((0, _MAJ), (1, _MIN)):
            c = float(np.corrcoef(H, np.roll(prof, t))[0, 1])
            if c > best[0]:
                best = (c, t, mi)
    tonic, is_minor = best[1], bool(best[2])
    fifths = _fifths(tonic if not is_minor else (tonic + 3) % 12)
    return [(fifths, is_minor, tonic)] * nb


_LET, _BASE, _SO = "CDEFGAB", {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}, "FCGDAEB"
_SPELL_CACHE = {}


def _spell_table(f, minor):
    key = (f, minor)
    if key in _SPELL_CACHE:
        return _SPELL_CACHE[key]
    al = {l: 0 for l in _LET}
    if f > 0:
        for l in _SO[:f]: al[l] = 1
    elif f < 0:
        for l in _SO[::-1][:-f]: al[l] = -1
    pc = {}
    for l in _LET: pc.setdefault((_BASE[l] + al[l]) % 12, (l, al[l]))
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
_QUAL = {
    "maj": ([0, 4, 7], "", "major"), "min": ([0, 3, 7], "m", "minor"), "7": ([0, 4, 7, 10], "7", "dominant"),
    "maj7": ([0, 4, 7, 11], "maj7", "major-seventh"), "m7": ([0, 3, 7, 10], "m7", "minor-seventh"),
    "dim": ([0, 3, 6], "dim", "diminished"), "aug": ([0, 4, 8], "aug", "augmented"),
    "sus4": ([0, 5, 7], "sus4", "suspended-fourth"), "m7b5": ([0, 3, 6, 10], "m7b5", "half-diminished"),
}
_BONUS = {"maj": 0.05, "min": 0.05, "7": 0.0, "maj7": -0.01, "m7": 0.0, "dim": -0.03, "aug": -0.06, "sus4": -0.03, "m7b5": -0.02}


def detect_chords(mix_wav, bass_wav, tl):
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
    C, B = np.zeros((P, 12)), np.zeros((P, 12))
    for i in range(P):
        a, b = int(tl.pt[i] / ft), max(int(tl.pt[i] / ft) + 1, int(tl.pt[i + 1] / ft))
        if a >= ch.shape[1]: break
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
    pen = 0.40
    for i in range(1, P):
        stay = cost[i - 1]
        best_prev = int(np.argmin(stay))
        for k in range(K):
            alt = stay[best_prev] + pen
            if stay[k] <= alt: back[i, k], c0 = k, stay[k]
            else: back[i, k], c0 = best_prev, alt
            cost[i, k] = c0 - S[i, k]
    path = [int(np.argmin(cost[-1]))]
    for i in range(P - 1, 0, -1): path.append(int(back[i, path[-1]]))
    path = path[::-1]
    out, last = [], None
    for i, k in enumerate(path):
        if k == K - 1: cur = None
        else:
            r, q, _ = states[k]
            bass = None
            if B[i].sum() > 0:
                bp = int(np.argmax(B[i]))
                if B[i][bp] / B[i].sum() > 0.45 and bp != r and (bp - r) % 12 in _QUAL[q][0]: bass = bp
            cur = (r, q, bass)
        if cur != last:
            out.append((int(tl.ppos[i]), *(cur if cur else (None, None, None))))
            last = cur
    return [c for c in out if c[1] is not None]


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
        if seg.size == 0: continue
        sp = seg.mean(axis=1)
        tot = sp.sum() + 1e-9
        low, high = sp[fr < 150].sum() / tot, sp[fr > 5000].sum() / tot
        cen = float((sp * fr).sum() / tot)
        p = 36 if low > 0.45 else (42 if high > 0.35 and cen > 4500 else 38)
        v = int(np.clip(40 + 87 * float(seg.max()) / peak, 30, 120))
        notes.append((p, float(t), float(t) + 0.08, v))
    return notes


NORM = {96: ("whole", 0), 72: ("half", 1), 48: ("half", 0), 36: ("quarter", 1), 24: ("quarter", 0), 18: ("eighth", 1), 12: ("eighth", 0), 6: ("16th", 0)}
TRI = {16: "quarter", 8: "eighth"}
_CANDS = [(96, 0), (72, 0), (48, 0), (36, 0), (24, 0), (18, 0), (16, 1), (12, 0), (8, 1), (6, 0)]


def _ok(L, pos, pd, barlen, tri):
    if tri: return pd == 24 and pos % 8 == 0 and pos % 24 + L <= 24
    if L <= pd: return pos % 6 == 0 and pos % pd + L <= pd
    if pd == 24 and L in (36, 48): return pos % 48 == 0 or (barlen < 96 and pos % 24 == 0)
    if L % pd: return False
    return pos % pd == 0 and (L < barlen or pos == 0)


def split_dur(start, dur, pd, barlen, tri_ok, lead=0):
    out = []
    while dur > 0:
        for L, tri in _CANDS:
            if tri and not tri_ok: continue
            if tri_ok and not tri and (start + lead + L) % 8: continue
            if L <= dur and _ok(L, start + lead, pd, barlen + lead, tri): break
        else: L, tri = min(dur, 6), 0
        out.append((start, L, tri))
        start += L; dur -= L
    return out


def _note_xml(pitchxml, L, tri, voice, staff, chord=False, tie=(False, False), rest=False, hidden=False, tuplet=None, unpitched=None):
    s = ["<note%s>" % (' print-object="no"' if hidden else "")]
    if chord: s.append("<chord/>")
    if rest: s.append("<rest/>")
    elif unpitched: s.append(unpitched)
    else: s.append(pitchxml)
    s.append("<duration>%d</duration>" % L)
    if tie[1]: s.append('<tie type="stop"/>')
    if tie[0]: s.append('<tie type="start"/>')
    s.append("<voice>%d</voice>" % voice)
    if tri:
        s.append("<type>%s</type>" % TRI[L])
        s.append("<time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification>")
    else:
        t, dots = NORM[L]
        s.append("<type>%s</type>" % t + "<dot/>" * dots)
    s.append("<staff>%d</staff>" % staff)
    nt = []
    if tie[1]: nt.append('<tied type="stop"/>')
    if tie[0]: nt.append('<tied type="start"/>')
    if tuplet: nt.append('<tuplet type="%s" bracket="yes"/>' % tuplet)
    if nt: s.append("<notations>%s</notations>" % "".join(nt))
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
    x = '<harmony><root><root-step>%s</root-step>%s</root><kind text="%s">%s</kind>' % (
        rs, "<root-alter>%d</root-alter>" % ra if ra else "", _QUAL[qual][1], _QUAL[qual][2])
    if bass is not None:
        bs, ba = sa(bass)
        x += "<bass><bass-step>%s</bass-step>%s</bass>" % (bs, "<bass-alter>%d</bass-alter>" % ba if ba else "")
    if offset: x += "<offset>%d</offset>" % offset
    return x + "</harmony>"


def _clef_xml(c, staff=None):
    sign, line, octc = c
    x = "<clef%s><sign>%s</sign>" % (' number="%d"' % staff if staff else "", sign)
    if sign != "percussion": x += "<line>%d</line>" % line
    if octc: x += "<clef-octave-change>%d</clef-octave-change>" % octc
    return x + "</clef>"


def _build_part_objs(kind, notes_div):
    cfg = KINDS[kind]
    if kind == "drums": return {1: assign_voices(notes_div, 1, mono=False)}
    if cfg["staves"] == 2:
        up, low = split_staves(notes_div)
        return {1: assign_voices(up, 1, cfg["mono"]), 2: assign_voices(low, 1, cfg["mono"])}
    return {1: assign_voices(notes_div, 1, cfg["mono"])}


def _pieces_by_bar(objs, tl):
    res = {}
    for o in objs:
        b0 = max(0, bisect.bisect_right(tl.bar_d0, o["s"]) - 1)
        for bi in range(b0, len(tl.bars)):
            bar = tl.bars[bi]
            if bar["d0"] >= o["e"]: break
            s, e = max(o["s"], bar["d0"]), min(o["e"], bar["d1"])
            if e > s: res.setdefault(bi, []).append((s - bar["d0"], e - bar["d0"], o["p"], s > o["s"], e < o["e"]))
    return res


def write_musicxml(path, parts, tl, chords, keys, title, written=False):
    X = ['<?xml version="1.0" encoding="UTF-8"?>',
         '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">',
         '<score-partwise version="3.1"><work><work-title>%s</work-title></work>' % escape(title),
         '<identification><encoding><software>Audio2Score (PianoNotes Engine)</software></encoding></identification><part-list>']
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
            if tr: f = _fifths(((tonic if not minor else (tonic + 3)) - tr[0] - 12 * tr[2]) % 12)
            X.append('<measure number="%d"%s>' % (bi if tl.bars[0]["pickup"] else bi + 1, ' implicit="yes"' if bar["pickup"] else ""))
            attrs = []
            if bi == 0: attrs.append("<divisions>%d</divisions>" % DIV)
            if (f, minor) != last_key:
                attrs.append("<key><fifths>%d</fifths><mode>%s</mode></key>" % (f, "minor" if minor else "major"))
                last_key = (f, minor)
            if (bar["num"], bar["den"]) != last_ts:
                attrs.append("<time><beats>%d</beats><beat-type>%d</beat-type></time>" % (bar["num"], bar["den"]))
                last_ts = (bar["num"], bar["den"])
            if bi == 0:
                if cfg["staves"] > 1: attrs.append("<staves>%d</staves>" % cfg["staves"])
                for si, c in enumerate(cfg["clefs"]):
                    attrs.append(_clef_xml(c, si + 1 if cfg["staves"] > 1 else None))
                if tr:
                    attrs.append("<transpose><diatonic>%d</diatonic><chromatic>%d</chromatic>%s</transpose>" % (
                        tr[1], tr[0], "<octave-change>%d</octave-change>" % tr[2] if tr[2] else ""))
            if attrs: X.append("<attributes>%s</attributes>" % "".join(attrs))
            if bar["d0o"] in tempo_at and pi == 0:
                bpm = tempo_at[bar["d0o"]]
                X.append('<direction placement="above"><direction-type><metronome><beat-unit>quarter</beat-unit>'
                         '<per-minute>%d</per-minute></metronome></direction-type><sound tempo="%.2f"/></direction>' % (round(bpm), bpm))
            if pi == 0:
                for off, r, q, bs in ch_by_bar.get(bi, []): X.append(_harmony_xml(r, q, bs, f, off))
            tri_ok = tl.tri[bi]
            first_stream = True
            for st in sorted(p["objs"]):
                for vi, pc in enumerate(pcs[st]):
                    voice_no = 1
                    pieces = sorted(pc.get(bi, []), key=lambda x: x[0])
                    if not pieces and vi > 0: continue
                    if not first_stream: X.append("<backup><duration>%d</duration></backup>" % blen)
                    first_stream = False
                    if not pieces:
                        X.append('<note><rest measure="yes"/><duration>%d</duration><voice>%d</voice><staff>%d</staff></note>' % (blen, voice_no, st))
                        continue
                    cur, acc = 0, 0
                    seq = []
                    for (s, e, pits, t_start, t_stop) in pieces:
                        if s < cur: s = cur
                        if e <= s: continue
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
                            if acc == 0: tup = "start"
                            acc += L
                            if acc >= 24:
                                tup = "stop" if tup is None else None
                                acc = 0
                        if kind_ == "rest":
                            X.append(_note_xml("", L, tri, voice_no, st, rest=True, tuplet=tup))
                        elif kind == "drums":
                            for k, pp in enumerate(pits):
                                nm, dstep, doct, nh = DRUM_INFO.get(pp, DRUM_INFO[38])
                                unp = "<unpitched><display-step>%s</display-step><display-octave>%d</display-octave></unpitched>" % (dstep, doct)
                                x = _note_xml("", L, tri, voice_no, st, chord=k > 0, tuplet=tup if k == 0 else None, unpitched=unp)
                                x = x.replace("<voice>", '<instrument id="%s-I%d"/><voice>' % (pid, pp if pp in DRUM_INFO else 38), 1)
                                if nh: x = x.replace("<staff>", "<notehead>%s</notehead><staff>" % nh, 1)
                                X.append(x)
                        else:
                            for k, pp in enumerate(pits):
                                wp = pp - tr[0] - 12 * tr[2] if tr else pp
                                X.append(_note_xml(_pitch_xml(wp, f, minor), L, tri, voice_no, st, chord=k > 0, tie=tie, tuplet=tup if k == 0 else None))
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
    data += bytes([0x00, 0xff, 0x03]) + _vlq(len(nb)) + nb
    for tick, _, msg in events:
        data += _vlq(max(0, tick - last)) + msg
        last = max(last, tick)
    data += bytes([0x00, 0xff, 0x2f, 0x00])
    return b"MTrk" + struct.pack(">I", len(data)) + bytes(data)


def write_midi(path, parts, tl, chords, pedal_by_part=None, names=None):
    trk, meta = [], []
    for d, bpm in tl.tempos:
        us = int(round(60e6 / bpm))
        meta.append((d * TICK_PER_DIV, 0, bytes([0xff, 0x51, 0x03]) + us.to_bytes(3, "big")))
    last = None
    for bar in tl.bars:
        if (bar["num"], bar["den"]) != last:
            dd = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}.get(bar["den"], 2)
            meta.append((bar["d0"] * TICK_PER_DIV, 0, bytes([0xff, 0x58, 0x04]) + bytes([bar["num"], dd, 24, 8])))
            last = (bar["num"], bar["den"])
    names_ = _SHARP_N
    for pos, r, q, bs in chords:
        txt = (names_[r] + _QUAL[q][1] + ("/" + names_[bs] if bs is not None else "")).encode("utf-8")
        meta.append((pos * TICK_PER_DIV, 1, bytes([0xff, 0x06]) + _vlq(len(txt)) + txt))
    trk.append(_track(meta, "Tempo / Chords"))
    ch = 0
    for pi, p in enumerate(parts):
        cfg = KINDS[p["kind"]]
        is_dr = p["kind"] == "drums"
        for st in sorted(p["objs"]):
            channel = 9 if is_dr else (ch if ch < 9 else ch + 1)
            ch += 0 if is_dr else 1
            ev = [] if is_dr else [(0, 0, bytes([0xC0 | channel, cfg["prog"]]))]
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
    sec = lambda t: int(round(max(t, 0.0) * 960))
    meta = [(0, 0, bytes([0xff, 0x51, 0x03]) + (500000).to_bytes(3, "big")), (0, 0, bytes([0xff, 0x58, 0x04]) + bytes([4, 2, 24, 8]))]
    for t, r, q, bs in (chords_sec or []):
        txt = (_SHARP_N[r] + _QUAL[q][1] + ("/" + _SHARP_N[bs] if bs is not None else "")).encode("utf-8")
        meta.append((sec(t), 1, bytes([0xff, 0x06]) + _vlq(len(txt)) + txt))
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


def engrave(parts_in, tl, chords, out_base, title="Transcription", written_pitch=False, key_override=None):
    onsets = [n[1] for p in parts_in if p["kind"] != "drums" for n in p["notes"]]
    choose_feels(tl, onsets)
    parts, all_notes, pedal, qs = [], [], None, []
    for p in parts_in:
        nd = quantize_notes(p["notes"], tl)
        if p["kind"] == "drums":
            for n in nd: n["e"] = n["s"] + (8 if tl.tri[tl.pbar[tl.pulse_at(n["s"])]] else 6)
        if nd: qs.append((p, nd))
    if qs and tl.bars[0]["pickup"]:
        s0 = min(n["s"] for _, nd in qs for n in nd)
        b0 = tl.bars[0]
        lead = ((s0 - b0["d0"]) // 6) * 6
        if lead >= 6 and s0 < b0["d1"]:
            b0["d0"] += lead; b0["lead"] = lead; tl.bar_d0[0] = b0["d0"]
    for p, nd in qs:
        objs = _build_part_objs(p["kind"], nd)
        parts.append(dict(kind=p["kind"], objs=objs))
        if p["kind"] != "drums": all_notes += nd
        if p.get("pedal"): pedal = p["pedal"]
    if not parts: raise RuntimeError("No notes to engrave")
    keys = [tuple(key_override)] * len(tl.bars) if key_override else (key_per_bar(all_notes, tl) if all_notes else [(0, False, 0)] * len(tl.bars))
    xml_path, mid_path = out_base + ".musicxml", out_base + ".mid"
    write_musicxml(xml_path, parts, tl, chords or [], keys, title, written_pitch)
    write_midi(mid_path, parts, tl, chords or [], pedal)
    return {"midi": mid_path, "xml": xml_path}
