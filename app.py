from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import io
import shutil
import subprocess
import torch
import gc
import threading
import traceback
import urllib.parse
import uuid
import mido
import numpy as np
import copy
import notation as nt   # score engine: beats/meter/tempo -> MusicXML + MIDI + chords

app = FastAPI(title="Audio to MIDI Pipeline")

# ---------- key detection + tempo-map quantization (shared with the local-pipeline project) ----------
MAJ_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MIN_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
PPQ_OUT = 480

# model key -> Demucs stem, playable MIDI range, monophonic?
INSTR = {
    "piano":     dict(stem="piano",  lo=21, hi=108, mono=False),
    "bass":      dict(stem="bass",   lo=28, hi=67,  mono=True),
    "guitar":    dict(stem="guitar", lo=40, hi=88,  mono=False),
    "vocals":    dict(stem="vocals", lo=48, hi=84,  mono=True),
    "sax_sop":   dict(stem="other",  lo=56, hi=88,  mono=True),
    "sax_alto":  dict(stem="other",  lo=49, hi=81,  mono=True),
    "sax_tenor": dict(stem="other",  lo=44, hi=76,  mono=True),
    "sax_bari":  dict(stem="other",  lo=36, hi=69,  mono=True),
    "brass":     dict(stem="other",  lo=52, hi=94,  mono=True),
    "violin":    dict(stem="other",  lo=55, hi=100, mono=True),
    "cello":     dict(stem="other",  lo=36, hi=81,  mono=True),
    "strings":   dict(stem="other",  lo=36, hi=100, mono=False),
    "other":     dict(stem="other",  lo=21, hi=108, mono=False),
    "drums":     dict(stem="drums",  lo=35, hi=82,  mono=False),
}
ALL_PARTS = ["vocals", "piano", "guitar", "bass", "other", "drums"]   # what "all instruments" transcribes


def read_midi_notes_and_map(path):
    """Read any standard MIDI file -> full tempo map, time-sig map, notes as (pitch, start_sec, end_sec)."""
    mid = mido.MidiFile(str(path))
    ppq = mid.ticks_per_beat
    tempo_events, timesig_events, notes, pedal_events = [], [], [], []
    for track in mid.tracks:
        time = 0
        active = {}
        for msg in track:
            time += msg.time
            if msg.type == "set_tempo":
                tempo_events.append((time, msg.tempo))
            elif msg.type == "time_signature":
                timesig_events.append((time, msg.numerator, msg.denominator))
            elif msg.type == "control_change" and msg.control == 64:
                pedal_events.append((time, msg.value))  # sustain pedal, kept apart from note durations
            elif msg.type == "note_on" and msg.velocity > 0:
                active[msg.note] = (time, msg.velocity)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                if msg.note in active:
                    start_tick, vel = active.pop(msg.note)
                    notes.append((msg.note, start_tick, time, vel))
    if not tempo_events:
        tempo_events = [(0, 500000)]
    tempo_events.sort()
    if tempo_events[0][0] > 0:
        tempo_events.insert(0, (0, tempo_events[0][1]))
    if not timesig_events:
        timesig_events = [(0, 4, 4)]
    timesig_events.sort()
    if timesig_events[0][0] > 0:
        timesig_events.insert(0, (0, timesig_events[0][1], timesig_events[0][2]))

    def tick_to_sec(tick):
        sec = 0.0
        for i, (t_start, us) in enumerate(tempo_events):
            t_end = tempo_events[i + 1][0] if i + 1 < len(tempo_events) else float("inf")
            if tick <= t_start:
                break
            sec += (min(tick, t_end) - t_start) * (us / 1e6) / ppq
            if tick <= t_end:
                break
        return sec

    tempo_map = [(tick_to_sec(t), round(60e6 / us, 2)) for t, us in tempo_events]
    timesig_map = [(tick_to_sec(t), n, d) for t, n, d in timesig_events]
    note_list = [(p, tick_to_sec(s), tick_to_sec(e), v) for p, s, e, v in notes]

    # pedal-down intervals in seconds: [(down_sec, up_sec), ...]
    pedal_intervals, down_at = [], None
    for t, val in sorted(pedal_events):
        if val >= 64 and down_at is None:
            down_at = t
        elif val < 64 and down_at is not None:
            a, b = tick_to_sec(down_at), tick_to_sec(t)
            if b > a:
                pedal_intervals.append((a, b))
            down_at = None
    if down_at is not None and pedal_events:
        a, b = tick_to_sec(down_at), tick_to_sec(max(t for t, _ in pedal_events))
        if b > a:
            pedal_intervals.append((a, b))
    return {"tempo_map": tempo_map, "timesig_map": timesig_map, "notes": note_list, "pedal": pedal_intervals}


def filter_ghost_notes(notes_with_vel, min_duration=0.03, min_velocity=15, max_duration=4.0):
    """Drop too-short / too-quiet notes (ghost notes), and cap note length to tame
    sustain-pedal smearing. Returns (pitch, start, end, velocity) tuples -- velocity is
    carried through the pipeline so it can be smoothed and written to the output."""
    out = []
    for pitch, start, end, vel in notes_with_vel:
        dur = end - start
        if dur < min_duration or vel < min_velocity or (vel < 30 and dur < 0.15):
            continue
        if dur > max_duration:
            end = start + max_duration
        out.append((pitch, start, end, vel))
    return out


def remove_octave_overtones(notes, time_tolerance=0.06, vel_ratio=0.8):
    """Drop a weaker note that starts together with a note 1/2 octaves (or a 12th) below it:
    typical harmonic false-positives. Real octave doubling (similar velocity) is kept."""
    import bisect
    ns = sorted(notes, key=lambda n: n[1])
    starts = [n[1] for n in ns]
    drop = set()
    for i, n in enumerate(ns):
        v = n[3] if len(n) > 3 else 80
        lo = bisect.bisect_left(starts, n[1] - time_tolerance)
        hi = bisect.bisect_right(starts, n[1] + time_tolerance)
        for j in range(lo, hi):
            if j == i:
                continue
            m = ns[j]
            mv = m[3] if len(m) > 3 else 80
            if (n[0] - m[0]) in (12, 19, 24, 28) and m[2] > n[1] and (
                    (n[2] - n[1]) <= (m[2] - m[1]) + time_tolerance or v < mv * vel_ratio) and v <= mv * 1.05:
                drop.add(i)
                break
    return [n for i, n in enumerate(ns) if i not in drop]


def make_monophonic(notes):
    """Single-line instruments: simultaneous onsets keep the strongest, overlaps are trimmed."""
    out = []
    for n in sorted(notes, key=lambda n: (n[1], -(n[3] if len(n) > 3 else 80))):
        if out and n[1] - out[-1][1] < 0.04:
            continue
        if out and out[-1][2] > n[1]:
            p = out[-1]
            out[-1] = (p[0], p[1], n[1]) + tuple(p[3:])
        out.append(n)
    return [n for n in out if n[2] - n[1] >= 0.05]


def snap_to_key_dynamic(notes, key_notes=None, win=12.0):
    """Key detection per ~12s window (3s overlap, with hysteresis so it only flips on real evidence),
    then snaps weak/short out-of-key notes to the key of THEIR window. Long, loud notes are kept
    (real accidentals). key_notes (e.g. the anchor MIDI) can drive detection if given."""
    if not notes:
        return notes
    src_notes = key_notes or notes
    total = max(n[2] for n in list(src_notes) + list(notes))
    nw = int(total // win) + 1
    keys, prev = [], None
    for w in range(nw):
        a, b = w * win - 3, (w + 1) * win + 3
        hist = [0.0] * 12
        for n in src_notes:
            ov = min(n[2], b) - max(n[1], a)
            if ov > 0:
                hist[n[0] % 12] += ov
        if sum(hist) <= 0:
            keys.append(None)
            continue
        scores, best = {}, (-1e18, 0, False)
        for tonic in range(12):
            for prof, mi in ((MAJ_PROFILE, False), (MIN_PROFILE, True)):
                sc = sum(hist[(i + tonic) % 12] * prof[i] for i in range(12))
                scores[(tonic, mi)] = sc
                if sc > best[0]:
                    best = (sc, tonic, mi)
        cur = (best[1], best[2])
        if prev is not None and scores[prev] >= best[0] * 0.95:
            cur = prev
        prev = cur
        keys.append(cur)
    first = next((k for k in keys if k is not None), None)
    if first is None:
        return notes
    last = first
    for i in range(nw):
        if keys[i] is None:
            keys[i] = last
        else:
            last = keys[i]
    out = []
    for n in notes:
        tonic, mi = keys[min(int(max(n[1], 0) // win), nw - 1)]
        scale = {(x + tonic) % 12 for x in ([0, 2, 3, 5, 7, 8, 10, 11] if mi else [0, 2, 4, 5, 7, 9, 11])}
        pitch, pc = n[0], n[0] % 12
        vel = n[3] if len(n) > 3 else 80
        if pc not in scale and ((n[2] - n[1]) < 0.4 or vel < 60):
            for d in (1, -1):
                if (pc + d) % 12 in scale:
                    pitch += d
                    break
        out.append((pitch, n[1], n[2]) + tuple(n[3:]))
    return out


# ---------- chroma DTW: only used when a reference MIDI is uploaded, to align its
# tempo/structure to the ACTUAL timing of the performance (tempo drift, rubato, etc.) ----------
def _audio_chroma(wav_path, hop_length=1024, sr_target=22050):
    import librosa
    y, sr = librosa.load(wav_path, sr=sr_target)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    return chroma, hop_length, sr


def _midi_notes_to_chroma(notes4, duration_sec, hop_length, sr):
    import numpy as np
    frame_time = hop_length / sr
    n_frames = max(1, int(duration_sec / frame_time) + 1)
    chroma = np.zeros((12, n_frames))
    for pitch, start, end, *_ in notes4:
        f0, f1 = int(start / frame_time), min(n_frames, int(end / frame_time) + 1)
        if f1 > f0:
            chroma[pitch % 12, f0:f1] += 1.0
    norms = np.linalg.norm(chroma, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    return chroma / norms


def build_perf_to_ref(anchor_notes4, perf_wav_path):
    """Chroma DTW between the anchor MIDI and the performance. Returns to_ref(t): maps a
    performance time (sec) onto the anchor's timeline, so transcribed notes can wear the
    anchor's exact tempo map / time signatures. Raises on failure (caller falls back)."""
    import numpy as np
    import librosa
    chroma_perf, hop, sr = _audio_chroma(perf_wav_path)
    ref_dur = max((e for _, _, e, *_ in anchor_notes4), default=1.0)
    chroma_ref = _midi_notes_to_chroma(anchor_notes4, ref_dur, hop, sr)
    _, wp = librosa.sequence.dtw(X=chroma_ref, Y=chroma_perf, metric="cosine")
    wp = np.asarray(wp)[::-1]
    n_perf = chroma_perf.shape[1]
    cnt = np.bincount(wp[:, 1], minlength=n_perf)
    sm = np.bincount(wp[:, 1], weights=wp[:, 0], minlength=n_perf)
    idx = np.where(cnt > 0)[0]
    ft = hop / sr
    xs = idx * ft
    ys = np.maximum.accumulate(sm[idx] / cnt[idx]) * ft
    fwd = lambda t: float(np.interp(t, xs, ys))
    fwd.inv = lambda t: np.interp(t, ys, xs)
    return fwd


def truncate_sustain_overlaps(notes, pitch_range=4):
    """If a note's sustain runs past the start of another note within a few
    semitones of it, shorten the earlier note to end where the next one begins --
    mitigates pedal-smeared notes bleeding into later phrases."""
    out = list(notes)
    for i, n in enumerate(out):
        pitch, start, end = n[0], n[1], n[2]
        for m in out:
            pitch2, start2 = m[0], m[1]
            if start2 > start and start2 < end and abs(pitch2 - pitch) <= pitch_range:
                end = min(end, start2)
        out[i] = (pitch, start, end) + tuple(n[3:])
    return out


def _timesig_for_segment(y_seg, sr):
    """One heuristic guess for a short audio segment's time signature, via
    beat-accent periodicity. Candidate numerators: 2,3,4,5,6,7,9. This is a
    real heuristic, not a guarantee -- it works reasonably for a clear, steady
    meter, and is NOT reliable for irregular/mixed meters (7/8, 13/8, etc.):
    no simple algorithm (this one included) detects those with real confidence."""
    import numpy as np
    import librosa
    onset_env = librosa.onset.onset_strength(y=y_seg, sr=sr)
    _, beat_frames = librosa.beat.beat_track(y=y_seg, sr=sr, onset_envelope=onset_env)
    if len(beat_frames) < 8:
        return 4
    beat_strength = onset_env[beat_frames]
    best_n, best_score = 4, -1.0
    overall_std = float(np.std(beat_strength)) + 1e-6
    for n in (2, 3, 4, 5, 6, 7, 9):
        if len(beat_strength) < n * 2:
            continue
        groups = [beat_strength[i:i + n] for i in range(0, len(beat_strength) - n, n)]
        if len(groups) < 2:
            continue
        downbeat_avg = float(np.mean([g[0] for g in groups]))
        rest_avg = float(np.mean([np.mean(g[1:]) if len(g) > 1 else 0.0 for g in groups]))
        score = (downbeat_avg - rest_avg) / overall_std
        if score > best_score:
            best_score, best_n = score, n
    return best_n


def detect_time_signature_map(wav_path, window_sec=20.0):
    """Time-signature map across the whole track: re-runs the segment heuristic
    every window_sec seconds and records a new time_signature event whenever the
    guess changes. Denominator is always reported as 4 (this heuristic doesn't
    attempt to distinguish simple vs compound subdivision)."""
    import librosa
    y, sr = librosa.load(wav_path, sr=None)
    total_dur = len(y) / sr
    events, t, last_n = [], 0.0, None
    while t < total_dur:
        seg = y[int(t * sr):int(min(t + window_sec, total_dur) * sr)]
        if len(seg) < sr * 4:
            break
        n = _timesig_for_segment(seg, sr)
        if n != last_n:
            events.append((t, n, 4))
            last_n = n
        t += window_sec
    return events if events else [(0.0, 4, 4)]


def detect_bpm_from_audio(wav_path):
    """Audio-only tempo estimate, for the 'auto-detect' mode when no anchor MIDI is given."""
    import numpy as np
    import librosa
    y, sr = librosa.load(wav_path, sr=None)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(np.atleast_1d(tempo)[0])


def detect_key(notes):
    hist = [0.0] * 12
    for n in notes:
        pitch, start, end = n[0], n[1], n[2]
        hist[pitch % 12] += max(0.05, end - start)
    best = (-1e18, 0, False)
    for tonic in range(12):
        for profile, is_minor in ((MAJ_PROFILE, False), (MIN_PROFILE, True)):
            score = sum(hist[(i + tonic) % 12] * profile[i] for i in range(12))
            if score > best[0]:
                best = (score, tonic, is_minor)
    return best[1], best[2]


def snap_to_key(notes, tonic, is_minor):
    scale = {(x + tonic) % 12 for x in ([0, 2, 3, 5, 7, 8, 10] if is_minor else [0, 2, 4, 5, 7, 9, 11])}
    out = []
    for n in notes:
        pitch, start, end = n[0], n[1], n[2]
        extra = tuple(n[3:])
        pc = pitch % 12
        if pc in scale:
            out.append((pitch, start, end) + extra)
            continue
        best_s, best_d = pc, 99
        for s in scale:
            d = min(abs(s - pc), 12 - abs(s - pc))
            if d < best_d:
                best_d, best_s = d, s
        new_pitch = pitch + ((best_s - pc + 18) % 12 - 6) if best_d <= 1 else pitch
        out.append((new_pitch, start, end) + extra)
    return out


def build_segments(tempo_map, ppq):
    segs, tick = [], 0.0
    for i, (t, bpm) in enumerate(tempo_map):
        if i > 0:
            prev_t, prev_bpm = tempo_map[i - 1]
            tick += (t - prev_t) * (prev_bpm / 60.0) * ppq
        segs.append((t, bpm, tick))
    return segs


def sec_to_tick(sec, segs):
    for t, bpm, tick in reversed(segs):
        if sec >= t:
            return tick + (sec - t) * (bpm / 60.0) * PPQ_OUT
    return 0.0


def smooth_velocities(notes, window=3, blend=0.6, lo=35, hi=110):
    """Velocity smoothing: pull every note's velocity toward the average of its
    neighbours (in time order) and clamp it, so the output has an even dynamic
    shape instead of jumpy model noise. Notes without a velocity get 80."""
    if not notes:
        return notes
    order = sorted(range(len(notes)), key=lambda i: notes[i][1])
    vels = [(notes[i][3] if len(notes[i]) > 3 else 80) for i in order]
    out = list(notes)
    n = len(order)
    for k, i in enumerate(order):
        a, b = max(0, k - window), min(n, k + window + 1)
        avg = sum(vels[a:b]) / (b - a)
        v = int(round((1 - blend) * vels[k] + blend * avg))
        out[i] = tuple(notes[i][:3]) + (max(lo, min(hi, v)),)
    return out


def quantize_notes(bp_notes, tempo_map, anchor_notes, fine_starts=None):
    segs = build_segments(tempo_map, PPQ_OUT)
    grid = PPQ_OUT // 4
    fine = PPQ_OUT // 12      # 16ths AND triplets, used only for notes matched to the anchor
    phase = 0
    if anchor_notes:
        offsets = sorted(int(sec_to_tick(s, segs)) % grid for _, s, _ in anchor_notes)
        phase = offsets[len(offsets) // 2]
    out = []
    for n in bp_notes:
        pitch, start, end = n[0], n[1], n[2]
        is_fine = bool(fine_starts) and (pitch, start) in fine_starts
        g = fine if is_fine else grid
        ph = phase % g
        s_tick = sec_to_tick(start, segs)
        e_tick = sec_to_tick(end, segs)
        q_s = round((s_tick - ph) / g) * g + ph
        q_d = max(60 if is_fine else grid, round((e_tick - s_tick) / g) * g)
        out.append((pitch, max(0, int(q_s)), int(q_d)) + tuple(n[3:]))
    return out, segs


def dedupe_quantized(notes):
    """Same pitch, same tick -> one note; overlapping same-pitch notes are trimmed (no smear/doubles)."""
    by = {}
    for n in notes:
        by.setdefault(n[0], []).append(n)
    out = []
    for p, lst in by.items():
        lst.sort(key=lambda n: n[1])
        cur = lst[0]
        for n in lst[1:]:
            if n[1] == cur[1]:
                cv = max(cur[3] if len(cur) > 3 else 80, n[3] if len(n) > 3 else 80)
                cur = (p, cur[1], max(cur[2], n[2]), cv)
                continue
            if cur[1] + cur[2] > n[1]:
                cur = (p, cur[1], n[1] - cur[1]) + tuple(cur[3:])
            if cur[2] > 0:
                out.append(cur)
            cur = n
        out.append(cur)
    return sorted(out, key=lambda n: (n[1], n[0]))


def _tick_to_sec(tick, segs):
    for t, bpm, tk in reversed(segs):
        if tick >= tk:
            return t + (tick - tk) / ((bpm / 60.0) * PPQ_OUT)
    return 0.0


def measure_boundaries(segs, timesig_map, end_sec):
    """Measure start times (sec) built from the anchor's tempo map + time-signature map."""
    ts = sorted((sec_to_tick(t, segs), n, d) for t, n, d in timesig_map) or [(0, 4, 4)]
    end_tick, tick, bounds = sec_to_tick(end_sec, segs), 0.0, []
    while tick <= end_tick + 1:
        bounds.append(_tick_to_sec(tick, segs))
        cur = next((x for x in reversed(ts) if x[0] <= tick + 1), ts[0])
        tick += cur[1] * PPQ_OUT * 4 / cur[2]
    return bounds


def refine_notes_to_anchor(notes, anchor_notes, bounds):
    """Phrase/measure matching: per measure, shift transcribed notes by the median offset to the
    anchor's notes (pitch-class match, +-0.25s, capped +-0.15s), then snap matched notes to the
    anchor's onsets one-to-one (never duplicates). Returns (notes, fine_starts)."""
    import bisect, statistics
    a = sorted(anchor_notes, key=lambda x: x[1])
    a_st = [x[1] for x in a]
    ns = sorted(notes, key=lambda n: n[1])
    n_st = [n[1] for n in ns]
    shifted = list(ns)
    for m in range(len(bounds) - 1):
        i0, i1 = bisect.bisect_left(n_st, bounds[m]), bisect.bisect_left(n_st, bounds[m + 1])
        if i1 - i0 < 3:
            continue
        offs = []
        for k in range(i0, i1):
            n = ns[k]
            lo, hi = bisect.bisect_left(a_st, n[1] - 0.25), bisect.bisect_right(a_st, n[1] + 0.25)
            c = [a[j][1] - n[1] for j in range(lo, hi) if a[j][0] % 12 == n[0] % 12]
            if c:
                offs.append(min(c, key=abs))
        if len(offs) >= 3:
            off = max(-0.15, min(0.15, statistics.median(offs)))
            for k in range(i0, i1):
                n = shifted[k]
                shifted[k] = (n[0], n[1] + off, n[2] + off) + tuple(n[3:])
    used, fine, out = set(), set(), []
    for n in sorted(shifted, key=lambda n: n[1]):
        lo, hi = bisect.bisect_left(a_st, n[1] - 0.08), bisect.bisect_right(a_st, n[1] + 0.08)
        c = [j for j in range(lo, hi) if j not in used and a[j][0] == n[0]] or \
            [j for j in range(lo, hi) if j not in used and a[j][0] % 12 == n[0] % 12]
        if c:
            j = min(c, key=lambda j: abs(a[j][1] - n[1]))
            used.add(j)
            d, s = n[2] - n[1], a[j][1]
            n = (n[0], s, s + d) + tuple(n[3:])
            fine.add((n[0], s))
        out.append(n)
    return out, fine


def split_hands_dynamic(notes, window_ticks=PPQ_OUT * 8, lo=55, hi=65, default=60):
    """Dynamic hand split: instead of a fixed middle C, the split point follows the
    median pitch of the notes around each note (about +-8 beats), clamped to
    [lo, hi]. Notes at/above the local split go to the right hand (treble), the rest
    to the left hand (bass). Sparse passages (<3 nearby notes) fall back to middle C.
    Notes are (pitch, start_tick, dur_tick[, velocity])."""
    import bisect
    if not notes:
        return [], []
    order = sorted(range(len(notes)), key=lambda i: notes[i][1])
    starts = [notes[i][1] for i in order]
    pitches = [notes[i][0] for i in order]
    right, left = [], []
    for k, i in enumerate(order):
        a = bisect.bisect_left(starts, starts[k] - window_ticks)
        b = bisect.bisect_right(starts, starts[k] + window_ticks)
        local = sorted(pitches[a:b])
        split = default if len(local) < 3 else max(lo, min(hi, local[len(local) // 2]))
        (right if notes[i][0] >= split else left).append(notes[i])
    return right, left


def _pedal_track(intervals_ticks, channel=0):
    """Sustain pedal as separate CC64 events -- independent of the real note lengths."""
    tr = mido.MidiTrack()
    evs = []
    for a, b in intervals_ticks:
        if b > a:
            evs.append((a, mido.Message("control_change", channel=channel, control=64, value=127)))
            evs.append((b, mido.Message("control_change", channel=channel, control=64, value=0)))
    evs.sort(key=lambda e: e[0])
    last_t = 0
    for tick, msg in evs:
        msg.time = max(0, round(tick - last_t))
        tr.append(msg)
        last_t = max(last_t, tick)
    tr.append(mido.MetaMessage("end_of_track"))
    return tr


def _notes_track(notes, channel=0):
    tr = mido.MidiTrack()
    evs = []
    for n in notes:
        pitch, s_tick, d_tick = n[0], n[1], n[2]
        vel = max(1, min(127, int(n[3]))) if len(n) > 3 else 90
        evs.append((s_tick, 1, mido.Message("note_on", note=int(pitch), velocity=vel, channel=channel)))
        evs.append((s_tick + d_tick, 0, mido.Message("note_off", note=int(pitch), velocity=0, channel=channel)))
    evs.sort(key=lambda e: (e[0], e[1]))  # at equal ticks: note_off before note_on
    last_t = 0
    for tick, _order, msg in evs:
        msg.time = max(0, round(tick - last_t))
        tr.append(msg)
        last_t = max(last_t, tick)
    tr.append(mido.MetaMessage("end_of_track"))
    return tr


def build_output_midi_bytes(quantized_notes, tempo_map, timesig_map, segs, split_hands=False, pedal_ticks=None):
    events = []
    for t, bpm in tempo_map:
        events.append((sec_to_tick(t, segs), mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm))))
    for t, num, den in timesig_map:
        events.append((sec_to_tick(t, segs), mido.MetaMessage("time_signature", numerator=num, denominator=den)))
    events.sort(key=lambda e: e[0])
    meta_track = mido.MidiTrack()
    last = 0.0
    for tick, msg in events:
        msg.time = max(0, round(tick - last))
        meta_track.append(msg)
        last = max(last, tick)
    meta_track.append(mido.MetaMessage("end_of_track"))

    out_mid = mido.MidiFile(ticks_per_beat=PPQ_OUT)
    out_mid.tracks.append(meta_track)
    if split_hands:
        # Right hand (treble) / left hand (bass), split point follows the music (see split_hands_dynamic).
        right, left = split_hands_dynamic(quantized_notes)
        out_mid.tracks.append(_notes_track(right, channel=0))
        out_mid.tracks.append(_notes_track(left, channel=1))
    else:
        out_mid.tracks.append(_notes_track(quantized_notes, channel=0))
    if pedal_ticks:
        out_mid.tracks.append(_pedal_track(pedal_ticks, channel=0))
    buf = io.BytesIO()
    out_mid.save(file=buf)
    return buf.getvalue()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Download-Name", "X-Used-Fallback", "X-DTW-Used"],
)

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"⚡ מפעיל התקן עיבוד: {DEVICE}")

PIANO_MODEL = None
try:
    from piano_transcription_inference import PianoTranscription
    print("⏳ טוען את מודל הפסנתר ל-GPU מראש...")
    PIANO_MODEL = PianoTranscription(device=DEVICE)
    print("✅ מודל הפסנתר נטען בהצלחה!")
except Exception:
    import traceback
    print("⚠️ שגיאה בטעינת מודל הפסנתר -- המודל הכללי (Basic Pitch) ישמש כגיבוי אוטומטי:")
    traceback.print_exc()  # full traceback, not just str(e) -- needed to actually diagnose this

BP_PREDICT, BP_MODEL = None, None
try:
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict as BP_PREDICT
    try:
        from basic_pitch.inference import Model as _BPModel
        BP_MODEL = _BPModel(ICASSP_2022_MODEL_PATH)      # load once, reuse
    except Exception:
        BP_MODEL = ICASSP_2022_MODEL_PATH                # path is accepted as model_or_model_path
except Exception:
    traceback.print_exc()


def basic_pitch_to_midi(wav_path, out_midi, model_key):
    """FIX: uses predict() with the model passed explicitly (no predict_and_save), tuned per instrument."""
    import librosa
    if BP_PREDICT is None:
        raise RuntimeError("Basic Pitch לא מותקן")
    cfg = INSTR.get(model_key, INSTR["other"])
    kw = dict(onset_threshold=0.6 if cfg["mono"] else 0.5,
              frame_threshold=0.35 if cfg["mono"] else 0.3,
              minimum_note_length=100,
              minimum_frequency=float(librosa.midi_to_hz(cfg["lo"])),
              maximum_frequency=float(librosa.midi_to_hz(cfg["hi"])))
    try:
        _, midi_data, _ = BP_PREDICT(wav_path, model_or_model_path=BP_MODEL, **kw)
    except TypeError:
        _, midi_data, _ = BP_PREDICT(wav_path, **kw)
    midi_data.write(out_midi)


HTML_CONTENT = """<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ממיר שמע לתווים ול-MIDI</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background-color: #0f172a; color: #f8fafc; margin: 0; padding: 20px; display: flex; justify-content: center; align-items: center; min-height: 100vh; overflow: hidden; }
        #bgCanvas { position: fixed; inset: 0; z-index: 0; pointer-events: none; }
        .card { position: relative; z-index: 1; background: #1e293bcc; backdrop-filter: blur(6px); border-radius: 16px; padding: 28px; max-width: 480px; width: 100%; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5); text-align: center; border: 1px solid #334155; }
        h1 { font-size: 1.6rem; margin-bottom: 8px; color: #38bdf8; }
        p { color: #94a3b8; font-size: 0.95rem; margin-bottom: 24px; line-height: 1.4; }
        .upload-box { border: 2px dashed #0284c7; border-radius: 12px; padding: 24px; cursor: pointer; margin-bottom: 16px; background: #0f172a; transition: 0.2s; }
        .upload-box:hover { border-color: #38bdf8; background: #1e293b; }
        input[type="file"] { display: none; }
        select, button { width: 100%; padding: 14px; border-radius: 10px; border: 1px solid #475569; margin-top: 12px; font-size: 1rem; font-weight: 600; box-sizing: border-box; }
        select { background: #0f172a; color: white; outline: none; }
        button { background: #0284c7; color: white; border: none; cursor: pointer; transition: 0.2s; margin-top: 20px; }
        button:hover { background: #0369a1; }
        button:active { transform: scale(.97); }
        #status { margin-top: 18px; font-size: 0.95rem; font-weight: bold; }
        .success { color: #4ade80; }
        .error { color: #f87171; font-size: 0.85rem; word-break: break-all; }
        .loading { color: #facc15; }
        .modebar { display: flex; gap: 8px; margin-top: 16px; }
        .modebar label { flex: 1; text-align: center; padding: 9px 6px; border: 1px solid #475569; border-radius: 8px; font-size: 0.82rem; cursor: pointer; color: #94a3b8; }
        .modebar input { display: none; }
        .modebar label:has(input:checked) { border-color: #38bdf8; color: #38bdf8; font-weight: 700; background: rgba(56,189,248,0.08); }
        .modepane { display: none; text-align: right; }
        .modepane.active { display: block; }
        .row2 { display: flex; gap: 10px; }
        .row2 > div { flex: 1; }
        .fieldlabel { display: block; font-size: 0.8rem; color: #94a3b8; margin: 10px 0 4px; }
    </style>
</head>
<body>
    <canvas id="bgCanvas"></canvas>
    <div class="card">
        <h1>🎹 ממיר שמע לתווים ול-MIDI</h1>
        <p>העלה קובץ MP3 או שמע מהאייפד והמר אותו לקובץ MIDI נקי</p>
        <form id="transcribeForm">
            <div class="upload-box" onclick="document.getElementById('audioFile').click()">
                <span id="fileName">📁 לחץ כאן לבחירת קובץ MP3</span>
                <input type="file" id="audioFile" name="file" accept="audio/*,.mp3,.wav,.m4a" required onchange="updateFileName(this)">
            </div>
            <select name="model" id="modelSelect">
                <optgroup label="הכל">
                    <option value="all">🎼 כל הכלים (שירה, פסנתר, גיטרה, בס, כלים נוספים, תופים) + אקורדים</option>
                </optgroup>
                <optgroup label="פסנתר">
                    <option value="piano">🎹 פסנתר (Demucs + Piano Transcription)</option>
                </optgroup>
                <optgroup label="סקסופון (סטם other)">
                    <option value="sax_alto">🎷 סקסופון אלט</option>
                    <option value="sax_tenor">🎷 סקסופון טנור</option>
                    <option value="sax_bari">🎷 סקסופון בריטון</option>
                    <option value="sax_sop">🎷 סקסופון סופרן</option>
                </optgroup>
                <optgroup label="נחושת ומיתרים (סטם other)">
                    <option value="brass">🎺 חצוצרה וכלי נשיפה מתכתיים</option>
                    <option value="violin">🎻 כינור</option>
                    <option value="cello">🎻 צ'לו</option>
                    <option value="strings">🎻 מיתרים (אנסמבל)</option>
                </optgroup>
                <optgroup label="בס / גיטרה / שירה">
                    <option value="bass">🎸 בס (Demucs + Basic Pitch)</option>
                    <option value="guitar">🎸 גיטרה (Demucs + Basic Pitch)</option>
                    <option value="vocals">🎤 שירה (Demucs + Basic Pitch)</option>
                </optgroup>
                <optgroup label="אחר">
                    <option value="drums">🥁 תופים (בסיסי: קיק / סנייר / האט)</option>
                    <option value="other">🎼 שאר הכלים (סטם "other")</option>
                </optgroup>
            </select>
            <p style="color:#64748b;font-size:0.78rem;margin-top:6px">כל בחירה מפעילה קודם הפרדת Demucs אמיתית, ורק אז מתמללת את הכלי המבודד -- לא את כל התערובת.</p>

            <div class="modebar">
                <label><input type="radio" name="mode" value="detect" checked onchange="setMode('detect')"><span>זיהוי אוטומטי</span></label>
                <label><input type="radio" name="mode" value="manual" onchange="setMode('manual')"><span>קביעה ידנית</span></label>
                <label><input type="radio" name="mode" value="anchor" onchange="setMode('anchor')"><span>MIDI בסיס</span></label>
            </div>

            <div class="modepane active" id="pane-detect">
                <p style="color:#64748b;font-size:0.78rem;margin-top:4px">הסולם יזוהה מתוך התווים שתומללו, והקצב (BPM) מתוך האודיו עצמו (librosa.beat_track). לא צריך להעלות שום דבר נוסף.</p>
            </div>

            <div class="modepane" id="pane-manual">
                <div class="row2">
                    <div>
                        <label class="fieldlabel">סולם</label>
                        <select id="keyTonic">
                            <option value="0">C</option><option value="1">C#</option><option value="2">D</option>
                            <option value="3">D#</option><option value="4">E</option><option value="5">F</option>
                            <option value="6">F#</option><option value="7">G</option><option value="8">G#</option>
                            <option value="9">A</option><option value="10">A#</option><option value="11">B</option>
                        </select>
                    </div>
                    <div>
                        <label class="fieldlabel">מז'ור/מינור</label>
                        <select id="keyMode"><option value="major">מז'ור</option><option value="minor">מינור</option></select>
                    </div>
                </div>
                <label class="fieldlabel">קצב (BPM)</label>
                <input type="number" id="manualBpm" value="120" style="width:100%;padding:12px;border-radius:8px;border:1px solid #475569;background:#0f172a;color:white;box-sizing:border-box;">
            </div>

            <div class="modepane" id="pane-anchor">
                <label class="fieldlabel">קובץ MIDI בסיס</label>
                <input type="file" id="anchorFile" accept=".mid,.midi" style="display:block;width:100%;color:#94a3b8;font-size:0.85rem;">
                <p style="color:#64748b;font-size:0.78rem;margin-top:6px">הסולם, מפת הטמפו ומקצב (כולל שינויים באמצע השיר) ייקראו אוטומטית מהקובץ הזה.</p>
            </div>

            <label id="splitRow" style="display:flex;align-items:center;gap:8px;font-size:0.85rem;color:#94a3b8;margin-top:14px;text-align:right">
                <input type="checkbox" id="splitHands" style="width:auto">
                פיצול לשתי ידיים (ליד ימין/שמאל, רק לפסנתר)
            </label>

            <label id="pedalRow" style="display:flex;align-items:center;gap:8px;font-size:0.85rem;color:#94a3b8;margin-top:10px;text-align:right">
                <input type="checkbox" id="keepPedal" style="width:auto">
                שמור פדל סאסטיין כאירועים נפרדים (CC64), בלי למתוח את התווים
            </label>

            <label style="display:flex;align-items:center;gap:8px;font-size:0.85rem;color:#94a3b8;margin-top:10px;text-align:right">
                <input type="checkbox" id="addChords" checked style="width:auto">
                הוסף סימני אקורדים מעל התווים (lead sheet)
            </label>
            <label style="display:flex;align-items:center;gap:8px;font-size:0.85rem;color:#94a3b8;margin-top:10px;text-align:right">
                <input type="checkbox" id="writtenPitch" style="width:auto">
                תווים כתובים לכלי מתועתק (סקסופון / חצוצרה ב-Bb / Eb)
            </label>

            <button type="submit" id="submitBtn">המר לתווים + MIDI 🚀</button>
        </form>
        <div id="status"></div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
    <script>
    (function(){
        if (window.matchMedia('(prefers-reduced-motion: reduce)').matches || !window.THREE) return;
        const canvas = document.getElementById('bgCanvas');
        const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.setSize(window.innerWidth, window.innerHeight);
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);
        camera.position.z = 8;

        scene.add(new THREE.AmbientLight(0x3a2410, 1.4));
        const key = new THREE.PointLight(0xffb347, 2.2, 60); key.position.set(6, 6, 6); scene.add(key);
        const rim = new THREE.PointLight(0xff7a1a, 1.2, 60); rim.position.set(-6, -3, -4); scene.add(rim);

        const amberMat = () => new THREE.MeshStandardMaterial({ color: 0xe08a2e, roughness: 0.35, metalness: 0.55, transparent: true, opacity: 0.55 });
        const shapes = [
            new THREE.Mesh(new THREE.IcosahedronGeometry(1.7, 0), amberMat()),
            new THREE.Mesh(new THREE.TorusGeometry(1.1, 0.35, 16, 64), amberMat()),
            new THREE.Mesh(new THREE.IcosahedronGeometry(0.9, 1), amberMat()),
        ];
        shapes[0].position.set(-3.6, 1.6, -2);
        shapes[1].position.set(3.3, -1.8, -3);
        shapes[2].position.set(2.6, 2.3, -4.5);
        shapes.forEach(s => scene.add(s));

        function animate(t) {
            shapes.forEach((m, i) => {
                m.rotation.x = t * 0.00018 * (i + 1);
                m.rotation.y = t * 0.00013 * (i + 1);
                m.position.y += Math.sin(t * 0.0004 + i) * 0.0012;
            });
            renderer.render(scene, camera);
            requestAnimationFrame(animate);
        }
        requestAnimationFrame(animate);
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
    })();
    </script>
    <script>
        function updateFileName(input) {
            if (input.files.length > 0) {
                document.getElementById('fileName').innerText = "📄 " + input.files[0].name;
            }
        }
        function setMode(m) {
            document.getElementById('pane-detect').classList.toggle('active', m === 'detect');
            document.getElementById('pane-manual').classList.toggle('active', m === 'manual');
            document.getElementById('pane-anchor').classList.toggle('active', m === 'anchor');
        }
        document.getElementById('modelSelect').addEventListener('change', function(){
            const isPiano = (this.value === 'piano' || this.value === 'all');
            document.getElementById('splitRow').style.display = isPiano ? 'flex' : 'none';
            document.getElementById('pedalRow').style.display = isPiano ? 'flex' : 'none';
        });

        document.getElementById('transcribeForm').onsubmit = async (e) => {
            e.preventDefault();
            const status = document.getElementById('status');
            const btn = document.getElementById('submitBtn');
            const mode = document.querySelector('input[name="mode"]:checked').value;
            if (mode === 'anchor' && !document.getElementById('anchorFile').files.length) {
                status.className = "error";
                status.innerText = "❌ יש לבחור קובץ MIDI בסיס, או לעבור למצב ידני.";
                return;
            }

            status.className = "loading";
            status.innerText = "⏳ מפריד כלים ומתמלל... זה יכול לקחת כמה דקות, אל תסגור את הדף";
            btn.disabled = true;
            btn.style.opacity = "0.5";

            const formData = new FormData();
            formData.append('file', document.getElementById('audioFile').files[0]);
            formData.append('model', document.getElementById('modelSelect').value);
            formData.append('mode', mode);
            formData.append('split_hands', document.getElementById('splitHands').checked);
            formData.append('keep_pedal', document.getElementById('keepPedal').checked);
            formData.append('add_chords', document.getElementById('addChords').checked);
            formData.append('written_pitch', document.getElementById('writtenPitch').checked);
            if (mode === 'manual') {
                formData.append('manual_key_tonic', document.getElementById('keyTonic').value);
                formData.append('manual_key_mode', document.getElementById('keyMode').value);
                formData.append('manual_bpm', document.getElementById('manualBpm').value);
            } else if (mode === 'anchor') {
                formData.append('anchor_midi', document.getElementById('anchorFile').files[0]);
            }

            const sleep = ms => new Promise(r => setTimeout(r, ms));
            try {
                // 1) start the job -- the server answers right away, the heavy work runs in the background
                const res = await fetch('/transcribe', { method: 'POST', body: formData });
                const raw = await res.text();
                let data = null;
                try { data = JSON.parse(raw); } catch (e) {}
                if (!res.ok || !data || !data.job_id) {
                    const why = (data && data.detail) ? data.detail : ('HTTP ' + res.status + ' ' + raw.slice(0, 160));
                    throw new Error(why);
                }
                const jobId = data.job_id;

                // 2) poll until done (no long-lived request, so the tunnel cannot time it out)
                let fails = 0, info = null;
                while (true) {
                    await sleep(2500);
                    let r;
                    try {
                        r = await fetch('/status/' + jobId);
                        fails = 0;
                    } catch (e) {
                        fails++;
                        if (fails >= 8) throw new Error('החיבור לשרת נפל (השרת או המנהרה כבו). הרץ שוב את תא ההפעלה ב-Colab.');
                        continue;
                    }
                    if (r.status === 404) throw new Error('המשימה נעלמה -- השרת הופעל מחדש באמצע. נסה שוב.');
                    if (!r.ok) throw new Error('בדיקת הסטטוס נכשלה (HTTP ' + r.status + ')');
                    info = await r.json();
                    if (info.status === 'error') throw new Error(info.error || 'העיבוד נכשל');
                    status.className = "loading";
                    status.innerText = '⏳ ' + (info.stage || 'מעבד...');
                    if (info.status === 'done') break;
                }

                // 3) download buttons (MIDI / MusicXML for MuseScore / PDF when available)
                const labels = {midi: '⬇ MIDI', xml: '⬇ MusicXML (MuseScore)', pdf: '⬇ PDF'};
                const base = (document.getElementById('audioFile').files[0].name || 'transcribed').replace(/\\.[^.]+$/, '');
                const extOf = {midi: '.mid', xml: '.musicxml', pdf: '.pdf'};
                status.className = info.used_fallback ? "loading" : "success";
                let msg = info.used_fallback ? "⚠️ מודל הפסנתר הייעודי לא היה זמין -- נעשה שימוש ב-Basic Pitch.<br>" : "✅ ההמרה הושלמה!<br>";
                msg += (info.n_bars || 0) + ' תיבות | משקל: ' + (info.meter || []).join(', ') + ' | טמפו: ' + (info.tempo || []).join(', ') + ' | אקורדים: ' + (info.n_chords || 0);
                if (info.beat_engine) msg += ' | פעימות: ' + info.beat_engine;
                if (info.dtw_used === true) msg += ' | יושר ל-MIDI הבסיס (DTW)';
                else if (info.dtw_used === false) msg += ' | יישור DTW נכשל';
                status.innerHTML = msg + '<br>';
                for (const f of (info.formats || [])) {
                    const a = document.createElement('a');
                    a.textContent = labels[f] || f;
                    a.href = '/download/' + jobId + '?fmt=' + f;
                    a.download = base + extOf[f];
                    a.style.cssText = 'display:inline-block;margin:8px 4px 0;padding:8px 14px;border-radius:8px;background:#2563eb;color:#fff;text-decoration:none;font-size:0.9rem';
                    status.appendChild(a);
                }
            } catch (err) {
                status.className = "error";
                status.innerText = "❌ שגיאה: " + (err && err.message ? err.message : err);
            } finally {
                btn.disabled = false;
                btn.style.opacity = "1";
            }
        };
    </script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_CONTENT

JOBS = {}                      # job_id -> {status, stage, error, path, download_name, ...}
JOB_LOCK = threading.Lock()    # one conversion at a time on the GPU


def _set(job_id, **kw):
    JOBS[job_id].update(kw)


def _find_musescore():
    for c in ("mscore4", "musescore4", "mscore3", "musescore3", "mscore", "musescore"):
        p = shutil.which(c)
        if p:
            return p
    return None


def _stem_notes(model, stem_wav, work_prefix, job_id):
    """One isolated stem -> cleaned notes [(pitch,start,end,vel)], pedal intervals, used_fallback."""
    cfg = INSTR[model]
    if model == "drums":
        return nt.transcribe_drums(stem_wav), [], False
    used_fallback, pedal, out_mid = False, [], f"{work_prefix}.mid"
    if model == "piano" and PIANO_MODEL is not None:
        from piano_transcription_inference import sample_rate
        import librosa
        audio, _ = librosa.load(stem_wav, sr=sample_rate)
        PIANO_MODEL.transcribe(audio, out_mid)
    else:
        used_fallback = (model == "piano")
        if used_fallback:
            print("⚠️ מודל הפסנתר לא זמין -- עובר ל-Basic Pitch על הסטם המבודד")
        basic_pitch_to_midi(stem_wav, out_mid, model)
    if not os.path.exists(out_mid):
        raise HTTPException(status_code=500, detail="קובץ ה-MIDI לא נוצר")
    parsed = read_midi_notes_and_map(out_mid)
    if model == "piano" and not used_fallback:
        pedal = parsed["pedal"]
    notes = filter_ghost_notes(parsed["notes"], min_duration=0.06, min_velocity=15, max_duration=4.0)
    notes = [n for n in notes if cfg["lo"] <= n[0] <= cfg["hi"]]
    notes = remove_octave_overtones(notes)
    if cfg["mono"]:
        notes = make_monophonic(notes)
    notes = truncate_sustain_overlaps(notes)
    return notes, pedal, used_fallback


def _process_job(job_id, raw_path, anchor_path, orig_filename, model, mode,
                 manual_key_tonic, manual_key_mode, manual_bpm, split_hands, keep_pedal,
                 add_chords=True, written_pitch=False):
    """Whole pipeline in a background thread (no long HTTP request -> no Cloudflare timeout).
    Demucs -> per-stem transcription -> beats/meter/tempo -> beat-synchronous quantization
    -> staves/voices/keys/chords -> MusicXML + MIDI (+ PDF when MuseScore is installed)."""
    out_base = f"outputs/{job_id}_score"
    sep_dir = f"outputs/{job_id}_sep"
    mix_wav = f"uploads/{job_id}_mix.wav"
    orig_stem = os.path.splitext(orig_filename or "transcribed")[0] or "transcribed"
    try:
        _set(job_id, status="queued", stage="ממתין לתורו...")
        with JOB_LOCK:
            _set(job_id, status="running", stage="מפריד כלים עם Demucs (השלב הארוך, בפעם הראשונה גם יורד מודל)...")
            r = subprocess.run(["demucs", "-n", "htdemucs_6s", "-o", sep_dir, raw_path], capture_output=True, text=True)
            if r.returncode != 0:
                raise HTTPException(status_code=500, detail=f"הפרדת Demucs נכשלה: {r.stderr[-1500:]}")
            base = os.path.splitext(os.path.basename(raw_path))[0]
            stem_dir = os.path.join(sep_dir, "htdemucs_6s", base)
            conv = subprocess.run(["ffmpeg", "-y", "-i", raw_path, "-ar", "44100", "-ac", "1", mix_wav], capture_output=True, text=True)
            if conv.returncode != 0:
                raise HTTPException(status_code=500, detail="המרת המיקס נכשלה")

            kinds = ALL_PARTS if model == "all" else [model]
            parts, used_fallback, pedal_events = [], False, None
            for k_i, kind in enumerate(kinds):
                src_stem = os.path.join(stem_dir, INSTR[kind]["stem"] + ".wav")
                if not os.path.exists(src_stem):
                    raise HTTPException(status_code=500, detail=f"הסטם '{INSTR[kind]['stem']}' לא נמצא אחרי ההפרדה")
                _set(job_id, stage=f"מתמלל {kind} ({k_i + 1}/{len(kinds)})...")
                stem_wav = f"uploads/{job_id}_{kind}_stem.wav"
                subprocess.run(["ffmpeg", "-y", "-i", src_stem, "-ar", "32000", "-ac", "1", stem_wav], capture_output=True, text=True)
                notes, pedal, fb = _stem_notes(kind, stem_wav, f"outputs/{job_id}_{kind}", job_id)
                used_fallback = used_fallback or fb
                if notes:
                    parts.append(dict(kind=kind, notes=notes, pedal=pedal, stem_wav=stem_wav))
            if not parts:
                raise HTTPException(status_code=500, detail="לא נותרו תווים אחרי סינון רעשים -- נסה קובץ אחר")

            _set(job_id, stage="מזהה פעימות, משקל וטמפו...")
            dtw_used, key_notes, to_ref, beat_engine = None, None, None, None
            all_t = [n[1] for p in parts for n in p["notes"]]
            all_e = [n[2] for p in parts for n in p["notes"]]
            if mode == "anchor":
                if not anchor_path:
                    raise HTTPException(status_code=400, detail="מצב 'MIDI בסיס' דורש להעלות קובץ MIDI")
                anchor = read_midi_notes_and_map(anchor_path)
                tempo_map, timesig_map = anchor["tempo_map"], anchor["timesig_map"]
                anchor_notes = [(p, s, e) for p, s, e, v in anchor["notes"]]
                key_notes = anchor_notes
                ref_part = next((p for p in parts if p["kind"] != "drums"), parts[0])
                try:   # chroma DTW: performance time -> anchor timeline
                    to_ref = build_perf_to_ref(anchor["notes"], ref_part["stem_wav"])
                    dtw_used = True
                except Exception:
                    dtw_used = False
                    to_ref = lambda t: t
                    to_ref.inv = lambda t: t
                    traceback.print_exc()
                for p in parts:
                    p["notes"] = [(n[0], to_ref(n[1]), max(to_ref(n[2]), to_ref(n[1]) + 0.03)) + tuple(n[3:]) for n in p["notes"]]
                    p["pedal"] = [(to_ref(a), to_ref(b)) for a, b in p["pedal"]]
                t_end = max(max(n[2] for p in parts for n in p["notes"]), max(a[2] for a in anchor_notes)) + 2.0
                tl = nt.timeline_from_anchor(tempo_map, timesig_map, t_end)
            else:
                tl = None
                if mode == "detect":
                    try:
                        beats, downs, beat_engine = nt.track_beats(mix_wav, "cuda" if torch.cuda.is_available() else "cpu")
                        tl = nt.timeline_from_beats(beats, downs, min(all_t), max(all_e), all_t)
                    except Exception:
                        traceback.print_exc()
                        print("⚠️ זיהוי פעימות נכשל -- עובר לטמפו קבוע")
                if tl is None:
                    bpm = float(manual_bpm) if mode == "manual" else detect_bpm_from_audio(mix_wav)
                    step = 60.0 / bpm
                    beats = np.arange(0.0, max(all_e) + 4 * step, step)
                    tl = nt.timeline_from_beats(beats, beats[::4], min(all_t), max(all_e), all_t)

            _set(job_id, stage="מצמיד לסולמות, מיישר ומקוונטז...")
            key_override = None
            if mode == "manual":
                tonic, is_minor = int(manual_key_tonic), (manual_key_mode == "minor")
                key_override = (nt._fifths(tonic if not is_minor else (tonic + 3) % 12), is_minor, tonic)
            for p in parts:
                if p["kind"] == "drums":
                    continue
                if mode == "manual":
                    p["notes"] = snap_to_key(p["notes"], tonic, is_minor)
                else:
                    p["notes"] = snap_to_key_dynamic(p["notes"], key_notes)
                p["notes"] = smooth_velocities(p["notes"])
                if mode == "anchor":
                    bounds = measure_boundaries(build_segments(tempo_map, PPQ_OUT), timesig_map, max(n[2] for n in p["notes"]) + 4.0)
                    p["notes"], _ = refine_notes_to_anchor(p["notes"], anchor_notes, bounds)
                if p["kind"] == "piano" and keep_pedal and p["pedal"]:
                    p["pedal"] = [e for a, b in p["pedal"] for e in ((a, 127), (b, 0))]
                else:
                    p["pedal"] = None

            chords = []
            if add_chords:
                _set(job_id, stage="מזהה אקורדים...")
                try:
                    tl_audio = tl
                    if to_ref is not None:                     # anchor mode: score time -> performance time
                        tl_audio = copy.copy(tl)
                        tl_audio.pt = np.asarray(to_ref.inv(tl.pt), float)
                    bass_wav = os.path.join(stem_dir, "bass.wav")
                    chords = nt.detect_chords(mix_wav, bass_wav, tl_audio)
                except Exception:
                    traceback.print_exc()
                    print("⚠️ זיהוי אקורדים נכשל -- ממשיך בלי אקורדים")

            _set(job_id, stage="בונה תווים (MusicXML) ו-MIDI...")
            if model != "all" and model == "piano" and not split_hands:
                pass   # piano keeps its grand staff (two staves) in the score; the MIDI hand split is always dynamic
            res = nt.engrave([dict(kind=p["kind"], notes=p["notes"], pedal=p["pedal"]) for p in parts], tl, chords,
                             out_base, title=orig_stem, written_pitch=bool(written_pitch), key_override=key_override)
            files = {"midi": res["midi"], "xml": res["xml"]}
            ms = _find_musescore()
            if ms:
                try:
                    pdf = out_base + ".pdf"
                    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
                    subprocess.run([ms, "-o", pdf, res["xml"]], capture_output=True, text=True, timeout=240, env=env)
                    if os.path.exists(pdf):
                        files["pdf"] = pdf
                except Exception:
                    traceback.print_exc()
            _set(job_id, status="done", stage="הושלם", files=files, download_name=orig_stem,
                 used_fallback=used_fallback, dtw_used=dtw_used, beat_engine=beat_engine,
                 n_bars=len(tl.bars), n_chords=len(chords),
                 meter=sorted({f"{b['num']}/{b['den']}" for b in tl.bars}),
                 tempo=[round(b) for _, b in tl.tempos][:6])
    except HTTPException as e:
        print(f"❌ משימה {job_id} נכשלה: {e.detail}")
        _set(job_id, status="error", error=str(e.detail))
    except Exception as err:
        traceback.print_exc()
        _set(job_id, status="error", error=f"{type(err).__name__}: {err}")
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("piano"),
    mode: str = Form("detect"),
    manual_key_tonic: int = Form(0),
    manual_key_mode: str = Form("major"),
    manual_bpm: float = Form(120.0),
    split_hands: bool = Form(False),
    keep_pedal: bool = Form(False),
    add_chords: bool = Form(True),
    written_pitch: bool = Form(False),
    anchor_midi: UploadFile = File(None),
):
    """Saves the uploads, starts the pipeline in the background and answers immediately."""
    if model not in INSTR and model != "all":
        raise HTTPException(status_code=400, detail=f"כלי לא מוכר: {model}")
    job_id = uuid.uuid4().hex[:8]
    src_ext = os.path.splitext(file.filename or "")[1] or ".mp3"
    raw_path = f"uploads/{job_id}_raw{src_ext}"
    with open(raw_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    anchor_path = None
    if mode == "anchor":
        if anchor_midi is None:
            raise HTTPException(status_code=400, detail="מצב 'MIDI בסיס' דורש להעלות קובץ MIDI")
        anchor_path = f"uploads/{job_id}_anchor.mid"
        with open(anchor_path, "wb") as f:
            f.write(await anchor_midi.read())
    JOBS[job_id] = {"status": "queued", "stage": "התקבל", "error": None}
    threading.Thread(
        target=_process_job,
        args=(job_id, raw_path, anchor_path, file.filename, model, mode,
              manual_key_tonic, manual_key_mode, manual_bpm, split_hands, keep_pedal, add_chords, written_pitch),
        daemon=True,
    ).start()
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="המשימה לא נמצאה (ייתכן שהשרת הופעל מחדש)")
    out = {k: job.get(k) for k in ("status", "stage", "error", "download_name", "used_fallback", "dtw_used", "beat_engine", "n_bars", "n_chords", "meter", "tempo")}
    out["formats"] = list((job.get("files") or {}).keys())
    return out


@app.get("/download/{job_id}")
def job_download(job_id: str, fmt: str = "midi"):
    job = JOBS.get(job_id)
    if job is None or job.get("status") != "done":
        raise HTTPException(status_code=404, detail="הקובץ עדיין לא מוכן או שהמשימה לא נמצאה")
    path = (job.get("files") or {}).get(fmt)
    if not path:
        raise HTTPException(status_code=404, detail=f"הפורמט {fmt} לא זמין")
    ext = {"midi": ".mid", "xml": ".musicxml", "pdf": ".pdf"}[fmt]
    mt = {"midi": "audio/midi", "xml": "application/vnd.recordare.musicxml+xml", "pdf": "application/pdf"}[fmt]
    name = job["download_name"] + ext
    resp = FileResponse(path, filename=name, media_type=mt)
    resp.headers["X-Download-Name"] = urllib.parse.quote(name)
    return resp
