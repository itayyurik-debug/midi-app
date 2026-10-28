from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import io
import shutil
import subprocess
import torch
import gc
import mido

app = FastAPI(title="Audio to MIDI Pipeline")

# ---------- key detection + tempo-map quantization (shared with the local-pipeline project) ----------
MAJ_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MIN_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
PPQ_OUT = 480


def read_midi_notes_and_map(path):
    """Read any standard MIDI file -> full tempo map, time-sig map, notes as (pitch, start_sec, end_sec)."""
    mid = mido.MidiFile(str(path))
    ppq = mid.ticks_per_beat
    tempo_events, timesig_events, notes = [], [], []
    for track in mid.tracks:
        time = 0
        active = {}
        for msg in track:
            time += msg.time
            if msg.type == "set_tempo":
                tempo_events.append((time, msg.tempo))
            elif msg.type == "time_signature":
                timesig_events.append((time, msg.numerator, msg.denominator))
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

    tempo_map = [(tick_to_sec(t), round(60e6 / us)) for t, us in tempo_events]
    timesig_map = [(tick_to_sec(t), n, d) for t, n, d in timesig_events]
    note_list = [(p, tick_to_sec(s), tick_to_sec(e), v) for p, s, e, v in notes]
    return {"tempo_map": tempo_map, "timesig_map": timesig_map, "notes": note_list}


def filter_ghost_notes(notes_with_vel, min_duration=0.03, min_velocity=15, max_duration=4.0):
    """Drop too-short / too-quiet notes (ghost notes), and cap note length to tame
    sustain-pedal smearing. Returns plain (pitch, start, end) tuples (velocity dropped
    after filtering -- the output MIDI always writes a fixed velocity)."""
    out = []
    for pitch, start, end, vel in notes_with_vel:
        dur = end - start
        if dur < min_duration or vel < min_velocity:
            continue
        if dur > max_duration:
            end = start + max_duration
        out.append((pitch, start, end))
    return out


def remove_octave_overtones(notes, time_tolerance=0.05):
    """Drop a note that starts almost exactly when another note starts one or two
    octaves below it and lasts no longer -- a common false-positive when a bass
    note's harmonic resonance gets mis-detected as a separate note."""
    out = []
    for i, (pitch, start, end) in enumerate(notes):
        is_overtone = False
        for j, (pitch2, start2, end2) in enumerate(notes):
            if i == j or pitch2 == pitch:
                continue
            if (pitch - pitch2) in (12, 24) and abs(start - start2) <= time_tolerance and (end - start) <= (end2 - start2) + time_tolerance:
                is_overtone = True
                break
        if not is_overtone:
            out.append((pitch, start, end))
    return out


# ---------- chroma DTW: only used when a reference MIDI is uploaded, to align its
# tempo/structure to the ACTUAL timing of the performance (tempo drift, rubato, etc.) ----------
def _audio_chroma(wav_path, hop_length=512, sr_target=22050):
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


def align_reference_to_performance(anchor_notes4, anchor_tempo_map, anchor_timesig_map, perf_wav_path):
    """Returns (warped_tempo_map, warped_timesig_map, warped_notes3) with every time
    value remapped from the reference MIDI's own timeline onto the real timing of
    perf_wav_path, via chroma + DTW. Raises on failure -- caller should catch and
    fall back to the unwarped anchor data rather than block the user."""
    import librosa
    chroma_perf, hop_length, sr = _audio_chroma(perf_wav_path)
    ref_duration = max((e for _, _, e, *_ in anchor_notes4), default=1.0)
    chroma_ref = _midi_notes_to_chroma(anchor_notes4, ref_duration, hop_length, sr)
    _, wp = librosa.sequence.dtw(X=chroma_ref, Y=chroma_perf, metric="cosine")
    frame_time = hop_length / sr
    path = sorted(((r * frame_time, p * frame_time) for r, p in wp), key=lambda x: x[0])

    def warp(t_ref):
        if t_ref <= path[0][0]:
            return path[0][1]
        if t_ref >= path[-1][0]:
            return path[-1][1]
        for i in range(1, len(path)):
            r0, p0 = path[i - 1]
            r1, p1 = path[i]
            if r0 <= t_ref <= r1:
                return p0 if r1 == r0 else p0 + (t_ref - r0) / (r1 - r0) * (p1 - p0)
        return path[-1][1]

    warped_tempo_map = [(warp(t), bpm) for t, bpm in anchor_tempo_map]
    warped_timesig_map = [(warp(t), n, d) for t, n, d in anchor_timesig_map]
    warped_notes = [(p, warp(s), warp(e)) for p, s, e, v in anchor_notes4]
    return warped_tempo_map, warped_timesig_map, warped_notes


def detect_bpm_from_audio(wav_path):
    """Audio-only tempo estimate, for the 'auto-detect' mode when no anchor MIDI is given."""
    import numpy as np
    import librosa
    y, sr = librosa.load(wav_path, sr=None)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(np.atleast_1d(tempo)[0])


def detect_key(notes):
    hist = [0.0] * 12
    for pitch, start, end in notes:
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
    for pitch, start, end in notes:
        pc = pitch % 12
        if pc in scale:
            out.append((pitch, start, end))
            continue
        best_s, best_d = pc, 99
        for s in scale:
            d = min(abs(s - pc), 12 - abs(s - pc))
            if d < best_d:
                best_d, best_s = d, s
        new_pitch = pitch + ((best_s - pc + 18) % 12 - 6) if best_d <= 1 else pitch
        out.append((new_pitch, start, end))
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


def quantize_notes(bp_notes, tempo_map, anchor_notes):
    segs = build_segments(tempo_map, PPQ_OUT)
    grid = PPQ_OUT // 4
    phase = 0
    if anchor_notes:
        offsets = sorted(int(sec_to_tick(s, segs)) % grid for _, s, _ in anchor_notes)
        phase = offsets[len(offsets) // 2]
    out = []
    for pitch, start, end in bp_notes:
        s_tick = sec_to_tick(start, segs)
        e_tick = sec_to_tick(end, segs)
        q_s = round((s_tick - phase) / grid) * grid + phase
        q_d = max(grid, round((e_tick - s_tick) / grid) * grid)
        out.append((pitch, max(0, int(q_s)), int(q_d)))
    return out, segs


def _notes_track(notes, channel=0):
    tr = mido.MidiTrack()
    evs = []
    for pitch, s_tick, d_tick in notes:
        evs.append((s_tick, mido.Message("note_on", note=int(pitch), velocity=90, channel=channel)))
        evs.append((s_tick + d_tick, mido.Message("note_off", note=int(pitch), velocity=0, channel=channel)))
    evs.sort(key=lambda e: e[0])
    last_t = 0
    for tick, msg in evs:
        msg.time = max(0, round(tick - last_t))
        tr.append(msg)
        last_t = max(last_t, tick)
    tr.append(mido.MetaMessage("end_of_track"))
    return tr


def build_output_midi_bytes(quantized_notes, tempo_map, timesig_map, segs, split_hands=False):
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
        # Right hand (treble): pitch >= 60 (middle C). Left hand (bass): below.
        out_mid.tracks.append(_notes_track([n for n in quantized_notes if n[0] >= 60], channel=0))
        out_mid.tracks.append(_notes_track([n for n in quantized_notes if n[0] < 60], channel=1))
    else:
        out_mid.tracks.append(_notes_track(quantized_notes, channel=0))
    buf = io.BytesIO()
    out_mid.save(file=buf)
    return buf.getvalue()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

HTML_CONTENT = """<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ממיר שמע ל-MIDI</title>
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
        <h1>🎹 ממיר שמע ל-MIDI</h1>
        <p>העלה קובץ MP3 או שמע מהאייפד והמר אותו לקובץ MIDI נקי</p>
        <form id="transcribeForm">
            <div class="upload-box" onclick="document.getElementById('audioFile').click()">
                <span id="fileName">📁 לחץ כאן לבחירת קובץ MP3</span>
                <input type="file" id="audioFile" name="file" accept="audio/*,.mp3,.wav,.m4a" required onchange="updateFileName(this)">
            </div>
            <select name="model" id="modelSelect">
                <option value="piano">🎹 פסנתר (Demucs + Piano Transcription)</option>
                <option value="bass">🎸 בס (Demucs + Basic Pitch)</option>
                <option value="guitar">🎸 גיטרה (Demucs + Basic Pitch)</option>
                <option value="vocals">🎤 שירה (Demucs + Basic Pitch)</option>
                <option value="other">🎻 שאר הכלים (סטם "other" של Demucs + Basic Pitch)</option>
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

            <button type="submit" id="submitBtn">המר ל-MIDI 🚀</button>
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
            document.getElementById('splitRow').style.display = (this.value === 'piano') ? 'flex' : 'none';
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
            status.innerText = "⏳ מעבד שמע ב-GPU... ייקח כ-3 שניות";
            btn.disabled = true;
            btn.style.opacity = "0.5";

            const formData = new FormData();
            formData.append('file', document.getElementById('audioFile').files[0]);
            formData.append('model', document.getElementById('modelSelect').value);
            formData.append('mode', mode);
            formData.append('split_hands', document.getElementById('splitHands').checked);
            if (mode === 'manual') {
                formData.append('manual_key_tonic', document.getElementById('keyTonic').value);
                formData.append('manual_key_mode', document.getElementById('keyMode').value);
                formData.append('manual_bpm', document.getElementById('manualBpm').value);
            } else if (mode === 'anchor') {
                formData.append('anchor_midi', document.getElementById('anchorFile').files[0]);
            }

            try {
                const res = await fetch('/transcribe', { method: 'POST', body: formData });
                if (res.ok) {
                    const usedFallback = res.headers.get('X-Used-Fallback');
                    const dtwUsed = res.headers.get('X-DTW-Used');
                    const disp = res.headers.get('Content-Disposition') || '';
                    const m = disp.match(/filename="?([^"]+)"?/);
                    const outName = m ? m[1] : 'transcribed.mid';
                    const blob = await res.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = outName;
                    a.click();
                    status.className = usedFallback ? "loading" : "success";
                    let msg = usedFallback
                        ? "⚠️ מודל הפסנתר הייעודי לא היה זמין -- נעשה שימוש ב-Basic Pitch במקום."
                        : "✅ ההמרה הושלמה!";
                    if (dtwUsed === "1") msg += " (יושר מול ה-MIDI הבסיס עם DTW)";
                    else if (dtwUsed === "0") msg += " (יישור DTW נכשל, נעשה שימוש ב-MIDI הבסיס הלא-מיושר)";
                    status.innerText = msg + " " + outName + " ירד.";
                } else {
                    const errData = await res.json();
                    status.className = "error";
                    status.innerText = "❌ שגיאה: " + (errData.detail || "עיבוד השמע נכשל");
                }
            } catch (err) {
                status.className = "error";
                status.innerText = "❌ שגיאה בהתקשרות לשרת.";
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

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("piano"),
    mode: str = Form("detect"),
    manual_key_tonic: int = Form(0),
    manual_key_mode: str = Form("major"),
    manual_bpm: float = Form(120.0),
    split_hands: bool = Form(False),
    anchor_midi: UploadFile = File(None),
):
    import uuid
    job_id = uuid.uuid4().hex[:8]
    src_ext = os.path.splitext(file.filename or "")[1] or ".mp3"
    raw_path = f"uploads/{job_id}_raw{src_ext}"
    stem_wav_path = f"uploads/{job_id}_stem.wav"
    output_midi = f"outputs/{job_id}.mid"
    basic_pitch_out = f"outputs/{job_id}_stem_basic_pitch.mid"
    final_midi = f"outputs/{job_id}_final.mid"
    sep_dir = f"outputs/{job_id}_sep"

    DEMUCS_STEMS = {"piano", "bass", "guitar", "vocals", "other"}
    if model not in DEMUCS_STEMS:
        raise HTTPException(status_code=400, detail=f"כלי לא מוכר: {model}")

    # output filename follows the original upload's name, swapped to .mid
    orig_stem = os.path.splitext(file.filename or "transcribed")[0] or "transcribed"
    download_name = f"{orig_stem}.mid"

    try:
        with open(raw_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # ---- Demucs: real stem separation on the ORIGINAL full-quality audio
        # (not downsampled first -- that would hurt separation quality) ----
        demucs_result = subprocess.run(
            ["demucs", "-n", "htdemucs_6s", "-o", sep_dir, raw_path],
            capture_output=True, text=True,
        )
        if demucs_result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"הפרדת Demucs נכשלה: {demucs_result.stderr[-1500:]}")

        raw_stem_name = os.path.splitext(os.path.basename(raw_path))[0]
        stem_source = os.path.join(sep_dir, "htdemucs_6s", raw_stem_name, f"{model}.wav")
        if not os.path.exists(stem_source):
            raise HTTPException(status_code=500, detail=f"הסטם '{model}' לא נמצא אחרי ההפרדה")

        # המרה ל-WAV תקני -- על הסטם המבודד, לא על כל התערובת
        conv = subprocess.run(["ffmpeg", "-y", "-i", stem_source, "-ar", "32000", "-ac", "1", stem_wav_path], capture_output=True, text=True)
        if conv.returncode != 0:
            raise HTTPException(status_code=500, detail="המרה ראשונית נכשלה")

        used_fallback = False
        if model == "piano" and PIANO_MODEL is not None:
            from piano_transcription_inference import sample_rate
            import librosa
            audio, _ = librosa.load(stem_wav_path, sr=sample_rate)
            PIANO_MODEL.transcribe(audio, output_midi)
        else:
            if model == "piano":
                used_fallback = True
                print("⚠️ מודל הפסנתר לא זמין -- עובר ל-Basic Pitch על הסטם המבודד")
            from basic_pitch.inference import predict_and_save
            predict_and_save([stem_wav_path], "outputs", True, False, False, False)
            output_midi = basic_pitch_out

        if not os.path.exists(output_midi):
            raise HTTPException(status_code=500, detail="קובץ ה-MIDI לא נוצר")

        # ---- ghost-note cleanup: drop too-short / too-quiet notes, cap pedal-smeared length ----
        raw_notes_vel = read_midi_notes_and_map(output_midi)["notes"]
        raw_notes = filter_ghost_notes(raw_notes_vel, min_duration=0.06, min_velocity=15, max_duration=4.0)
        raw_notes = remove_octave_overtones(raw_notes)
        if not raw_notes:
            raise HTTPException(status_code=500, detail="לא נותרו תווים אחרי סינון רעשים -- נסה קובץ אחר")

        # ---- key + tempo: auto-detected, set manually, or read from an anchor/reference MIDI ----
        dtw_used = False
        if mode == "anchor":
            if anchor_midi is None:
                raise HTTPException(status_code=400, detail="מצב 'MIDI בסיס' דורש להעלות קובץ MIDI")
            anchor_bytes = await anchor_midi.read()
            anchor_path = f"uploads/{job_id}_anchor.mid"
            with open(anchor_path, "wb") as f:
                f.write(anchor_bytes)
            anchor = read_midi_notes_and_map(anchor_path)
            anchor_notes3 = [(p, s, e) for p, s, e, v in anchor["notes"]]
            tonic, is_minor = detect_key(anchor_notes3)
            tempo_map, timesig_map, anchor_notes = anchor["tempo_map"], anchor["timesig_map"], anchor_notes3
            # Chroma DTW: warp the reference's tempo/structure onto the performance's
            # actual timing (handles tempo drift/rubato). Falls back to the plain
            # (unwarped) anchor data -- never blocks the user if DTW itself fails.
            try:
                tempo_map, timesig_map, anchor_notes = align_reference_to_performance(
                    anchor["notes"], anchor["tempo_map"], anchor["timesig_map"], stem_wav_path
                )
                dtw_used = True
            except Exception:
                import traceback
                print("⚠️ יישור Chroma DTW נכשל -- ממשיך עם ה-MIDI הבסיס הלא-מיושר:")
                traceback.print_exc()
        elif mode == "manual":
            tonic, is_minor = int(manual_key_tonic), (manual_key_mode == "minor")
            tempo_map, timesig_map, anchor_notes = [(0, manual_bpm)], [(0, 4, 4)], []
        else:  # "detect": key from the transcription itself, BPM from the audio
            tonic, is_minor = detect_key(raw_notes)
            detected_bpm = detect_bpm_from_audio(raw_path)  # full mix tracks tempo more reliably than an isolated stem
            tempo_map, timesig_map, anchor_notes = [(0, detected_bpm)], [(0, 4, 4)], []

        snapped = snap_to_key(raw_notes, tonic, is_minor)
        quantized, segs = quantize_notes(snapped, tempo_map, anchor_notes)
        final_bytes = build_output_midi_bytes(quantized, tempo_map, timesig_map, segs, split_hands=bool(split_hands))
        with open(final_midi, "wb") as f:
            f.write(final_bytes)

        resp = FileResponse(final_midi, filename=download_name, media_type="audio/midi")
        if used_fallback:
            resp.headers["X-Used-Fallback"] = "basic_pitch_instead_of_piano_model"
        if mode == "anchor":
            resp.headers["X-DTW-Used"] = "1" if dtw_used else "0"
        return resp
    except HTTPException:
        raise
    except Exception as err:
        print(f"❌ שגיאה בזמן המרה: {err}")
        raise HTTPException(status_code=500, detail=str(err))
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
