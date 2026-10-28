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

MAJ_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MIN_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
PPQ_OUT = 480

# ==========================================
# 1. MIDI PARSING & UTILITIES
# ==========================================
def read_midi_notes_and_map(path):
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
    
    if not tempo_events: tempo_events = [(0, 500000)]
    tempo_events.sort()
    if tempo_events[0][0] > 0: tempo_events.insert(0, (0, tempo_events[0][1]))
    if not timesig_events: timesig_events = [(0, 4, 4)]
    timesig_events.sort()
    if timesig_events[0][0] > 0: timesig_events.insert(0, (0, timesig_events[0][1], timesig_events[0][2]))

    def tick_to_sec(tick):
        sec = 0.0
        for i, (t_start, us) in enumerate(tempo_events):
            t_end = tempo_events[i + 1][0] if i + 1 < len(tempo_events) else float("inf")
            if tick <= t_start: break
            sec += (min(tick, t_end) - t_start) * (us / 1e6) / ppq
            if tick <= t_end: break
        return sec

    return {
        "tempo_map": [(tick_to_sec(t), round(60e6/us)) for t, us in tempo_events],
        "timesig_map": [(tick_to_sec(t), n, d) for t, n, d in timesig_events],
        "notes": [(p, tick_to_sec(s), tick_to_sec(e), v) for p, s, e, v in notes]
    }

# ==========================================
# 2. CLEANUP, CHORDS & QUANTIZATION
# ==========================================
def filter_noise(notes, min_duration=0.035, min_velocity=12):
    return [(p, s, e, v) for p, s, e, v in notes if (e - s) >= min_duration and v >= min_velocity]

def merge_micro_gaps(notes, max_gap_sec=0.065):
    if not notes: return []
    sorted_notes = sorted(notes, key=lambda x: (x[0], x[1]))
    merged = []
    curr_p, curr_s, curr_e, curr_v = sorted_notes[0]
    for n_p, n_s, n_e, n_v in sorted_notes[1:]:
        if n_p == curr_p and (n_s - curr_e) <= max_gap_sec:
            curr_e = max(curr_e, n_e)
            curr_v = max(curr_v, n_v)
        else:
            merged.append((curr_p, curr_s, curr_e, curr_v))
            curr_p, curr_s, curr_e, curr_v = n_p, n_s, n_e, n_v
    merged.append((curr_p, curr_s, curr_e, curr_v))
    return sorted(merged, key=lambda x: x[1])

def quantize_for_sheet_music(notes, tempo_map, grid_division=16):
    bpm = tempo_map[0][1] if tempo_map else 120.0
    grid_ticks = PPQ_OUT // (grid_division // 4)
    sec_per_tick = 60.0 / (bpm * PPQ_OUT)
    quantized = []
    for pitch, start_sec, end_sec, vel in notes:
        s_tick = int(start_sec / sec_per_tick)
        e_tick = int(end_sec / sec_per_tick)
        q_start = round(s_tick / grid_ticks) * grid_ticks
        q_end = round(e_tick / grid_ticks) * grid_ticks
        if q_end <= q_start: q_end = q_start + grid_ticks
        quantized.append((pitch, max(0, q_start), q_end - q_start, vel))
    return quantized

def detect_chords(quantized_notes):
    beats = {}
    for pitch, s_tick, d_tick, vel in quantized_notes:
        beat_idx = s_tick // PPQ_OUT
        if beat_idx not in beats: beats[beat_idx] = []
        beats[beat_idx].append(pitch % 12)
    
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    chord_markers = []
    for beat, pitches in beats.items():
        unique_pitches = list(set(pitches))
        if len(unique_pitches) >= 3:
            root = unique_pitches[0]
            chord_markers.append((beat * PPQ_OUT, mido.MetaMessage('text', text=note_names[root])))
    return chord_markers

def detect_bpm_from_audio(wav_path):
    import librosa, numpy as np
    y, sr = librosa.load(wav_path, sr=None)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(np.atleast_1d(tempo)[0])

def detect_key(notes):
    hist = [0.0] * 12
    for p, s, e, _ in notes:
        hist[p % 12] += max(0.05, e - s)
    best = (-1e18, 0, False)
    for tonic in range(12):
        for profile, is_minor in ((MAJ_PROFILE, False), (MIN_PROFILE, True)):
            score = sum(hist[(i + tonic) % 12] * profile[i] for i in range(12))
            if score > best[0]: best = (score, tonic, is_minor)
    return best[1], best[2]

# ==========================================
# 3. BUILD OUTPUT MIDI
# ==========================================
def build_output_midi_bytes(notes, tempo_map, timesig_map, is_quantized=True, add_chords=False):
    out_mid = mido.MidiFile(ticks_per_beat=PPQ_OUT)
    meta_track = mido.MidiTrack()
    
    bpm = tempo_map[0][1] if tempo_map else 120.0
    meta_track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    meta_track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))

    if is_quantized and add_chords:
        chords = detect_chords(notes)
        last_t = 0
        for tick, msg in chords:
            msg.time = max(0, tick - last_t)
            meta_track.append(msg)
            last_t = max(last_t, tick)

    out_mid.tracks.append(meta_track)

    tr = mido.MidiTrack()
    evs = []
    sec_per_tick = 60.0 / (bpm * PPQ_OUT)

    for p, s, d_or_e, v in notes:
        if is_quantized:
            s_tick = s
            d_tick = d_or_e
        else:
            s_tick = int(s / sec_per_tick)
            d_tick = int((d_or_e - s) / sec_per_tick)
        evs.append((s_tick, mido.Message("note_on", note=int(p), velocity=int(v), channel=0)))
        evs.append((s_tick + d_tick, mido.Message("note_off", note=int(p), velocity=0, channel=0)))

    evs.sort(key=lambda e: e[0])
    last_t = 0
    for tick, msg in evs:
        msg.time = max(0, round(tick - last_t))
        tr.append(msg)
        last_t = max(last_t, tick)
    tr.append(mido.MetaMessage("end_of_track"))
    out_mid.tracks.append(tr)

    buf = io.BytesIO()
    out_mid.save(file=buf)
    return buf.getvalue()

# ==========================================
# 4. FASTAPI SERVER & UI
# ==========================================
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PIANO_MODEL = None
try:
    from piano_transcription_inference import PianoTranscription
    PIANO_MODEL = PianoTranscription(device=DEVICE)
except Exception:
    pass

HTML_CONTENT = """<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
    <meta charset="UTF-8">
    <title>ממיר שמע ל-MIDI</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
        .card { background: #1e293bcc; backdrop-filter: blur(6px); border-radius: 16px; padding: 28px; max-width: 480px; width: 100%; text-align: center; border: 1px solid #334155; }
        h1 { font-size: 1.5rem; color: #38bdf8; margin-bottom: 8px; }
        p { color: #94a3b8; font-size: 0.88rem; margin-bottom: 20px; }
        .upload-box { border: 2px dashed #0284c7; border-radius: 12px; padding: 20px; cursor: pointer; background: #0f172a; margin-bottom: 16px; }
        input[type="file"] { display: none; }
        select, button, input[type="number"] { width: 100%; padding: 12px; border-radius: 8px; border: 1px solid #475569; margin-top: 10px; font-size: 0.95rem; box-sizing: border-box; }
        select, input { background: #0f172a; color: white; }
        button { background: #0284c7; color: white; border: none; cursor: pointer; font-weight: bold; margin-top: 18px; }
        #status { margin-top: 16px; font-size: 0.9rem; font-weight: bold; }
        .modebar { display: flex; gap: 8px; margin-top: 14px; }
        .modebar label { flex: 1; padding: 8px; border: 1px solid #475569; border-radius: 8px; font-size: 0.8rem; cursor: pointer; background: #0f172a; }
        .modebar input { display: none; }
        .modebar label:has(input:checked) { border-color: #38bdf8; color: #38bdf8; font-weight: bold; }
        .modepane { display: none; text-align: right; margin-top: 10px; }
        .modepane.active { display: block; }
    </style>
</head>
<body>
    <div class="card">
        <h1>🎹 ממיר שמע ל-MIDI</h1>
        <p>תמלול שמע מקצועי עם יישור לתווים וזיהוי אקורדים</p>
        <form id="transcribeForm">
            <div class="upload-box" onclick="document.getElementById('audioFile').click()">
                <span id="fileName">📁 לחץ לבחירת קובץ MP3 / WAV</span>
                <input type="file" id="audioFile" required onchange="document.getElementById('fileName').innerText = '📄 ' + this.files[0].name">
            </div>

            <select id="modelSelect">
                <option value="piano">🎹 פסנתר (Demucs + Piano Engine)</option>
                <option value="bass">🎸 בס (Demucs + Basic Pitch)</option>
                <option value="guitar">🎸 גיטרה (Demucs + Basic Pitch)</option>
                <option value="vocals">🎤 שירה (Demucs + Basic Pitch)</option>
                <option value="other">🎻 שאר הכלים (Demucs + Basic Pitch)</option>
            </select>

            <select id="outputMode">
                <option value="sheet">🎼 מיושר לתווים (יישור לקצב וסיומת נקייה)</option>
                <option value="sheet_chords">🎼 מיושר לתווים + זיהוי אקורדים</option>
                <option value="raw">⚡ MIDI גולמי (ללא יישור לקצב)</option>
            </select>

            <div class="modebar">
                <label><input type="radio" name="mode" value="detect" checked onchange="setMode('detect')"><span>זיהוי אוטומטי</span></label>
                <label><input type="radio" name="mode" value="manual" onchange="setMode('manual')"><span>קביעה ידנית</span></label>
                <label><input type="radio" name="mode" value="anchor" onchange="setMode('anchor')"><span>MIDI בסיס</span></label>
            </div>

            <div class="modepane active" id="pane-detect">
                <p style="color:#64748b;font-size:0.75rem;">זיהוי מקצב וסולם אוטומטי מהקובץ.</p>
            </div>

            <div class="modepane" id="pane-manual">
                <label style="font-size:0.8rem;color:#94a3b8">קצב (BPM)</label>
                <input type="number" id="manualBpm" value="120">
            </div>

            <div class="modepane" id="pane-anchor">
                <label style="font-size:0.8rem;color:#94a3b8">קובץ MIDI בסיס לסינכרון</label>
                <input type="file" id="anchorFile" accept=".mid,.midi">
            </div>

            <button type="submit" id="submitBtn">המר ל-MIDI 🚀</button>
        </form>
        <div id="status"></div>
    </div>

    <script>
        function setMode(m) {
            document.getElementById('pane-detect').classList.toggle('active', m === 'detect');
            document.getElementById('pane-manual').classList.toggle('active', m === 'manual');
            document.getElementById('pane-anchor').classList.toggle('active', m === 'anchor');
        }

        document.getElementById('transcribeForm').onsubmit = async (e) => {
            e.preventDefault();
            const btn = document.getElementById('submitBtn');
            const status = document.getElementById('status');
            const fileInput = document.getElementById('audioFile');

            btn.disabled = true;
            status.style.color = "#facc15";
            status.innerText = "⏳ מעבד שמע...";

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            formData.append('model', document.getElementById('modelSelect').value);
            formData.append('output_mode', document.getElementById('outputMode').value);
            
            const mode = document.querySelector('input[name="mode"]:checked').value;
            formData.append('mode', mode);
            if (mode === 'manual') {
                formData.append('manual_bpm', document.getElementById('manualBpm').value);
            } else if (mode === 'anchor' && document.getElementById('anchorFile').files.length) {
                formData.append('reference_midi', document.getElementById('anchorFile').files[0]);
            }

            const baseName = fileInput.files[0].name.split('.').slice(0, -1).join('.') || 'audio';

            try {
                const res = await fetch('/transcribe', { method: 'POST', body: formData });
                if (res.ok) {
                    const blob = await res.blob();
                    const a = document.createElement('a');
                    a.href = window.URL.createObjectURL(blob);
                    a.download = baseName + ".mid";
                    a.click();
                    status.style.color = "#4ade80";
                    status.innerText = "✅ ירד בהצלחה: " + baseName + ".mid";
                } else {
                    status.style.color = "#f87171";
                    status.innerText = "❌ שגיאה בזמן העיבוד";
                }
            } catch (err) {
                status.style.color = "#f87171";
                status.innerText = "❌ שגיאה בהתקשרות לשרת";
            } finally {
                btn.disabled = false;
            }
        };
    </script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
def index(): return HTML_CONTENT

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("piano"),
    output_mode: str = Form("sheet"),
    mode: str = Form("detect"),
    manual_bpm: float = Form(120.0),
    reference_midi: UploadFile = File(None)
):
    import uuid
    job_id = uuid.uuid4().hex[:8]
    src_ext = os.path.splitext(file.filename or "")[1] or ".mp3"
    raw_path = f"uploads/{job_id}_raw{src_ext}"
    stem_wav_path = f"uploads/{job_id}_stem.wav"
    output_midi = f"outputs/{job_id}.mid"
    final_midi = f"outputs/{job_id}_final.mid"
    sep_dir = f"outputs/{job_id}_sep"

    try:
        with open(raw_path, "wb") as f: shutil.copyfileobj(file.file, f)

        # 1. Demucs Stem Isolation
        subprocess.run(["demucs", "-n", "htdemucs_6s", "-o", sep_dir, raw_path], capture_output=True)
        raw_stem_name = os.path.splitext(os.path.basename(raw_path))[0]
        stem_source = os.path.join(sep_dir, "htdemucs_6s", raw_stem_name, f"{model}.wav")

        if not os.path.exists(stem_source):
            subprocess.run(["ffmpeg", "-y", "-i", raw_path, "-ar", "32000", "-ac", "1", stem_wav_path], capture_output=True)
        else:
            subprocess.run(["ffmpeg", "-y", "-i", stem_source, "-ar", "32000", "-ac", "1", stem_wav_path], capture_output=True)

        # 2. Transcription
        if model == "piano" and PIANO_MODEL is not None:
            from piano_transcription_inference import sample_rate
            import librosa
            audio, _ = librosa.load(stem_wav_path, sr=sample_rate)
            PIANO_MODEL.transcribe(audio, output_midi)
        else:
            from basic_pitch.inference import predict_and_save
            predict_and_save([stem_wav_path], "outputs", True, False, False, False)
            output_midi = f"outputs/{job_id}_stem_basic_pitch.mid"

        # 3. Reading Notes & BPM
        raw_notes = read_midi_notes_and_map(output_midi)["notes"]
        
        if mode == "manual":
            bpm = manual_bpm
        elif mode == "anchor" and reference_midi is not None:
            ref_bytes = await reference_midi.read()
            ref_path = f"uploads/{job_id}_ref.mid"
            with open(ref_path, "wb") as f: f.write(ref_bytes)
            ref_map = read_midi_notes_and_map(ref_path)
            bpm = ref_map["tempo_map"][0][1] if ref_map["tempo_map"] else 120.0
        else:
            bpm = detect_bpm_from_audio(raw_path)

        tempo_map = [(0, bpm)]
        timesig_map = [(0, 4, 4)]

        # 4. Processing
        cleaned = filter_noise(raw_notes)
        cleaned = merge_micro_gaps(cleaned)

        is_quant = output_mode in ["sheet", "sheet_chords"]
        add_chords = output_mode == "sheet_chords"

        if is_quant:
            processed_notes = quantize_for_sheet_music(cleaned, tempo_map)
        else:
            processed_notes = cleaned

        final_bytes = build_output_midi_bytes(processed_notes, tempo_map, timesig_map, is_quantized=is_quant, add_chords=add_chords)

        with open(final_midi, "wb") as f: f.write(final_bytes)
        return FileResponse(final_midi, filename="transcribed.mid", media_type="audio/midi")

    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))
    finally:
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
