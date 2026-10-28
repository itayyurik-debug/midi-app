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
import math

app = FastAPI(title="Professional Audio to Sheet-Music MIDI Pipeline")

MAJ_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MIN_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
PPQ_OUT = 480

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

def quantize_for_sheet_music(notes, bpm, grid_division=16):
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

def build_output_midi_bytes(notes, tempo_map_events, timesig_events, is_quantized=True, add_chords=False):
    out_mid = mido.MidiFile(ticks_per_beat=PPQ_OUT)
    meta_track = mido.MidiTrack()
    
    for t_sec, bpm in tempo_map_events:
        meta_track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    
    for t_sec, num, den in timesig_events:
        midi_den = int(math.log2(den)) if den > 0 else 2
        meta_track.append(mido.MetaMessage("time_signature", numerator=num, denominator=midi_den, time=0))

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
    
    bpm_start = tempo_map_events[0][1] if tempo_map_events else 120.0
    sec_per_tick = 60.0 / (bpm_start * PPQ_OUT)

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
        .card { background: #1e293bcc; backdrop-filter: blur(6px); border-radius: 16px; padding: 28px; max-width: 500px; width: 100%; text-align: center; border: 1px solid #334155; }
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
        .field-group { margin-top: 10px; text-align: right; }
        .field-group label { display: block; font-size: 0.8rem; color: #94a3b8; margin-bottom: 4px; }
        .row2 { display: flex; gap: 10px; }
        .row2 > div { flex: 1; }
        optgroup { background: #1e293b; color: #94a3b8; font-weight: bold; }
        option { color: #f8fafc; }
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
                <option value="raw">⚡ MIDI ללא יישור לקצב</option>
            </select>

            <label style="display:flex;align-items:center;gap:8px;font-size:0.85rem;color:#94a3b8;margin-top:10px;text-align:right">
                <input type="checkbox" id="addChords" checked style="width:auto">
                ייצוא עם שמות אקורדים
            </label>

            <div class="modebar">
                <label><input type="radio" name="mode" value="detect" checked onchange="setMode('detect')"><span>זיהוי אוטומטי</span></label>
                <label><input type="radio" name="mode" value="manual" onchange="setMode('manual')"><span>קביעה ידנית</span></label>
                <label><input type="radio" name="mode" value="anchor" onchange="setMode('anchor')"><span>MIDI בסיס</span></label>
            </div>

            <div class="modepane active" id="pane-detect">
                <p style="color:#64748b;font-size:0.75rem;">זיהוי מקצב, סולם ומשקל אוטומטי מהקובץ.</p>
            </div>

            <div class="modepane" id="pane-manual">
                <div class="field-group">
                    <label>קצב (BPM)</label>
                    <select id="bpmMode" onchange="document.getElementById('manualBpmInput').style.display = this.value === 'custom' ? 'block' : 'none'">
                        <option value="auto">זיהוי אוטומטי</option>
                        <option value="custom">קביעה ידנית</option>
                    </select>
                    <input type="number" id="manualBpmInput" value="120" style="display:none;" placeholder="הזן BPM">
                </div>

                <div class="field-group">
                    <label>משקל</label>
                    <select id="timeSigSelect">
                        <option value="auto">זיהוי אוטומטי (4/4)</option>
                        <optgroup label="רבעים">
                            <option value="1/4">1/4</option><option value="2/4">2/4</option><option value="3/4">3/4</option><option value="4/4">4/4</option><option value="5/4">5/4</option><option value="6/4">6/4</option><option value="7/4">7/4</option>
                        </optgroup>
                        <optgroup label="שמיניות">
                            <option value="3/8">3/8</option><option value="5/8">5/8</option><option value="6/8">6/8</option><option value="7/8">7/8</option><option value="9/8">9/8</option><option value="11/8">11/8</option><option value="12/8">12/8</option><option value="13/8">13/8</option><option value="15/8">15/8</option>
                        </optgroup>
                        <optgroup label="חלקי 16">
                            <option value="5/16">5/16</option><option value="7/16">7/16</option><option value="9/16">9/16</option>
                        </optgroup>
                    </select>
                </div>
            </div>

            <div class="modepane" id="pane-anchor">
                <div class="field-group">
                    <label>קובץ MIDI בסיס לסינכרון</label>
                    <div class="upload-box" style="margin-top:6px;padding:12px;" onclick="document.getElementById('anchorFile').click()">
                        <span id="anchorFileName">🎼 לחץ לבחירת קובץ MIDI</span>
                        <input type="file" id="anchorFile" accept=".mid,.midi" onchange="document.getElementById('anchorFileName').innerText = '📄 ' + this.files[0].name">
                    </div>
                </div>
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
            formData.append('add_chords', document.getElementById('addChords').checked);
            
            const mode = document.querySelector('input[name="mode"]:checked').value;
            formData.append('mode', mode);

            if (mode === 'manual') {
                formData.append('bpm_mode', document.getElementById('bpmMode').value);
                formData.append('manual_bpm', document.getElementById('manualBpmInput').value);
                formData.append('time_sig_select', document.getElementById('timeSigSelect').value);
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
                    const errData = await res.json();
                    status.style.color = "#f87171";
                    status.innerText = "❌ שגיאה: " + (errData.detail || "עיבוד נכשל");
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
    add_chords: bool = Form(True),
    mode: str = Form("detect"),
    bpm_mode: str = Form("auto"),
    manual_bpm: float = Form(120.0),
    time_sig_select: str = Form("auto"),
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

        subprocess.run(["demucs", "-n", "htdemucs_6s", "-o", sep_dir, raw_path], capture_output=True)
        raw_stem_name = os.path.splitext(os.path.basename(raw_path))[0]
        stem_source = os.path.join(sep_dir, "htdemucs_6s", raw_stem_name, f"{model}.wav")

        if not os.path.exists(stem_source):
            subprocess.run(["ffmpeg", "-y", "-i", raw_path, "-ar", "32000", "-ac", "1", stem_wav_path], capture_output=True)
        else:
            subprocess.run(["ffmpeg", "-y", "-i", stem_source, "-ar", "32000", "-ac", "1", stem_wav_path], capture_output=True)

        if model == "piano" and PIANO_MODEL is not None:
            from piano_transcription_inference import sample_rate
            import librosa
            audio, _ = librosa.load(stem_wav_path, sr=sample_rate)
            PIANO_MODEL.transcribe(audio, output_midi)
        else:
            from basic_pitch.inference import predict_and_save
            predict_and_save([stem_wav_path], "outputs", True, False, False, False)
            output_midi = f"outputs/{job_id}_stem_basic_pitch.mid"

        raw_notes = read_midi_notes_and_map(output_midi)["notes"]
        
        tempo_map_events = []
        timesig_events = []

        if mode == "anchor" and reference_midi is not None:
            ref_bytes = await reference_midi.read()
            ref_path = f"uploads/{job_id}_ref.mid"
            with open(ref_path, "wb") as f: f.write(ref_bytes)
            ref_map = read_midi_notes_and_map(ref_path)
            
            tempo_map_events = ref_map["tempo_map"]
            timesig_events = ref_map["timesig_map"]
            bpm = tempo_map_events[0][1] if tempo_map_events else 120.0
        else:
            if mode == "manual" and bpm_mode == "custom":
                bpm = manual_bpm
            else:
                bpm = detect_bpm_from_audio(raw_path)
            tempo_map_events = [(0, bpm)]

            if mode == "manual" and time_sig_select != "auto":
                num, den = map(int, time_sig_select.split('/'))
                timesig_events = [(0, num, den)]
            else:
                timesig_events = [(0, 4, 4)]

        cleaned = filter_noise(raw_notes)
        cleaned = merge_micro_gaps(cleaned)

        is_quant = output_mode == "sheet"

        if is_quant:
            processed_notes = quantize_for_sheet_music(cleaned, bpm=bpm)
        else:
            processed_notes = cleaned

        final_bytes = build_output_midi_bytes(processed_notes, tempo_map_events, timesig_events, is_quantized=is_quant, add_chords=bool(add_chords))

        with open(final_midi, "wb") as f: f.write(final_bytes)
        return FileResponse(final_midi, filename="transcribed.mid", media_type="audio/midi")

    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))
    finally:
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
