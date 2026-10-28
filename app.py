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

app = FastAPI(title="Professional Audio to Sheet-Music MIDI Pipeline")

PPQ_OUT = 480

# ==========================================
# 1. MIDI READING & BASIC PARSING
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
    if not timesig_events: timesig_events = [(0, 4, 4)]

    def tick_to_sec(tick):
        sec = 0.0
        for i, (t_start, us) in enumerate(tempo_events):
            t_end = tempo_events[i + 1][0] if i + 1 < len(tempo_events) else float("inf")
            if tick <= t_start: break
            sec += (min(tick, t_end) - t_start) * (us / 1e6) / ppq
        return sec

    return {"tempo_map": [(tick_to_sec(t), round(60e6/us)) for t, us in tempo_events], 
            "notes": [(p, tick_to_sec(s), tick_to_sec(e), v) for p, s, e, v in notes]}

# ==========================================
# 2. SHEET MUSIC PROCESSING & CHORDS
# ==========================================
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

def detect_chords_for_midi(quantized_notes, bpm):
    """מזהה אקורדים בסיסיים לפי חלוקה לרבעים ויוצר אירועי טקסט"""
    ticks_per_beat = PPQ_OUT
    chord_events = []
    
    # קיבוץ תווים לפי פעימות
    beats = {}
    for pitch, s_tick, d_tick, vel in quantized_notes:
        beat_idx = s_tick // ticks_per_beat
        if beat_idx not in beats: beats[beat_idx] = []
        beats[beat_idx].append(pitch % 12)
        
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    for beat, pitches in beats.items():
        unique_pitches = list(set(pitches))
        if len(unique_pitches) >= 3:
            root = unique_pitches[0] # הערכה בסיסית
            chord_name = note_names[root]
            # שתילת מרקר טקסט ב-MIDI
            chord_events.append((beat * ticks_per_beat, mido.MetaMessage('marker', text=chord_name)))
            
    return sorted(chord_events, key=lambda x: x[0])

def detect_bpm_from_audio(wav_path):
    import librosa
    import numpy as np
    y, sr = librosa.load(wav_path, sr=None)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(np.atleast_1d(tempo)[0])

# ==========================================
# 3. MIDI GENERATION
# ==========================================
def build_output_midi_bytes(notes, bpm, is_quantized=False, add_chords=False):
    out_mid = mido.MidiFile(ticks_per_beat=PPQ_OUT)
    
    meta_track = mido.MidiTrack()
    meta_track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    meta_track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    
    if is_quantized and add_chords:
        chord_events = detect_chords_for_midi(notes, bpm)
        last_t = 0
        for tick, msg in chord_events:
            msg.time = max(0, tick - last_t)
            meta_track.append(msg)
            last_t = max(last_t, tick)
            
    out_mid.tracks.append(meta_track)

    # יצירת ערוץ אחד (Track) כדי למנוע 4 חמשות בתוכנות תווים
    tr = mido.MidiTrack()
    evs = []
    
    if not is_quantized:
        # המרה לטיקים עבור קובץ גולמי
        sec_per_tick = 60.0 / (bpm * PPQ_OUT)
        for pitch, start_sec, end_sec, vel in notes:
            s_tick = int(start_sec / sec_per_tick)
            d_tick = int((end_sec - start_sec) / sec_per_tick)
            evs.append((s_tick, mido.Message("note_on", note=int(pitch), velocity=int(vel), channel=0)))
            evs.append((s_tick + d_tick, mido.Message("note_off", note=int(pitch), velocity=0, channel=0)))
    else:
        for pitch, s_tick, d_tick, vel in notes:
            evs.append((s_tick, mido.Message("note_on", note=int(pitch), velocity=int(vel), channel=0)))
            evs.append((s_tick + d_tick, mido.Message("note_off", note=int(pitch), velocity=0, channel=0)))

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
# 4. FASTAPI & UI
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
        body { font-family: sans-serif; background: #0f172a; color: #f8fafc; padding: 20px; text-align: center; }
        .card { background: #1e293b; padding: 28px; border-radius: 12px; max-width: 500px; margin: auto; }
        .upload-box { border: 2px dashed #0284c7; padding: 20px; cursor: pointer; margin-bottom: 15px; }
        input[type="file"] { display: none; }
        select, button { width: 100%; padding: 10px; margin-top: 10px; }
        button { background: #0284c7; color: white; border: none; font-weight: bold; cursor: pointer; }
    </style>
</head>
<body>
    <div class="card">
        <h1>🎹 ממיר שמע ל-MIDI</h1>
        <form id="transcribeForm">
            <div class="upload-box" onclick="document.getElementById('audioFile').click()">
                <span id="fileName">📁 בחר קובץ שמע</span>
                <input type="file" id="audioFile" required onchange="document.getElementById('fileName').innerText = this.files[0].name">
            </div>
            <select id="processMode">
                <option value="raw">MIDI גולמי ומדויק (ללא סידור אוטומטי)</option>
                <option value="sheet">MIDI מסודר לתווים (יישור לקצב)</option>
                <option value="sheet_chords">MIDI מסודר לתווים + זיהוי אקורדים</option>
            </select>
            <button type="submit" id="submitBtn">המר ל-MIDI</button>
        </form>
        <div id="status" style="margin-top: 15px;"></div>
    </div>
    <script>
        document.getElementById('transcribeForm').onsubmit = async (e) => {
            e.preventDefault();
            const btn = document.getElementById('submitBtn');
            const status = document.getElementById('status');
            btn.disabled = true;
            status.innerText = "⏳ מעבד שמע...";
            
            const formData = new FormData();
            formData.append('file', document.getElementById('audioFile').files[0]);
            formData.append('mode', document.getElementById('processMode').value);

            const fileInput = document.getElementById('audioFile');
            const baseName = fileInput.files[0].name.split('.').slice(0, -1).join('.') || 'audio';

            try {
                const res = await fetch('/transcribe', { method: 'POST', body: formData });
                if (res.ok) {
                    const blob = await res.blob();
                    const a = document.createElement('a');
                    a.href = window.URL.createObjectURL(blob);
                    a.download = baseName + ".mid";
                    a.click();
                    status.innerText = "✅ ירד בהצלחה!";
                } else {
                    status.innerText = "❌ שגיאה בהמרה";
                }
            } catch (err) {
                status.innerText = "❌ שגיאה בשרת";
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
async def transcribe(file: UploadFile = File(...), mode: str = Form("raw")):
    import uuid
    job_id = uuid.uuid4().hex[:8]
    raw_path = f"uploads/{job_id}_raw.mp3"
    stem_wav_path = f"uploads/{job_id}_stem.wav"
    output_midi = f"outputs/{job_id}.mid"
    final_midi = f"outputs/{job_id}_final.mid"

    try:
        with open(raw_path, "wb") as f: shutil.copyfileobj(file.file, f)

        # Demucs (Simplified for speed)
        subprocess.run(["demucs", "-n", "htdemucs_6s", "-o", "outputs", raw_path], capture_output=True)
        stem_source = f"outputs/htdemucs_6s/{job_id}_raw/piano.wav"
        
        if not os.path.exists(stem_source):
            # אם Demucs נכשל, נשתמש בקובץ המקורי כגיבוי
            subprocess.run(["ffmpeg", "-y", "-i", raw_path, "-ar", "32000", "-ac", "1", stem_wav_path])
        else:
            subprocess.run(["ffmpeg", "-y", "-i", stem_source, "-ar", "32000", "-ac", "1", stem_wav_path])

        if PIANO_MODEL is not None:
            from piano_transcription_inference import sample_rate
            import librosa
            audio, _ = librosa.load(stem_wav_path, sr=sample_rate)
            PIANO_MODEL.transcribe(audio, output_midi)
        else:
            from basic_pitch.inference import predict_and_save
            predict_and_save([stem_wav_path], "outputs", True, False, False, False)
            output_midi = f"outputs/{job_id}_stem_basic_pitch.mid"

        raw_notes = read_midi_notes_and_map(output_midi)["notes"]
        bpm = detect_bpm_from_audio(raw_path)

        is_quant = mode in ["sheet", "sheet_chords"]
        add_chords = mode == "sheet_chords"

        if is_quant:
            notes_to_process = quantize_for_sheet_music(raw_notes, bpm=bpm, grid_division=16)
        else:
            notes_to_process = raw_notes

        final_bytes = build_output_midi_bytes(notes_to_process, bpm=bpm, is_quantized=is_quant, add_chords=add_chords)
        
        with open(final_midi, "wb") as f: f.write(final_bytes)
        return FileResponse(final_midi, filename="output.mid", media_type="audio/midi")

    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))
    finally:
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
