"""
Local transcription pipeline: Demucs (stem separation) -> PianoTranscription / Basic Pitch
"""
import io
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import mido
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict as basic_pitch_predict

# ייבוא מנוע הפסנתר החדש ברמת piano2notes
from piano_transcription_inference import PianoTranscription, sample_rate
from piano_transcription_inference.utils import load_audio

app = FastAPI(title="Local Transcription Pipeline")

OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)

STEM_MAP = {
    "bass": "bass", "guitar": "guitar", "piano": "piano", "vocals": "vocals",
    "strings": "other", "brass": "other", "trumpet": "other", "sax": "other",
}

# טעינת מודל הפסנתר המתקדם לזיכרון ה-GPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
piano_transcriber = PianoTranscription(device=DEVICE)

FORM_HTML = """
<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
<meta charset="UTF-8">
<title>High Quality Transcription Pipeline</title>
<style>
body{font-family:sans-serif;background:#15130f;color:#ede6d6;display:flex;justify-content:center;padding-top:50px;}
.card{background:#201c15;padding:30px;border-radius:10px;border:1px solid #3a3326;width:400px;}
h2{margin-top:0;color:#d9a441;}
.field{margin-bottom:15px;}
label{display:block;margin-bottom:5px;font-size:0.9rem;}
input,select,button{width:100%;padding:10px;border-radius:5px;border:1px solid #3a3326;background:#2a2419;color:#fff;}
button{background:#d9a441;color:#161108;font-weight:bold;cursor:pointer;margin-top:10px;}
</style>
</head>
<body>
<div class="card">
<h2>תמלול אודיו ל-MIDI</h2>
<form action="/transcribe" method="post" enctype="multipart/form-data">
  <div class="field"><label>קובץ אודיו (MP3/WAV)</label><input type="file" name="file" required></div>
  <div class="field"><label>כלי נגינה</label>
  <select name="instrument">
    <option value="piano">Piano (High Quality Engine)</option>
    <option value="bass">Bass</option>
    <option value="guitar">Guitar</option>
    <option value="vocals">Vocals</option>
  </select></div>
  <button type="submit">הפרד ותמלל</button>
</form>
</div>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def root():
    return FORM_HTML

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    instrument: str = Form(...)
):
    if instrument not in STEM_MAP:
        raise HTTPException(400, "Instrument not supported")
    
    stem_name = STEM_MAP[instrument]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        input_path = tmp / file.filename
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 1. הפרדת סטמים באמצעות Demucs
        separated_dir = tmp / "separated"
        result = subprocess.run(
            ["demucs", "-n", "htdemucs_6s", "-o", str(separated_dir), str(input_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise HTTPException(500, f"Demucs failed: {result.stderr[-500:]}")

        found_stems = list(separated_dir.rglob(f"{stem_name}.wav"))
        if not found_stems:
            raise HTTPException(500, f"Stem '{stem_name}.wav' not found.")
        stem_path = found_stems[0]

        out_name = f"{uuid.uuid4().hex[:8]}_{instrument}.mid"
        out_path = OUTPUT_DIR / out_name

        # 2. תמלול בהתאם לכלי
        if instrument == "piano":
            # תמלול ברמת אולפן לפסנתר (Onsets & Frames / CRNN)
            (audio, _) = load_audio(str(stem_path), sr=sample_rate, mono=True)
            piano_transcriber.transcribe(audio, str(out_path))
        else:
            # תמלול כללי באמצעות Basic Pitch לכל שאר הכלים
            _, midi_data, _ = basic_pitch_predict(
                str(stem_path),
                model_or_model_path=ICASSP_2022_MODEL_PATH,
                onset_threshold=0.65,
                frame_threshold=0.55,
                minimum_note_length=150.0,
            )
            midi_data.write(str(out_path))

    return FileResponse(out_path, media_type="audio/midi", filename=out_path.name)
