"""
Local transcription pipeline: Demucs (stem separation) -> Basic Pitch (MIDI transcription).

Runs entirely on YOUR computer. Nothing here talks to Claude or to any outside
server except the one-time model downloads Demucs/Basic Pitch do on first use.

Setup (once):
    pip install fastapi "uvicorn[standard]" python-multipart demucs basic-pitch

Run:
    uvicorn app:app --reload

Then open http://127.0.0.1:8000 in your browser, pick a file and an instrument,
and submit. The finished .mid file downloads automatically when it's done --
Demucs has to process the whole song first, so this can take a while.
"""
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict

app = FastAPI(title="Local Transcription Pipeline")

OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Demucs' 6-stem model (htdemucs_6s) separates: drums, bass, other, vocals, guitar, piano.
# Instruments it can't isolate on their own fall back to "other" (everything left
# after drums/bass/vocals/guitar/piano are pulled out) -- not a clean solo track,
# but the closest available stem.
STEM_MAP = {
    "bass": "bass",
    "guitar": "guitar",
    "piano": "piano",
    "vocals": "vocals",
    "strings": "other",
    "brass": "other",
    "trumpet": "other",
    "sax": "other",
}

FORM_HTML = """
<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Local Transcription Pipeline</title></head>
<body style="font-family:-apple-system,sans-serif;max-width:480px;margin:48px auto;line-height:1.5">
<h2>Local Transcription Pipeline</h2>
<p>Demucs separates the stem, then Basic Pitch transcribes it to MIDI with your tuned settings
(onset 0.6, frame 0.5, min note length 127.7ms).</p>
<form action="/transcribe" method="post" enctype="multipart/form-data">
  <p><input type="file" name="file" accept="audio/*" required></p>
  <p><select name="instrument">
    <option value="bass">Bass</option>
    <option value="guitar">Guitar</option>
    <option value="piano">Piano</option>
    <option value="vocals">Vocals</option>
    <option value="strings">Strings (uses the "other" stem)</option>
    <option value="brass">Brass (uses the "other" stem)</option>
    <option value="trumpet">Trumpet (uses the "other" stem)</option>
    <option value="sax">Saxophone (uses the "other" stem)</option>
  </select></p>
  <p><button type="submit">Separate &amp; Transcribe</button></p>
</form>
<p style="color:#777;font-size:.9em">This can take a few minutes on a slower computer -- Demucs has to
process the whole track before Basic Pitch even starts. The .mid file will download automatically
when it's ready.</p>
</body></html>
"""


@app.get("/", response_class=HTMLResponse)
def root():
    return FORM_HTML


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), instrument: str = Form(...)):
    if instrument not in STEM_MAP:
        raise HTTPException(400, f"Unknown instrument '{instrument}'. Choose from: {list(STEM_MAP)}")
    stem_name = STEM_MAP[instrument]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        input_path = tmp / file.filename
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 1) Demucs stem separation
        separated_dir = tmp / "separated"
        result = subprocess.run(
            ["demucs", "-n", "htdemucs_6s", "-o", str(separated_dir), str(input_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise HTTPException(500, f"Demucs failed:\n{result.stderr[-2000:]}")

        stem_path = separated_dir / "htdemucs_6s" / input_path.stem / f"{stem_name}.wav"
        if not stem_path.exists():
            raise HTTPException(500, f"Expected stem not found at {stem_path}")

        # 2) Basic Pitch transcription with the tuned parameters
        _, midi_data, _ = predict(
            str(stem_path),
            model_or_model_path=ICASSP_2022_MODEL_PATH,
            onset_threshold=0.6,
            frame_threshold=0.5,
            minimum_note_length=127.7,
        )

        out_name = f"{uuid.uuid4().hex[:8]}_{instrument}.mid"
        out_path = OUTPUT_DIR / out_name
        midi_data.write(str(out_path))

    return FileResponse(out_path, media_type="audio/midi", filename=out_path.name)
