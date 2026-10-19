"""
Local transcription pipeline: Demucs (stem separation) -> Basic Pitch (MIDI transcription)
-> tempo/key-aware quantization, mirroring the claude.ai "Interactive Transcription Studio".

Runs entirely on YOUR computer / your own server. Nothing here talks to Claude.

Setup (once):
    pip install fastapi "uvicorn[standard]" python-multipart demucs basic-pitch mido

Run:
    uvicorn app:app --reload

Then open http://127.0.0.1:8000 -- pick a file and an instrument, and choose one of two
ways to set the key/tempo:
  1) Manual key: pick the key yourself and type a BPM.
  2) Anchor MIDI: upload a basic MIDI file and the key, tempo map (including mid-song
     tempo changes) and time-signature map (including mid-song changes) are detected
     from it automatically and used to build the final MIDI -- same logic as the
     browser Studio.
"""
import io
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import mido
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict

app = FastAPI(title="Local Transcription Pipeline")

OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)

STEM_MAP = {
    "bass": "bass", "guitar": "guitar", "piano": "piano", "vocals": "vocals",
    "strings": "other", "brass": "other", "trumpet": "other", "sax": "other",
}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJ_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MIN_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


# ---------- anchor MIDI parsing: full tempo map + time-signature map, ticks -> seconds ----------
def parse_anchor_midi(path):
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
                active[msg.note] = time
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                if msg.note in active:
                    notes.append((msg.note, active.pop(msg.note), time))
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
    note_list = [(p, tick_to_sec(s), tick_to_sec(e)) for p, s, e in notes]
    return {"tempo_map": tempo_map, "timesig_map": timesig_map, "notes": note_list}


def detect_key(notes):
    """notes: list of (pitch, start_sec, end_sec)."""
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


# ---------- tempo-map-aware quantization + MIDI writing (mirrors the browser Studio) ----------
def build_segments(tempo_map, ppq):
    segs, tick = [], 0.0
    for i, (t, bpm) in enumerate(tempo_map):
        if i > 0:
            prev_t, prev_bpm = tempo_map[i - 1]
            tick += (t - prev_t) * (prev_bpm / 60.0) * ppq
        segs.append((t, bpm, tick))
    return segs


PPQ_GLOBAL = 480  # fixed resolution for all output files


def sec_to_tick(sec, segs):
    for t, bpm, tick in reversed(segs):
        if sec >= t:
            return tick + (sec - t) * (bpm / 60.0) * PPQ_GLOBAL
    return 0.0


def quantize_notes(bp_notes, tempo_map, anchor_notes):
    segs = build_segments(tempo_map, PPQ_GLOBAL)
    grid = PPQ_GLOBAL // 4
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


def build_output_midi_bytes(quantized_notes, tempo_map, timesig_map, segs, split_piano=False):
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

    def notes_track(notes):
        tr = mido.MidiTrack()
        evs = []
        for pitch, s_tick, d_tick in notes:
            evs.append((s_tick, mido.Message("note_on", note=int(pitch), velocity=90)))
            evs.append((s_tick + d_tick, mido.Message("note_off", note=int(pitch), velocity=0)))
        evs.sort(key=lambda e: e[0])
        last_t = 0
        for tick, msg in evs:
            msg.time = max(0, round(tick - last_t))
            tr.append(msg)
            last_t = max(last_t, tick)
        tr.append(mido.MetaMessage("end_of_track"))
        return tr

    mid = mido.MidiFile(ticks_per_beat=PPQ_GLOBAL)
    mid.tracks.append(meta_track)
    if split_piano:
        mid.tracks.append(notes_track([n for n in quantized_notes if n[0] >= 60]))
        mid.tracks.append(notes_track([n for n in quantized_notes if n[0] < 60]))
    else:
        mid.tracks.append(notes_track(quantized_notes))

    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


# ---------- frontend ----------
FORM_HTML = """
<!DOCTYPE html>
<html lang="en" dir="ltr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Local Transcription Pipeline</title>
<style>
:root{
  --bg:#15130f; --surface:#201c15; --surface2:#2a2419; --border:#3a3326;
  --text:#ede6d6; --muted:#9c9382; --accent:#d9a441; --accent-dim:#8a6b2e;
  --ok:#4f9b8c; --bad:#c1584a; color-scheme: dark;
}
@media (prefers-color-scheme: light){
  :root{
    --bg:#f1efea; --surface:#ffffff; --surface2:#e9e4d8; --border:#ddd5c2;
    --text:#201c15; --muted:#746a58; --accent:#a87820; --accent-dim:#c9a34f;
    --ok:#2e7a6c; --bad:#a6412f; color-scheme: light;
  }
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,"Segoe UI",Arial,sans-serif;overflow-x:hidden}
.bg-orb{position:fixed;border-radius:50%;filter:blur(70px);z-index:-1;pointer-events:none}
.bg-orb.a{width:420px;height:420px;background:var(--accent);opacity:.28;top:-120px;left:-100px;animation:drift1 24s ease-in-out infinite alternate}
.bg-orb.b{width:380px;height:380px;background:var(--ok);opacity:.22;bottom:-130px;right:-100px;animation:drift2 28s ease-in-out infinite alternate}
@keyframes drift1{from{transform:translate(0,0)}to{transform:translate(60px,80px)}}
@keyframes drift2{from{transform:translate(0,0)}to{transform:translate(-70px,-50px)}}
@media (prefers-reduced-motion: reduce){ .bg-orb{animation:none} }
.wrap{max-width:480px;margin:0 auto;padding:40px 20px 60px}
h1{font-size:1.4rem;font-weight:800;margin:0 0 4px}
.sub{color:var(--muted);font-size:.9rem;margin:0 0 24px}
.chain{display:flex;align-items:center;gap:4px;margin-bottom:24px}
.chain .seg{flex:1;height:2px;background:var(--border)}
.chain .seg.done{background:var(--accent-dim)}
.stage{display:flex;flex-direction:column;align-items:center;gap:6px}
.stage .dot{width:11px;height:11px;border-radius:50%;background:var(--surface2);border:2px solid var(--border);transition:all .25s ease}
.stage.active .dot{background:var(--accent);border-color:var(--accent);transform:scale(1.15)}
.stage.done .dot{background:var(--accent-dim);border-color:var(--accent-dim)}
.stage .lbl{font-size:.68rem;color:var(--muted)}
.stage.active .lbl{color:var(--accent);font-weight:700}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:22px}
.field{margin-bottom:16px}
.field label{display:block;font-size:.85rem;color:var(--muted);margin-bottom:6px}
input[type=file],input[type=number],select{width:100%;background:var(--surface2);border:1px solid var(--border);color:var(--text);padding:9px 10px;border-radius:6px;font-size:.9rem;font-family:inherit}
.row2{display:flex;gap:10px}
.row2>div{flex:1}
.modebar{display:flex;gap:8px;margin-bottom:16px}
.modebar label{flex:1;text-align:center;padding:9px 6px;border:1px solid var(--border);border-radius:6px;font-size:.82rem;cursor:pointer;color:var(--muted)}
.modebar input{display:none}
.modebar input:checked + span{color:var(--accent);font-weight:700}
.modebar label:has(input:checked){border-color:var(--accent);background:rgba(217,164,65,.08)}
.modepane{display:none}
.modepane.active{display:block}
.chk{display:flex;align-items:center;gap:8px;font-size:.85rem;color:var(--muted);margin-bottom:16px}
.chk input{width:auto}
button{width:100%;font-family:inherit;font-size:.95rem;font-weight:700;border:none;border-radius:6px;padding:12px 18px;cursor:pointer;background:var(--accent);color:#161108;transition:transform .15s ease,opacity .15s ease}
button:active{transform:scale(.97)}
button:disabled{opacity:.5;cursor:not-allowed}
.msg{font-size:.86rem;padding:10px 12px;border-radius:6px;margin-top:14px;animation:fadeIn .3s ease}
.msg.ok{background:rgba(79,155,140,.15);color:var(--ok)}
.msg.err{background:rgba(193,88,74,.15);color:var(--bad)}
@keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
.spin{width:16px;height:16px;border:2px solid currentColor;border-top-color:transparent;border-radius:50%;display:inline-block;vertical-align:middle;margin-right:8px;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.hint{font-size:.78rem;color:var(--muted);margin-top:14px}
</style>
</head>
<body>
<div class="bg-orb a"></div><div class="bg-orb b"></div>
<div class="wrap">
<h1>Local Transcription Pipeline</h1>
<p class="sub">Demucs separates the stem, then Basic Pitch transcribes it (onset 0.6, frame 0.5, min note length 127.7ms), quantized to the key and tempo you set below.</p>
<div class="chain" id="chain"></div>
<div class="card">
<form id="f">
  <div class="field"><label>Audio file (MP3)</label><input type="file" id="file" accept="audio/*" required></div>
  <div class="field"><label>Instrument</label>
  <select id="instrument">
    <option value="bass">Bass</option><option value="guitar">Guitar</option>
    <option value="piano">Piano</option><option value="vocals">Vocals</option>
    <option value="strings">Strings (uses the "other" stem)</option>
    <option value="brass">Brass (uses the "other" stem)</option>
    <option value="trumpet">Trumpet (uses the "other" stem)</option>
    <option value="sax">Saxophone (uses the "other" stem)</option>
  </select></div>

  <div class="modebar">
    <label><input type="radio" name="mode" value="manual" checked onchange="setMode('manual')"><span>1. Set key manually</span></label>
    <label><input type="radio" name="mode" value="anchor" onchange="setMode('anchor')"><span>2. Upload anchor MIDI</span></label>
  </div>

  <div class="modepane active" id="pane-manual">
    <div class="row2">
      <div class="field"><label>Key</label>
        <select id="keyTonic">
          <option value="0">C</option><option value="1">C#</option><option value="2">D</option>
          <option value="3">D#</option><option value="4">E</option><option value="5">F</option>
          <option value="6">F#</option><option value="7">G</option><option value="8">G#</option>
          <option value="9">A</option><option value="10">A#</option><option value="11">B</option>
        </select>
      </div>
      <div class="field"><label>Mode</label>
        <select id="keyMode"><option value="major">Major</option><option value="minor">Minor</option></select>
      </div>
    </div>
    <div class="field"><label>Tempo (BPM)</label><input type="number" id="manualBpm" value="120"></div>
  </div>

  <div class="modepane" id="pane-anchor">
    <div class="field"><label>Anchor MIDI file</label><input type="file" id="anchorFile" accept=".mid,.midi"></div>
    <p class="hint" style="margin-top:-8px">Key, tempo map, and time-signature map (including mid-song changes) are detected from this file automatically.</p>
  </div>

  <div class="chk"><input type="checkbox" id="splitPiano"><label for="splitPiano">Split into two hands (piano only)</label></div>

  <button type="submit" id="go">Separate &amp; Transcribe</button>
</form>
<div id="msg"></div>
</div>
<p class="hint">Demucs has to process the whole track before Basic Pitch even starts, so this can take a few minutes &mdash; that's normal, not stuck.</p>
</div>
<script>
function setMode(m){
  document.getElementById('pane-manual').classList.toggle('active', m==='manual');
  document.getElementById('pane-anchor').classList.toggle('active', m==='anchor');
}
const STAGES=[{l:'Upload'},{l:'Separate'},{l:'Transcribe'},{l:'Download'}];
function renderChain(active,done){
  const c=document.getElementById('chain'); c.innerHTML='';
  STAGES.forEach((s,i)=>{
    const d=document.createElement('div'); d.className='stage'+(i===active?' active':i<done?' done':'');
    d.innerHTML='<div class="dot"></div><div class="lbl">'+s.l+'</div>'; c.appendChild(d);
    if(i<STAGES.length-1){ const seg=document.createElement('div'); seg.className='seg'+(i<done?' done':''); c.appendChild(seg); }
  });
}
renderChain(0,0);
document.getElementById('f').addEventListener('submit', async (e)=>{
  e.preventDefault();
  const fileInput=document.getElementById('file'), inst=document.getElementById('instrument').value;
  const mode=document.querySelector('input[name="mode"]:checked').value;
  const msg=document.getElementById('msg'), go=document.getElementById('go');
  if(!fileInput.files.length){ msg.innerHTML='<div class="msg err">Choose an audio file first.</div>'; return; }
  if(mode==='anchor' && !document.getElementById('anchorFile').files.length){ msg.innerHTML='<div class="msg err">Choose an anchor MIDI file, or switch to manual key.</div>'; return; }
  go.disabled=true;
  renderChain(1,1);
  msg.innerHTML='<div class="msg ok"><span class="spin"></span>Demucs is separating the stem... this can take a few minutes.</div>';
  try{
    const fd=new FormData();
    fd.append('file',fileInput.files[0]);
    fd.append('instrument',inst);
    fd.append('mode',mode);
    fd.append('split_piano', document.getElementById('splitPiano').checked);
    if(mode==='manual'){
      fd.append('manual_key_tonic', document.getElementById('keyTonic').value);
      fd.append('manual_key_mode', document.getElementById('keyMode').value);
      fd.append('manual_bpm', document.getElementById('manualBpm').value);
    } else {
      fd.append('anchor_midi', document.getElementById('anchorFile').files[0]);
    }
    renderChain(2,2);
    const res=await fetch('/transcribe',{method:'POST',body:fd});
    if(!res.ok){
      let detail='Request failed.';
      try{ const j=await res.json(); detail=j.detail||detail; }catch(e){}
      throw new Error(detail);
    }
    renderChain(3,3);
    msg.innerHTML='<div class="msg ok">Done! Starting download...</div>';
    const disp=res.headers.get('Content-Disposition')||'';
    const m=disp.match(/filename="?([^"]+)"?/);
    const filename=m?m[1]:(inst+'.mid');
    const blob=await res.blob();
    const url=URL.createObjectURL(blob);
    const a=document.createElement('a'); a.href=url; a.download=filename; document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
    msg.innerHTML='<div class="msg ok">Saved as '+filename+'.</div>';
  }catch(err){
    msg.innerHTML='<div class="msg err">'+(err.message||'Something went wrong.')+'</div>';
    renderChain(0,0);
  }finally{
    go.disabled=false;
  }
});
</script>
</body></html>
"""


@app.get("/", response_class=HTMLResponse)
def root():
    return FORM_HTML


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    instrument: str = Form(...),
    mode: str = Form("manual"),
    manual_key_tonic: int = Form(0),
    manual_key_mode: str = Form("major"),
    manual_bpm: float = Form(120.0),
    split_piano: bool = Form(False),
    anchor_midi: UploadFile = File(None),
):
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
            capture_output=True, text=True,
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
        bp_notes = [(n.pitch, n.start, n.end) for inst in midi_data.instruments for n in inst.notes]
        if not bp_notes:
            raise HTTPException(500, "No notes were transcribed from the isolated stem.")

        # 3) Key + tempo map: either from the anchor MIDI, or manual
        if mode == "anchor":
            if anchor_midi is None:
                raise HTTPException(400, "mode=anchor requires an anchor_midi file.")
            anchor_bytes = await anchor_midi.read()
            anchor_path = tmp / "anchor.mid"
            anchor_path.write_bytes(anchor_bytes)
            anchor = parse_anchor_midi(anchor_path)
            tonic, is_minor = detect_key(anchor["notes"])
            tempo_map, timesig_map, anchor_notes = anchor["tempo_map"], anchor["timesig_map"], anchor["notes"]
        else:
            tonic, is_minor = int(manual_key_tonic), (manual_key_mode == "minor")
            tempo_map, timesig_map, anchor_notes = [(0, manual_bpm)], [(0, 4, 4)], []

        # 4) Snap to key, quantize to the tempo map, write final MIDI
        snapped = snap_to_key(bp_notes, tonic, is_minor)
        quantized, segs = quantize_notes(snapped, tempo_map, anchor_notes)
        midi_bytes = build_output_midi_bytes(quantized, tempo_map, timesig_map, segs, split_piano=bool(split_piano))

        out_name = f"{uuid.uuid4().hex[:8]}_{instrument}.mid"
        out_path = OUTPUT_DIR / out_name
        out_path.write_bytes(midi_bytes)

    return FileResponse(out_path, media_type="audio/midi", filename=out_path.name)
