from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import subprocess
import torch
import gc

app = FastAPI(title="Audio to MIDI Pipeline")

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
except Exception as e:
    print(f"⚠️ שגיאה בטעינת המודל: {e}")

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
                <option value="piano">🎹 Piano Transcription (פסנתר בלבד)</option>
                <option value="basic">🎸 Basic Pitch (כללי)</option>
            </select>
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

        document.getElementById('transcribeForm').onsubmit = async (e) => {
            e.preventDefault();
            const status = document.getElementById('status');
            const btn = document.getElementById('submitBtn');
            
            status.className = "loading";
            status.innerText = "⏳ מעבד שמע ב-GPU... ייקח כ-3 שניות";
            btn.disabled = true;
            btn.style.opacity = "0.5";

            const formData = new FormData();
            formData.append('file', document.getElementById('audioFile').files[0]);
            formData.append('model', document.getElementById('modelSelect').value);

            try {
                const res = await fetch('/transcribe', { method: 'POST', body: formData });
                if (res.ok) {
                    const blob = await res.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'transcribed_music.mid';
                    a.click();
                    status.className = "success";
                    status.innerText = "✅ ההמרה הושלמה! קובץ ה-MIDI הורד בהצלחה.";
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
async def transcribe(file: UploadFile = File(...), model: str = Form("piano")):
    import uuid
    job_id = uuid.uuid4().hex[:8]
    raw_path = f"uploads/{job_id}_raw"
    clean_wav_path = f"uploads/{job_id}_clean.wav"
    output_midi = f"outputs/{job_id}.mid"
    basic_pitch_out = f"outputs/{job_id}_clean_basic_pitch.mid"

    try:
        with open(raw_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # המרה ל-WAV תקני
        conv = subprocess.run(["ffmpeg", "-y", "-i", raw_path, "-ar", "32000", "-ac", "1", clean_wav_path], capture_output=True, text=True)
        if conv.returncode != 0:
            raise HTTPException(status_code=500, detail="המרה ראשונית נכשלה")

        if model == "piano":
            if PIANO_MODEL is None:
                raise HTTPException(status_code=500, detail="מודל הפסנתר לא זמין")
            from piano_transcription_inference import sample_rate
            import librosa
            audio, _ = librosa.load(clean_wav_path, sr=sample_rate)
            PIANO_MODEL.transcribe(audio, output_midi)
        else:
            from basic_pitch.inference import predict_and_save
            predict_and_save([clean_wav_path], "outputs", True, False, False, False)
            output_midi = basic_pitch_out

        if os.path.exists(output_midi):
            return FileResponse(output_midi, filename="transcribed.mid", media_type="audio/midi")

        raise HTTPException(status_code=500, detail="קובץ ה-MIDI לא נוצר")
    except Exception as err:
        print(f"❌ שגיאה בזמן המרה: {err}")
        raise HTTPException(status_code=500, detail=str(err))
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
