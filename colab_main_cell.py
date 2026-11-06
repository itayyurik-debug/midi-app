# ===== תא אחד: התקנה + משיכה מגיטהאב + בדיקה + הפעלה =====
import os, subprocess, time

print("📦 [1/5] תוכנות מערכת...")
!apt-get -qq update && apt-get -qq install -y ffmpeg musescore3 xvfb > /dev/null 2>&1

print("📦 [2/5] ספריות Python...")
!pip install -q fastapi uvicorn python-multipart mido pretty_midi librosa soundfile torchaudio demucs sphn julius lameenc
!pip install -q einops soxr rotary-embedding-torch tqdm
!pip install -q https://github.com/CPJKU/beat_this/archive/main.zip
!pip install -q muscriptor
!pip install -q --no-deps basic-pitch piano_transcription_inference torchlibrosa mir-eval resampy

print("📋 [3/5] מושך קבצים מגיטהאב...")
GITHUB_USER, GITHUB_REPO, BRANCH = "itayyurik-debug", "midi-app", "main"
!wget -q -O app.py "https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{BRANCH}/app.py?nocache=$(date +%s)"
!wget -q -O notation.py "https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{BRANCH}/notation.py?nocache=$(date +%s)"
!test -f cloudflared || (wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared && chmod +x cloudflared)

print("🔍 [4/5] בדיקת שלמות...")
for m in ["fastapi", "demucs", "sphn", "muscriptor", "torchlibrosa", "piano_transcription_inference",
          "basic_pitch", "librosa", "mido", "beat_this"]:
    r = subprocess.run(["python3", "-c", f"import {m}"], capture_output=True, text=True)
    print(("✅ " if r.returncode == 0 else "❌ ") + m, "" if r.returncode == 0 else r.stderr.strip().splitlines()[-1])
chk = subprocess.run(["python3", "-c", "import app"], capture_output=True, text=True)
if chk.returncode != 0:
    print("\n❌ app.py לא נטען -- שלח לי את זה:\n", chk.stderr)
else:
    print("\n✅ app.py נטען. מפעיל שרת...\n")
    !fuser -k 8000/tcp || true
    !pkill -9 -f uvicorn || true
    !pkill -9 -f cloudflared || true
    log = open("log.txt", "w")
    subprocess.Popen(["xvfb-run", "-a", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"],
                     stdout=log, stderr=subprocess.STDOUT)
    time.sleep(8)
    print("🌐 [5/5] מחכה לקישור למטה (trycloudflare.com), לחץ עליו:")
    !./cloudflared tunnel --url http://127.0.0.1:8000
