# Midi‑App

Audio‑to‑MIDI pipeline built with **FastAPI**.

## תיאור
הפרויקט מקבל קבצי אודיו, מריץ מודלים של למידת מכונה (torch, spacy, …) ומחזיר תוצאה בפורמט MIDI. בנוסף הוא מציע ממשק API להעלאת קבצים והורדת הפלט.

## דרישות
- Python 3.11 (או גרסה תואמת)
- `pip` מודע ל‑`requirements.txt`
- `FastAPI` + `uvicorn`

## התקנה (ב‑venv)
```powershell
# יצירת סביבת וירטואלית
python -m venv venv
.\venv\Scripts\Activate.ps1   # (או activate.bat ל‑cmd)

# עדכון pip
python -m pip install --upgrade pip

# התקנת תלויות – רק גלגלים בינאריים
python -m pip install --only-binary=:all: -r requirements.txt
```

## הרצה מקומית
```powershell
# הפעלת ה‑API
uvicorn app:app --reload
```
ה‑API יהיה זמין ב‑`http://127.0.0.1:8000`.
תיעוד אוטומטי של Swagger נמצא ב‑`/docs`.

## Docker
קובץ `Dockerfile` כבר נמצא במאגר. לבנייה והרצה:
```bash
docker build -t midi-app .
docker run -p 8000:8000 midi-app
```

## קבצים חשובים
- `app.py` – קוד ה‑FastAPI.
- `requirements.txt` – תלויות הפרויקט (גרסאות מעודכנות, חבילות שלא זמינות במצב בנייה מושבתות בקובץ עם `#`).
- `.gitignore` – מתעדכן אוטומטית כדי לא לכלול קבצים גדולים (Python‑311, קבצי EXEC, סביבת‑venv ועוד).


---
תיקון בטא לבנייה מהירה – מתאריך 2026-11-07 00:05.

