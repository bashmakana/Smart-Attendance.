from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
import uvicorn
import numpy as np
import cv2
import torch
from facenet_pytorch import InceptionResnetV1
from typing import Dict, Optional
import os
import io
import logging
import csv
from datetime import datetime, timedelta

app = FastAPI(title="Face Recognition Attendance Server")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n======== DEVICE STATUS: {device} ========\n")

model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if face_cascade.empty():
    raise RuntimeError("Failed to load Haar cascade XML file!")

users: Dict[str, np.ndarray] = {}

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
EMB_DIR = os.path.join(DATA_DIR, 'embeddings')
IMG_DIR = os.path.join(DATA_DIR, 'images')
TEAMMATES_SOURCE_DIR = os.path.join(BASE_DIR, 'teammates_source_images')

ATTENDANCE_FILE = os.path.join(DATA_DIR, 'attendance.csv')
ATTENDANCE_COOLDOWN = 60 # Seconds
last_logged_time: Dict[str, datetime] = {} 

is_attendance_active = False

os.makedirs(EMB_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(TEAMMATES_SOURCE_DIR, exist_ok=True)

if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])

latest_frame: Optional[np.ndarray] = None
RECOGNITION_THRESHOLD = 0.8

def _safe_name(name: str) -> str:
    safe = ''.join(c for c in name if c.isalnum() or c in (' ', '_', '-')).strip()
    return safe.replace(' ', '_')

def save_embedding(name: str, embedding: np.ndarray):
    fname = os.path.join(EMB_DIR, f"{_safe_name(name)}.npy")
    np.save(fname, embedding)

def load_embeddings():
    loaded_count = 0
    for fname in os.listdir(EMB_DIR):
        if not fname.lower().endswith('.npy'):
            continue
        name = os.path.splitext(fname)[0]
        path = os.path.join(EMB_DIR, fname)
        try:
            emb = np.load(path)
            users[name] = emb
            loaded_count += 1
        except Exception as e:
            logging.warning(f"Failed to load embedding for {name}: {e}")
    print(f"Loaded {loaded_count} existing users from disk.")

def get_embedding(face_img: np.ndarray) -> np.ndarray:
    face_img = cv2.resize(face_img, (160, 160))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = face_img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(face_img).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(tensor)
    return embedding.cpu().numpy()[0]

def compare_embedding(embedding: np.ndarray, threshold: float = RECOGNITION_THRESHOLD) -> str:
    best_name = "Unknown"
    best_dist = float('inf')
    for name, ref_emb in users.items():
        if ref_emb is None: continue
        dist = np.linalg.norm(embedding - ref_emb)
        if dist < best_dist and dist < threshold:
            best_dist = dist
            best_name = name
    return best_name

def draw_results(img: np.ndarray, results: list):
    for res in results:
        x, y, w, h = res["box"]
        name = res["name"]
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    if is_attendance_active:
        cv2.putText(img, "REC: ON", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(img, "REC: PAUSED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

def log_attendance(name: str):
    """Logs the user to CSV if they haven't been logged recently."""
    if name == "Unknown": return

    now = datetime.now()
    if name in last_logged_time:
        elapsed = (now - last_logged_time[name]).total_seconds()
        if elapsed < ATTENDANCE_COOLDOWN: return

    try:
        with open(ATTENDANCE_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])
        last_logged_time[name] = now
        print(f"✅ ATTENDANCE MARKED: {name} at {now.strftime('%H:%M:%S')}")
    except Exception as e:
        print(f"Error logging attendance: {e}")

def register_local_image(name: str, image_path: str):
    safe_name = _safe_name(name)
    if safe_name in users: return

    img = cv2.imread(image_path)
    if img is None: return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100,100))
    
    if len(faces) == 0: return
    if len(faces) > 1:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

    x, y, w, h = faces[0]
    face_crop = img[y:y+h, x:x+w]
    embedding = get_embedding(face_crop)
    users[safe_name] = embedding
    save_embedding(safe_name, embedding)
    print(f"Successfully registered: {safe_name}")


@app.on_event("startup")
async def startup_event():
    print("\n--- SERVER STARTUP ---")
    load_embeddings()
    if os.path.exists(TEAMMATES_SOURCE_DIR):
        print(f"Scanning images in: {TEAMMATES_SOURCE_DIR}")
        for f in os.listdir(TEAMMATES_SOURCE_DIR):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                register_local_image(os.path.splitext(f)[0], os.path.join(TEAMMATES_SOURCE_DIR, f))
    print(f"Total users recognized: {len(users)}")
    print("--- SERVER READY ---\n")


@app.post("/start_attendance")
def start_attendance():
    global is_attendance_active
    is_attendance_active = True
    print(">>> ATTENDANCE STARTED <<<")
    return {"status": "Attendance Started", "active": True}

@app.post("/stop_attendance")
def stop_attendance():
    global is_attendance_active
    is_attendance_active = False
    print(">>> ATTENDANCE PAUSED <<<")
    return {"status": "Attendance Paused", "active": False}

@app.post("/recognize")
async def recognize(request: Request):
    global latest_frame
    
    body = await request.body() 
    
    if not body: 
        raise HTTPException(status_code=400, detail="Empty image data")

    try:
        nparr = np.frombuffer(body, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Image Decode Error: {e}")
        raise HTTPException(status_code=400, detail="Invalid image data")

    if img is None: 
        raise HTTPException(status_code=400, detail="Could not decode image")

    height, width = img.shape[:2]
    if width > 320:
        new_width = 320
        new_height = int(height * (320 / width))
        img = cv2.resize(img, (new_width, new_height))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    results = []
    for (x, y, w, h) in faces:
        face_crop = img[y:y+h, x:x+w]
        if face_crop.size == 0: continue
            
        embedding = get_embedding(face_crop)
        name = compare_embedding(embedding)
        
        if is_attendance_active:
            log_attendance(name)

        confidence = None
        if name != "Unknown" and users.get(name) is not None:
            confidence = float(np.linalg.norm(embedding - users.get(name)))

        results.append({
            "name": name,
            "confidence": confidence,
            "box": [int(x), int(y), int(w), int(h)]
        })

    annotated = img.copy()
    draw_results(annotated, results)
    latest_frame = annotated

    return JSONResponse({"results": results, "face_count": len(results)})


@app.get("/latest_frame")
def get_latest_frame():
    global latest_frame
    if latest_frame is None:
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        _, buf = cv2.imencode('.jpg', blank)
    else:
        _, buf = cv2.imencode('.jpg', latest_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")

@app.get("/get_attendance")
def get_attendance_sheet():
    if os.path.exists(ATTENDANCE_FILE):
        return FileResponse(ATTENDANCE_FILE, media_type='text/csv', filename="attendance.csv")
    else:
        return {"error": "No attendance records found yet."}

if __name__ == "__main__":
    import os
    host = os.getenv("HOST", "0.0.0.0") 
    port = int(os.getenv("PORT", 5001)) 
    print(f"Starting server on http://{host}:{port}")
    uvicorn.run("server:app", host=host, port=port, reload=False)