# app.py
from flask import Flask, render_template, Response, request, jsonify
import cv2
import datetime
import csv
import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import base64
import io
from collections import Counter

from model import SiameseEfficientNet  # ตอนนี้ข้างในใช้ FaceNet (InceptionResnetV1)

print(">>> START app.py (FaceNet Version)")

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on Device: {device}")

# ================== Transform สำหรับ FaceNet ==================
# รูปขนาด 160x160, ค่าพิกเซล normalize เป็น [-1, 1]
transform = transforms.Compose(
    [
        transforms.Resize((160, 160)),
        transforms.ToTensor(),  # [0,1]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [0,1] -> [-1,1]
    ]
)

# ================== โหลด FaceNet ผ่าน SiameseEfficientNet ==================
model = SiameseEfficientNet().to(device)
model.eval()
print("✅ Loaded FaceNet backbone (InceptionResnetV1, pretrained)")

# ================== โหลด Database ใบหน้า ==================
face_db = {}
if os.path.exists("face_db.pt"):
    try:
        loaded_db = torch.load("face_db.pt", map_location=device)
        for name, data in loaded_db.items():
            if isinstance(data, list):
                face_db[name] = [emb.to(device) for emb in data]
            else:
                face_db[name] = [data.to(device)]
        print(f"Loaded face_db: {len(face_db)} users")
    except Exception as e:
        print(f"❌ Error loading face_db: {e}")

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# เก็บเวลาที่เช็คชื่อครั้งล่าสุดของแต่ละคน (ใช้สำหรับ cooldown)
last_checkin = {}  # {"6610201234": datetime, ...}


def can_checkin(name, cooldown_minutes=10):
    """
    ตรวจว่าคนนี้สามารถเช็คชื่อได้ไหมตาม cooldown
    คืนค่า: (True/False, เวลาเช็คที่จะใช้ หรือ เวลาเก่าที่เช็คไปแล้ว)
    """
    now = datetime.datetime.now()
    last = last_checkin.get(name)

    if last is None:
        # ยังไม่เคยเช็คชื่อเลย
        last_checkin[name] = now
        return True, now

    diff_sec = (now - last).total_seconds()
    if diff_sec >= cooldown_minutes * 60:
        # เวลาห่างกันมากพอแล้ว ให้เช็คได้อีกครั้ง
        last_checkin[name] = now
        return True, now

    # ยังไม่ถึงเวลา cooldown
    return False, last


# ===== Identify Function (Multi-view) =====
def identify_face(pil_img, threshold=0.8):
    """
    รับ PIL Image -> คืน (name, distance)
    threshold อาจต้องจูนใหม่เมื่อใช้ FaceNet (ลอง print dist มาดูก่อน)
    """
    if model is None:
        return "System Error", 999.0

    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        q_emb = model.forward_once(img_tensor)

    best_name = "unknown"
    best_dist = 999.0

    for name, db_embs in face_db.items():
        for db_emb in db_embs:
            dist = F.pairwise_distance(q_emb, db_emb.unsqueeze(0)).item()
            if dist < best_dist:
                best_dist = dist
                best_name = name

    if best_dist > threshold:
        return "unknown", best_dist
    return best_name, best_dist


# ===== Loop หลัก + Smoothing (โหมด Live) =====
def gen_frames():
    cap = cv2.VideoCapture(0)

    csv_file = "attendance.csv"
    if not os.path.isfile(csv_file):
        with open(csv_file, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(["name", "datetime", "status"])

    history = []
    max_history = 7  # จำ 7 เฟรม

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            if len(history) > 0:
                history.pop(0)

        for x, y, w, h in faces:
            face_bgr = frame[y : y + h, x : x + w]
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_rgb)

            # 1. Identify
            raw_name, dist = identify_face(pil_img, threshold=0.8)

            # 2. Voting
            history.append(raw_name)
            if len(history) > max_history:
                history.pop(0)

            final_name = raw_name
            if history:
                count = Counter(history)
                best, vote = count.most_common(1)[0]
                # ต้องชนะโหวตเกินครึ่ง
                if vote >= (len(history) // 2) + 1:
                    final_name = best
                else:
                    final_name = "unknown"

            # 3. Draw
            color = (0, 255, 0) if final_name != "unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                f"{final_name} ({dist:.2f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

            # 4. Record (เช็คด้วย cooldown)
            if final_name != "unknown":
                ok, check_time = can_checkin(final_name, cooldown_minutes=10)
                if ok:
                    now_str = check_time.strftime("%Y-%m-%d %H:%M:%S")
                    with open(
                        "attendance.csv", "a", newline="", encoding="utf-8-sig"
                    ) as f:
                        csv.writer(f).writerow([final_name, now_str, "check-in"])
                    print(f"[Check-in] {final_name} เวลา {now_str}")
                else:
                    # แค่แจ้งใน console เฉย ๆ ว่าเพิ่งเช็คไป ยังไม่ครบ cooldown
                    print(f"[Cooldown] ข้าม {final_name} (ยังไม่ครบเวลาเช็คซ้ำ)")

        ret, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )

    cap.release()


def scan_once(num_frames=15, threshold=0.8):
    """
    เปิดกล้องสั้น ๆ เก็บหลายเฟรม → ใช้ identify_face + voting
    คืน (final_name, best_dist หรือ None ถ้าไม่มีหน้า)
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ ไม่สามารถเปิดกล้องได้ใน scan_once")
        return None, None

    history = []
    dist_history = []

    print(f">>> เริ่มสแกนครั้งเดียว (เก็บ {num_frames} เฟรม)")

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            continue

        # เลือกหน้าใหญ่สุดในเฟรม (กันหลายคน)
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])

        face_bgr = frame[y : y + h, x : x + w]
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb)

        raw_name, dist = identify_face(pil_img, threshold=threshold)
        history.append(raw_name)
        dist_history.append(dist)

    cap.release()

    if not history:
        print("❌ ไม่เจอหน้าเลยในระหว่างสแกน")
        return "unknown", None

    # ใช้ voting จาก history (คล้ายใน gen_frames)
    count = Counter(history)
    best_name, votes = count.most_common(1)[0]

    # ถ้าส่วนใหญ่เป็น unknown อยู่ดี → ถือว่าไม่รู้จัก
    if best_name == "unknown":
        return "unknown", min(d for d in dist_history if d is not None)

    # บังคับให้ต้องชนะเกิน 70% ของเฟรมที่มี
    req_ratio = 0.7
    if votes < int(len(history) * req_ratio):
        print(f"ไม่มั่นใจ: {best_name} ได้โหวต {votes}/{len(history)}")
        return "unknown", min(
            d for d, n in zip(dist_history, history) if n == best_name
        )

    # เลือก dist ที่ดีที่สุดของชื่อที่ชนะโหวต
    best_dist = min(d for d, n in zip(dist_history, history) if n == best_name)

    print(
        f"[SCAN ONCE RESULT] name={best_name}, dist={best_dist:.4f}, votes={votes}/{len(history)}"
    )
    return best_name, best_dist


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/logs")
def logs():
    rows = []
    if os.path.exists("attendance.csv"):
        with open("attendance.csv", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
    return render_template("logs.html", rows=rows)


@app.route("/enroll")
def enroll():
    return render_template("enroll.html")


@app.route("/scan")
def scan():
    """Render a simple Scan page (placeholder).
    The page can call the existing API `/api/checkin_once` to perform a camera scan.
    """
    return render_template("scan.html")


# ============== API Enroll ==============
@app.route("/api/enroll", methods=["POST"])
def api_enroll():
    global face_db
    data = request.get_json()
    name = data.get("name")
    img_data = data.get("image")

    if not name or not img_data:
        return jsonify({"ok": False}), 400

    try:
        header, encoded = img_data.split(",", 1)
        pil_img = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")

        img_tensor = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.forward_once(img_tensor)

        if name in face_db:
            face_db[name].append(emb.to(device))
        else:
            face_db[name] = [emb.to(device)]

        cpu_db = {}
        for k, v_list in face_db.items():
            cpu_db[k] = [e.cpu() for e in v_list]
        torch.save(cpu_db, "face_db.pt")

        return jsonify({"ok": True, "message": f"Saved {name}"})
    except Exception as e:
        print(e)
        return jsonify({"ok": False}), 500


# ============== API Scan Once (ยังใช้ได้ ถึงหน้าเว็บจะไม่เรียกแล้ว) ==============
@app.route("/api/checkin_once", methods=["POST"])
def api_checkin_once():
    name, dist = scan_once(num_frames=15, threshold=0.8)

    if name is None:
        return jsonify({"ok": False, "error": "camera_error"}), 500

    if name == "unknown" or dist is None:
        return jsonify({"ok": False, "name": "unknown"}), 200

    # เตรียมไฟล์ CSV ถ้ายังไม่มี
    csv_file = "attendance.csv"
    if not os.path.isfile(csv_file):
        with open(csv_file, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(["name", "datetime", "status"])

    # ===== ตรวจ cooldown =====
    ok, check_time = can_checkin(name, cooldown_minutes=10)

    if not ok:
        return jsonify(
            {"ok": True, "name": name, "dist": round(dist, 3), "message": "cooldown"}
        )

    now_str = check_time.strftime("%Y-%m-%d %H:%M:%S")
    with open(csv_file, "a", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow([name, now_str, "check-in"])
    print(f"[Scan Once Check-in] {name} เวลา {now_str}")

    return jsonify(
        {"ok": True, "name": name, "dist": round(dist, 3), "message": "new_checkin"}
    )


@app.route("/api/list_faces", methods=["GET"])
def api_list_faces():
    return jsonify({"ok": True, "names": sorted(face_db.keys())})


@app.route("/api/delete_face", methods=["POST"])
def api_delete_face():
    global face_db
    data = request.get_json()
    name = (data.get("name") or "").strip()
    if name in face_db:
        del face_db[name]
        cpu_db = {}
        for k, v_list in face_db.items():
            cpu_db[k] = [e.cpu() for e in v_list]
        torch.save(cpu_db, "face_db.pt")
        return jsonify({"ok": True})
    return jsonify({"ok": False}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
