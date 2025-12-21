# attendance.py
import cv2
import csv
import os
import datetime
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# นำเข้า Class โมเดลใหม่
from model import SiameseEfficientNet

# ===== 0) Config การแปลงภาพ (ต้องเหมือน enroll.py เป๊ะ) =====
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 1) โหลดโมเดลใหม่ =====
print(f"Loading Siamese Model on {device}...")
model = SiameseEfficientNet().to(device)

if os.path.exists("siamese_model.pth"):
    model.load_state_dict(torch.load("siamese_model.pth", map_location=device))
    model.eval()
else:
    print("❌ Error: ไม่พบไฟล์ siamese_model.pth")
    exit()

# ===== 2) โหลด Database และย้ายข้อมูลไปบน Device เดียวกับ Model =====
if os.path.exists("face_db.pt"):
    face_db = torch.load("face_db.pt", map_location=device)
    print("Loaded face_db:", list(face_db.keys()))
    
    # ย้าย Embedding ใน DB ไปอยู่บน GPU/CPU ให้หมด เพื่อความเร็ว
    for name in face_db:
        face_db[name] = face_db[name].to(device)
else:
    print("❌ Error: ไม่พบไฟล์ face_db.pt (กรุณารัน enroll.py ก่อน)")
    exit()

# ===== 3) เตรียมไฟล์ CSV =====
csv_file = "attendance.csv"
file_exists = os.path.isfile(csv_file)

f = open(csv_file, "a", newline="", encoding="utf-8-sig")
writer = csv.writer(f)

if not file_exists:
    writer.writerow(["name", "datetime", "status"])

marked_names = set()

# ===== 4) ฟังก์ชันระบุตัวตน (Core Logic) =====
def identify_face(pil_img, threshold=0.8): # ปรับ Threshold ตามความเหมาะสม (0.7 - 1.0)
    # 1. Preprocess ภาพ
    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    # 2. แปลงเป็น Embedding
    with torch.no_grad():
        q_emb = model.forward_once(img_tensor)

    best_name = "unknown"
    best_dist = 999.0

    # 3. เทียบกับทุกคนใน DB
    for name, db_emb in face_db.items():
        # คำนวณระยะห่าง (Euclidean Distance)
        # เนื่องจากเรา Normalize vector แล้ว ระยะห่างควรจะอยู่ช่วง 0.0 - 2.0
        dist = F.pairwise_distance(q_emb, db_emb.unsqueeze(0)).item()
        
        if dist < best_dist:
            best_dist = dist
            best_name = name

    # 4. ตัดสินด้วย Threshold
    if best_dist > threshold:
        return "unknown", best_dist
    
    return best_name, best_dist

# ===== 5) Main Loop เปิดกล้อง =====
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ไม่สามารถเปิดกล้องได้")
    f.close()
    exit()

print(">>> เริ่มระบบเช็คชื่อ (กด 'q' เพื่อออก)")
print(f"Threshold ที่ตั้งไว้: 0.8 (ถ้าจำผิดคนให้ลดค่า, ถ้าจำไม่ได้ให้เพิ่มค่า)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_bgr = frame[y:y+h, x:x+w]

        # แปลงเป็น PIL เพื่อส่งเข้าโมเดล
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb)

        # ทำนายชื่อ
        name, dist = identify_face(pil_img, threshold=0.8)

        # UI: สีเขียว=รู้จัก, สีแดง=ไม่รู้จัก
        color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
        
        # วาดกรอบ
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # เขียนชื่อและระยะห่าง (Distance ช่วยให้เราจูน Threshold ได้ง่ายขึ้น)
        label = f"{name} ({dist:.2f})"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # บันทึกเวลา (Check-in)
        if name != "unknown" and name not in marked_names:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([name, now, "check-in"])
            f.flush() # บันทึกลงไฟล์ทันทีกันโปรแกรมค้าง
            marked_names.add(name)
            print(f"[✅ CHECK-IN] {name} เวลา {now} (Dist: {dist:.4f})")

    cv2.imshow("Face Attendance Siamese", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
f.close()
cv2.destroyAllWindows()