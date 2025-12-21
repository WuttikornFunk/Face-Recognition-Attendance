# utils.py
import os
import torch
from torchvision import transforms
from PIL import Image

# นำเข้า Class จาก model.py
from model import SiameseEfficientNet

# เลือก device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 1) Transform (ต้องมี Normalize เหมือนตอน Train) =====
inference_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model(model_path="siamese_model.pth"):
    """โหลด Model และ Weight"""
    print(f"Loading model from {model_path} on {device}...")
    
    # สร้าง instance ของโมเดล
    model = SiameseEfficientNet().to(device)
    
    if os.path.exists(model_path):
        # โหลด weight
        state_dict = torch.load(model_path, map_location=device)
        
        # ลองโหลดแบบปกติ
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            # กรณี key ไม่ตรงกันเป๊ะๆ (เช่น ติด prefix 'module.') ให้ลองโหลดแบบไม่ strict
            print("⚠️ Warning: Key mismatch, loading with strict=False")
            model.load_state_dict(state_dict, strict=False)
            
        model.eval() # เข้าโหมดใช้งาน (ปิด Dropout)
        return model
    else:
        print(f"❌ Error: Model file not found at {model_path}")
        return None

def load_face_db(db_path="face_db.pt"):
    """โหลด Database ใบหน้า"""
    if os.path.exists(db_path):
        print(f"Loading face_db from {db_path}")
        # โหลดมาแล้วส่งเข้า device เลย เพื่อความเร็วตอนเทียบ
        db = torch.load(db_path, map_location=device)
        return db
    else:
        return {}

def save_face_db(face_db, db_path="face_db.pt"):
    """บันทึก Database (แปลงกลับเป็น CPU ก่อนเซฟ เพื่อให้ย้ายเครื่องได้ง่าย)"""
    cpu_db = {name: emb.cpu() for name, emb in face_db.items()}
    torch.save(cpu_db, db_path)
    print(f"Saved face_db to {db_path}")

def get_embedding_from_pil(img_pil, model):
    """
    รับรูป PIL -> คืน embedding tensor
    ใช้ forward_once แทนการส่งรูปคู่
    """
    # 1. Preprocess
    img_tensor = inference_transform(img_pil).unsqueeze(0).to(device)
    
    # 2. Forward Pass (ใช้ forward_once)
    with torch.no_grad():
        embedding = model.forward_once(img_tensor)
        
    # คืนค่ากลับมา (ยังอยู่บน GPU ก็ได้ ถ้าจะเอาไปเทียบต่อเลย)
    return embedding.squeeze(0)