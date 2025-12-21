# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision import models   # ไม่จำเป็นแล้ว ถ้าไม่ใช้ EfficientNet
from facenet_pytorch import InceptionResnetV1

class SiameseEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        # ใช้ FaceNet (InceptionResnetV1) pretrained บน VGGFace2
        self.backbone = InceptionResnetV1(
            pretrained='vggface2',  # หรือ 'casia-webface'
            classify=False
        )

        # ถ้าไม่ได้เทรนต่อ ให้ freeze น้ำหนักทั้งหมด
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward_once(self, x):
        """
        รับภาพขนาด (B, 3, 160, 160) ค่า pixel อยู่ในช่วง [-1, 1]
        คืนค่า embedding ขนาด (B, 512)
        """
        x = self.backbone(x)          # (B,512)
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, img1, img2):
        feat1 = self.forward_once(img1)
        feat2 = self.forward_once(img2)
        return feat1, feat2
