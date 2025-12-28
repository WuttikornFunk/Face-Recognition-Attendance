# 📸 Face Recognition Attendance System

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Framework-Flask-black?style=for-the-badge&logo=flask)
![OpenCV](https://img.shields.io/badge/Computer_Vision-OpenCV-green?style=for-the-badge&logo=opencv)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

**A smart, contactless attendance management system leveraging Deep Learning technology.**

This project automates the attendance tracking process using real-time face recognition. Built with **Python** and **Flask**, it utilizes **MTCNN** for face detection and **FaceNet (InceptionResnetV1)** for generating high-quality face embeddings, ensuring accurate identification and automated logging.
---

## ✨(Key Features)
⚡ Real-time Detection: Instant face detection and recognition from a live webcam feed.

🧠 Deep Learning Core: Powered by FaceNet (InceptionResnetV1) for state-of-the-art accuracy in feature extraction.

🛡️ Basic Anti-Spoofing: Includes measures to distinguish between real faces and static photographs.

📝 Auto-Logging: Automatically records entry times, names, and dates into a CSV/Excel file (logs.csv) using Pandas.

💻 Web Dashboard: A user-friendly web interface built with Flask for monitoring and management.

## 🛠️ (Tech Stack)
* **Core:** Python 3.10
* **AI Model:** FaceNet (InceptionResnetV1) + MTCNN
* **Computer Vision:** OpenCV
* **Web Framework:** Flask
* **Data Handling:** Pandas

## 🚀(Installation)

1. **Clone project**
   ```bash
   git clone [https://github.com/Wuttikorn777/Face-Recognition-Attendance.git](https://github.com/Wuttikorn777/Face-Recognition-Attendance.git)

2. **install Library **
   ```bash
    pip install -r requirements.txt

3. **RUN**
    ```bash
    python app.py
    ระบบจะเปิดเว็บที่ URL: http://127.0.0.1:5000
