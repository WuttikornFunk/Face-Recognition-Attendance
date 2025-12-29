# 📸 Face Recognition Attendance System

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Framework-Flask-black?style=for-the-badge&logo=flask)
![OpenCV](https://img.shields.io/badge/Computer_Vision-OpenCV-green?style=for-the-badge&logo=opencv)
![Pandas](https://img.shields.io/badge/Data-Pandas-150458?style=for-the-badge&logo=pandas)

> **A smart, contactless attendance management system leveraging Deep Learning technology for real-time identification.**

This project automates the attendance tracking process using a webcam. Built with **Python** and **Flask**, it integrates state-of-the-art Computer Vision models (**MTCNN** & **FaceNet**) to detect faces, verify identities, and automatically log entry times into an Excel/CSV file.

---

## ⚙️ System Architecture 

The system operates through a specialized pipeline to ensure accurate recognition:


🛠️ Tech Stack & Tools 
1. Programming Languages
   
Python (3.10+): The core logic, backend processing, and AI implementation.

HTML5 / CSS3: Frontend interface for the camera feed and dashboard.

JavaScript: Client-side handling for real-time interactions.

2. AI & Computer Vision Modules
OpenCV (cv2): Handles image processing, video stream capture, and frame manipulation.

MTCNN: A deep learning model used for accurate Face Detection (locating faces in the frame).

FaceNet (InceptionResnetV1): Used for Face Recognition (generating 128-dimensional embeddings to identify unique facial features).

NumPy: Performs high-speed matrix calculations for image data.

3. Web Framework & Data Management
Flask: A lightweight web framework serving the application and routing requests.

Pandas: Manages attendance logs and exports data to .csv or .xlsx formats.

✨ Key Features
Real-time Recognition: Instantly identifies registered users via live video feed.

Anti-Spoofing (Basic): Algorithm to distinguish between real faces and static photos.

Automated Logging: Saves "Time-In", "Name", and "Date" automatically without manual input.

Dashboard UI: A user-friendly web interface to view the camera and status.

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
