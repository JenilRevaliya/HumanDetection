# 👁️ Human Detection System
> **Advanced AI-Powered Monitoring Dashboard**

Welcome to the **Human Detection System**, a cutting-edge computer vision application designed to monitor human presence, engagement, and safety in real-time. 

Developed by **[Jenil](https://jenilsoni.vercel.app)**.

---

## 🚀 Key Features

### 👤 Human & Face Detection
- **Real-Time Tracking**: Instantly detects humans and tracks face landmarks.
- **Status Logging**: Logs when a face is "Detected", "Lost", or "Recovered" after an absence.

### 📱 Phone Usage Detection
- **Smart AI**: Detects if a person is holding a mobile phone.
- **Anti-False Positive**: Uses a secondary **SVM Classifier** to verify phone detections, reducing errors.
- **Alerts**: Logs "Phone Detected" events if usage persists.

### 😴 Drowsiness / Eye Monitor
- **Blink Tracking**: Monitors eye aspect ratio to detect blinks.
- **Eyes Closed Alert**: Visually alerts (Pink Box) and logs events if eyes remain closed (drowsiness/sleeping).

### 📏 Distance Estimation
- **Proximity Sensor**: Estimates if a person is "Near", "Medium", or "Far" from the camera.
- **Out-of-Range Logs**: Tracks when a user moves "Too Far" for extended periods.

### 💾 Robust Data & Privacy
- **Database Logging**: All events are securely stored in a local SQLite database (`human_logs.db`).
- **Watermark**: Secure video feed with ownership watermark.
- **Privacy First**: All processing happens **locally** on your device.

---

## 🛠️ Controls & Usage

| Control | Function |
| :--- | :--- |
| **Camera Selector** | Choose between multiple connected webcams. |
| **Auto-Improve** | Automatically collects training data to improve phone detection. |
| **Train Model** | Retrains the AI classifier with new data for better accuracy. |
| **Clear Data / DB** | Resets the training dataset or wipes the event logs. |
| **View Database** | Opens a table view of the last 50 detected events. |
| **Open Readme** | Opens this guide! |

---

## ⚙️ Installation & Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the App**:
    ```bash
    python main.py
    ```
3.  **Build (Optional)**:
    Use `PyInstaller` to create a standalone executable.

---

## 👨‍💻 Developer

**Jenil**  
[Check out my Portfolio](https://jenilsoni.vercel.app)

*Built with Python, OpenCV, MediaPipe, and PyQt6.*
