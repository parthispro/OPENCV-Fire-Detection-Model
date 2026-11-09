# OPENCV Fire Detection Model

A real-time fire detection system built with YOLOv8, OpenCV, and Flask. The system analyzes video input from uploaded files or a live webcam feed to detect fire, trigger an audio alarm, and dispatch an email alert.

---

## Features

- Video upload mode for fire detection on pre-recorded footage
- Real-time webcam monitoring with live inference
- Audio alarm triggered on positive fire detection
- Automated email alerts via SMTP on detection events
- Custom-trained YOLOv8 model optimized for fire classification

---

## Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv8 (Ultralytics) |
| Computer Vision | OpenCV, CVZone |
| Web Framework | Flask |
| Email Dispatch | SMTP (SendGrid) |
| Configuration | python-dotenv |

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip
- A SendGrid account for email alerts

### Installation

```bash
git clone https://github.com/parthispro/OPENCV-Fire-Detection-Model.git
cd OPENCV-Fire-Detection-Model
python -m venv venv
source venv/bin/activate
pip install flask ultralytics opencv-python cvzone python-dotenv pygame
```

### Configuration

```bash
cp .env.example .env
```

Open `.env` and fill in your values:

```env
SECRET_KEY=your-secret-key
MODEL_PATH=fire.pt
SMTP_HOST=smtp.sendgrid.net
SMTP_PORT=587
SMTP_USERNAME=apikey
SMTP_PASSWORD=your-sendgrid-api-key
EMAIL_SENDER=your-email@gmail.com
EMAIL_RECEIVER=receiver@gmail.com
ALARM_FILE=firealarm.mp3
CONFIDENCE_THRESHOLD=50
```

### Running the Application

```bash
python app.py
```

Navigate to `http://127.0.0.1:5000` in your browser.

---

## Project Structure

```
OPENCV-Fire-Detection-Model/
├── app.py                  # Flask application and routing
├── fire.py                 # Standalone detection script
├── fire.pt                 # YOLOv8 model weights
├── firealarm.mp3           # Alert audio file
├── templates/
│   ├── index.html          # Video upload interface
│   └── realtime.html       # Realtime monitoring interface
├── static/
│   └── css/style.css       # Stylesheet
├── uploads/                # Uploaded video storage (auto-created)
├── .env                    # Local environment config (not committed)
├── .env.example            # Environment variable reference template
└── .gitignore
```

---

## How It Works

Each video frame is passed through the YOLOv8 model for inference. When a detection exceeds the configured confidence threshold, the system draws a bounding box around the detected region, plays an audio alarm, and sends an email notification with the timestamp of the event.

---

## Security

The `.env` file containing credentials is excluded from version control via `.gitignore`. Use `.env.example` as a reference when configuring the project on a new machine. Never commit API keys or passwords directly to source code.

---

## License

MIT License
