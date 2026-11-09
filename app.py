import flask
from flask import Flask, render_template, request, redirect, url_for, flash, Response
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cvzone
import cv2
import math
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import ssl
from datetime import datetime
import time
import threading
import pygame
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'fallback-dev-key')

MODEL_PATH            = os.getenv('MODEL_PATH', 'fire.pt')
SMTP_HOST             = os.getenv('SMTP_HOST', 'smtp.sendgrid.net')
SMTP_PORT             = int(os.getenv('SMTP_PORT', 587))
SMTP_USERNAME         = os.getenv('SMTP_USERNAME', 'apikey')
SMTP_PASSWORD         = os.getenv('SMTP_PASSWORD', '')
EMAIL_SENDER          = os.getenv('EMAIL_SENDER', '')
EMAIL_RECEIVER        = os.getenv('EMAIL_RECEIVER', '')
ALARM_FILE            = os.getenv('ALARM_FILE', 'firealarm.mp3')
CONFIDENCE_THRESHOLD  = int(os.getenv('CONFIDENCE_THRESHOLD', 50))
GPS_LOCATION          = os.getenv('GPS_LOCATION', 'https://maps.app.goo.gl/DP8jKnojixhz1j4q6')

# Initialize pygame mixer for alert sound
try:
    pygame.mixer.init()
except Exception as e:
    print(f"Pygame mixer init warning: {e}")

model = YOLO(MODEL_PATH)
classnames = ['Fire']

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Alert tracking to prevent spamming audio and emails
last_alarm_time = 0
last_email_time = 0

def send_alert_email_async():
    if not (SMTP_HOST and SMTP_USERNAME and SMTP_PASSWORD and EMAIL_SENDER and EMAIL_RECEIVER):
        return
    try:
        context = ssl.create_default_context()
        message = MIMEMultipart()
        message["From"] = EMAIL_SENDER
        message["To"] = EMAIL_RECEIVER
        current_time = str(datetime.now())
        message["Subject"] = "Allen EmberEye - Fire detect Alert!"
        body = (
            f"We've detected a potential fire incident at {current_time}.\n\n"
            f"Location: {GPS_LOCATION}\n\n"
            "Immediate action may be required to ensure safety. "
            "Please investigate and take appropriate measures. "
            "Your prompt attention to this matter is greatly appreciated."
        )
        message.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls(context=context)
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(message)
    except Exception as e:
        print(f"Error sending email: {e}")

def trigger_alerts():
    global last_alarm_time, last_email_time
    now = time.time()

    # Play alarm audio (cooldown: 5 seconds)
    if now - last_alarm_time > 5:
        last_alarm_time = now
        try:
            if os.path.exists(ALARM_FILE):
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.load(ALARM_FILE)
                    pygame.mixer.music.play()
        except Exception as e:
            print(f"Audio playback error: {e}")

    # Send email alert (cooldown: 60 seconds)
    if now - last_email_time > 60:
        last_email_time = now
        threading.Thread(target=send_alert_email_async, daemon=True).start()

def generate_frames(video_source):
    cap = cv2.VideoCapture(video_source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        result = model(frame, stream=True)
        fire_detected = False

        for info in result:
            boxes = info.boxes
            for box in boxes:
                confidence = math.ceil(box.conf[0] * 100)
                Class = int(box.cls[0])
                if confidence > CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                    cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                       scale=1.5, thickness=2)
                    fire_detected = True

        if fire_detected:
            trigger_alerts()

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    f = request.files.get('file')
    if not f or f.filename == '':
        return 'No file selected.', 400
    if not allowed_file(f.filename):
        return 'Invalid file type. Please upload a video file.', 400

    filename = secure_filename(f.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(video_path)
    return render_template('index.html', video_file=filename)

@app.route('/video_feed/<filename>')
def video_feed(filename):
    safe_filename = secure_filename(filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
    if not os.path.exists(video_path):
        return 'Video not found', 404
    return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/realtime')
def realtime():
    streaming = request.args.get('stream') == '1'
    return render_template('realtime.html', streaming=streaming)

@app.route('/webcam_feed')
def webcam_feed():
    return Response(generate_frames(0), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true')
