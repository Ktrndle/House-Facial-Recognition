import cv2
import numpy as np
import os
import threading
import time
import speech_recognition as sr
import pyttsx3
import serial
from google import genai

# ---------- CONFIGURATION ----------
AI_API_KEY = "AIzaSyDMXx-F-5Ig5Z1YjvKz4pbg7-jgXaRcdPo"
SERIAL_PORT = 'COM4'
BAUDRATE = 9600

# ---------- SETUP ----------
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1)
genai_client = genai.Client(api_key=AI_API_KEY)
model_name = "gemini-2.5-flash-preview-04-17"

arduino = serial.Serial(port=SERIAL_PORT, baudrate=BAUDRATE, timeout=1)
time.sleep(2)

speech_lock = threading.Lock()

# ---------- STORAGE ----------
if not os.path.exists("registered_faces"):
    os.makedirs("registered_faces")

registered_users = {}
label_user_map = {}
user_label_map = {}
admin_labels = set()
current_label = 0
next_user_id = 1

ai_active_flag = False
ai_thread_running = False

# ---------- UTILITY FUNCTIONS ----------
def get_user_details():
    name = input("Enter Name: ")
    age = input("Enter Age: ")
    gender = input("Enter Gender (M/F): ")
    return {"name": name, "age": age, "gender": gender}

def capture_multiple_faces(user_id, label_name):
    print(f"[Capturing for {label_name}] Move slowly, vary lighting...")
    captured = 0
    while captured < 60:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(cv2.GaussianBlur(gray, (5, 5), 0))
        faces = face_cascade.detectMultiScale(gray, 1.1, 6, minSize=(90, 90))
        if len(faces) == 1:
            x, y, w, h = faces[0]
            face_crop = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
            folder_path = f"registered_faces/{user_id}"
            os.makedirs(folder_path, exist_ok=True)
            cv2.imwrite(f"{folder_path}/{captured}.png", face_crop)
            captured += 1
            print(f"Captured {captured}/60")
        cv2.imshow('Capturing Face', frame)
        cv2.waitKey(300)
    cv2.destroyWindow('Capturing Face')

def prepare_training_data():
    global current_label
    faces, labels = [], []
    label_user_map.clear()
    user_label_map.clear()
    current_label = 0
    for folder in os.listdir("registered_faces"):
        folder_path = os.path.join("registered_faces", folder)
        if os.path.isdir(folder_path):
            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.equalizeHist(img)
                    img = cv2.GaussianBlur(img, (3, 3), 0)
                    faces.append(img)
                    labels.append(current_label)
                    faces.append(cv2.convertScaleAbs(img, alpha=1.2, beta=30))
                    labels.append(current_label)
                    faces.append(cv2.convertScaleAbs(img, alpha=0.8, beta=-30))
                    labels.append(current_label)
            label_user_map[current_label] = folder
            user_label_map[folder] = current_label
            current_label += 1
    return faces, labels

def retrain_recognizer():
    faces, labels = prepare_training_data()
    if faces and labels:
        recognizer.train(faces, np.array(labels))
        print("[Recognizer Retrained]")

# ---------- AI FUNCTIONS ----------
command_map = {
    "open main door": "open main door",
    "open front door": "open main door",
    "open back door": "open back door",
    "close main door": "close main door",
    "close back door": "close back door",
    "close all doors": "close all doors",
    "turn on lights": "turn on lights",
    "turn off lights": "turn off lights"
}

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening... Please speak clearly.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=None)
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            print("Could not understand the audio.")
        except sr.RequestError:
            print("Could not request results.")
        except Exception as e:
            print(f"Unexpected error: {e}")
    return ""

def speak_text(text):
    speech_lock.acquire()
    try:
        engine.say(text)
        engine.runAndWait()
    finally:
        speech_lock.release()

def chat_with_gemini(prompt):
    try:
        response = genai_client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"[Gemini Error] {e}")
        return "Sorry, I'm having trouble connecting to Gemini API."

def find_best_command(user_text):
    user_text = user_text.lower()
    for phrase in command_map:
        if phrase in user_text:
            return command_map[phrase]
    return None

def send_to_arduino(command):
    try:
        arduino.write((command + '\n').encode())
        print(f"Sent to Arduino: {command}")
    except Exception as e:
        print(f"Failed to send to Arduino: {e}")

def ai_command_loop():
    global ai_thread_running
    ai_thread_running = True
    while True:
        if not ai_active_flag:
            time.sleep(0.2)
            continue

        if speech_lock.locked():
            time.sleep(0.1)
            continue

        user_input = recognize_speech()
        if not user_input:
            continue

        print(f"You: {user_input}")

        if user_input.lower() in ['exit', 'quit', 'bye']:
            speak_text("Goodbye!")
            continue

        matched_command = find_best_command(user_input)

        if matched_command:
            send_to_arduino(matched_command)
            speak_text(f"Executing command: {matched_command}")
        else:
            response = chat_with_gemini(user_input)
            print(f"Chatbot: {response}")
            speak_text(response)
    ai_thread_running = False

# ---------- ADMIN REGISTRATION ----------
print("[Waiting for Admin Registration...]")
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(cv2.GaussianBlur(gray, (5, 5), 0))
    faces = face_cascade.detectMultiScale(gray, 1.1, 6, minSize=(90, 90))
    if len(faces) == 1:
        print("[Admin Detected - Start Registration]")
        details = get_user_details()
        details['admin'] = True
        user_id = f"user_{next_user_id}"
        registered_users[user_id] = details
        next_user_id += 1
        capture_multiple_faces(user_id, details['name'])
        retrain_recognizer()
        label = user_label_map[user_id]
        admin_labels.add(label)
        print(f"[Admin {details['name']} Registered]")
        break
    elif len(faces) > 1:
        print("[Only Admin Should be in Frame!]")
    cv2.imshow("Register Admin", frame)
    if cv2.waitKey(1) == ord('q'):
        exit()

# ---------- MAIN LOOP ----------
print("[System Running]")
if not ai_thread_running:
    threading.Thread(target=ai_command_loop, daemon=True).start()

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(cv2.GaussianBlur(gray, (5,5), 0))
    faces = face_cascade.detectMultiScale(gray, 1.1, 6, minSize=(90,90))

    admin_detected = False

    for (x, y, w, h) in faces:
        face_crop = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
        label_text = "Unknown"
        color = (0, 0, 255)
        confidence = 999
        user_details = {}

        if len(label_user_map) > 0:
            label, confidence = recognizer.predict(face_crop)
            user_id = label_user_map.get(label, None)
            if confidence < 90 and user_id and user_id in registered_users:
                user_details = registered_users[user_id]
                label_text = user_details.get("name", "Unknown")
                if user_label_map.get(user_id) in admin_labels:
                    label_text += " [Admin]"
                    admin_detected = True
                    color = (0, 255, 0)
                else:
                    label_text += f" [{user_details.get('gender','')}, {user_details.get('age','')}]"
                    color = (255, 255, 0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if admin_detected:
        if not ai_active_flag:
            print("[Admin In View] AI Activated")
        ai_active_flag = True
    else:
        if ai_active_flag:
            print("[Admin Not Detected] AI Paused")
        ai_active_flag = False

    cv2.imshow("Face Recognition + AI Control", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
