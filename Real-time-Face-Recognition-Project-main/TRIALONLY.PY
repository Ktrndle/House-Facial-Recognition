import cv2
import numpy as np
import os

# Initialize
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Storage
if not os.path.exists("registered_faces"):
    os.makedirs("registered_faces")

registered_users = {}
label_user_map = {}
user_label_map = {}
admin_labels = set()
current_label = 0
next_user_id = 1

# Functions
def get_user_details():
    name = input("Enter Name: ")
    age = input("Enter Age: ")
    gender = input("Enter Gender (M/F): ")
    return {"name": name, "age": age, "gender": gender}

def ask_register_relation():
    while True:
        relation = input("Register this person as (family/friend/relative)? (Type 'no' to skip): ").lower()
        if relation in ['family', 'friend', 'relative', 'no']:
            return relation
        print("Please type: family, friend, relative, or no.")

def capture_multiple_faces(user_id, label_name):
    print(f"[Capturing for {label_name}] Look around slowly while facing the camera...")

    captured = 0
    capture_delay = 300  # milliseconds between frames

    while captured < 50:  # Increased from 30 to 50
        ret, frame = cap.read()
        if not ret:
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        gray_frame = cv2.equalizeHist(gray_frame)

        faces = face_cascade.detectMultiScale(gray_frame, 1.1, 6, minSize=(90, 90))

        if len(faces) == 1:
            (x, y, w, h) = faces[0]
            face_crop = gray_frame[y:y+h, x:x+w]
            face_crop = cv2.resize(face_crop, (100, 100))

            folder_path = f"registered_faces/{user_id}"
            os.makedirs(folder_path, exist_ok=True)

            img_path = f"{folder_path}/{captured}.png"
            cv2.imwrite(img_path, face_crop)

            captured += 1
            print(f"Captured sample {captured}/50")

        cv2.imshow('Capturing Face Data', frame)
        cv2.waitKey(capture_delay)  # Wait to let user move/change angle

    cv2.destroyWindow('Capturing Face Data')
    print("[Done] Face data collection complete!")


def prepare_training_data():
    faces = []
    labels = []
    label_user_map.clear()
    user_label_map.clear()
    global current_label
    current_label = 0

    for folder_name in os.listdir("registered_faces"):
        folder_path = os.path.join("registered_faces", folder_name)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    faces.append(img)
                    labels.append(current_label)
            label_user_map[current_label] = folder_name
            user_label_map[folder_name] = current_label
            current_label += 1

    return faces, labels

def retrain_recognizer():
    faces, labels = prepare_training_data()
    if faces and labels:
        recognizer.train(faces, np.array(labels))
        print("[Recognizer retrained]")

# ---- MAIN ----

print("\n[System Starting] Please stand in front of the camera...")

# Admin registration
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    gray_frame = cv2.equalizeHist(gray_frame)

    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 6, minSize=(90, 90))

    if len(faces) == 1:
        print("\n[Admin Registration]")
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
        print("[Multiple faces detected] Only admin should be present!")

    cv2.imshow('Face Recognition System', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        exit()

print("\n[System Running] Admin is registered.")

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    gray_frame = cv2.equalizeHist(gray_frame)

    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 6, minSize=(90, 90))

    known_faces = []
    unknown_faces = []

    for (x, y, w, h) in faces:
        face_crop = gray_frame[y:y+h, x:x+w]
        face_crop = cv2.resize(face_crop, (100, 100))

        label_text = "Unknown"
        confidence = 999  # Default high

        if len(label_user_map) > 0:
            label, confidence = recognizer.predict(face_crop)
            user_id = label_user_map.get(label)

            if confidence < 65:  # Lower threshold = stricter
                details = registered_users.get(user_id, {})
                label_text = details.get('name', 'Unknown')
                known_faces.append((x, y, w, h, label_text))
            else:
                unknown_faces.append((x, y, w, h))
        else:
            unknown_faces.append((x, y, w, h))

        color = (0, 255, 0) if label_text != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Register new face if 1 known and 1 unknown detected
    if len(known_faces) == 1 and len(unknown_faces) == 1:
        print("\n[Registered person detected] 1 unknown nearby.")
        (x, y, w, h) = unknown_faces[0]
        relation = ask_register_relation()
        if relation != 'no':
            details = get_user_details()
            details['relation'] = relation
            details['admin'] = False
            user_id = f"user_{next_user_id}"
            registered_users[user_id] = details
            next_user_id += 1
            capture_multiple_faces(user_id, details['name'])
            retrain_recognizer()
            print(f"[Registered] {details['name']} as {relation}.")
        else:
            print("[Unknown skipped]")

    cv2.imshow('Face Recognition System', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        print("\n[Exit] Shutting down...")
        break

cap.release()
cv2.destroyAllWindows()

print("\nFinal Registered Users:")
for uid, info in registered_users.items():
    print(f"ID {uid}: {info}")
