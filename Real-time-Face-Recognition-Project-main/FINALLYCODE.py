import cv2
import numpy as np
import os
import datetime

# Initialize
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Set the main window
window_name = 'Face Recognition System'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

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
    print(f"[Capturing for {label_name}] Please move your face slowly and vary lighting if possible...")

    captured = 0
    capture_delay = 300  # milliseconds between frames

    while captured < 60:  # capturing more for lighting robustness
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # Normalize brightness/contrast
        gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise

        faces = face_cascade.detectMultiScale(gray, 1.1, 6, minSize=(90, 90))

        if len(faces) == 1:
            (x, y, w, h) = faces[0]
            face_crop = gray[y:y+h, x:x+w]
            face_crop = cv2.resize(face_crop, (100, 100))

            folder_path = f"registered_faces/{user_id}"
            os.makedirs(folder_path, exist_ok=True)

            # Save original and a brightened version
            cv2.imwrite(f"{folder_path}/{captured}.png", face_crop)
            captured += 1
            print(f"Captured sample {captured}/60")

        cv2.imshow('Capturing Face Data', frame)
        cv2.waitKey(capture_delay)

    cv2.destroyWindow('Capturing Face Data')
    print("[Done] Captured with lighting normalization.")


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
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.equalizeHist(img)
                    img = cv2.GaussianBlur(img, (3, 3), 0)
                    faces.append(img)
                    labels.append(current_label)

                    # Add brightened version
                    brighter = cv2.convertScaleAbs(img, alpha=1.2, beta=30)
                    faces.append(brighter)
                    labels.append(current_label)

                    # Add darkened version
                    darker = cv2.convertScaleAbs(img, alpha=0.8, beta=-30)
                    faces.append(darker)
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
        user_details = {}

        if len(label_user_map) > 0:
            label, confidence = recognizer.predict(face_crop)
            user_id = label_user_map.get(label)

            if confidence < 90:
                user_details = registered_users.get(user_id, {})
                label_text = user_details.get('name', 'Unknown')
                known_faces.append((x, y, w, h, label_text))
            else:
                print(f"[Uncertain match: {confidence:.2f}] Possibly known, but too dim/bright?")
                unknown_faces.append((x, y, w, h))
        else:
            unknown_faces.append((x, y, w, h))

        # DESIGN: Draw futuristic box with info panel
        color = (0, 255, 0) if label_text != "Unknown" else (0, 0, 255)
        thickness = 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)

        info_panel = [
            f"Name: {label_text}",
            f"Confidence: {confidence:.2f}",
        ]

        if label_text != "Unknown":
            info_panel += [
                f"Age: {user_details.get('age', '-')}",
                f"Gender: {user_details.get('gender', '-')}",
                f"Relation: {user_details.get('relation', 'Admin' if user_details.get('admin') else '-')}"
            ]

        # Determine if there's enough space below, otherwise place it above
        panel_height = len(info_panel) * 20
        frame_height = frame.shape[0]
        base_y = y + h + 10 if (y + h + 10 + panel_height < frame_height) else y - panel_height - 10
        base_y = max(base_y, 10)  # Prevent going too high up

        # Adjust vertical position to avoid overlap with the bounding box
        if y - 60 > 20:
            base_y = y - 120  # Display above face if there's enough space
        else:
            base_y = y + h + 30  # Display below with extra padding to prevent overlap

        for i, line in enumerate(info_panel):
            text_pos = (x, base_y + i * 22)  # Increase line spacing slightly
            cv2.putText(frame, line, text_pos, cv2.FONT_HERSHEY_PLAIN, 1.3, color, 2, cv2.LINE_AA)




    # Register new face only if an admin is seen and 1 unknown face
    if len(unknown_faces) == 1 and len(known_faces) == 1:
        known_label_text = known_faces[0][4]
        
        # Search if that known face is an admin
        matched_admin_id = None
        for uid, details in registered_users.items():
            if details.get('name') == known_label_text and details.get('admin') == True:
                matched_admin_id = uid
                break

        if matched_admin_id:
            print(f"\n[Admin Detected: {known_label_text}] 1 unknown nearby.")
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


    cv2.imshow(window_name, frame)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    key = cv2.waitKey(1)

    # Exit conditions
    if key == ord('q'):
        print("\n[Exit] Shutting down by 'q' press...")
        break

    # NEW: Handle manual window close
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        print("\n[Exit] Window manually closed.")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

print("\nFinal Registered Users:")
for uid, info in registered_users.items():
    print(f"ID {uid}: {info}")
