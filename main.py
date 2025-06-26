import cv2
import numpy as np
import os
from datetime import datetime
import insightface
from insightface.app import FaceAnalysis
from numpy.linalg import norm
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
import time

# Load ArcFace model
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

# Load known face embeddings
known_faces = {}
for filename in os.listdir("embeddings"):
    if filename.endswith(".npy"):
        name = filename.split(".")[0]
        embedding = np.load(f"embeddings/{filename}")
        known_faces[name] = embedding

# Cosine similarity function
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# Logging function
def log_attendance(name):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("attendance_log.csv", "a") as f:
        f.write(f"{name},{now}\n")
    print(f"[LOGGED] {name} at {now}")

# Initialize anti-spoofing predictor
anti_spoof = AntiSpoofPredict(device_id=0)
image_cropper = CropImage()
model_dir = "resources/anti_spoof_models"
model_names = os.listdir(model_dir)

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Press 'q' to quit")

def is_real_face(frame, box):
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    prediction = np.zeros((1, 3))
    for model_name in model_names:
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": frame,
            "bbox": [x1, y1, x2-x1, y2-y1],
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        prediction += anti_spoof.predict(img, os.path.join(model_dir, model_name))
    label = np.argmax(prediction)
    value = prediction[0][label] / len(model_names)
    return label == 1



previous_present = set()


anti_spoof_results = {}  # Store last spoofing result for each face (by bbox or name)
frame_count = 0
N = 3  # Process anti-spoofing every Nth frame

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)
    current_present = set()
    frame_count += 1

    for face in faces:
        box = tuple(face.bbox.astype(int))
        # Use face.bbox as key; you could use face id/name if available
        spoof_key = box

        # Only run anti-spoofing every Nth frame
        if frame_count % N == 0 or spoof_key not in anti_spoof_results:
            is_real = is_real_face(frame, box)
            anti_spoof_results[spoof_key] = is_real
        else:
            is_real = anti_spoof_results[spoof_key]

        if not is_real:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "Fake", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            continue

        emb = face.embedding
        name = "Unknown"
        best_score = 0

        for known_name, known_emb in known_faces.items():
            score = cosine_similarity(emb, known_emb)
            if score > 0.6 and score > best_score:
                best_score = score
                name = known_name

        # Draw
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(frame, name, (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if name != "Unknown":
            current_present.add(name)
            if name not in previous_present:
                log_attendance(name)

    previous_present = current_present



    # Calculate and display FPS
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Face Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()