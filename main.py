import cv2
import numpy as np
import os
from datetime import datetime
import insightface
from insightface.app import FaceAnalysis
from numpy.linalg import norm

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

# Start webcam
cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)
    for face in faces:
        emb = face.embedding
        name = "Unknown"
        best_score = 0

        for known_name, known_emb in known_faces.items():
            score = cosine_similarity(emb, known_emb)
            if score > 0.6 and score > best_score:
                best_score = score
                name = known_name

        # Draw and log
        box = face.bbox.astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(frame, name, (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if name != "Unknown":
            log_attendance(name)

    cv2.imshow("Face Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
