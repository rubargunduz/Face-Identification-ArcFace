import os
import numpy as np
import cv2
from insightface.app import FaceAnalysis

# Initialize ArcFace model
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

image_base = "images"     # folder with subfolders per person
save_base = "embeddings" # where averaged embeddings will be saved

os.makedirs(save_base, exist_ok=True)

for person in os.listdir(image_base):
    person_path = os.path.join(image_base, person)
    if not os.path.isdir(person_path):
        continue  # skip files if any

    embeddings = []

    for img_file in os.listdir(person_path):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(person_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: could not read {img_path}")
            continue

        faces = app.get(img)
        if not faces:
            print(f"No face found in {img_path}")
            continue

        # Use first detected face's embedding
        emb = faces[0].embedding
        embeddings.append(emb)

    if embeddings:
        avg_emb = np.mean(embeddings, axis=0)
        np.save(os.path.join(save_base, f"{person}.npy"), avg_emb)
        print(f"Saved averaged embedding for {person} with {len(embeddings)} images")
    else:
        print(f"No valid faces found for {person}")
