import os
import face_recognition
import pickle
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True, help="Path to directory with face images")
args = ap.parse_args()

encodings = []
names = []

for filename in os.listdir(args.images):
    if filename.endswith((".jpg", ".png")):
        image_path = os.path.join(args.images, filename)
        name = os.path.splitext(filename)[0]
        image = face_recognition.load_image_file(image_path)
        face_enc = face_recognition.face_encodings(image)
        if face_enc:
            encodings.append(face_enc[0])
            names.append(name)
            print(f"[INFO] Encoded {name}")

data = {"encodings": encodings, "names": names}
with open("models/faces/face_encodings.pkl", "wb") as f:
    pickle.dump(data, f)

print("[INFO] Training complete. Encodings saved to models/faces/face_encodings.pkl")
