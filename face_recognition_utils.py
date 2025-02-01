import os
import face_recognition
import numpy as np

# Path to dataset
DATASET_PATH = "dataset/"

def load_known_faces():
    """Load images from dataset and encode them."""
    known_face_encodings = []
    known_face_names = []

    for file in os.listdir(DATASET_PATH):
        if file.endswith(".jpg") or file.endswith(".png"):
            image_path = os.path.join(DATASET_PATH, file)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:  # Ensure face encoding exists
                known_face_encodings.append(encodings[0])
                known_face_names.append(file.split("_")[0])  # Extract name from file

    return known_face_encodings, known_face_names

def recognize_face(image_path, known_face_encodings, known_face_names):
    """Recognize faces in an image."""
    unknown_image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    results = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances) if face_distances.size > 0 else None

        if best_match_index is not None and matches[best_match_index]:
            name = known_face_names[best_match_index]

        results.append(name)

    return results, face_locations
