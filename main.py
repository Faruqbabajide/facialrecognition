import cv2
import os
import sys
from face_recognition_utils import load_known_faces, recognize_face

# Load known faces
known_face_encodings, known_face_names = load_known_faces()

# Check if dataset is empty
if not known_face_encodings:
    print("No known faces found in the dataset. Please add images to 'dataset/'.")
    sys.exit()

# Test Image (Ensure correct file name and path)
test_image_path = "dataset/tomholland.png"  # Change to an actual image file

if not os.path.exists(test_image_path):
    print(f"Test image '{test_image_path}' not found.")
    sys.exit()

# Recognize faces
recognized_names, face_locations = recognize_face(test_image_path, known_face_encodings, known_face_names)

# Load the image using OpenCV
image = cv2.imread(test_image_path)

# Draw rectangles and labels around detected faces
for (top, right, bottom, left), name in zip(face_locations, recognized_names):
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the result
cv2.imshow("Face Recognition", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
