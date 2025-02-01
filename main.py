import cv2
import sys
from face_recognition_utils import load_known_faces, recognize_face

# Load known faces
known_face_encodings, known_face_names = load_known_faces()

# Test Image
test_image_path = "test_images/test_face.jpg"

# Recognize faces
recognized_names, face_locations = recognize_face(test_image_path, known_face_encodings, known_face_names)

# Display the results
image = cv2.imread(test_image_path)
for (top, right, bottom, left), name in zip(face_locations, recognized_names):
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.imshow("Face Recognition", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
