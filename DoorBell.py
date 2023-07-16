import cv2
import os
import numpy as np

# Path to the folder containing the training images
train_folder = 'C:/users/Tabish/Desktop/4AI3 - Artifical Intelligence/Project/train_images/'

# Load the pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained face recognition model
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Function to train the face recognition model
def train_model():
    faces = []
    labels = []
    for root, dirs, files in os.walk(train_folder):
        for file in files:
            if file.endswith('jpg') or file.endswith('jpeg'):
                img_path = os.path.join(root, file)
                label = int(os.path.basename(root))
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                for (x, y, w, h) in faces_rect:
                    faces.append(gray[y:y+h, x:x+w])
                    labels.append(label)

    recognizer.train(faces, np.array(labels))

# Function to recognize faces in the input image
def recognize_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi = gray[y:y+w, x:x+h]
        label, confidence = recognizer.predict(roi)
        if confidence < 70:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, 'User Recognized. Access Granted', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(image, 'Unrecognized Person. Access Denied', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return image

# Train the model with the provided training images
train_model()

# Load and process the test image
test_image_path = 'C:/users/Tabish/Desktop/4AI3 - Artifical Intelligence/Project/test_image.png'
test_image = cv2.imread(test_image_path)
result_image = recognize_faces(test_image)

# Display the result
cv2.imshow('Face Recognition', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
