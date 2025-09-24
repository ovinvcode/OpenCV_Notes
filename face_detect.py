import os
import cv2 as cv
import numpy as np

# A list of people's names, matching the directory names
people = ['anne', 'sydney', "rebecca"]

# This section is redundant since you have 'people' hardcoded.
# If you want to use the list from the directory, replace 'people' with 'p'.
# p = []
# for i in os.listdir(r"M:\learn code\opencv\people"):
#     p.append(i)
# print(p)

dir = r"M:\learn code\opencv\people"

# Load the Haar Cascade classifier for face detection
haar_cascade = cv.CascadeClassifier("haar_face_default.xml")

features = []
labels = []

def create_train():
    # Loop over each person's folder
    for person in people:
        path = os.path.join(dir, person)
        label = people.index(person)  # Get the numeric label for the person

        # Loop over each image in the person's folder
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)

            # Read the image and convert it to grayscale
            img_array = cv.imread(img_path)
            if img_array is None:
                continue # Skip if image is not loaded correctly
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            # Detect faces in the grayscale image
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4)

            # Loop through each detected face
            for (x, y, w, h) in faces_rect:
                # Crop the face region of interest (ROI)
                faces_roi = gray[y:y+h, x:x+w]
                
                # Append the cropped face ROI and the corresponding label
                # This is the crucial correction.
                features.append(faces_roi)
                labels.append(label)

                # The following two lines were for visualization and should be outside the training loop if used
                # cv.rectangle(gray, (x, y), (x + w, y + h), (0, 0, 0), thickness=2)
                # cv.imshow(f"{y+x+h+w}n",gray)

# Execute the function to build our training dataset
print("--- Starting Training Data Creation ---")
create_train()
print("--- Training Data Creation Complete ---")
print("--- Starting Model Training ---")

# Convert the feature and label lists to NumPy arrays, as required by OpenCV
# We use dtype='object' for features because the face ROIs might have different sizes.
features = np.array(features, dtype="object")
labels = np.array(labels)

# Initialize the LBPH Face Recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the recognizer with our features and labels
face_recognizer.train(features, labels)

# Save the trained model to a file for later use
face_recognizer.save("face_trained.yml")

# Optional: Save the features and labels arrays
# This can be useful for debugging or analysis later.
np.save("features.npy", features)
np.save("labels.npy", labels)

print("--- Model Trained and Saved Successfully! --- ✅")
print(f"Number of features trained: {len(features)}")
print(f"Number of labels trained: {len(labels)}")