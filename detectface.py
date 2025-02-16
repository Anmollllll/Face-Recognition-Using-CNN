
import cv2
import numpy as np
import pandas as pd
import os
import csv
from tensorflow.keras.models import load_model
from datetime import datetime

# Load the trained CNN model
model_path = 'recognizer/face_recognizer_cnn.h5'
if not os.path.isfile(model_path):
    print('First train the model and save it.')
    exit(0)

model = load_model(model_path)

names = {}
labels = []
students = []

def from_excel_to_csv():
    # Convert Excel file to CSV without adding an index column
    df = pd.read_excel('data.xlsx')
    df.to_csv('data.csv', index=False)  # Prevent adding "Unnamed" column
def getdata():
    with open('data.csv', 'r') as f:
        data = csv.reader(f)
        next(data)  # Skip header row
        # Populate the dictionary with numeric labels as keys
        for idx, line in enumerate(data):  # Assign numeric labels starting from 0
            names[idx] = line[0]  # Use the student's name from the first column


def markPresent(name):
    # Load the CSV file into a Pandas DataFrame
    if os.path.exists('data.csv'):
        df = pd.read_csv('data.csv')
    else:
        print("Error: 'data.csv' does not exist!")
        return

    # Get today's date as a column name
    today = datetime.now().strftime('%Y-%m-%d')

    # If the date column doesn't exist, create it
    if today not in df.columns:
        df[today] = 'A'  # Default to 'A' (Absent) for all students

    # Mark the student as 'P'
    df.loc[df['Name'] == name, today] = 'P'

    # Save the updated DataFrame back to the CSV file without adding the index
    df.to_csv('data.csv', index=False)
    print(f"Marked {name} as present for {today}.")

def update_Excel():
    # Convert the updated 'data.csv' back to 'data.xlsx'
    df = pd.read_csv('data.csv')
    df.to_excel('student_attendance_data.xlsx', index=False)  # Prevent adding "Unnamed" column
    print("Updated 'student_attendance_data.xlsx' with the latest attendance data.")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

from_excel_to_csv()  # Convert Excel to CSV
getdata()  # Load names from the CSV into a dictionary
print('Total students:', names)
cap = cv2.VideoCapture(0)  # Use 0 for default webcam, 1 for external
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(0)




while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Preprocessing for CNN
        face_img = gray[y:y + h, x:x + w]  # Crop the detected face
        resized_face = cv2.resize(face_img, (100, 100))  # Resize to match CNN input
        normalized_face = resized_face / 255.0  # Normalize pixel values
        reshaped_face = normalized_face.reshape(1, 100, 100, 1)  # Reshape for CNN

        # Make prediction using the CNN model
        predictions = model.predict(reshaped_face)
        confidence = np.max(predictions) * 100  # Get the highest confidence
        predicted_label = np.argmax(predictions)  # Get the index of the highest probability
        predicted_name = names.get(predicted_label, "Unknown")

        # Display the prediction on the image
        if confidence > 70:  # Confidence threshold
            cv2.putText(img, f"{predicted_name} {confidence:.2f}%", (x + 2, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 255, 0), 2)
            labels.append(predicted_label)
            students.append(predicted_name)

            # Mark presence
            total_students = set(students)
            just_labels = set(labels)
            print('Student Recognized:', total_students, just_labels)
            for i in just_labels:
                if labels.count(i) > 20:
                    markPresent(predicted_name)
                    update_Excel()

    cv2.imshow('Face Recognizer', img)
    if cv2.waitKey(33) == ord('a'):  # Exit on pressing 'a'
        print('Pressed a')
        break

cap.release()
cv2.destroyAllWindows()

