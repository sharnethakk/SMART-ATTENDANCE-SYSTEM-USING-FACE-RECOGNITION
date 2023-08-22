import face_recognition
import cv2
import numpy as np
import pandas as pd
from datetime import date, datetime

# Load face images and encodings
known_face_encodings = []
known_face_names = []

# Define the names and image paths of known faces
known_faces = {
    "smrithi": "smrithi.jpg",
    "swetha": "swetha.jpg"
}

# Load known face encodings
for name, image_path in known_faces.items():
    image = face_recognition.load_image_file(image_path)
    face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(name)

# Initialize variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
now = datetime.now()

# Create a new DataFrame for the attendance
attendance_df = pd.DataFrame(columns=['Name', 'Date', 'Time'])

# Open the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Resize frame to improve performance
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color to RGB
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        # Find all faces and their encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Compare the current face encoding with the known face encodings
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match is found, use the known face name
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

            # Add attendance data if a new face is detected
            if name not in attendance_df['Name'].values:
                current_time = now.strftime("%H:%M:%S")
                attendance_df = attendance_df.append({'Name': name, 'Date': str(date.today()), 'Time': current_time},
                                                     ignore_index=True)
                print("Attendance taken for:", name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up the face locations
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with the name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the attendance DataFrame to an Excel file
attendance_df.to_excel('attendance.xlsx', index=False)

# Release the webcam and destroy all windows
video_capture.release()
cv2.destroyAllWindows()
