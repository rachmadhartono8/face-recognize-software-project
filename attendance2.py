import cv2
import numpy as np
import face_recognition
import os
import torch
import mysql.connector
from datetime import datetime, timedelta

print(torch.cuda.is_available())

db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'keyperformance'
}

# Create a MySQL connection
conn = mysql.connector.connect(**db_config)

# Create a cursor to execute SQL queries
cursor = conn.cursor()

# ...

# Define a dictionary to store attendance information
attendance_info = {}
encodeListKnown = findEncodings(images)

print('Encoding Complete')

# Uncomment the following line based on your use case:
# cap = cv2.VideoCapture(0)  # For webcam
# cap = cv2.VideoCapture('lib/videos/BERTANDANG KE SPACE X.mp4')  # For video file

last_seen_time = datetime.now()
discipline_status = 0
attendance_inserted = {}

# ...

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    while True:
    # success, img = cap.read()

    # ... (Your existing code)

    # for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
    #     matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
    #     faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)


    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # Update attendance information
            if name not in attendance_info:
                attendance_info[name] = {
                    'entry_time': datetime.now(),
                    'exit_count': 0,
                    'last_seen_time': datetime.now(),
                }
            else:
                # Check if 15 minutes have passed since the last detection
                if datetime.now() - attendance_info[name]['last_seen_time'] > timedelta(minutes=1):
                    attendance_info[name]['exit_count'] += 1

                attendance_info[name]['last_seen_time'] = datetime.now()

            # Insert the attendance record into the database
            try:
                sql_insert = "INSERT INTO attendance (name, discipline_status, timestamp) VALUES (%s, %s, CURRENT_TIMESTAMP)"
                val_insert = (name, discipline_status)
                cursor.execute(sql_insert, val_insert)
                conn.commit()

            except Exception as e:
                print(f"Error inserting into database: {e}")

    cv2.imshow("Attendance", img)
    cv2.waitKey(1)
