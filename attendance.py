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

path = 'lib/attendance'
images = []
classNames = []

myList = os.listdir(path)

print(myList)

for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])

print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)

print('Encoding Complete')

# cap = cv2.VideoCapture('lib/videos/BERTANDANG KE SPACE X.mp4')
cap = cv2.VideoCapture(0)

last_seen_time = datetime.now()
discipline_status = 0  # Assume the p
attendance_inserted = {}

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    current_time = datetime.now()

    if facesCurFrame:
        last_seen_time = datetime.now()  # Update the last seen time when a face is detected

    # Check if 15 minutes have passed since the last detection
    if datetime.now() - last_seen_time > timedelta(minutes=1):
        discipline_status = 1  # Set discipline status to 1


    for encodeFace, faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

            if name in attendance_inserted:
                entry_time = attendance_inserted[name]['entry_time']
                working_hours = (current_time - entry_time).total_seconds() / 3600.0

                # Update the database with working hours
                try:
                    sql_update = "UPDATE attendance SET working_hours = %s WHERE name = %s AND discipline_status = 1"
                    val_update = (working_hours, name)
                    cursor.execute(sql_update, val_update)
                    conn.commit()

                except Exception as e:
                    print(f"Error updating working hours in the database: {e}")

            if name not in attendance_inserted:
                try:
                    # Insert the attendance record into the database
                    sql_insert = "INSERT INTO attendance (name, discipline_status, entry_time, timestamp) VALUES (%s, %s, %s, CURRENT_TIMESTAMP)"
                    val_insert = (name, discipline_status, current_time)
                    cursor.execute(sql_insert, val_insert)
                    conn.commit()

                    # Update the flag to indicate that attendance has been inserted for this person
                    attendance_inserted[name] = {
                        'entry_time': current_time,
                        'exit_count': 0,
                        'last_seen_time': current_time,
                    }


            # # Insert the attendance record into the database
            # try:
            #     sql = "INSERT INTO attendance (name) VALUES (%s)"
            #     val = (name,)
            #     cursor.execute(sql, val)
            #     conn.commit()
            # except Exception as e:
            #     print(f"Error inserting into database: {e}")


            # Insert the attendance record into the database
            # bisa ini 
            # try:
            #     sql = "INSERT INTO attendance (name, timestamp) VALUES (%s, CURRENT_TIMESTAMP)"
            #     val = (name,)
            #     cursor.execute(sql, val)
            #     conn.commit()
            # except Exception as e:
            #     print(f"Error inserting into database: {e}")

            # sekali insert saja
            # try:
            #     sql_check = "SELECT * FROM attendance WHERE name = %s AND timestamp = CURRENT_TIMESTAMP"
            #     val_check = (name,)
            #     cursor.execute(sql_check, val_check)
            #     existing_record = cursor.fetchone()

            #     if not existing_record:
            #         # Insert the attendance record into the database
            #         sql_insert = "INSERT INTO attendance (name, discipline_status, timestamp) VALUES (%s, %s, CURRENT_TIMESTAMP)"
            #         val_insert = (name, discipline_status)
            #         cursor.execute(sql_insert, val_insert)
            #         conn.commit()

            # except Exception as e:
            #     print(f"Error checking or inserting into database: {e}")

            # last
                        # Check if the attendance record already exists


            # if name not in attendance_inserted:
            #     try:
            #         # Insert the attendance record into the database
            #         sql_insert = "INSERT INTO attendance (name, discipline_status, timestamp) VALUES (%s, %s, CURRENT_TIMESTAMP)"
            #         val_insert = (name, discipline_status)
            #         cursor.execute(sql_insert, val_insert)
            #         conn.commit()

            #         # Update the flag to indicate that attendance has been inserted for this person
            #         attendance_inserted[name] = True

            #     except Exception as e:
            #         print(f"Error inserting into database: {e}")
                except Exception as e:
                    print(f"Error inserting into database: {e}")


    cv2.imshow("Attendance", img)
    cv2.waitKey(1)