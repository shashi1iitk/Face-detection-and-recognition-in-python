import os
import cv2
import numpy as np
import core as fr


#This module captures images via webcam and performs face recognition
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingData.yml')#Load saved training data

name = {0 : "Priyanka",1 : "Kangana", 2 : "Shashi", 3: "Pushkar"}


cap=cv2.VideoCapture(1)

while True:
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
    faces_detected,gray_img=fr.faceDetection(test_img)



    for (x,y,w,h) in faces_detected:
      cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Face Detection Tutorial ',resized_img)   #Screen to see detected faces
    cv2.waitKey(10)


    for face in faces_detected:
        (x,y,w,h)=face
        roi_gray=gray_img[y:y+w, x:x+h]
        label,uncertainty = face_recognizer.predict(roi_gray)#predicting the label of given image
        print("confidence:",uncertainty)
        print("label:",label)
        fr.draw_rect(test_img,face)
        predicted_name=name[label]
        if uncertainty < 100:#If uncertainty greater than than 100 then don't print predicted face text on screen
           fr.put_text(test_img,predicted_name,x,y)


    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Face Recognition Tutorial ',resized_img) #Screen to see recognished faces
    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break


cap.release()
cv2.destroyAllWindows