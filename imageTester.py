import cv2
import os
import numpy as np
import core as fr



#load trained model
face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingData.yml')





#This module takes images  stored in disk and detect face
test_img=cv2.imread('testImages/Kangana.jpg')#test_img path
faces_detected,gray_img=fr.faceDetection(test_img)
print("faces_detected:",faces_detected)


for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label,uncertainty=face_recognizer.predict(roi_gray)#predicting the label of given image
    print("confidence:",uncertainty)
    print("label:",label)
    fr.draw_rect(test_img,face)
    predicted_name=fr.name[label]
    if(uncertainty > 50):#If confidence more than 51 then don't print predicted face text on screen
        continue
    fr.put_text(test_img,predicted_name,x,y)

resized_img=cv2.resize(test_img,(1000,1000))
cv2.imshow("face detection tutorial",resized_img)
cv2.waitKey(0)#Waits indefinitely until a key is pressed
cv2.destroyAllWindows





