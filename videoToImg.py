import cv2

cap=cv2.VideoCapture(1)

count = 0
while True:
    ret,test_img=cap.read()
    if not ret :
        continue
    resized_img = cv2.resize(test_img, (1000, 700))#Resizing Images before saving them
    cv2.imwrite("trainingImages/4/frame%d.jpg" % count, resized_img)# save frame as JPG file
    count += 1
    cv2.imshow('face detection Tutorial ',resized_img)
    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break


cap.release()
cv2.destroyAllWindows
