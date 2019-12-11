import os
import cv2
import numpy as np
import core as fr

#Training the model with the images in given directory
directory ='resizedTrainingImages'
faces,faceID=fr.labels_for_training_data(directory)
face_recognizer=fr.train_classifier(faces,faceID)
face_recognizer.write('trainingData.yml')