import cv2
import numpy as np
import face_recognition

imgVj1 = face_recognition.load_image_file('imgs/Vj1.jpg')
imgVj1 = cv2.cvtColor(imgVj1,cv2.COLOR_BGR2RGB)
imgVj2 = face_recognition.load_image_file('imgs/ok3.jpg')
imgVj2 = cv2.cvtColor(imgVj2,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgVj1)[0]
encodeVj1 = face_recognition.face_encodings(imgVj1)[0]
cv2.rectangle(imgVj1,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLoc = face_recognition.face_locations(imgVj2)[0]
encodeVj2 = face_recognition.face_encodings(imgVj2)[0]
cv2.rectangle(imgVj2,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeVj1],encodeVj2)
faceDis = face_recognition.face_distance([encodeVj1],encodeVj2)
print(results,faceDis)
cv2.putText(imgVj2,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Vedant Jakarwar',imgVj1)
cv2.imshow('Vedant 2',imgVj2)
cv2.waitKey(0)