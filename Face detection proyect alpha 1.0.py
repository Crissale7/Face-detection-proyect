#Face detection proyect alpha 1.0

#funciones:  detección de ojos, estimación de pose, reconocimiento facial básico y detección en tiempo real con webcam.

import cv2
import dlib
from scipy.spatial import distance

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

# Reconocimiento facial
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 

# Seguimiento facial 
tracker = dlib.correlation_tracker()

while True:
    _, frame = cap.read()
    
    # Detección de cara    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    for (x,y,w,h) in faces:
       cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

       # Estimar pose
       landmarks = predictor(gray, dlib.rectangle(x,y,x+w,y+h))
       pose_landmarks = landmarks.parts()
       # ... calcular yaw, pitch, roll
       
       # Detectar emociones
       # ... entrenar y aplicar modelo
       
       # Reconocimiento facial
       face_desc = predictor(gray, dlib.rectangle(x,y,x+w,y+h))
       # ... identificar persona

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()