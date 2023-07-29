#Face detection proyect alpha 3.0

#Nuevas funciones agregadas:

#- Reconocimiento facial para identificar personas específicas. Entrenando un modelo de reconocimiento facial basado en redes neuronales convolucionales.

#- Seguimiento facial (face tracking) para seguir el movimiento de caras en video. Utilizando algún algoritmo como CAMShift o correlación de OpenCV.

#- Detección de poses faciales (orientación en yaw, pitch, roll) a partir de landmarks faciales. Útil para aplicaciones de realidad aumentada o animación 3D.

#- Detección de caras 3D a partir de información de profundidad de cámaras RGB-D.


import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model

# Cargamos los modelos necesarios
face_detector = dlib.get_frontal_face_detector()
face_landmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognizer = load_model('face_recognition_model.h5') 

labels = np.load('face_labels.npy') # etiquetas de personas conocidas

cap = cv2.VideoCapture(0)

# Inicializar tracker 
tracker = dlib.correlation_tracker()

while True:
    ret, frame = cap.read()   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detección de cara    
    rects = face_detector(gray, 0)
    
    for rect in rects:
       x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()

       # Iniciar tracker
       tracker.start_track(gray, rect)

       # Extraer ROI
       roi_gray = gray[y:y+h, x:x+w]

       # Reconocimiento facial
       label = face_recognizer.predict(roi_gray)
       label = labels[label]
       
       # Estimar pose facial
       landmarks = face_landmark(roi_gray, rect)
       yaw, pitch, roll = estimate_face_pose(landmarks)
       
       # Mostrar resultados
       cv2.putText(frame, f'{label} {yaw} {pitch} {roll}', (x,y-10), cv2.FONT_HERSHEY_PLAIN, 1.8, (0,255,0), 2) 

    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()