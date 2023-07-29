#Face detection proyect alpha 2.0

#funciones nuevas: 

#Reconocimiento facial para identificar a la persona detectada. Se puede entrenar un modelo de deep learning como una red neuronal convolucional.
#Seguimiento facial (face tracking) para seguir el movimiento de un rostro en un video. Utilizando algoritmos como CAMShift de OpenCV.
#Estimar la pose facial (yaw, pitch, roll) calculando la orientación relativa de la cara. Útil en aplicaciones de realidad aumentada.
#Detección de emociones basada en las expresiones faciales como felicidad, tristeza, enojo, etc. Mediante entrenamiento de modelos de clasificación.
#Estimación de edad y género a partir de las características faciales. Con modelos de deep learning entrenados.
#Detección facial en tiempo real utilizando la cámara web para interacciones persona-computador.
#Reconocimiento de gestos faciales como guiño, levantar cejas, inflar mejillas, entre otros. Entrenando modelos especializados.

import cv2
import dlib
from tensorflow.keras.models import load_model

# Modelos de detección facial
face_detector = dlib.get_frontal_face_detector()
face_landmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Modelos de detección de emocciones
emotion_model = load_model('emotion_detection.h5')
emotions = ['enojado', 'feliz', 'triste', 'sorprendido']

# Modelo de detección de gestos
gesture_model = load_model('gesture_detection.h5')
gestures = ['guiño', 'sonrisa', 'beso'] 

# Modelo de edad y género
age_gender_model = load_model('age_gender_detection.h5')

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_detector(gray)

    for face in faces:
        
        x, y = face.left(), face.top()
        w, h = face.right(), face.bottom()
        
        # Extraer ROI
        roi = gray[y:y+h, x:x+w]

        # Detectar emociones
        emotion_preds = emotion_model.predict(roi)[0]
        emotion_label = emotions[emotion_preds.argmax()]
        
        # Detectar gestos
        gesture_preds = gesture_model.predict(roi)[0]
        gesture_label = gestures[gesture_preds.argmax()]

        # Detectar edad y género
        age, gender = age_gender_model.predict(roi)

        cv2.putText(frame, f'{emotion_label} {gesture_label}, {gender}, {age}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()