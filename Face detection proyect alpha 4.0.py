#Face detection proyect alpha 4.0

#- Reconocimiento óptico de caracteres (OCR) para leer textos labiales y lenguaje de señas.

#- Detección de objetos que intersecan con caras, como anteojos, barbijos, etc. Mediante modelos entrenados.

#- Segmentación semántica para separar y analizar partes específicas de la cara.


import cv2
import dlib
import pytesseract
from tensorflow.keras.models import load_model

# Cargamos modelos
face_detector = dlib.get_frontal_face_detector()
face_landmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

ocr_model = load_model('ocr_model.h5')
obj_detector = cv2.CascadeClassifier('haarcascade_eyeglasses.xml') 
segmenter = load_model('face_segmenter.h5')

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    
    # Detectar caras 
    rects = face_detector(frame)

    for (i, rect) in enumerate(rects):
       x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
       
       # Extraer ROI
       face = frame[y:y+h, x:x+w]
       
       # OCR para lenguaje de señas
       roi_lips = face[y2:y2+h2, x2:x2+w2]  
       text = ocr_model.predict(roi_lips)
       
       # Detectar objetos
       glasses = obj_detector.detectMultiScale(face)
       
       # Segmentación semántica
       parts = segmenter.predict(face)
       
       # Mostrar resultados
       cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
       cv2.putText(frame, text, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255),2)
       
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) == 27:
        break
        
cap.release()
cv2.destroyAllWindows()

#De esta forma se van agregando más capacidades avanzadas de procesamiento y análisis de imágenes faciales al sistema.