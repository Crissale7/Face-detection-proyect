# Face-detection-proyect
Face detection proyect alpha 1.0 - 4.0 (in progress)

funciones a desarrollar a futuro(buscar mas documentacion)

- Reconocimiento óptico de caracteres (OCR) para leer textos labiales y lenguaje de señas.

- Detección de objetos que intersecan con caras, como anteojos, barbijos, etc. Mediante modelos entrenados.

- Segmentación semántica para separar y analizar partes específicas de la cara.
-----------------------------------------------------------------------------------------------------------------------
Aquí está una explicación paso a paso del código de detección facial en Python:

1. Se importan las librerías necesarias:

- cv2: OpenCV, librería principal para procesamiento de imágenes y visión artificial.
- dlib: Biblioteca con algoritmos de machine learning, incluye detectores faciales.
- numpy: Fundamental para cómputo científico en Python, usado para arrays.
- tensorflow.keras: API de Keras para modelos de deep learning.

2. Se cargan los modelos pre-entrenados:

- face_detector: Detector frontal de caras de dlib.
- face_landmark: Predictor de landmarks faciales de dlib.
- face_recognizer: Modelo de reconocimiento facial en Keras.

3. Se inicializa la captura de video desde la cámara web.

4. En un loop se procesa cada frame:

- Detección de caras con face_detector.
- Por cada cara encontrada:
  - Obtener coordenadas (x, y, ancho, alto).
  - Extraer ROI (región de interés) recortando el frame.
  - Aplicar face_recognizer para identificar a la persona.
  - Obtener landmarks con face_landmark.
  - Estimar orientación de la cara (pose) a partir de los landmarks.
  - Mostrar los resultados en el frame.
  
5. Se muestra el frame procesado y se espera tecla ESC para salir.

6. Liberar la cámara y destruir ventanas.

En resumen, con OpenCV, Dlib y Keras se implementa un sistema de detección, reconocimiento y análisis facial en tiempo real utilizando la webcam.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Guia:

Para poner en funcionamiento el código de detección facial en Python, los pasos serían:

1. Tener Python 3.6 o superior instalado, así como OpenCV y Dlib. Se pueden instalar con pip:

```
pip install opencv-python dlib
```

2. Tener un clasificador entrenado para detectar caras frontales. OpenCV incluye algunos como haarcascade_frontalface_default.xml.

3. Ejecutar el script Python con el código de detección facial:

```
python face_detector.py
``` 

4. Asegurarse que la cámara web está disponible y habilitada. En Linux puede requerir darle permisos al usuario o aplicación.

5. Mostrar el rostro frente a la cámara web y ver cómo se detecta en vivo en la ventana de salida.

6. Para compilar a un ejecutable standalone, se puede usar PyInstaller:

```
pip install pyinstaller
pyinstaller --onefile face_detector.py
```

7. Esto generará un archivo ejecutable .exe o .app que se puede distribuir y ejecutar en cualquier equipo, sin necesidad de tener Python instalado.

8. Opcionalmente se puede entrenar modelos más precisos de detección facial con frameworks como TensorFlow/Keras o PyTorch.

9. También se puede integrar con otros sistemas y aplicaciones que requieran detección facial.
