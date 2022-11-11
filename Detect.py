import torch
import cv2
import numpy as np
import pandas as pd

# Leemos el modelo (especificando que es un modelo personalizado)
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='model/best.pt')

# Realizamos video captura (especificando el numero de camara a usar)
cap = cv2.VideoCapture(0)

# Empezamos
while True:

    # Realizar lectura de la videocaptura
    # "Frame" obtendrá el siguiente fotograma en la cámara (a través de "cap").
    # "Ret" obtendrá el valor de retorno al obtener el marco de la cámara, ya sea True o False.
    ret, frame = cap.read()

    # Realizamos deteccion
    detect = model(frame)

    # Muestra las coordenadas en la captura de cuadros delimitadores
    info = detect.pandas().xyxy[0]
    print(info)

    # Mostramos FPS
    cv2.imshow('Detector de Carros', np.squeeze(detect.render()))

    # Leer el teclado
    # cv2.waitKey(delay) permite a los usuarios mostrar una ventana durante milisegundos determinados o
    # hasta que se presione cualquier tecla
    t = cv2.waitKey(5)
    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()