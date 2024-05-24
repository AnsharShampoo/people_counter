# Importamos las dependencias necesarias:

import os #interacción con el sistema operativo, para la manipulación de archivos locales
import cv2 #manipulación de imagenes y videos
import random #generador de números aleatorios
from ultralytics import YOLO #versión optimizada del modelo de detección de imagenes YOLO 
from tracker import Tracker #clase tracker para hacer el seguimiento de objetos

# Video a tratar:
video_path = os.path.join('.', 'media', 'subway_stairs.mp4')
video_out_path = os.path.join('.', 'media', 'out_stairs.mp4')
cap = cv2.VideoCapture(video_path) #Abrimos el video mediante cv2

# El objeto cap contiene dos elementos, ret (true cuando se lee el frame de forma correcta) y frame (el fotograma actual)
# cada que se hace cap.read y mientras haya un fotograma contiguo, frame retiene el fotograma siguiente.
ret, frame = cap.read()

#Variable que permite guardar el video de salida
cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))

model = YOLO("yolov8n.pt") #Utilizamos la versión 8 del modelo YOLO.

tracker = Tracker() #Inicializamos clase tracker

# Creamos una lista de colores aleatorios que nos ayudarán a enmarcar a las personas, que sean
# de colores diferentes nos permite debuggear si en algún momento una persona es detectada como
# otra tras cambiar de frame.
colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for j in range(10)]
i=0

# Diccionario que marca el la coordenada x del pixel inferior donde se dibujó por primera vez la bounding-box de cierta persona (track_id:coord_x)
first_time_observed={}
# Set que nos permite saber si cierta persona ya se contabilizó a partir de su track_id}
counted_tracks = set()
count = 100
delimiter = 490
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# Mientras haya un frame por analizar entramos al bucle
while ret:
    i+=1
    results = model(frame) #Análisis del frame actual mediante YOLOv8

    result = results[0] 
    detections = [] #Lista vacía que guarda todas las personas detectadas en el frame

    #Iteramos sobre cada detección encontrada a manera de lista.
    # x1, y1, x2, y2 son las coordenadas del rectangulo que enmarca nuestra detección (a la persona)
    # score es el puntaje de confianza de la detección (Qué tan seguro está el modelo de haber detectado una persona)
    # class_id determina el tipo de detección, YOLOv8 está preentrenado con diferentes cosas por detectar
    # (personas, animales, etc), en este caso presuponemos que en el video sólo se encontrarán personas, por 
    # lo que descartamos este valor al agregar la detección a la lista detectios.
    for r in result.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = map(int, r)  # transformamos los valores en enteros
        if class_id == 0:
            detections.append([x1, y1, x2, y2, score]) # Agregamos la detección a nuestra lista si se trata de una persona
        
    tracker.update(frame, detections) # Actualizamos nuestro tracker
    
    #Dibujamos en el video el delimitador de entrada y salida.
    cv2.line(frame, (0,delimiter), (frame_width, delimiter), (0,255,0), 3)

    #Iteramos sobre cada persona de la que tenemos registro en tracker y dibujamos una bounding-box 
    #enmarcandolos en este frame.
    for track in tracker.tracks:
        bbox = track.bbox
        x1, y1, x2, y2 = bbox
        track_id = track.track_id
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
        #Hacemos el conteo del track dependiendo de la posición en la que se vio por primera vez, su posición actual y el delimitador:
        if(track_id not in counted_tracks and track_id not in first_time_observed):
            first_time_observed[track_id] = y2
        elif(track_id not in counted_tracks and track_id in first_time_observed and delimiter>first_time_observed[track_id] and y2>=delimiter):
            count+=1
            counted_tracks.add(track_id)
        elif(track_id not in counted_tracks and track_id in first_time_observed and delimiter<first_time_observed[track_id] and y2<=delimiter):
            count-=1
            counted_tracks.add(track_id)
    #Dibujamos el delimitador
    cv2.putText(frame, f'personas: {count}', (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 3, cv2.LINE_AA)        
    #Mostramos el frame mientras procesamos para asegurarnos de que todo va como debe.
    cv2.imshow('frame', frame)
    #Esperamos 25ms para que de tiempo de ver lo que pasa en el procesamiento
    cv2.waitKey(25)
    #Escribimos el frame actual en el video de salida
    cap_out.write(frame)
    #Avanamos al siguiente frame
    ret, frame = cap.read()
#Liberamos el video con el que trabajamos
cap.release()
cap_out.release()
#Cerramos las imagenes abiertas con las que vemos el procesamiento mientras el resultado se procesa.
cv2.destroyAllWindows()

