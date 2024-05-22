# Importamos las dependencias necesarias:

from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
import numpy as np

# Clase TRacker que permite el seguimiento de objetos sobre un video
class Tracker:

    tracker = None
    encoder = None
    tracks = None

    #Método constructor de Tracker. 
    def __init__(self):
        
        max_cosine_distance = 0.4 #Umbral máximo de la distancia coseno para el cálculo de similitudes
        nn_budget = None #Quitamos el límite de muestras históricas posibles para hacer asociación de similitudes
        encoder_model_filename = 'model_data/mars-small128.pb' #Modelo preentrenado a utilizar

        # Creamos la instancia de NearestNeighborDistanceMetric, con las características antes inicializadas
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = DeepSortTracker(metric) # Inicializamos el tracker con las métricas definidas
        self.encoder = gdet.create_box_encoder(encoder_model_filename, batch_size=1) #Encoder encargado de hacer el análisis de características de las bounding boxes

    #Método de actualización de detecciones
    def update(self, frame, detections):

        #En caso de no tener nuevas detecciones, simplemente se hace el update de las tracks pre-existentes. 
        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])  
            self.update_tracks()
            return

        # Tomamos todas las bounding boxes detectadas, solamente se toman los pixels delimitadores y se ignora la puntiación de confianza.
        bboxes = np.asarray([d[:-1] for d in detections])
        # Transformamos las bonding boxes del formato [x1, y1, x2, y2] (coordenadas de las esquinas superior izquierda e inferior derecha) 
        # al formato [x, y, w, h] (coordenadas de la esquina superior izquierda, ancho y alto)
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
        # Guradamos las puntuaciones de confianza antes ignoradas:
        scores = [d[-1] for d in detections]

        # Genera características (features) para cada bounding-box en el cuadro actual utilizando el codificador "encoder".
        features = self.encoder(frame, bboxes)

        # Crea una lista de objetos Detection donde cada objeto Detection contiene una boundig-box, 
        # una puntuación (scores[bbox_id]) y características (features[bbox_id]).
        dets = []
        for bbox_id, bbox in enumerate(bboxes):
            dets.append(Detection(bbox, scores[bbox_id], features[bbox_id]))


        self.tracker.predict() #Predice las nuevas posiciones de las tracks actuales (dónde están ahora los objetos antes detectados)
        # Actualiza el tracker con las nuevas detecciones (dets), 
        # asociando cada detección con las pistas existentes basándose en la métrica de distancia (como la distancia coseno).
        self.tracker.update(dets)
        self.update_tracks()

    # Método encargado de actualizar las pistas confirmadas y remover las pistas no actualizadas.
    def update_tracks(self):
        tracks = []

        for track in self.tracker.tracks:

        # Solo procesamos las tracks que están confirmadas (track.is_confirmed()) y que han sido actualizadas en el último cuadro (donde track.time_since_update <= 1). 
        # Si una pista no está confirmada o no ha sido actualizada en más de un cuadro, se omite (continue).
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr() # convertimos track actual al formato [x1, y1, x2, y2]
            id = track.track_id #identificador único
            tracks.append(Track(id, bbox)) #Crea un nuevo objeto Track utilizando el identificador y la bounding-box, y lo añade a la lista tracks.

        self.tracks = tracks #Actualizamos tracks

# Clase que implementa las tracks
class Track:
    track_id = None
    bbox = None

    #Método constructor
    def __init__(self, id, bbox):
        self.track_id = id
        self.bbox = bbox
