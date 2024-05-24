# ***People Counter***
Prototipo de visión por computadora que permite el conteo en "tiempo real" de las personas que entran y salen de cierto espacio. Desarrollado con la idea de utilzarlo para el conteo de usuarios dentro de los sistemas de transporte público con estaciones definidas.

**Nota importante:** Este primer prototipo no hace un conteo de usuarios constante a través de un mecanismo de grabación, utiliza un video pre-grabado y devuelve el análisis en otro video donde se muestra el conteo. 

Aquí una demo pequeña de cómo se ve el video después de ser procesado! 

<div align="center">
  <img src="media/out_stairs.gif" alt="Example GIF">
</div>

## ¿Cómo funciona?
El proyecto hace uso de dos características principales; el reconocimiento de objetos mediante [Ultralytics YOLOv8](https://docs.ultralytics.com/) combinado con el seguimiento de objetos a partir de [Deep Sort](https://github.com/nwojke/deep_sort). \
Para el conteo de personas entrando y saliendo se hace uso de un delimitador en el video, cuando la parte inferior de la caja que delimita a una persona pasa por dicho delimitador se hace una comparación con respecto a la primera vez que se vió a la persona en el video, así podemos saber si dicha persona está entrando o saliendo del lugar. 

**NOTA IMPORTANTE:** Si quieres hacer uso de este código para procesar tus propios videos, debes cambiar el delimitador "delimiter" en [app.py](https://github.com/AnsharShampoo/people_counter/blob/main/app.py#L37) según dónde quieras marcar la entrada/salida en tu caso particular.

## Instalación y Ejecución
### Linux
Para poder utilizar este proyecto se recomienda ampliamente crear un ambiente virtual de Python para manejar las dependencias: 

```
python3 -m venv venv
```
Posteriormente inicializa tu ambiente virtual con:
```
source venv/bin/activate
```
Una vez inicializado el ambiente virtual descarga todas las bibliotecas requeridas para correr el proyecto mediante:
```
pip install -r requirements.txt
```
¡Todo listo! Ya puedes correr el proyecto mediante el comando:
```
python3 app.py
```
En este repositorio ya está preparado un video de muestra para que veas cómo es el procesamiento, el archivo de salida se encuentra en la carpeta **media** con el nombre **out_stairs.mp4**

## Información Importante
Este proyecto hace uso de la biblioteca [deep_sort]() y está basado en el trabajo previo de [computervisioneng](https://github.com/computervisioneng) que puedes consultar [aquí](https://github.com/computervisioneng/object-tracking-yolov8-deep-sort).