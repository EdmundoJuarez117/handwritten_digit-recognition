from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow import keras
import cv2
import base64

#Inicializar la app
app = Flask(__name__)

#Cargar el modelo preconstruido
model = keras.models.load_model('app/mnist_classification.h5')

#Manejar los GET request
@app.route('/', methods=['GET'])
def drawing():
    return render_template('drawing.html')

#Manejar los POST request
@app.route('/',methods=['POST'])
def canvas():
    #Se recibe datos en base64 del el formulario del usuario
    canvasdata = request.form['canvasimg']#Trae la imagen en el formulario
    #Separacion de los datos codificados de la imagen separados por una coma y tomados en la posicion 1 del arreglo correspondiente a datos base64
    encoded_data = request.form['canvasimg'].split(',')[1]

    #Metodo de decodificacion de imagen base64 a un arreglo de python
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)#toString() es para crear un arreglo de 1 dimension para datos binarios o datos de texto a una cadena / uint8 es para datos enteros no definidos de 0 a 255
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)#imdecode Se usa para leer datos de una imagen de una memoria cache y convertirla en formato de imagen / IMREAD_COLOR Se usa para leer los colores en RGB de una imagen pero no la transparencia

    #Convertir los 3 canales de imagen (RGB) a un canal de imagen en escala de grises (GRAY)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Cambiar el tamaño a 28 x 28
    gray_image = cv2.resize(gray_image, (28,28), interpolation=cv2.INTER_LINEAR)#interpolation es una tecnica de python utilizada para estimar puntos de datos desconocidos entre dos puntos de datos conocidos y cv2.INTER_LINEAR es usado cuando se necesita de un acercamiento (ZOOM)

    #Expandir la dimension del arreglo a (1,28,28) porque tenemos 1 imagen con tamaño 28x28
    img = np.expand_dims(gray_image, axis=0)

    try:
            prediction = np.argmax(model.predict(img))
            print(f"Prediction Result : {str(prediction)}")
            return render_template('drawing.html', response=str(prediction), canvasdata=canvasdata, success=True)
    except Exception as e:
            return render_template('drawing.html', response=str(e), canvasdata=canvasdata)
