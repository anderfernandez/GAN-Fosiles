import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from functions import *

# Data
tamaño_imagen = 64

# Load Files

print('Procesar imagenes')

ficheros = os.listdir('images/Marcas_Dientes')
number_files = len(ficheros)
print(number_files)
datagen = ImageDataGenerator()
imagenes = datagen.flow_from_directory('images/', class_mode=None, color_mode="grayscale", batch_size=number_files, target_size = (tamaño_imagen,tamaño_imagen))
x_train = imagenes.next()
x_train = (x_train - 127.5) / 127.5


# Create generator
print('Crear Generador')
modelo_generador = generador_de_imagenes()

# Create discriminator
print('Crear Discriminador')
modelo_discriminador = discriminador_de_imagenes()

# Train discriminator
print('Entrenar Discriminador')
entrenar_discriminador(modelo_discriminador, x_train)

print('Crear GAN')
gan = crear_gan(modelo_discriminador,modelo_generador, None)

# Train model
print('Entrenar GAN')
dataset = x_train.copy()
entrenamiento(gan, dataset, modelo_generador, modelo_discriminador, epochs = 7200, n_batch= 32, inicio = 0)


