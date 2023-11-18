
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


#conexión con google drive

from google.colab import drive
drive.mount('/content/drive')
path="/content/drive/MyDrive/Proyecto Metodologías ágiles para el desarrollo de aplicaciones con ML/Data/Imágenes/"

import cv2
def image_generator(folder_path):
    for filename in os.listdir(folder_path):
        image = cv2.imread(os.path.join(folder_path, filename))
        yield image




#preprocesamiento del conjunto de imágenes#

from PIL import Image

def load_images_in_directory(directory, batch_size):
    image_list = []
    for dirname, _, filenames in os.walk(directory):
        for filename in filenames:
            image = Image.open(os.path.join(dirname, filename))
            image_array = np.divide(np.array(image), 255)
            image_list.append(image_array)
            if len(image_list) >= batch_size:
                yield np.array(image_list)
                image_list = []
    if image_list:
        yield np.array(image_list)



list_name_train_benign=[]
list_name_train_malignant=[]
list_name_test_benign=[]
list_name_test_malignant=[]

for dirname, _, filenames in os.walk(path + 'train/malignant'):
    for filename in filenames:
        list_name_train_malignant.append(os.path.join(dirname, filename))

for dirname, _, filenames in os.walk(path + 'train/benign'):
    for filename in filenames:
        list_name_train_benign.append(os.path.join(dirname, filename))

for dirname, _, filenames in os.walk(path + 'test/malignant'):
    for filename in filenames:
        list_name_test_malignant.append(os.path.join(dirname, filename))

for dirname, _, filenames in os.walk(path + 'test/benign'):
    for filename in filenames:
        list_name_test_benign.append(os.path.join(dirname, filename))



from PIL import Image
image_list_train_benign= []
image_list_train_malignant= []
image_list_test_benign= []
image_list_test_malignant= []

#-----------------------------------------------------------------------------------
for dir_images in list_name_train_benign[:2000]:
  # Convert the image to a NumPy array
  image_list_train_benign.append(np.divide(np.array(Image.open(dir_images)),255))

#-----------------------------------------------------------------------------------
for dir_images in list_name_train_malignant[:2000]:
  # Convert the image to a NumPy array
  image_list_train_malignant.append(np.divide(np.array(Image.open(dir_images)),255))

#-----------------------------------------------------------------------------------
for dir_images in list_name_test_benign:
  # Convert the image to a NumPy array
  image_list_test_benign.append(np.divide(np.array(Image.open(dir_images)),255))

#-----------------------------------------------------------------------------------
for dir_images in list_name_test_malignant:
  # Convert the image to a NumPy array
  image_list_test_malignant.append(np.divide(np.array(Image.open(dir_images)),255))


image_list_train_benign= np.array(image_list_train_benign)
image_list_train_malignant= np.array(image_list_train_malignant)
image_list_test_benign= np.array(image_list_test_benign)
image_list_test_malignant= np.array(image_list_test_malignant)

print(image_list_train_benign.shape)
print(image_list_train_malignant.shape)
print(image_list_test_benign.shape)
print(image_list_test_malignant.shape)



#se concatena las imágenes de melanomas benignos y malignos en su conjunto de datos de entrenamiento y prueba correspondientes#

X_train = np.vstack((image_list_train_benign, image_list_train_malignant))
X_test = np.vstack((image_list_test_benign, image_list_test_malignant))

array_labels_train_benign = np.full((len(image_list_train_benign), 1), 0)
array_labels_train_malignant = np.full((len(image_list_train_malignant), 1), 1)
Y_train = np.vstack((array_labels_train_benign, array_labels_train_malignant))

array_labels_test_benign = np.full((len(image_list_test_benign), 1), 0)
array_labels_test_malignant = np.full((len(image_list_test_malignant), 1), 1)
Y_test = np.vstack((array_labels_test_benign, array_labels_test_malignant))

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)



# Divide el conjunto de entrenamiento oringinal en conjuntos de entrenamiento y validación con el 80% de los datos usados para entrenamiento y el 20% para validación.

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=5, stratify=Y_train)

# Veamos cuantas imagenes hay en cada partición
print(f'Datos de entrenamiento: {X_train.shape[0]}\nDatos de prueba: {X_test.shape[0]}\nDatos de validación: {X_val.shape[0]}\nTotal imágenes: {len(X_train)+len(X_test)+len(X_val)} ')



#Codificamos las etiquetas usando one-hot representation#

Y_train = tf.keras.utils.to_categorical(Y_train)
Y_val = tf.keras.utils.to_categorical(Y_val)
Y_test = tf.keras.utils.to_categorical(Y_test)

