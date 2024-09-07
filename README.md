# Project1
from tensorflow.keras.datasets import mnist #Загружаем базу mnist
from tensorflow.keras.datasets import cifar10 #Загружаем базу cifar10
from tensorflow.keras.datasets import cifar100 #Загружаем базу cifar100

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.models import Sequential #Сеть прямого распространения
#Базовые слои для счёрточных сетей
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator # работа с изображениями
from tensorflow.keras.optimizers import Adam, Adadelta # оптимизаторы
from tensorflow.keras import utils #Используем дял to_categoricall
from tensorflow.keras.preprocessing.image import load_img #Для отрисовки изображений и загрузки
from tensorflow.keras.preprocessing import image #Для отрисовки изображений
from google.colab import files #Для загрузки своей картинки
import numpy as np #Библиотека работы с массивами
import matplotlib.pyplot as plt #Для отрисовки графиков
from PIL import Image #Для отрисовки изображений
import random #Для генерации случайных чисел 
import math # Для округления
import os #Для работы с файлами 
# подключем диск
from google.colab import drive

%matplotlib inline
import os
import sys
from PIL import Image
!pip install matplotlib-venn
!apt-get -qq install -y libfluidsynth1
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ZeroPadding2D, Add, UpSampling2D, Concatenate, Lambda, LeakyReLU # Стандартные слои Keras
from tensorflow.keras.regularizers import l2 # Регуляризатор l2
from tensorflow.keras.optimizers import Adam # Оптимизатор Adam
from tensorflow.keras.models import Model # Абстрактный класс Model
from PIL import Image, ImageDraw, ImageFont # Модули работы с изображениями
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb # Модули конвертации между RGB и HSV
from google.colab import files # Модуль работы с файловой системой google

import tensorflow.keras.backend as K # бэкенд Keras
import math # Импортируем модуль math
import pandas as pd # Пандас для работы с csv
import tensorflow as tf # TensorFlow
import numpy as np # numpy массивы
import matplotlib.pyplot as plt # графики
import os # модуль работы с файловой системой
import time # модуль работы со 
import cv2
from google.colab import drive
drive.mount('/content/drive')
def ImageCreatorGenerator(fileSrc,fileTo): # Создание функции и обозначение пути исходного и пути папки куда складываем
  image2=load_img(fileSrc)      # Загрузка изображения
  image2=np.array(image2)       # Перевод изображения в numpy-массив
  image2=np.expand_dims(image2,axis=0)  # Добавление массиву одной размерности
  augmentation2=ImageDataGenerator(rotation_range=45,zoom_range=[1.0,1.6], #Создание экземпляра генератора
                 width_shift_range=0.0,height_shift_range=0.0,       
                 shear_range=0.15,
                 horizontal_flip=False,fill_mode="nearest")
  imagegen2=augmentation2.flow(image2,batch_size=1, save_to_dir=fileTo,   #Создание экземпляра генератора что сохраняет изображения
              save_format='jpg')
  next(imagegen2)               # Действие для сохранение генератором одной картинки
  def wholeSave(pathFrom, pathTo):   # Создане функции и обозначение папки откуда берутся изображения и куда складываются изображения
    for currPath in os.listdir(pathFrom):  # Цикл перебора всех названий исходных изображений
        if currPath.endswith('jpg'):  # Проверка что файл с расширением jpg
            print(currPath)  # Фиксация загрузок картинок в папке
            oldPath = os.path.join(pathFrom, currPath) # Полный путь до исходного изображения
            ImageCreatorGenerator(oldPath,pathTo)  # Преобразование и сохранение нового изображения
  wholeSave('/content/drive/MyDrive/Микросхемы1/NE555P(Таймер)', '/content/drive/MyDrive/Микросхемы/Микросхемы') # Вызов функции
  !unzip -q "/content/drive/MyDrive/МИКРОСХЕМЫ.zip" -d /content/microchips #Указываем путь к базе в Google Drive
  train_path = '/content/microchips' #Папка с папками картинок, рассортированных по категориям
batch_size = 25 #Размер выборки
img_width = 300 #Ширина изображения
img_height = 300 #Высота изображения
#Генератор изображений
datagen = ImageDataGenerator(
    rescale=1. / 255, #Значения цвета меняем на дробные показания
    rotation_range=30, #Поворачиваем изображения при генерации выборки
    width_shift_range=0.2, #Двигаем изображения по ширине при генерации выборки
    height_shift_range=0.2, #Двигаем изображения по высоте при генерации выборки
    zoom_range=0.1, #Зумируем изображения при генерации выборки
    horizontal_flip=True, #Включаем отзеркаливание изображений
    fill_mode='nearest', #Заполнение пикселей вне границ ввода
    validation_split=0.2 #Указываем разделение изображений на обучающую и тестовую выборку
)
# обучающая выборка
train_generator = datagen.flow_from_directory(
    train_path, #Путь ко всей выборке выборке
    target_size=(img_width, img_height), #Размер изображений
    batch_size=batch_size, #Размер batch_size
    class_mode='categorical', #Категориальный тип выборки. Разбиение выборки по маркам авто 
    shuffle=True, #Перемешивание выборки
    subset='training' # устанавливаем как набор для обучения
)

# проверочная выборка
validation_generator = datagen.flow_from_directory(
    train_path, #Путь ко всей выборке выборке
    target_size=(img_width, img_height), #Размер изображений
    batch_size=batch_size, #Размер batch_size
    class_mode='categorical', #Категориальный тип выборки. Разбиение выборки по маркам авто 
    shuffle=True, #Перемешивание выборки
    subset='validation' # устанавливаем как валидационный набор
)
#Выводим для примера картинки по каждому классу

fig, axs = plt.subplots(1, 3, figsize=(25, 5)) #Создаем полотно из 3 графиков
for i in range(3): #Проходим по всем классам
  microchip_path = train_path + '/' + os.listdir(train_path)[i] + '/'#Формируем путь к выборке
  img_path = microchip_path + random.choice(os.listdir(microchip_path)) #Выбираем случайное фото для отображения
  axs[i].imshow(image.load_img(img_path, target_size=(img_height, img_width))) #Отображение фотографии

plt.show() #Показываем изображения
#Создаем последовательную модель
model = Sequential()
#Первый сверточный слой
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(img_width, img_height, 3)))
#Второй сверточный слой
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
#Третий сверточный слой
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
#Слой регуляризации Dropout
model.add(Dropout(0.2))
#Четвертый сверточный слой
model.add(Conv2D(128, (3, 3), padding='same', activation='tanh'))
model.add(MaxPooling2D(pool_size=(3, 3)))
#Слой регуляризации Dropout
model.add(Dropout(0.3))
#Пятый свёрточный слой
model.add(Conv2D(256, (3, 3), padding='same', activation='tanh'))
#Шестой свёрточный слой
model.add(Conv2D(256, (3, 3), padding='same'))
#Седьмой свёрточный слой
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
#Слой регуляризации Dropout
model.add(Dropout(0.3))
#Восьмой свёрточный слой
model.add(Conv2D(512, (3, 3), padding='same'))
#Девятый свёрточный слой
model.add(Conv2D(1024, (3, 3), padding='same', activation='tanh'))
model.add(MaxPooling2D(pool_size=(3, 3)))
#Десятый свёрточный слой
model.add(Conv2D(2048, (3, 3), padding='same', activation='relu'))
#Одиннадцатый свёрточный слой
model.add(Conv2D(2048,(3,3), padding='same'))
#Слой преобразования двухмерных данных в одномерные
model.add(Flatten())
#Полносвязный слой
model.add(Dense(2048, activation='relu'))
# Полносвязный слой
model.add(Dense(1024, activation='relu'))
#Вызодной полносвязный слой
model.add(Dense(len(train_generator.class_indices), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.00007), metrics=['accuracy'])
history = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // batch_size,
    epochs=50,
    verbose=1
)
# Дообучаем модель ещё на эпохах с уменьшенным шагом
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.00001), metrics=['accuracy'])
history = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // batch_size,
    epochs=5,
    verbose=1
)
# Строим график для отображения динамики обучения и точности предсказания сети
plt.figure(figsize = (14, 7))
plt.plot(history.history['accuracy'], 
         label='Доля верных ответов на обучающем наборе')
plt.plot(history.history['val_accuracy'], 
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()
