import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image, ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


training_path = os.path.dirname(r'C:\Users\justi\OneDrive\Documents\Software\Python\PycharmProjects\IgniteMinds'
                                r'\brain_tumor_diagnosis\brain_tumor_image_set\training\\')
testing_path = os.path.dirname(r'C:\Users\justi\OneDrive\Documents\Software\Python\PycharmProjects\IgniteMinds'
                                r'\brain_tumor_diagnosis\brain_tumor_image_set\testing\\')

# Generate paths from root into each type of cancer.
# glioma_tumor_path = os.path.join(training_path, r'glioma_tumor')
# meningioma_tumor_path = os.path.join(training_path, r'meningioma_tumor')
# pituitary_tumor_path = os.path.join(training_path, r'pituitary_tumor')
# normal_path = os.path.join(training_path, r'normal')

img_width = 500
img_length = 500
batch_size = 100

train_generator = ImageDataGenerator().flow_from_directory(training_path,
                                                              target_size = (img_width, img_length),
                                                              classes = ['glioma', 'meningioma', 'pituitary', 'normal'],
                                                              batch_size = batch_size)

test_generator = ImageDataGenerator().flow_from_directory(testing_path,
                                                              target_size = (img_width, img_length),
                                                              classes = ['glioma', 'meningioma', 'pituitary', 'normal'],
                                                              batch_size = batch_size)

# Create a CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (500, 500, 3)))
model.add(Dropout(0.2))
model.add(Conv2D(20, kernel_size=(3, 3), strides = 2, activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
# model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

# Train and fit the model
model.fit(train_generator,
          steps_per_epoch = 10,
          validation_data = test_generator,
          validation_steps = 10,
          epochs = 5,
          verbose = 2)

# Evaluate image and predictions?
# model.predict()
