# import numpy as np
# import pandas as pd
import os
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.preprocessing import image, ImageDataGenerator
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, GlobalAveragePooling2D
# from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg


# Generate path from root into lung image folder
# training_path = os.path.dirname(r'\brain_tumor_image_set\training\')
training_path = os.path.dirname(r'C:\Users\justi\OneDrive\Documents\Software\Python\PycharmProjects\IgniteMinds'
                                r'\brain_tumor_diagnosis\brain_tumor_image_set\training\\')

# Generate paths from root into each type of cancer.
glioma_tumor_path = os.path.join(training_path, r'glioma_tumor')
meningioma_tumor_path = os.path.join(training_path, r'meningioma_tumor')
pituitary_tumor_path = os.path.join(training_path, r'pituitary_tumor')
normal_path = os.path.join(training_path, r'normal')

# print(glioma_tumor_path)


# Returns all file paths within a directory.
def get_tumor_paths(my_directory):
    file_paths = []

    for folder, subs, files in os.walk(my_directory):
        for filename in files:
            file_paths.append(os.path.abspath(os.path.join(folder, filename)))

    # for i in file_paths:
    #     print(i)

    return file_paths

glioma = get_tumor_paths(glioma_tumor_path)
meningioma = get_tumor_paths(meningioma_tumor_path)
pituitary = get_tumor_paths(pituitary_tumor_path)
normal = get_tumor_paths(normal_path)
#
# for i in glioma:
#     print(i)


# Generate Dataframes
# adeno = list(zip(adenocarc, ['adenocarc']*len(adenocarc)))
# adeno_df = pd.DataFrame(adeno, columns=['file', 'label'])
# print(adeno_df.head())
#
# norm = list(zip(normal, ['normal']*len(normal)))
# normal_df = pd.DataFrame(norm, columns=['file', 'label'])
# print(normal_df.head())
#
# squamouscarcinoma=list(zip(squamouscarcinoma, ['squamous']*len(squamouscarcinoma)*2))
# squamous_df = pd.DataFrame(squamouscarcinoma, columns=['file', 'label'])
# print(squamous_df.head())

# Combine into one dataframe
# dataframe=pd.concat([adeno_df, normal_df, squamous_df])
# dataframe.sample(5)

# Convert to style for data_gen
# df = pd.get_dummies(dataframe['label'])
# df=pd.concat([dataframe, df], axis=1)
# df.sample(5)

# Create the generator
# from ImageDataAugmentor.image_data_augmentor import *
# data_gen= ImageDataAugmentor(rescale = 1/255.)

# Split into training and test
# img_shape = 300
# batch_size = 100
# length = len(df)
# train_df = df.iloc[0:int(length*0.8), :]
# test_df = df.iloc[int(length*0.8):int(length), :]
# train_generator=data_gen.flow_from_dataframe(train_df,
#                                              target_size = (img_shape, img_shape),
#                                              x_col = 'file',
#                                              y_col = ['glioma', 'meningioma', 'pituitary', 'normal'],
#                                              class_mode = 'raw',
#                                              shuffle=True,
#                                              batch_size = batch_size)
# test_generator=data_gen.flow_from_dataframe(test_df,
#                                              target_size = (img_shape, img_shape),
#                                              x_col = 'file',
#                                              y_col = ['glioma', 'meningioma', 'pituitary', 'normal'],
#                                              class_mode = 'raw',
#                                            shuffle=False,
#                                              batch_size = batch_size)

# Preview the MRI Scans?


# Create a CNN
# model = Sequential()
# model.add(Conv2D(input_shape=(img_rows, img_cols, 3),
#                  filters = 4,
#                  kernel_size = 3,
#                  activation = 'relu'))
# model.add(Dropout(0.2))
# model.add(Conv2D(20, kernel_size=(3, 3), strides = 2, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(num_classes, activation='softmax'))
#
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])


# Train and fit the model
# model.fit(train_generator,
#           epochs = 5,
#           steps_per_epoch = train_generator,
#           validation_data = test_generator)


# Evaluate image and predictions?
# model.predict()
