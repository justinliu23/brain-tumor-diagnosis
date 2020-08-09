import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image, ImageDataGenerator
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, GlobalAveragePooling2D
# from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

trainpath = os.path.dirname('/content/train/lung_colon_image_set')
main_path_string = '/content/train/lung_colon_image_set/lung_image_sets/'
main_path = os.path.dirname(main_path_string)

# Generate paths from root into each type of tumor
glioma_path = os.path.join(main_path, '')
meningioma_path = os.path.join(main_path, '')
pituitary_path = os.path.join(main_path, '')
normal_path = os.path.join(main_path, '')

def preprocess_data(directory_list, image_size):
    X = []
    y = []
    image_width, image_height = image_size

    for directory in directory_list:
        for filename in directory_list(directory):
            # Load the image
            image = cv2.imread(directory + '\\' + filename)
            # crop the brain and ignore the unnecessary rest part of the image
            # image = crop_brain_contour(image, plot=False)
            # resize image
            # image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
            # normalize values
            image /= 255.
            # convert image to numpy array and append it to X
            X.append(image)
            # append a value of 1 to the target array if the image
            # is in the folder named 'yes', otherwise append 0.
            if directory[-3:] == 'yes':
                y.append([1])
            else:
                y.append([0])

    X = np.array(X)
    y = np.array(y)

    # Shuffle the data
    X, y = shuffle(X, y)

    print(f'Number of examples is: {len(X)}')
    print(f'X shape is: {X.shape}')
    print(f'y shape is: {y.shape}')

    return X, y

def predict_tumor():
