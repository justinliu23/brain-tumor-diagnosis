import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg


training_dir = r'brain_tumor_image_set/training'
testing_dir  = r'brain_tumor_image_set/testing'

img_width, img_height = 512, 512
input_shape = (img_width, img_height, 3)

train_samples = 2900
test_samples = 400
epochs = 10
batch_size = 100

# Data augmentation
train_data_gen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_data_gen = ImageDataGenerator(rescale=1. / 255)

# Create data generators
train_generator = train_data_gen.flow_from_directory(
    training_dir,
    target_size = (img_width, img_height),
    class_mode = 'categorical',
    batch_size = batch_size,
    shuffle = True)

test_generator = test_data_gen.flow_from_directory(
    testing_dir,
    target_size=(img_width, img_height),
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=False)


# Create a model
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = input_shape))
    model.add(Conv2D(32, (3, 3), activation = 'relu'))
    model.add(Flatten())
    model.add(Dense(32, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation = 'softmax'))

    model = Sequential([
    data_augmentation,
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
    ])

    return model


# Compile the model
def compile_model(model):
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = 'adam',
                  metrics = ['accuracy'])


# Train and fit the model
def fit_model(model):
    history = model.fit(
        train_generator,
        epochs = epochs,
        steps_per_epoch = train_samples / batch_size,
        validation_data = test_generator,
        validation_steps = test_samples / batch_size)
        # verbose = 2)


# Print an overall summary of the model
def print_model_summary(model):
    model.summary()


def predict_image(model):
    model.predict()

#   acc = history.history['accuracy']
#   val_acc = history.history['val_accuracy']

#   loss=history.history['loss']
#   val_loss=history.history['val_loss']

#   epochs_range = range(epochs)

#   plt.figure(figsize=(8, 8))
#   plt.subplot(1, 2, 1)
#   plt.plot(epochs_range, acc, label='Training Accuracy')
#   plt.plot(epochs_range, val_acc, label='Validation Accuracy')
#   plt.legend(loc='lower right')
#   plt.title('Training and Validation Accuracy')

#   plt.subplot(1, 2, 2)
#   plt.plot(epochs_range, loss, label='Training Loss')
#   plt.plot(epochs_range, val_loss, label='Validation Loss')
#   plt.legend(loc='upper right')
#   plt.title('Training and Validation Loss')
#   plt.show()


# Example on how to predict a given image
# sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
# sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)
#
# img = keras.preprocessing.image.load_img(
#     sunflower_path, target_size=(img_height, img_width)
# )
# img_array = keras.preprocessing.image.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0) # Create a batch
#
# predictions = model.predict(img_array)
# score = tf.nn.softmax(predictions[0])
#
# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )


model = create_model()
compile_model(model)
fit_model(model)
# print_model_summary(model)
# predict_image(model)
