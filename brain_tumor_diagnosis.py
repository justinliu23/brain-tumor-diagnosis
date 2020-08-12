import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Use this if running on Google Colab
training_dir = r'/content/drive/My Drive/Colab Notebooks/Brain Tumor Diagnosis/brain_tumor_image_set/Training'
testing_dir = r'/content/drive/My Drive/Colab Notebooks/Brain Tumor Diagnosis/brain_tumor_image_set/Testing'
# Use this if running on local machine
# training_dir = r'brain_tumor_image_set/training'
# testing_dir  = r'brain_tumor_image_set/testing'

num_classes = 4
img_width, img_height = 150, 150
input_shape = (img_width, img_height, 3)

train_samples = 2870
test_samples = 394
epochs = 5
batch_size = 20

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
    target_size=(img_width, img_height),
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True)

test_generator = test_data_gen.flow_from_directory(
    testing_dir,
    target_size=(img_width, img_height),
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=False)


# Create a CNN from the Sequential Keras API
def create_model():
    model = Sequential()

    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    # model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    # model.add(Conv2D(64, (3, 3), activation = 'relu', padding='same'))
    # model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    return model


# Print an overall summary of the model
def print_model_summary(model):
    model.summary()


# Standard model configuration
def compile_model(model):
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


# Train the model
def fit_model(model):
    history = model.fit(
        train_generator,
        epochs=epochs,
        steps_per_epoch=(train_samples / batch_size),
        validation_data=test_generator,
        validation_steps=(test_samples / batch_size),
        verbose=1)

    return history


def evaluate_model(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


# Diagnosis an MRI scan
def predict_image(model):
    model.predict()

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
print_model_summary(model)
compile_model(model)
history = fit_model(model)
evaluate_model(history)
# predict_image(model)
