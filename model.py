import pickle

import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from image_utils import meta_file, batch_generator
from sklearn.model_selection import train_test_split

learning_rate = 0.0001
batch_size = 40
images_per_sample = 15000
epoches = 10
img_shape = (66, 200, 3)

input_cols = ['center', 'left', 'right']
output_col = 'steering'


def load_data():
    """
    """
    meta_df = pd.read_csv(meta_file)
    X = meta_df[input_cols].values
    y = meta_df[output_col].values

    print("X data shape: ", X.shape)
    print("Y data shape: ", y.shape)
    return X, y


def build_model():
    """

    """

    model = keras.Sequential(
        [
            layers.Lambda(lambda x: x / 127.5 - 1.0, input_shape=img_shape),
            layers.Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='elu'),
            layers.Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='elu'),
            layers.Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='elu'),
            layers.Conv2D(filters=64, kernel_size=(3, 3), activation='elu'),
            layers.Conv2D(filters=64, kernel_size=(3, 3), activation='elu'),
            layers.Dropout(0.5),
            layers.Flatten(),
            layers.Dense(100, activation='elu'),
            layers.Dense(50, activation='elu'),
            layers.Dense(10, activation='elu'),
            layers.Dense(1)
        ])

    return model


def train_model(model: tf.keras.Sequential, X, y):
    """
    """
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=12)

    checkpoint = keras.callbacks.ModelCheckpoint(
        './models/model-{epoch:03d}.h5',
        monitor='val_loss',
        verbose=0,
        save_best_only='true',
        mode='auto')

    model.compile(loss='mse', optimizer=tf.optimizers.Adam(learning_rate))
    print(model.summary())

    training_generator = batch_generator(X_train, y_train, batch_size, True)
    validation_generator = batch_generator(X_valid, y_valid, batch_size, False)

    model.fit(training_generator,
              epochs=epoches,
              max_queue_size=10,
              workers=10,
              validation_data=validation_generator,
              shuffle=True,
              callbacks=[checkpoint],
              use_multiprocessing=True,
              verbose=2)

    model.save('model.h5')


if __name__ == '__main__':
    X, y = load_data()
    model = build_model()
    train_model(model, X, y)
