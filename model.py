import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

learning_rate = 1e-4
batch_size = 40
epoches = 10
img_shape = (66, 200, 3)


def load_data():
    """
    """
    with open('data/data.p', mode='rb') as f:
        train = pickle.load(f)
    X, y = train['X_train'], train['y_train']
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
            layers.Dense(1),
        ])

    return model


def train_model(model, X_train, y_train):
    """
    """
    checkpoint = keras.callbacks.ModelCheckpoint(
        './models/model-{epoch:03d}.h5',
        monitor='val_loss',
        verbose=0,
        save_best_only='true',
        mode='auto')

    model.compile(loss='mse', optimizer=tf.optimizers.Adam(learning_rate))
    print(model.summary())

    model.fit(X_train,
              y_train,
              epochs=epoches,
              validation_split=0.2,
              batch_size=batch_size,
              shuffle=True,
              callbacks=[checkpoint],
              use_multiprocessing=True,
              verbose=2)

    model.save('model.h5')


if __name__ == '__main__':
    X, y = load_data()
    model = build_model()
    train_model(model, X, y)
