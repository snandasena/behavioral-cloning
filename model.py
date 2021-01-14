import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

learning_rate = 0.0001
epoches = 20
img_shape = (66, 200, 3)


def load_data():
    """
    """
    with np.load('./numpy/train-data.npz') as data:
        X,y = data['X'], data['y']

    return X,y


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


def train_model(model, X,y):
    """
    """
    # X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=12)

    checkpoint = keras.callbacks.ModelCheckpoint('./models/model-{epoch:03d}.h5',
                                                 monitor='val_loss',
                                                 verbose=2,
                                                 save_best_only='true',
                                                 mode='auto')

    earlystop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                              mode='auto',
                                              verbose=2,
                                              patience=5)

    model.compile(loss='mse', optimizer=tf.optimizers.Adam(learning_rate))
    print(model.summary())

    model.fit(X,
              y,
              epochs=epoches,
              validation_split=0.2,
              shuffle=True,
              callbacks=[checkpoint, earlystop],
              use_multiprocessing=True,
              workers=8,
              verbose=2)

    model.save('model.h5')


if __name__ == '__main__':
    """
    Main driver function
    """
    X,y = load_data()
    model = build_model()
    train_model(model, X,y)
