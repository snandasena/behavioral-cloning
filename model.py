import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2

learning_rate = 1e-4


def preprocess_image(img):
    '''
    Method for preprocessing images: this method is the same used in drive.py, except this version uses
    BGR to YUV and drive.py uses RGB to YUV (due to using cv2 to read the image here, where drive.py images are
    received in RGB)
    '''
    # original shape: 160x320x3, input shape for neural net: 66x200x3
    # crop to 105x320x3
    # new_img = img[35:140,:,:]
    # crop to 40x320x3
    new_img = img[50:140, :, :]
    # apply subtle blur
    new_img = cv2.GaussianBlur(new_img, (3, 3), 0)
    # scale to 66x200x3 (same as nVidia)
    new_img = cv2.resize(new_img, (200, 66), interpolation=cv2.INTER_AREA)
    # scale to ?x?x3
    # new_img = cv2.resize(new_img,(80, 10), interpolation = cv2.INTER_AREA)
    # convert to YUV color space (as nVidia paper suggests)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2YUV)
    return new_img


model = keras.Sequential(
    [
        layers.Lambda(lambda x: x / 127.5 - 1.0, input_shape=(66, 200, 3)),
        layers.add(layers.Convolution2D())
    ]
)

model.compile(optimizer=tf.optimizers.Adam(learning_rate), loss='mse')

model.save_weights('./model.h5')
