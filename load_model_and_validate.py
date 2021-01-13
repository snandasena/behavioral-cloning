import cv2
import numpy as np
from tensorflow.keras.models import load_model
from image_utils import preprocess

model = load_model('./model.h5')

print(model.summary())

img = cv2.imread('/home/sajith/Documents/Acedamic/self-driving-car/data/data/IMG/center_2021_01_12_13_20_30_027.jpg')

cv2.imshow('tem', img)
img = preprocess(img)

img = np.asarray(img, dtype=np.float32)
print(img.shape)
print(float(model.predict(img[None,:,:,:])[0]))

