import cv2
import numpy as np
from tensorflow.keras.models import load_model
from image_utils import preprocess

model = load_model('./model.h5')

print(model.summary())

img = cv2.imread('./data/data/IMG/center_2016_12_01_13_30_48_287.jpg')
img = preprocess(img)

img = np.asarray(img, dtype=np.float32)
print(img.shape)
print(float(model.predict(img[None,:,:,:])[0]))

