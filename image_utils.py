# Import dependencies
import os
import pickle

import cv2
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from tqdm.contrib import tzip

# Image common params
i_height, i_width, i_channels = 66, 200, 3
img_shape = (i_height, i_width, i_channels)

# Dataset common params
data_dir = './data/data/'
meta_file = data_dir + 'driving_log.csv'
input_cols = ['center', 'left', 'right']
output_col = 'steering'
sample_per_image = 4


# Load an image
def load_image(img_path):
    """
    Load RGB image
    """
    return mpimg.imread(os.path.join(data_dir, img_path.strip()))


# Data Pre-processing
# Crop images to extract required road sections and to remove sky from the road
def crop_image(in_img):
    """
    This is used to cropping images
    """

    return in_img[60:-25, :, :]


# resize the images
def resize_image(in_img):
    """
    This is an utility function to resize images
    """
    return cv2.resize(in_img, (i_width, i_height), cv2.INTER_AREA)


# convert RGB to YUV image
def convert_rgb2yuv(in_img):
    """
    This is an utility function to convert RGB images to YUV.
    This technique was introduce by NVIDIA for their image pracessing pipeline
    """
    return cv2.cvtColor(in_img, cv2.COLOR_RGB2YUV)


# Define image prepocess pipeline
def preprocess(img):
    """
    This is a pipeline for image preprocesing
    """
    img = crop_image(img)
    img = resize_image(img)
    img = convert_rgb2yuv(img)
    return img


# Image Data Augmentation
# Flip images
def random_flip(img, streering_angle):
    """
    Flipping images randomly
    """
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        streering_angle = - streering_angle

    return img, streering_angle


# Translate images
def random_translate(img, streering_angle, range_x, range_y):
    """
    Randomly shift the image virtially and horizontally (translation).
    """

    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    streering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    h, w = img.shape[:2]
    img = cv2.warpAffine(img, trans_m, (w, h))
    return img, streering_angle


# Add random shadow
def random_shadow(img):
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = i_width * np.random.rand(), 0
    x2, y2 = i_width * np.random.rand(), i_height
    xm, ym = np.mgrid[0:i_height, 0:i_width]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line:
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(img[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio

    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


# Adgust brightness randomly
def random_brightness(img):
    """
    Randomly adjust brightness of the image.
    """

    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:, :, 2] = hsv[:, :, 2] * ratio

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


# Image data augmentation
def augment(img, streering_angle, range_x=100, range_y=10):
    """
    Augmenting images
    """
    img, streering_angle = random_flip(img, streering_angle)
    img, streering_angle = random_translate(img, streering_angle, range_x, range_y)
    img = crop_image(img)
    img = resize_image(img)
    img = random_shadow(img)
    img = random_brightness(img)
    img = convert_rgb2yuv(img)

    return img, streering_angle


def load_and_agment(iamge_path, streering_angles):
    """

    """
    x = []
    y = []
    img = load_image(iamge_path)
    print("Loaded image shape: ", img.shape)
    procced_img = preprocess(img)
    print("Pre-processed image shape: ", procced_img.shape)
    x.append(procced_img)
    y.append(streering_angles)

    for i in range(sample_per_image):
        st_a = streering_angles
        aug, st_a = augment(img, st_a)
        x.append(aug)
        y.append(st_a)

    return x, y


def image_data_augmentation(image_paths, streering_angles):
    """

    """
    X_train = []
    y_train = []
    for image_path, streering_angle in tzip(image_paths, streering_angles):
        for path in image_path:
            x, y = load_and_agment(path, streering_angle)
            X_train += x
            y_train += y

    return X_train, y_train


if __name__ == '__main__':
    meta_df = pd.read_csv(meta_file)
    X = meta_df[input_cols].values[:10]
    y = meta_df[output_col].values[:10]
    X_train, y_train = image_data_augmentation(X, y)

    X_train = np.array(X_train, np.float32)
    y_train = np.array(y_train, np.float32)

    image_data = {
        'X_train': X_train,
        'y_train': y_train
    }

    with open("data/data.p", 'wb') as p:
        pickle.dump(image_data, p, protocol=pickle.HIGHEST_PROTOCOL)
