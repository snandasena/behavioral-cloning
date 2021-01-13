# Import dependencies
import os

import cv2
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from tqdm import tqdm

np.random.seed(12)

# Image common params
i_height, i_width, i_channels = 66, 200, 3
img_shape = (i_height, i_width, i_channels)

# Dataset common params
data_dir = '/home/sajith/Documents/Acedamic/self-driving-car/data/data/'
meta_file = data_dir + 'driving_log.csv'
input_cols = ['center', 'left', 'right']
output_col = 'steering'
batch_size = 128
number_of_rows = 250


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


def select_random_image(image_path, steering_angle):
    """
    Randomly choose an image from the center, left or right, and adjust
    the steering angle.
    """
    center, left, right = image_path
    choice = np.random.choice(3)
    if choice == 0:
        return load_image(left), steering_angle + 0.2
    elif choice == 1:
        return load_image(right), steering_angle - 0.2
    return load_image(center), steering_angle


# Image data augmentation
def augment(image_path, in_streering_angle, range_x=100, range_y=10):
    """
    Augmenting images
    """
    img, steering_angle = select_random_image(image_path, in_streering_angle)

    img, steering_angle = random_flip(img, steering_angle)
    img, steering_angle = random_translate(img, steering_angle, range_x, range_y)
    img = crop_image(img)
    img = resize_image(img)
    img = random_shadow(img)
    img = random_brightness(img)
    img = convert_rgb2yuv(img)

    return img, steering_angle


# def batch_generator(image_paths, steering_angles, batch_size, is_training):
#     """
#     Generate training image give image paths and associated steering angles
#     """
#     images = np.empty([batch_size, i_height, i_width, i_channels], dtype=np.float32)
#     steers = np.empty(batch_size, dtype=np.float32)
#     while True:
#         i = 0
#         for index in np.random.permutation(image_paths.shape[0]):
#
#             image_path = image_paths[index]
#             steering_angle = steering_angles[index]
#             # argumentation
#             if is_training and np.random.rand() < 0.6:
#                 image, steering_angle = augment(image_path, steering_angle)
#
#             else:
#                 image = load_image(image_path[0])
#                 image = preprocess(image)
#                 # add the image and steering angle to the batch
#             images[i] = image
#             steers[i] = steering_angle
#
#             i += 1
#             if i == batch_size:
#                 break
#
#         yield images, steers

def batch_generator(image_paths, steering_angles, batch_size, total_samples, is_training):
    """
    Generate training image give image paths and associated steering angles
    """
    X = np.empty([total_samples * batch_size, i_height, i_width, i_channels], dtype=np.float32)
    y = np.empty(total_samples * batch_size, dtype=np.float32)

    row = 0
    for idx in tqdm(range(total_samples)):
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):

            image_path = image_paths[index]
            steering_angle = steering_angles[index]
            # argumentation
            if is_training and np.random.rand() < 0.6:
                image, steering_angle = augment(image_path, steering_angle)

            else:
                image = load_image(image_path[0])
                image = preprocess(image)
                # add the image and steering angle to the batch
            X[row] = image
            y[row] = steering_angle

            row += 1
            i += 1
            if i == batch_size:
                break

    np.savez_compressed("./numpy/train-data", X=X, y=y)

    print("X shape: ", X.shape)
    print("Y shape: ", y.shape)


def load_data():
    """
    """
    meta_df = pd.read_csv(meta_file)
    X = meta_df[input_cols].values
    y = meta_df[output_col].values

    print("X data shape: ", X.shape)
    print("Y data shape: ", y.shape)
    return X, y


if __name__ == '__main__':
    """
    Main driver function
    """
    X, y = load_data()
    batch_generator(X, y, batch_size, number_of_rows, True)
