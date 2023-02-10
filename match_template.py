import cv2 as cv
import numpy as np
import os
import scipy
from scipy.signal import fftconvolve
import sys
import matplotlib.pyplot as plt
from PIL import Image


def show(img, title=' '):
    cv.imshow(f'{title}', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def correlation(temp, img):
    temp = temp - np.mean(temp)
    img = img - np.mean(img)

    a1 = np.ones(temp.shape)
    out = fftconvolve(img, np.flipud(np.fliplr(temp)).conj(), mode="same")
    img = fftconvolve(np.square(img), a1, mode="same") - np.square(fftconvolve(img, a1, mode="same")) / (
        np.prod(temp.shape))

    temp = np.sum(np.square(temp))
    out = out / np.sqrt(img * temp)
    out[np.where(np.logical_not(np.isfinite(out)))] = 0
    return out


def resize(img, scale=1):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    img_resized = cv.resize(img, (width, height), interpolation=cv.INTER_AREA)
    return img_resized


def find_max(matrices):
    maxi = 0
    max_x = 0
    max_y = 0
    ind = 0
    for i, matrix in enumerate(matrices):
        if np.amax(matrix) > maxi:
            maxi = np.amax(matrix)
            max_x, max_y = np.unravel_index(np.argmax(matrix), np.asarray(matrix).shape)
            ind = i

    return ind, max_x, max_y


def my_rot(rotateImage, angle):
    h, w = rotateImage.shape[0], rotateImage.shape[1]
    rotationMatrix = cv.getRotationMatrix2D((h // 2, w // 2), angle, 1.0)
    new_h = int((h * np.abs(rotationMatrix[0][1])) + (w * np.abs(rotationMatrix[0][0])))
    new_w = int((h * np.abs(rotationMatrix[0][0])) + (w * np.abs(rotationMatrix[0][1])))
    rotationMatrix[0][2] += (new_w / 2) - w // 2
    rotationMatrix[1][2] += (new_h / 2) - h // 2
    return cv.warpAffine(rotateImage, rotationMatrix, (new_w, new_h))


templates_list = os.listdir('templates')
templates = [cv.imread(f'templates/{template_name}') for template_name in templates_list]

images_list = os.listdir('images')
images = [cv.imread(f'images/{image_name}') for image_name in images_list]
angles = range(0, 360, 45)
scales = np.arange(0.8, 1.2, 0.01)

for image, template in zip(images, templates):
    org_img = image
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    cor_matrices = [correlation(my_rot(template, i), image) for i in angles]
    [cor_matrices.append(correlation(resize(template, scale), image)) for scale in scales]

    sum_mat = np.sum(cor_matrices, axis=0)

    x, y = np.unravel_index(np.argmax(sum_mat), np.asarray(sum_mat).shape)

    cv.circle(org_img, (y, x), 100, color=(0, 0, 255), thickness=5)
    org_img = cv.pyrDown(org_img)
    org_img = cv.pyrDown(org_img)
    show(org_img)

