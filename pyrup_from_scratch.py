import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import os
from pathlib import Path
import timeit


def convolve(img, kern):
    k = len(kern)
    convolved_img = arr_zeros(img)
    img_with_padding = arr_zeros_2(len(img) + 2 * 2, len(img) + 2 * 2)
    img_with_padding = np.asarray(img_with_padding)
    img_with_padding[2:-2, 2:-2] = img

    for i in range((len(img))):
        for j in range((len(img))):
            mat = img_with_padding[i:(i+k), j:(j+k)]
            convolved_img[i][j] = summ(mult(mat, kern))

    convolved_img_normalized = normalize(convolved_img)
    return convolved_img_normalized


def normalize(inp):
    inp = np.asarray(inp)
    dfmax, dfmin = inp.max(), inp.min()
    normalized = (inp - dfmin) / (dfmax - dfmin)
    return normalized


def summ(m_1):
    suma = 0
    m_1 = np.asarray(m_1)
    for i in range(len(m_1)):
        for j in range(len(m_1[0])):
            suma += m_1[i][j]
    return suma


def mult(A, B):
    result = [[sum(a*b for a, b in zip(A_row, B_col)) for B_col in zip(*B)] for A_row in A]
    return result


def arr_zeros(size):
    convolved_img = []
    convolved_img_row = []
    for i in range(len(size[0])):
        for j in range(len(size[1])):
            convolved_img_row.append(0.)
        convolved_img.append(convolved_img_row)
        convolved_img_row = []
    return convolved_img


def arr_zeros_2(x, y, z=0):
    if z != 0:
        return [[[0. for _ in range(z)] for _ in range(y)] for _ in range(x)]
    else:
        return [[0. for _ in range(y)] for _ in range(x)]


def as_pyr_down(img, kern):
    maska = arr_zeros_2(len(img[0]), len(img[0]), 3)
    maska = np.asarray(maska)
    down_mask = maska[:round((len(maska[0])) / 2), :round(len(maska[0]) / 2)]
    for dim in range(3):
        maska[:, :, dim] = convolve(img[:, :, dim], kern)
        down_mask[:, :, dim] = pyr_down(maska[:, :, dim])
    return down_mask


def pyr_down(x):
    y = arr_zeros(x)
    y = np.asarray(y)
    for i, k in zip(range(0, len(x), 2), range(round(len(x) / 2))):
        for j, l in zip(range(0, len(x[0]), 2), range(round(len(x) / 2))):
            y[k][l] = x[i][j]
    y = y[:round((len(x))/2), :round(len(x)/2)]
    return y


def as_pyr_up(x):
    x = np.asarray(x)
    y = arr_zeros_2(len(x) * 2, len(x) * 2, 3)
    y = np.asarray(y)
    for dim in range(3):
        for i, k in zip(range(0, len(x)), range(0, len(x) * 2, 2)):
            for j, l in zip(range(0, len(x[0])), range(0, len(x) * 2, 2)):
                for m in range(2):
                    for n in range(2):
                        y[k + n, l + m, dim] = x[i, j, dim]
    return y


def gaussian_pyramid(img, levels, kern):
    lower = img.copy()
    gaussian_pyr = [lower]
    for i in range(levels):
        lower = as_pyr_down(lower, kern)
        gaussian_pyr.append(lower)
    return gaussian_pyr


def laplacian_pyramid(gaussian_pyr):
    laplacian_top = gaussian_pyr[-1]
    levels = len(gaussian_pyr) - 1
    laplacian_pyr = [laplacian_top]
    for i in range(levels, 0, -1):
        gaussian_expanded = as_pyr_up(gaussian_pyr[i])
        laplacian = normalize(sub(gaussian_pyr[i - 1], gaussian_expanded))
        laplacian_pyr.append(laplacian)
    return laplacian_pyr


def sub(matrix1, matrix2):
    result = [[matrix1[i][j] - matrix2[i][j] for j in range(len(matrix1[0]))] for i in range(len(matrix1))]
    return result


def add(matrix1, matrix2):
    result = [[matrix1[i][j] + matrix2[i][j] for j in range(len(matrix1[0]))] for i in range(len(matrix1))]
    return result


def blend(laplacian_A, laplacian_B, mask_pyr):
    LS = []
    for la, lb, mask in zip(laplacian_A, laplacian_B, mask_pyr):
        ls = lb * mask + la * (1.0 - mask)
        LS.append(ls)
    return LS


def reconstruct(laplacian_pyr):
    laplacian_top = laplacian_pyr[0]
    laplacian_lst = [laplacian_top]
    for i in range(len(laplacian_pyr) - 1):
        laplacian_expanded = as_pyr_up(laplacian_top)
        laplacian_top = normalize(add(laplacian_pyr[i+1], laplacian_expanded))
        laplacian_lst.append(laplacian_top)
    return laplacian_lst


def to_time():
    gaussian_pyr_1 = gaussian_pyramid(img1, lvl, kernel)
    laplacian_pyr_1 = laplacian_pyramid(gaussian_pyr_1)
    print('raz', end=' ')
    gaussian_pyr_2 = gaussian_pyramid(img2, lvl, kernel)
    laplacian_pyr_2 = laplacian_pyramid(gaussian_pyr_2)
    print('dwa', end=' ')
    mask_pyr_final = gaussian_pyramid(mask, lvl, kernel)
    mask_pyr_final.reverse()
    print('trzy')
    blended = blend(laplacian_pyr_1, laplacian_pyr_2, mask_pyr_final)
    final = reconstruct(blended)
    return final


lvl = 3
images = [p for p in Path('.').glob('*.jp*')]
head_tail = os.path.split(__file__)
filepath = head_tail[0]
A_path = str(Path(filepath,  images[0]))
B_path = str(Path(filepath,  images[1]))
img1 = plt.imread(A_path)
img2 = plt.imread(B_path)
img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
img1 = img1/255
img2 = img2/255
img1 = cv.resize(img1, (160, 160))
img2 = cv.resize(img2, (160, 160))
kernel = [[1/256,  4/256,  6/256,  4/256, 1/256],
          [4/256, 16/256, 24/256, 16/256, 4/256],
          [6/256, 24/256, 36/256, 24/256, 6/256],
          [4/256, 16/256, 24/256, 16/256, 4/256],
          [1/256,  4/256,  6/256,  4/256, 1/256]]
mask = arr_zeros_2(160, 160, 3)
mask = np.asarray(mask)
mask[:160, :80] = (1, 1, 1)

starttime = timeit.default_timer()
print("The start time is :", starttime)
for i in range(10):
    print(i+1, end=': ')
    to_time()
print("\nAverge time is :", (timeit.default_timer() - starttime)/10)



