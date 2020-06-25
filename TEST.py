# coding=utf-8
import numpy as np
import cv2 as cv
import numpy.ma as ma
import matplotlib.pyplot as plt
from skimage import io, measure, data, filters
from skimage import morphology
from skimage.morphology import square


def OTSU(img_gray):
    max_g = 0
    suitable_th = 0
    th_begin = 30
    th_end = 60
    for threshold in range(th_begin, th_end):
        bin_img = img_gray > threshold
        bin_img_inv = img_gray <= threshold
        fore_pix = np.sum(bin_img)
        back_pix = np.sum(bin_img_inv)
        if 0 == fore_pix:
            break
        if 0 == back_pix:
            continue

        w0 = float(fore_pix) / img_gray.size
        u0 = float(np.sum(img_gray * bin_img)) / fore_pix
        w1 = float(back_pix) / img_gray.size
        u1 = float(np.sum(img_gray * bin_img_inv)) / back_pix
        # intra-class variance
        g = w0 * w1 * (u0 - u1) * (u0 - u1)
        if g > max_g:
            max_g = g
            suitable_th = threshold

    return suitable_th


def area(input_img):
    temp_area = 0
    for p in range(input_img.shape[1]):
        for q in range(input_img.shape[0]):
            if input_img[p, q] == 1:
                temp_area += 1
            else:
                temp_area += 0
    return temp_area


def loc_thresh(img):
    threshold0 = filters.threshold_otsu(img)
    binary_img = (img >= threshold0)
    labels = measure.label(binary_img, connectivity=1)
    label_att = measure.regionprops(labels)
    C = len(label_att)
    for i in range(1, C + 1):
        temp_img = labels
        result = []
        threshold = []
        for p in range(labels.shape[0]):
            for q in range(labels.shape[1]):
                if temp_img[p, q] == i:
                    temp_img[p, q] = 1
                else:
                    temp_img[p, q] = 0
        mor_img = morphology.dilation(temp_img, square(3))
        r = area(temp_img)
        while area(mor_img) <= 2 * area(temp_img):
            mor_img = morphology.dilation(mor_img, square(3))
        pre_ostu = mor_img * img
        mask = (pre_ostu == 0) 
        zone = ma.masked_array(pre_ostu, mask=mask)
        temp_threshold = OTSU(zone)
        threshold.append(temp_threshold)
        temp_result = (zone >= threshold)
        result.append(temp_result)
    return result


if __name__ == "__main__":
    img = cv.imread(r'C:\Users\Lavine He\Desktop\test1111.jpg')
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    result = loc_thresh(img)