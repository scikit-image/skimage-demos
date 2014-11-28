#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Francois Boulogne

import os.path

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import skimage.io
from skimage.filter import threshold_otsu
from skimage.segmentation import clear_border
from skimage.morphology import closing, square
from skimage.measure import regionprops, label
from skimage.color import label2rgb


def segment_digit(image, filename, output_dir, digit_height=100, digit_width=52,
                  border=7, black_on_white=True, closingpx=4, show=False):
    """
    Segement each digit of a picture

    :param image: grey scale picture
    :param filename: filename of the picture source
    :param output_dir: path for the output
    :param digit_height: height of a digit
    :param digit_width: width of a digit
    :param border: pixels to shift the border
    :param black_on_white: black digit on clear background
    :param closingpx: number of pixels to close
    """
    # apply threshold
    thresh = threshold_otsu(image)
    if black_on_white:
        bw = closing(image > thresh, square(closingpx))
    else:
        bw = closing(image < thresh, square(closingpx))

    filled = ndimage.binary_fill_holes(bw)

    # remove artifacts connected to image border
    cleared = filled.copy()
    clear_border(cleared)

    # label image regions
    label_image = label(cleared, background=None)
    borders = np.logical_xor(filled, cleared)
    label_image[borders] = -1
    image_label_overlay = label2rgb(label_image, image=image)

    if show:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        ax.imshow(image_label_overlay)

    regions = regionprops(label_image)

    for item, region in enumerate(regions):
        # skip small elements
        if region['Area'] < 300:
            continue

        # draw rectangle around segmented digits
        minr, minc, maxr, maxc = region['BoundingBox']
        #minr = maxr - digit_height
        #minc = maxc - digit_width
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)

        # uniq size
        img = np.zeros((100, 75), 'uint8')
        img[bw[minr:maxr, minc:maxc] != 0] = 255
        newname = os.path.splitext(os.path.basename(filename))[0] + '-' + str(item) + '.png'
        skimage.io.imsave(os.path.join(output_dir, newname), img)
        if show:
            ax.add_patch(rect)

    if show:
        plt.show()
