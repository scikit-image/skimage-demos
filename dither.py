from __future__ import division
from skimage import img_as_float, io
from skimage.filters import threshold_otsu
import numpy as np


def quantize(image, L=1, N=4):
    """Quantize an image.

    Parameters
    ----------
    image : array_like
        Input image.
    L : float
        Maximum input value.
    N : int
        Number of quantization levels.

    """
    T = np.linspace(0, L, N, endpoint=False)[1:]
    return np.digitize(image.flat, T).reshape(image.shape)


def dither(image, N=4, positions=None, weights=None):
    """Quantize an image, using dithering.

    Parameters
    ----------
    image : ndarray
        Input image.
    N : int
        Number of quantization levels.
    positions : list of (i, j) offsets
        Position offset to which the quantization error is distributed.
        By default, implement Sierra's "Filter Lite".
    weights : list of ints
        Weights for propagated error.
        By default, implement Sierra's "Filter Lite".

    References
    ----------
    http://www.efg2.com/Lab/Library/ImageProcessing/DHALF.TXT

    """
    image = img_as_float(image.copy())

    if positions is None or weights is None:
        positions = [(0, 1), (1, -1), (1, 0)]
        weights = [2, 1, 1]

    weights = weights / np.sum(weights)

    T = np.linspace(0, 1, N, endpoint=False)[1:]
    rows, cols = image.shape

    out = np.zeros_like(image, dtype=float)
    for i in range(rows):
        for j in range(cols):
            # Quantize
            out[i, j], = np.digitize([image[i, j]], T)

            # Propagate quantization noise
            d = (image[i, j] - out[i, j] / (N - 1))
            for (ii, jj), w in zip(positions, weights):
                ii = i + ii
                jj = j + jj
                if ii < rows and jj < cols:
                    image[ii, jj] += d * w

    return out


def floyd_steinberg(image, N):
    offsets = [(0, 1), (1, -1), (1, 0), (1, 1)]
    weights = [      7,
               3, 5, 1]
    return dither(image, N, offsets, weights)


def stucki(image, N):
    offsets = [(0, 1), (0, 2), (1, -2), (1, -1),
               (1, 0), (1, 1), (1, 2),
               (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)]
    weights = [         8, 4,
               2, 4, 8, 4, 2,
               1, 2, 4, 2, 1]
    return dither(image, N, offsets, weights)


# Image with 255 color levels
img = img_as_float(io.imread('data/david.png'))

# Quantize to N levels
N = 2
img_quant = quantize(img, N=N)

img_dither_random = img + np.abs(np.random.normal(size=img.shape,
                                           scale=1./(3 * N)))
img_dither_random = quantize(img_dither_random, L=1, N=N)

img_dither_fs = floyd_steinberg(img, N=N)
img_dither_stucki = stucki(img, N=N)

import matplotlib.pyplot as plt
f, ax = plt.subplots(2, 3, subplot_kw={'xticks': [], 'yticks': []})
ax[0, 0].imshow(img, cmap=plt.cm.gray, interpolation='nearest')
ax[0, 1].imshow(img_quant, cmap=plt.cm.gray, interpolation='nearest')
ax[0, 2].imshow(img > threshold_otsu(img), cmap=plt.cm.gray, interpolation='nearest')
#ax[0, 2].set_visible(False)
ax[1, 0].imshow(img_dither_random, cmap=plt.cm.gray, interpolation='nearest')
ax[1, 1].imshow(img_dither_fs, cmap=plt.cm.gray, interpolation='nearest')
ax[1, 2].imshow(img_dither_stucki, cmap=plt.cm.gray, interpolation='nearest')

ax[0, 0].set_title('Input')
ax[0, 1].set_title('Quantization (N=%d)' % N)
ax[0, 2].set_title('Otsu threshold')
ax[1, 0].set_title('Dithering: Image + Noise')
ax[1, 1].set_title('Floyd-Steinberg')
ax[1, 2].set_title('Stucki')

plt.show()

