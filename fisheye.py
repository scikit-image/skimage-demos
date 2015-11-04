from skimage import transform, data, io

import numpy as np
import matplotlib.pyplot as plt

image = io.imread('data/stefan.jpg')
face = image[:185, 15:]


def fisheye(xy):
    center = np.mean(xy, axis=0)
    xc, yc = (xy - center).T

    # Polar coordinates
    r = np.sqrt(xc**2 + yc**2)
    theta = np.arctan2(yc, xc)

    r = 0.8 * np.exp(r**(1/2.1) / 1.8)

    return np.column_stack((
        r * np.cos(theta), r * np.sin(theta)
        )) + center

out = transform.warp(face, fisheye)

f, (ax0, ax1) = plt.subplots(1, 2,
                             subplot_kw=dict(xticks=[], yticks=[]))
ax0.imshow(face)
ax1.imshow(out)

plt.show()
