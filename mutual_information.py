from skimage.viewer import ImageViewer
from skimage import data, img_as_float

from skimage.viewer.plugins.plotplugin import PlotPlugin
from skimage.viewer.widgets import Slider
from skimage.exposure import rescale_intensity
from skimage.transform import rotate

import numpy as np
import matplotlib.pyplot as plt


class RotatedImageViewer(ImageViewer):
    def __init__(self, image, **kwargs):
        super(RotatedImageViewer, self).__init__(image, **kwargs)

        slider_kwds = dict(value=0, low=0, high=5, update_on='release',
                           callback=self.update_angle, value_type='float')

        self.slider = Slider('angle', **slider_kwds)
        self.layout.addWidget(self.slider)
        self.origin_image = image

    def update_angle(self, name, angle):
        self.image = rotate(self.original_image, angle)
        self.histogram.draw(angle=angle)


class Histogram(PlotPlugin):
    name = 'Histogram'

    def __init__(self, original_viewer, **kwargs):
        super(Histogram, self).__init__(height=400, **kwargs)
        self.bins = np.linspace(0, 1, 100)
        self.mpl_image = None
        self.original_viewer = original_viewer

    def attach(self, image_viewer):
        super(Histogram, self).attach(image_viewer)
        self.rotated_viewer = image_viewer

        self.ax.set_title('Histogram')
        self.ax.set_xlabel('Value in image 1')
        self.ax.set_ylabel('Value in image 2')

        self.draw(angle=0)

    def draw(self, angle=0):
        image1 = self.original_viewer.image
        image2 = self.rotated_viewer.image

        hist, x_edges, y_edges = np.histogram2d(image1.flatten(),
                                                image2.flatten(),
                                                self.bins, normed=True)

        hist = np.log(1 + hist)
        hist = rescale_intensity(hist, in_range=(0, 3))

        if self.mpl_image is None:
            self.mpl_image = self.ax.imshow(hist, extent=[0, 1, 0, 1],
                                            cmap=plt.cm.gray)
        else:
            self.mpl_image.set_data(hist)
            self.ax.figure.canvas.draw()

        return hist


image = img_as_float(data.camera())
viewer = ImageViewer(image)
rotated_viewer = RotatedImageViewer(image)

histogram = Histogram(viewer)
rotated_viewer += histogram
rotated_viewer.histogram = histogram

super(ImageViewer, viewer).show()
rotated_viewer.show()
