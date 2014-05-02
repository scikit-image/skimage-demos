"""Remove radial distortion.

"""

from __future__ import division

import scipy as sp
import scipy.optimize
import scipy.ndimage

from skimage.transform import warp

import matplotlib.pyplot as plt
import numpy as np
import math
import sys

class RadialDistortionInterface:
    """Mouse interaction interface for radial distortion removal.

    """
    def __init__(self, img):
      height, width = img.shape[:2]
      self.figure = plt.imshow(img, extent=(0, width, height, 0))
      plt.title('Removal of radial distortion')
      plt.xlabel('Select sets of three points with left mouse button,\n'
                 'click right button to process.')
      plt.connect('button_press_event', self.button_press)
      plt.connect('motion_notify_event', self.mouse_move)

      self.img = np.atleast_3d(img)
      self.points = []
      self.centre = np.array([(width - 1)/2., (height - 1)/2.])

      self.height = height
      self.width = width

      self.make_cursorline()
      self.figure.axes.set_autoscale_on(False)

      plt.show()
      plt.close()

    def make_cursorline(self):
        self.cursorline, = plt.plot([0],[0],'r:+',
                                    linewidth=2,markersize=15,markeredgecolor='b')

    def button_press(self,event):
        """Register mouse clicks.

        """
        if (event.button == 1 and event.xdata and event.ydata):
            self.points.append((event.xdata,event.ydata))
            print "Coordinate entered: (%f,%f)" % (event.xdata, event.ydata)

            if len(self.points) % 3 == 0:
                plt.gca().lines.append(self.cursorline)
                self.make_cursorline()

        if (event.button != 1 and len(self.points) >= 3):
            print "Removing distortion..."
            plt.gca().lines = []
            plt.draw()
            self.remove_distortion()
            self.points = []

    def mouse_move(self,event):
        """Handle cursor drawing.

        """
        pt_sets, pts_last_set = divmod(len(self.points),3)
        pts = np.zeros((3,2))
        if pts_last_set > 0:
            # Line follows up to 3 clicked points:
            pts[:pts_last_set] = self.points[-pts_last_set:]
            # The last point of the line follows the mouse cursor
        pts[pts_last_set:] = [event.xdata,event.ydata]
        self.cursorline.set_data(pts[:,0], pts[:,1])
        plt.draw()

    def remove_distortion(self, reshape=True):
        def radial_tf(xy, p=None):
            """Radially distort coordinates.

            Given a coordinate (x,y), apply the radial distortion defined by

            L(r) = 1 + p[2]r + p[3]r^2 + p[4]r^3

            where

            r = sqrt((x-p[0])^2 + (y-p[1])^2)

            so that

            x' = L(r)x   and   y' = L(r)y

            Parameters
            ----------
            xy : (M, 2) ndarray
                Input coordinates.
            p : tuple
                Warp parameters:
                - p[0],p[1]        -- Distortion centre
                - p[2], p[3], p[4] -- Radial distortion parameters

            Returns
            -------
            xy : (M, 2) ndarray
                Radially warped coordinates.

            """
            xy = np.array(xy, ndmin=2, copy=False)

            x = xy[:, 0]
            y = xy[:, 1]

            x = x - p[0]
            y = y - p[1]

            r = np.sqrt(x**2 + y**2)
            f = 1 + p[2]*r + p[3]*r**2 + p[4]*r**3

            return np.array([x*f + p[0], y*f + p[1]]).T

        def height_difference(p):
            """Measure deviation of distorted data points from straight line.

            References
            ----------
            http://paulbourke.net/geometry/pointlineplane/
            """
            out = 0
            for sets in 3 * np.arange(len(self.points) // 3):
                pts = np.array(self.points[sets:sets+3])
                xy = radial_tf(pts, p)

                x, y = xy[:, 0], xy[:, 1]
                x, y = xy.T

                # Find point on line (point0 <-> point2) closest to point1 (midpoint)
                u0 = ((x[0] - x[2])**2 + (y[0] - y[2])**2)
                if u0 == 0:
                    return 1

                u = ((x[1] - x[0]) * (x[2] - x[0]) + \
                     (y[1] - y[0]) * (y[2] - y[0])) / u0

                # Intersection point
                ip_x = x[0] + u * (x[2] - x[0])
                ip_y = y[0] + u * (y[2] - y[0])

                # Distance between tip of triangle and and midpoint
                out += (ip_x - x[1])**2 + (ip_y - y[1])**2

            return out

        # Find the distortion parameters for which the data points lie on a
        # straight line
        rc = sp.optimize.fmin(height_difference,
                              [self.centre[0], self.centre[1], 0., 0., 0.])

        # Determine inverse coefficient
        xy = np.array([np.linspace(0, self.width),
                       np.linspace(0, self.height)]).T

        def inv_min(p):
            # Take coordinates from a straight line and transform
            # to the "restored" domain with known rc
            xy_tf = radial_tf(xy, rc)

            # Transform back to the original image domain,
            # this time with the parameters p to be estimated
            xy_tf_back = radial_tf(xy_tf, p)

            return np.sum((xy_tf_back - xy)**2)

        rci = sp.optimize.fmin(inv_min, [rc[0], rc[1], 0., 0., 0.])

        # Perform reverse transformation on coordinates
        oshape = np.array(self.img.shape)
        if reshape:
            top_corner = radial_tf([0., 0.], rc)
            bottom_corner = radial_tf([self.width - 1, self.height-1], rc)
            out_shape = (bottom_corner - top_corner)[::-1]

        restored_image = warp(self.img, radial_tf, {'p': rci})

        plt.figure()
        plt.imshow(restored_image)

        # Plot forward and reverse transforms
        x = np.linspace(self.width / 2, self.width)
        y = np.linspace(self.height / 2, self.height)
        r = np.sqrt((x - self.centre[0])**2 + (y - self.centre[1])**2)

        xy = np.array([x, y]).T

        xyr = radial_tf(xy, rc) - self.centre
        xyri = radial_tf(xy, rci) - self.centre

        rf = np.hypot(*xyr.T)
        rr = np.hypot(*xyri.T)

        a = plt.axes([0.15,.15,.15,.15])
        plt.plot(r, rf, label='Forward mapping')
        plt.plot(r, rr, ':', label='Reverse mapping')
        plt.grid()
        #plt.xlabel('Input radius')
        #plt.ylabel('Transformed radius')
        #plt.legend()
        #plt.setp(a, xticks=[], yticks=[])

        plt.show()

from skimage.io import imread

if len(sys.argv) != 2:
    print "Usage: %s <image-file>" % sys.argv[0]
else:
    img = imread(sys.argv[1])
    rdi = RadialDistortionInterface(img)
