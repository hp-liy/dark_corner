import numpy as np
from scipy.optimize import leastsq
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def _ellipse_func_error(para, x, y):
    # a-j 都是参数。参数方程为 a*x^2+b*y^2+c*z^2+d*x*y
    # +e*y*x+f*z*x+g*x+h*y+i*z+j=0
    a, b, c, d, e, f, g, h, i, j = para

    # return a * np.power(x, 2) + b * np.power(y, 2) + c * np.power(z, 2) + \
    #        d * x * y + e * y * z + f * x * z + g * x + h * y + i * z + j
    return np.sqrt(( a * np.power(x, 2) + b * np.power(y, 2) + d * x * y + g * x + h * y + j)/c
                   + np.power((i + f * x + e * y)/(2 * c), 2)) - (i + f * x + e * y) / (2 * c)


def _ellipse_error(para, x, y, z):
    print("cal_error")
    return _ellipse_func_error(para, x, y) - z


class FitEllipse(object):
    def __init__(self, image):
        super(FitEllipse, self).__init__()
        self.img = image
        x = np.arange(0, self.img.shape[1])
        y = np.arange(0, self.img.shape[0])
        self.x_grid, self.y_grid = np.meshgrid(x, y)

        self.x = self.x_grid.flatten()
        self.y = self.y_grid.flatten()
        self.z = self.img.flatten()
        self.para = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

    def fit(self):
        func_paras = leastsq(_ellipse_error, self.para,
                             args=(self.x, self.y, self.z))

        paras = func_paras[0]
        print(paras)
        ellipse_img = np.sqrt((- paras[0] * np.power(self.x_grid, 2) - paras[1] * np.power(self.y_grid, 2) - paras[3]
                             * self.x_grid * self.y_grid - paras[6] * self.x_grid - paras[7] * self.y_grid - paras[9])/paras[2] +
                            np.power((paras[8] + paras[5] * self.x_grid + paras[4] * self.y_grid)/(2 * paras[2]), 2)) - \
                    (paras[8] + paras[5] * self.x_grid + paras[4] * self.y_grid) / (2 * paras[2])

        return ellipse_img


class FitCurve(object):
    def __init__(self, image):
        super(FitCurve, self).__init__()
        self.img = image
        self.x = np.arange(0, self.img.shape[1])
        self.y = np.sum(self.img, axis=0) / self.img.shape[0]

    def fit(self):

        polynomial_params = np.polyfit(self.x, self.y, 2)
        polynomial = np.poly1d(polynomial_params)
        Curve = polynomial(self.x)

        return np.expand_dims(Curve, 0).repeat(self.img.shape[0], axis=0)


if __name__ == "__main__":
    image_path = r"D:\hongpu_liyi_code_pycharm\dark_corner\sample_image\LRP904098200800312685_0.png"
    image = cv2.imread(image_path, 0)
    fit_ellipse = FitEllipse(image)
    ellipse_img = fit_ellipse.fit()

