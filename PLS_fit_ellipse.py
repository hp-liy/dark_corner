import numpy as np
from scipy.optimize import leastsq
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def _ellipse_func_error(para, x, y, z):
    x0, y0, z0, a0, b0, c0 = para
    return ((x.flatten() - x0) / a0) ** 2 + ((y.flatten() - y0) / b0) ** 2 + ((z.flatten() - z0) / c0) ** 2 - 1


class FitEllipse(object):
    """
    拟合椭球面
    """
    def __init__(self, image):
        super(FitEllipse, self).__init__()
        self.img = image
        x = np.arange(0, self.img.shape[1])
        y = np.arange(0, self.img.shape[0])
        self.x_grid, self.y_grid = np.meshgrid(x, y)
        # 坐标中心化，目的把椭球的中心放置到坐标中心。
        central_m = [np.where(self.img == np.max(self.img))[1][0], np.where(self.img == np.max(self.img))[0][0]]
        self.x = self.x_grid - central_m[1]
        self.y = self.y_grid - central_m[0]
        # print('0 is correct:', self.x[central_m[1], central_m[0]])  # 检查平移是否正确  为0正确
        # print('0 is correct:', self.y[central_m[1], central_m[0]])

    def fit(self):
        xm = np.mean(self.x.flatten())
        ym = np.mean(self.y.flatten())
        zm = np.mean(self.img.flatten())
        # 最小二乘拟合椭球。leastsq（损失函数，参数，除参数外的其他输入）
        tparas = leastsq(_ellipse_func_error, np.array([xm, ym, zm, 10, 10, 10]),
                         args=(self.x, self.y, self.img))
        paras = tparas[0]

        ellipse_img = np.sqrt((1-((self.x - paras[0])/paras[3]) ** 2 - ((self.y - paras[1])/paras[4]) ** 2 ))* paras[5]+paras[2]

        return [ellipse_img, self.x, self.y]

    def show_figure(self):
        [ellipse, m_x, m_y] = self.fit()

        fig = plt.figure()
        ax = Axes3D(fig)  # 在窗口添加一个3D坐标
        ax.plot_surface(m_x, m_y, self.img, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
        ax.contourf(m_x, m_y, self.img, zdir='Z', offset=-2, cmap='rainbow')
        plt.show()

        fig2 = plt.figure()
        ax2 = Axes3D(fig2)
        ax2.plot_surface(m_x, m_y, ellipse, cmap=plt.get_cmap('winter'))
        ax2.contourf(m_x, m_y, ellipse, zdir='Z', offset=-2, cmap='rainbow')
        plt.show()


class FitCurve(object):
    """
    拟合二维曲线
    """
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
    # image_path = r"D:\hongpu_liyi_code_pycharm\dark_corner\sample_image\LRP904098200800312685_0.png"
    image_path = r"D:\hongpu_liyi_code_pycharm\dark_corner\1.png"
    image = cv2.imread(image_path, 0)
    fit_ellipse = FitEllipse(image)
    [ellipse, m_x, m_y] = fit_ellipse.fit()

