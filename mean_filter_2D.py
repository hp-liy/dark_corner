import cv2
import numpy as np
from show_3D_dark_corner import show_3d_dark_corner
from show_2d_mean_gray_figure import Show2DMeanGrayFigure


def filter_2d_mean(kernel_size, img_path_or_img):
    """
    对原图进行均值滤波，采用的是cv2中自定义滤波方法
    :param kernel_size: 滤波器的size
    :param img_path_or_img: 两种选择：图片路径或者图片
    :return: 滤波后的图片
    """
    if isinstance(img_path_or_img, str):
        src = cv2.imread(img_path_or_img, 0)
    else:
        src = img_path_or_img
    if src is None:
        return -1

    kernel = np.ones((kernel_size, kernel_size))/(kernel_size*kernel_size)
    dst = cv2.filter2D(src, -1, kernel)
    # show_gray_figure = Show2DMeanGrayFigure(src, dst, "gray_figure_compared_with_src_dst.png")
    # show_gray_figure.show_figure_for_src_and_dst_img()

    # show_3d_dark_corner(None, dst)
    # cv2.imshow('img', dst)
    # cv2.waitKey(0)
    return src, dst


if __name__ == "__main__":
    image_path = r"D:\hongpu_liyi_code_pycharm\dark_corner\sample_image\LRP904098200800312685_0.png"
    filter_2d_mean(130, image_path)