import mean_filter_2D
import peak_filter
import numpy as np
import show_3D_dark_corner
import show_2d_mean_gray_figure
import PLS_fit_ellipse
import matplotlib.pyplot as plt
import cv2
import time
from show_result import show_result
import os
import glob


def main(image_path, model_path, filter_size):
    beg = time.time()
    src_img, filtered_img = mean_filter_2D.filter_2d_mean(filter_size, image_path)
    removed_peak_img = peak_filter.remove_peak_of_img(src_img, filtered_img)

    fit_figure_class = PLS_fit_ellipse.FitCurve(removed_peak_img)
    curve_img = fit_figure_class.fit()
    correction_matrix = np.ones((src_img.shape[0], src_img.shape[1])) * np.max(curve_img) - curve_img
    # removed_dark_corner_img = correction_matrix + src_img
    # show_3D_dark_corner.show_3d_dark_corner(None, removed_dark_corner_img)
    end = time.time()
    print("生成模型所用时间(min):", (end - beg)/60)
    show_result(model_path + "\\test\\*_0.png", model_path, correction_matrix)
    return correction_matrix


# def main(image_path, model_path, filter_size):
#     beg = time.time()
#     test_src_img_list = glob.glob(image_path)
#     correction_matrix = []
#     for i in range(len(test_src_img_list)):
#
#         img = cv2.imread(os.path.join(model_path + "\\test\\",
#                                       test_src_img_list[i]), 0)
#
#         src_img, filtered_img = mean_filter_2D.filter_2d_mean(filter_size, img)
#         removed_peak_img = peak_filter.remove_peak_of_img(src_img, filtered_img)
#
#         fit_figure_class = PLS_fit_ellipse.FitCurve(removed_peak_img)
#         curve_img = fit_figure_class.fit()
#         correction_matrix.append(np.ones((src_img.shape[0], src_img.shape[1])) * np.max(curve_img) - curve_img)
#     end = time.time()
#     print("生成模型所用时间(min):", (end - beg)/60)
#     show_result(model_path + "\\test\\*_0.png", model_path, correction_matrix)
#     return correction_matrix


if __name__ == '__main__':
    image_path = r"D:\hongpu_liyi_code_pycharm\dark_corner\sample_image\LRP904098200800312685_0.png"
    # image_path = r"D:\hongpu_liyi_code_pycharm\dark_corner\sample_for_view_0\model_2\test\*_0.png"
    model_path = r"D:\hongpu_liyi_code_pycharm\dark_corner\sample_for_view_0\model_1"
    correct_mat = main(image_path, model_path, 130)

