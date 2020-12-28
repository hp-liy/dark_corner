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
    # 使用自定义滤波函数对原始图片进行滤波——————》思想hample滤波方法
    src_img, filtered_img = mean_filter_2D.filter_2d_mean(filter_size, image_path)
    # 将滤波后的图片和原图比较，去除异常值
    removed_peak_img = peak_filter.remove_peak_of_img(src_img, filtered_img)
    # 二维曲线拟合算法如下：
    # fit_figure_class = PLS_fit_ellipse.FitCurve(removed_peak_img)
    # curve_img = fit_figure_class.fit()

    # 三维椭球拟合算法如下：
    fit_figure_class = PLS_fit_ellipse.FitEllipse(removed_peak_img)
    [curve_img, m_x, m_y] = fit_figure_class.fit()
    # ###############显示src和remove_peak和curve的平均灰度比较###########################
    # dst_col_gary_value = np.sum(removed_peak_img, axis=0) / removed_peak_img.shape[0]
    # dst_col_gary_value = dst_col_gary_value.tolist()
    # dst_row_gray_value = np.sum(removed_peak_img, axis=1) / removed_peak_img.shape[1]
    # dst_row_gray_value = dst_row_gray_value.tolist()
    # dst_col = np.arange(1, removed_peak_img.shape[1] + 1).tolist()
    #
    # src_col_gary_value = np.sum(src_img, axis=0) / src_img.shape[0]
    # src_col_gary_value = src_col_gary_value.tolist()
    # src_row_gray_value = np.sum(src_img, axis=1) / src_img.shape[1]
    # src_row_gray_value = src_row_gray_value.tolist()
    # src_col = np.arange(1, src_img.shape[1] + 1).tolist()
    #
    # cur_col_gary_value = np.sum(curve_img, axis=0) / curve_img.shape[0]
    # cur_col_gary_value = cur_col_gary_value.tolist()
    # cur_row_gray_value = np.sum(curve_img, axis=1) / curve_img.shape[1]
    # cur_row_gray_value = cur_row_gray_value.tolist()
    # cur_col = np.arange(1, curve_img.shape[1] + 1).tolist()
    #
    # plt.figure(figsize=(15, 15))
    # plt.subplot(121)
    # plt.plot(dst_col, dst_col_gary_value, color='red', linestyle='-', linewidth=1)
    # plt.plot(src_col, src_col_gary_value, color='blue', linestyle='--', linewidth=1)
    # plt.plot(cur_col, cur_col_gary_value, color="black", linestyle='-', linewidth =1)
    # plt.xlabel('number_col')
    # plt.ylabel('mean_gray')
    # plt.title('mean_gray_added_by_row[src->blue; dst->red, cur->black]')
    # plt.subplot(122)
    #
    # dst_row = np.arange(1, removed_peak_img.shape[0] + 1).tolist()
    # src_row = np.arange(1, src_img.shape[0] + 1).tolist()
    # cur_row = np.arange(1, curve_img.shape[0] + 1).tolist()
    # plt.plot(dst_row, dst_row_gray_value, color='red', linestyle='-', linewidth=1)
    # plt.plot(src_row, src_row_gray_value, color='blue', linestyle='--', linewidth=1)
    # plt.plot(cur_row, cur_row_gray_value, color="black", linestyle='-', linewidth=1)
    # plt.xlabel('number_row')
    # plt.ylabel('mean_gray')
    # plt.title('mean_gray_added_by_col[src->blue; dst->red; cur->black]')
    #
    # plt.savefig("doctor.png")
    # plt.close()

    #
    # 求出拟合矩阵
    correction_matrix = np.ones((src_img.shape[0], src_img.shape[1])) * np.max(curve_img) - curve_img
    end = time.time()
    print("生成模型所用时间(min):", (end - beg)/60)

    # 显示原图去黑角的效果。
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
    model_path = r"D:\hongpu_liyi_code_pycharm\dark_corner\sample_for_view_0\model_tuiqiu_1"
    correct_mat = main(image_path, model_path, 130)

