import mean_filter_2D
import peak_filter
import numpy as np
import PLS_fit_ellipse
import cv2
import time
import os
from read_image import read_image
import show_2d_mean_gray_figure


def train(view_img_dict, view_img_name_sort_list, model_save_path, filter_size):

    with open(model_save_path + "\\correct_matrix_list\\log.txt", mode='w') as fp:
            fp.write(view_img_name_sort_list[0])

    correction_matrix = list()
    for view in range(0, 12):
        # 使用自定义滤波函数对原始图片进行滤波——————》思想hample滤波方法
        src_img, filtered_img = mean_filter_2D.filter_2d_mean(filter_size, view_img_dict["view_"+str(view)][0])

        # 将滤波后的图片和原图比较，去除异常值
        removed_peak_img = peak_filter.remove_peak_of_img(src_img, filtered_img)
        # 二维曲线拟合算法如下：
        # fit_figure_class = PLS_fit_ellipse.FitCurve(removed_peak_img)
        # curve_img = fit_figure_class.fit()

        # 三维椭球拟合算法如下：
        fit_figure_class = PLS_fit_ellipse.FitEllipse(removed_peak_img)
        [curve_img, m_x, m_y] = fit_figure_class.fit()

        # 求出拟合矩阵
        correction_matrix.append(np.ones((src_img.shape[0], src_img.shape[1])) * np.max(curve_img) - curve_img)
        np.save(model_save_path + "\\correct_matrix_list"
                "\\view_" + str(view) + ".txt", correction_matrix[view])
        print("完成第"+str(view + 1)+"个模型拟合")
    return correction_matrix


def test(view_img_dict, view_img_name_sort_list, model_save_path, correct_matrix):
    for view in range(len(correct_matrix)):
        for sample_num in range(len(view_img_name_sort_list)):
            img = view_img_dict["view_"+str(view)][sample_num] + correct_matrix[view]
            cv2.imwrite(model_save_path + "\\result\\result_" + view_img_name_sort_list[sample_num]
                        + "_" + str(view) + ".png", img)
            show_2d_mean_gray_figure.show_figure_for_src_and_dst_img(view_img_dict["view_"+str(view)][sample_num], img
                                                                     , model_path+"\\gray_mean_compare_img\\" + view_img_name_sort_list[sample_num]
                        + "_" + str(view) + "gray_mean_compare_img.png")


def main(image_path, model_path, filter_size):
    """
    主函数
    :param image_path: 文件路径。应包含所有视角的图片。eg(L0.png;L1.png;...;L11.png)
    :param model_path: 结果存储的位置
    :param filter_size: 初始滤波器的大小（默认130）
    :return:
    """
    beg = time.time()
    print("#"*20 + "开始训练" + "#"*20)
    view_img_dict, sort_img_name_list = read_image(image_path)
    correction_matrix = []
    if os.path.exists(model_path + "\\correct_matrix_list\\log.txt"):
        print("模型已经训练过了")
    else:
        correction_matrix = train(view_img_dict, sort_img_name_list, model_path, filter_size)
    end = time.time()
    print("生成模型所用时间(min):", (end - beg)/60)
    beg = time.time()
    print("#" * 20 + "开始测试" + "#" * 20)
    if not correction_matrix:
        return -1
    test(view_img_dict, sort_img_name_list, model_path, correction_matrix)
    end = time.time()
    print("测试所用时间(min):", (end - beg) / 60)


if __name__ == '__main__':
    image_path = r"D:\hongpu_liyi_code_pycharm\dark_corner\sample_for_all\M1_L3"
    model_path = r"D:\hongpu_liyi_code_pycharm\dark_corner\sample_for_all"
    correct_mat = main(image_path, model_path, 130)

