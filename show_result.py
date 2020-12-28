import glob
import time
from typing import List
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import show_2d_mean_gray_figure


def show_result(image_path, model_path, correct_matrix):
    beg = time.time()
    test_src_img_list = glob.glob(image_path)
    test_src_img = []
    test_dst_img = []
    if isinstance(correct_matrix, list):
        for i in range(len(test_src_img_list)):
            test_src_img.append(
                cv2.imread(os.path.join(model_path+"\\test\\", test_src_img_list[i]), 0))
            test_dst_img.append(
                cv2.imread(os.path.join(model_path+"\\test\\", test_src_img_list[i]), 0)
                + correct_matrix[i])
    else:
        for i in range(len(test_src_img_list)):
            test_src_img.append(
                cv2.imread(os.path.join(model_path+"\\test\\", test_src_img_list[i]), 0))
            test_dst_img.append(
                cv2.imread(os.path.join(model_path+"\\test\\", test_src_img_list[i]), 0)
                + correct_matrix)

    end = time.time()
    print('处理测试所用时间(min):', (end - beg)/60)

    plt.figure(figsize=(10, 10))

    for i in range(len(test_src_img_list)):

        cv2.imwrite(model_path+"\\result\\"+"model1"+str(i)
            + "src_img.png", test_src_img[i])
        cv2.imwrite(model_path+"\\result\\"+"model1"+str(i)
            + "dst_img.png",test_dst_img[i])

        show_2d_figure = show_2d_mean_gray_figure.Show2DMeanGrayFigure(test_src_img[i], test_dst_img[i])
        show_2d_figure.show_figure_for_src_and_dst_img(model_path+"\\result\\"+"model1"+str(i)
             + ".png")


if __name__ == "__main__":
    image_path = r"D:\hongpu_liyi_code_pycharm\dark_corner\sample_for_view_0\model_1\test\*_0.png"
    model_path = r"D:\hongpu_liyi_code_pycharm\dark_corner\sample_for_view_0\model_1"
    correct_matrix = np.zeros((1000, 1500))
    show_result(image_path, model_path, correct_matrix, )
