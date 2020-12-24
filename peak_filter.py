import mean_filter_2D
import cv2
import numpy as np
import matplotlib.pyplot as plt
import show_2d_mean_gray_figure
import show_3D_dark_corner


def remove_peak_of_img(src_img, filter_img):
    removed_peak_img = src_img.copy()
    for row in range(src_img.shape[0]):
        for col in range(src_img.shape[1]):
            if np.abs(src_img[row][col].astype(np.float32)
                      - filter_img[row][col].astype(np.float32)) > 5 :
                removed_peak_img[row][col] = filter_img[row][col]
    # figure = show_2d_mean_gray_figure.Show2DMeanGrayFigure(src_img, removed_peak_img, "test.png")
    # figure.show_figure_for_src_and_dst_img()
    # show_3D_dark_cornet.show_3d_dark_corner(None, removed_peak_img)
    return removed_peak_img


def residual_hist(src_img, filter_img):
    residual_count = []
    for row in range(src_img.shape[0]):
        for col in range(src_img.shape[1]):
            residual_count.append(np.abs(src_img[row][col].astype(np.float32)
                                         - filter_img[row][col].astype(np.float32)))
    plt.hist(residual_count, color="red")
    plt.show()


if __name__ == "__main__":
    image_path = r"D:\hongpu_liyi_code_pycharm\dark_corner\sample_image\LRP904098200800312685_0.png"
    src, filter = mean_filter_2D.filter_2d_mean(130, image_path)
    # residual_hist(src, filter)
    remove_peak_of_img(src, filter)
