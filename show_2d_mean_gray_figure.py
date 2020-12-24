import numpy as np
import matplotlib.pyplot as plt


class Show2DMeanGrayFigure(object):
    def __init__(self, src_img, dst_img, ):
        super(Show2DMeanGrayFigure, self)
        self.src_img = src_img
        self.dst_img = dst_img

    def show_figure_for_src_img(self, save_name):
        col_gary_value = np.sum(self.src_img, axis=0) / self.src_img.shape[0]
        col_gary_value = col_gary_value.tolist()
        row_gray_value = np.sum(self.src_img, axis=1) / self.src_img.shape[1]
        row_gray_value = row_gray_value.tolist()
        x_col = np.arange(1, self.src_img.shape[1] + 1).tolist()
        plt.subplot(121)
        plt.plot(x_col, col_gary_value, color='red', linestyle='-', linewidth=1)
        plt.xlabel('number_col')
        plt.ylabel('mean_gray')
        plt.title('mean_gray_added_by_row')
        plt.subplot(122)
        x_row = np.arange(1, self.src_img.shape[0] + 1).tolist()
        plt.plot(x_row, row_gray_value, color='blue', linestyle='-', linewidth=1)
        plt.xlabel('number_row')
        plt.ylabel('mean_gray')
        plt.title('mean_gray_added_by_col')
        plt.savefig(save_name)
        plt.close()

    def show_figure_for_src_and_dst_img(self,  save_name):
        dst_col_gary_value = np.sum(self.dst_img, axis=0) / self.dst_img.shape[0]
        dst_col_gary_value = dst_col_gary_value.tolist()
        dst_row_gray_value = np.sum(self.dst_img, axis=1) / self.dst_img.shape[1]
        dst_row_gray_value = dst_row_gray_value.tolist()
        dst_col = np.arange(1, self.dst_img.shape[1] + 1).tolist()

        src_col_gary_value = np.sum(self.src_img, axis=0) / self.src_img.shape[0]
        src_col_gary_value = src_col_gary_value.tolist()
        src_row_gray_value = np.sum(self.src_img, axis=1) / self.src_img.shape[1]
        src_row_gray_value = src_row_gray_value.tolist()
        src_col = np.arange(1, self.src_img.shape[1] + 1).tolist()

        plt.figure(figsize=(15, 15))
        plt.subplot(121)
        plt.plot(dst_col, dst_col_gary_value, color='red', linestyle='-', linewidth=1)
        plt.plot(src_col, src_col_gary_value, color='blue', linestyle='--', linewidth=1)
        plt.xlabel('number_col')
        plt.ylabel('mean_gray')
        plt.title('mean_gray_added_by_row[src->blue; dst->red]')
        plt.subplot(122)

        dst_row = np.arange(1, self.dst_img.shape[0] + 1).tolist()
        src_row = np.arange(1, self.src_img.shape[0] + 1).tolist()
        plt.plot(dst_row, dst_row_gray_value, color='red', linestyle='-', linewidth=1)
        plt.plot(src_row, src_row_gray_value, color='blue', linestyle='--', linewidth=1)
        plt.xlabel('number_row')
        plt.ylabel('mean_gray')
        plt.title('mean_gray_added_by_col[src->blue; dst->red]')

        plt.savefig( save_name)
        plt.close()
