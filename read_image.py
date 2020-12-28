import glob
import cv2


def read_image(img_path):
    """
    将文件夹内的所有图片按照视角进行读取。最后存成字典。
    字典形式为:
    dict{
    "view_0":list(img0, img1, img2...)
    "view_1":list(img0, img1, img2...)
    ....
    "view_11":list(img0, img1, img2...)
    }
    :param img_path: 图片所在路径
    :return:存储的图像的字典， 按照字典存取顺序的图像名字。
    """
    sample_view = dict()
    sample_sort_name = list()
    for view in range(0, 12):
        view_img_path = glob.glob(img_path+"\\*_"+str(view)+".png")
        sample_view["view_" + str(view)] = list()
        for view_img_path_num in range(len(view_img_path)):
            sample_view["view_"+str(view)].append(cv2.imread(view_img_path[view_img_path_num], 0))
    for sample_name in glob.glob(img_path+"\\*_0.png"):
        sample_sort_name.append(sample_name[sample_name.find("LRP"):sample_name.find("_0.png")])

    return sample_view, sample_sort_name


if __name__ == "__main__":
    img_path = r"D:\hongpu_liyi_code_pycharm\dark_corner\sample_for_all\M1_L3"
    a, b = read_image(img_path)
    print(a, b)