

#D:\hongpu_liyi_code_pycharm\dark_corner\taizhou_sample\output_taizhou_1202am

import os


def create_save_path(sample_path):
    """
    创建存储路径，并返回源路径和存储路径
    源路径: image_path = ['/..../M1_L3', '/..../M1_L4'....]
    存储路径： save_path = ['/...../result_M1_L3', '/...../result_M1_L4']
    :param sample_path: 所有车间(Mx_Lx,Mx_Lx...)所在的路径
    :return: image_path, save_path
    """

    chejian_name = os.listdir(sample_path)
    save_path = list()
    image_path = list()
    for name in range(len(chejian_name)):
        os.mkdir(sample_path + "\\result_" + chejian_name[name])
        save_path.append(sample_path + "\\result_" + chejian_name[name])
        image_path.append(sample_path + "\\" + chejian_name[name])
    return image_path, save_path


if __name__ == "__main__":
    sample_path = r"D:\hongpu_liyi_code_pycharm\dark_corner\taizhou_sample\output_taizhou_1202am"
    create_save_path(sample_path)