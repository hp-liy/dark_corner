import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import numpy as np

"""
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'), 
                edgecolor='black')
    1. rstride和cstride表示跨度，r:row，c:column。rstride表示行跨，即两条线之间跨了多少行，
                                cstride表示两条线之间垮了多少列。

    2. cmap就是图像的颜色属性。

    3. edgecolor对应的就是图中一条条线的颜色。

ax.contourf(X,Y,Z,zdir='Z',offset=-2,cmap='rainbow')
    1. 这个函数是画等高线所用到的一个函数，在这个例子中相当于用这个函数将3d图像映射到xoy轴上。

    2. zdir='z'，offset=-2：设置一个z=-2的高度，在z轴的方向将这个3d图像压到一个平面上。
    
meshgrid(x,y)用于生成绘制3D图形所需的网格数据。在计算机中进行绘图操作时，往往需要一些采样点，
    然后根据这些采样点绘制出整个图形。在绘制3D图形时需要有x,y,z三组数据，x,y这两组数据可以看
    作是在XOY平面内对坐标进行采样而得到的坐标对(x,y)。

"""


def show_3d_dark_corner(img_path, image=None):
    if img_path is None:
        dark_corner_img = image
    else:
        dark_corner_img = cv2.imread(img_path, 0)
    fig = plt.figure()
    ax = Axes3D(fig)  # 在窗口添加一个3D坐标
    X = np.arange(0, dark_corner_img.shape[1])  # 坐标XYZ
    Y = np.arange(0, dark_corner_img.shape[0])
    X, Y = np.meshgrid(X, Y)  # 生成绘制3D图形所需要的网格数据
    # 将图画在3d坐标上
    ax.plot_surface(X, Y, dark_corner_img, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    ax.contourf(X, Y, dark_corner_img, zdir='Z', offset=-2, cmap='rainbow')
    plt.show()


if __name__ == "__main__":
    image_path = r"D:\hongpu_liyi_code_pycharm\dark_corner\sample_image\LRP904098200800312685_0.png"
    show_3d_dark_corner(image_path)
