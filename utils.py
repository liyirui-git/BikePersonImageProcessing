import os
from matplotlib import pyplot
import matplotlib.image as mpimg


# 创建文件夹
# folder_name_list: 字符串数组
# 返回 folder_name_list
def makedir(folder_name_list):
    for folder_name in folder_name_list:
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
    return folder_name_list


# 绘制曲线统计图
# value: 浮点数数组
# out_fig_name: 输出图片的名字
# 无返回值
def plot_data(value, out_fig_name):
    # 绘制折线图时区间采样的个数
    N = 200
    # 绘制图片的大小
    S = 5.12

    # 先求一下最大值跟最小值
    maxVal = value[0]
    minVal = value[0]
    for v in value:
        if v > maxVal: maxVal = v
        if v < minVal: minVal = v

    print("max value:" + str(maxVal))
    print("min value:" + str(minVal))

    delta = (maxVal - minVal) / N

    dataX = []
    dataCalcu = []
    for i in range(0, N + 2):
        dataCalcu.append(0)
        dataX.append(minVal + i * delta)

    for v in value:
        p = (int)((v - minVal) / delta)
        dataCalcu[p] = dataCalcu[p] + 1

    pyplot.figure(figsize=(S, S))
    pyplot.plot(dataX, dataCalcu)
    pyplot.savefig(out_fig_name)

    ct = 0
    for d in dataCalcu:
        ct = ct + d

    print("data number: " + str(ct))


# 并排显示多组图片
# img_name_list: 字符串数组
# out_fig_name: 输出图片的名字
# figsize: 显示图片的大小，缺省值为 (12, 9)
# 无返回值
def img_display(img_name_list, out_fig_name, tag_list=None, fig_show=True, fig_size=(12, 9)):
    folder_name = "img_display/"
    if tag_list is None:
        tag_list = []
    pyplot.figure(figsize=fig_size)
    img_array = []
    for img_name in img_name_list:
        img_array.append(mpimg.imread(img_name))

    length = len(img_array)

    for i in range(0, length):
        pyplot.subplot(1, length, i + 1)
        if i < len(tag_list):
            pyplot.text(20, -30, tag_list[i], fontdict={'size': 20})
        pyplot.imshow(img_array[i])
        pyplot.xticks([])
        pyplot.yticks([])

    pyplot.savefig(folder_name + out_fig_name)
    if fig_show:
        pyplot.show()


# 将以数字序号命名的，反映射到其原本的名字
# img_name: 字符串
# 返回值 字符串 是原本的名字
def get_origin_name(img_name):
    # 文件名的反映射
    name_list = []
    name_file = open("pictureNameList_Full.txt", "r")
    for line in name_file.readlines():
        name_list.append(line.split("\n")[0])
    img_num = int(img_name.split(".")[0]) - 1
    return name_list[img_num]