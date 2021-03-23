import os, sys
from matplotlib import pyplot
import matplotlib.image as mpimg


# 创建文件夹
# folder_name_list: 字符串数组
# 返回 folder_name_list
def makedir_from_name_list(folder_name_list):
    for folder_name in folder_name_list:
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
    return folder_name_list


# 绘制曲线统计图
# value: 浮点数数组
# out_fig_name: 输出图片的名字
# 无返回值
def plot_data(value, out_fig_name):
    # 新建一个文件夹
    makedir_from_name_list(["plot"])
    # 绘制折线图时区间采样的个数
    N = 200
    # 绘制图片的大小
    S = 8

    # 先求一下最大值跟最小值
    max_val = value[0]
    min_val = value[0]
    for v in value:
        if v > max_val: max_val = v
        if v < min_val: min_val = v

    print("max value:" + str(max_val))
    print("min value:" + str(min_val))

    delta = (max_val - min_val) / N

    dataX = []
    dataCalcu = []
    for i in range(0, N + 2):
        dataCalcu.append(0)
        dataX.append(min_val + i * delta)

    for v in value:
        p = int((v - min_val) / delta)
        dataCalcu[p] = dataCalcu[p] + 1

    pyplot.figure(figsize=(S, S))
    pyplot.plot(dataX, dataCalcu)
    pyplot.savefig(os.path.join("plot", out_fig_name))
    pyplot.show()
    ct = 0
    for d in dataCalcu:
        ct = ct + d

    print("scale of data: " + str(ct))
    pyplot.close()


# 显示多张图片
# img_name_list: 字符串数组
# out_fig_name: 输出图片的名字
# tag_list：图片的文本注释
# figsize: 显示图片的大小，缺省值为 (12, 9)
# in_line: 如果值为True，则并排显示，否则，显示在一列上
# 无返回值
def img_display(img_name_list, out_fig_name,
                tag_list=None, in_line=True, fig_show=True, fig_size=(12, 9), folder_name="img_display/"):

    if tag_list is None:
        tag_list = []
    pyplot.figure(figsize=fig_size)
    img_array = []
    for img_name in img_name_list:
        img_array.append(mpimg.imread(img_name))

    length = len(img_array)

    for i in range(0, length):
        if in_line:
            pyplot.subplot(1, length, i + 1)
        else:
            pyplot.subplot(length, 1, i+1)
        if i < len(tag_list):
            pyplot.text(20, -30, tag_list[i], fontdict={'size': 20})
        pyplot.imshow(img_array[i])
        pyplot.xticks([])
        pyplot.yticks([])

    pyplot.savefig(folder_name + out_fig_name)
    if fig_show:
        pyplot.show()
    pyplot.close()

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


# 传入一个数字，然后对它进行4位扩充，返回一个字符串
# 例如：
# 4 -> 0004
# 123 -> 0123
# 4444 -> 4444
def four_bit_num(num):
    if num >= 10000 or num < 0:
        exception_info = "Error: the number " + str(num) + "is invalid."
        raise Exception(exception_info)
    elif num >= 1000:
        return str(num)
    elif num >= 100:
        return "0"+str(num)
    elif num >= 10:
        return "00"+str(num)
    else:
        return "000"+str(num)


def get_all_file_path_in_reid_path_format(dir_path):
    picture_path_list = []
    sub_folder_list = ["bounding_box_test", "bounding_box_train", "query"]
    for sub_folder_name in sub_folder_list:
        folder_path = os.path.join(dir_path, sub_folder_name)
        for picture_name in os.listdir(folder_path):
            picture_path_list.append(os.path.join(folder_path, picture_name))
    return picture_path_list


def get_all_file_name_in_reid_path_format(dir_path):
    picture_name_list = []
    sub_folder_list = ["bounding_box_test", "bounding_box_train", "query"]
    for sub_folder_name in sub_folder_list:
        folder_path = os.path.join(dir_path, sub_folder_name)
        for picture_name in os.listdir(folder_path):
            picture_name_list.append(picture_name)
    return picture_name_list


def progress_bar(portion, total, length=50):
    """
    total 总数据大小，portion 已经传送的数据大小
    :param portion: 已经接收的数据量
    :param total: 总数据量
    :param length: 进度条的长度
    :return: 接收数据完成，返回True
    """
    sys.stdout.write('\r')
    temp_str = '[%-' + str(length) + 's] %d/%d %.2f%%'
    count = int(portion * length / total - 1)
    sys.stdout.write((temp_str % (('-' * count + '>'), portion, total, portion / total * 100)))
    sys.stdout.flush()

    if portion >= total:
        sys.stdout.write('\n')
        return True


# 写一个函数，在创建文件的时候，如果该文件存在，将该文件备份一下
# filename: 文件名，不带格式后缀
# fileformat: 文件格式，即后缀，例如，".txt"
# mode: 访问模式，例如，"w"，"r"
def open_file(filename, fileformat, mode):
    if os.path.exists(filename+fileformat):
        return open_file(filename+'_new', fileformat, mode)
    else:
        return open(filename+fileformat, mode)
