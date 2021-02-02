import os
from matplotlib import pyplot


# 创建文件夹
def makedir(folder_name_list):
    for folder_name in folder_name_list:
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
    return folder_name_list


# 绘制曲线统计图
def plot_data(value, fig_name):
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
    pyplot.savefig(fig_name)

    ct = 0
    for d in dataCalcu:
        ct = ct + d

    print("data number: " + str(ct))
