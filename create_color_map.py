import random
import numpy as np
from matplotlib import pyplot, patches


DATA_LOADER_SEED = 0

random.seed(DATA_LOADER_SEED)
# 注意，这里可能不是RGB编码，而是GBR编码
class_colors = [(random.randint(0, 255), random.randint(
    0, 255), random.randint(0, 255)) for _ in range(5000)]

color_name_en = ["Background", "Hat", "Hair", "Glove", "Sunglasses", "UpperClothes",
                 "Dress","Coat","Socks","Pants","Jumpsuits","Scarf","Skirt","Face",
                 "Left-arm","Right-arm","Left-leg","Right-leg","Left-shoe","Right-shoe"]

color_name_ch = ["背景", "帽子", "头发", "手套", "太阳镜", "上衣", "连衣裙", "外套", "袜子", "裤子",
                 "连身裤","围巾","短裙","脸部","左臂","右臂","左腿","右腿","左鞋","右鞋"]

color_hex = []

for i in range(0, 20):
    prt = "0x"
    for n in class_colors[i]:
        if n == 0:
            prt = prt + "00"
        elif n < 16:
            prt = prt + '0' + str(hex(n)).split("x")[1]
        else:
            prt = prt + str(hex(n)).split("x")[1]
    nprt = prt[0:2]
    nprt = nprt + prt[6:]
    nprt = nprt + prt[4:6]
    nprt = nprt + prt[2:4]
    # print(str(i) + ": " + nprt)
    color_hex.append(nprt)


# 做一个颜色对照表
def plot_color_map():
    pyplot.figure(figsize=(12, 12))
    for i in range(0, 4):
        for j in range (0, 5):
            k = i * 5 + j
            xy = np.array([0.05 + 0.2*j, 0.9 - 0.25*i])
            ax = pyplot.subplot()
            rect = patches.Rectangle(xy, 0.1, 0.05, color='#' + color_hex[k][2:])
            ax.add_patch(rect)
            pyplot.text(xy[0] + 0.02, xy[1] - 0.05, color_name_ch[k], fontdict={'family': 'SimHei', 'size':24})
            pyplot.text(xy[0] + 0.02, xy[1] - 0.1, color_name_en[k], fontdict={'size':18})
            #pyplot.text()

    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.savefig("colormap.png")
    pyplot.show()
