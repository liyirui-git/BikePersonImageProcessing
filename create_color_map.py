import random
import numpy as np
from matplotlib import pyplot, patches

color_name_en_lip = ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes',
                     'Dress', 'Coat', 'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face',
                     'Left-arm', 'Right-arm', 'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']

color_name_en_atr = ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt',
                     'Pants', 'Dress', 'Belt', 'Left-shoe', 'Right-shoe', 'Face', 'Left-leg',
                     'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']

color_name_en_pascal = ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs']


# Self Correction for Human Parsing 的颜色映射规则
def color_rules_of_schp(num_cls):
    n = num_cls
    color_hex = []
    palette = [0] * (n * 3)
    for j in range(0, n):
        color = '0x'
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
        
        for i in range(0, 3):
            if palette[j*3 + i] == 0:
                color = color + '00'
            elif palette[j*3 + i] < 16:
                color = color + '0' + str(hex(palette[j*3 + i])).split('x')[1]
            else:
                color = color + str(hex(palette[j*3 + i])).split('x')[1]

        color_hex.append(color)

    return color_hex


# image-segmentation-keras 的颜色映射规则
def color_rules_of_isk():
    DATA_LOADER_SEED = 0

    random.seed(DATA_LOADER_SEED)
    # 注意，这里可能不是RGB编码，而是GBR编码
    class_colors = [(random.randint(0, 255), random.randint(
            0, 255), random.randint(0, 255)) for _ in range(5000)]

    color_hex = []

    for i in range(0, 20):
        prt = '0x'
        for n in class_colors[i]:
            if n == 0:
                prt = prt + '00'
            elif n < 16:
                prt = prt + '0' + str(hex(n)).split('x')[1]
            else:
                prt = prt + str(hex(n)).split('x')[1]
        nprt = prt[0:2]
        nprt = nprt + prt[6:]
        nprt = nprt + prt[4:6]
        nprt = nprt + prt[2:4]
        # print(str(i) + ': ' + nprt)
        color_hex.append(nprt)

        return color_hex


# 做一个颜色对照表，最大不超过20种颜色
def plot_color_map(color_hex, color_name_ch, color_name_en, out_file_name):
    color_num = len(color_hex)
    pyplot.figure(figsize=(12, 12))
    rows = int(color_num/5) + 1
    count = 0
    for i in range(0, rows):
        for j in range (0, 5):
            if count == color_num: break
            k = i * 5 + j
            xy = np.array([0.05 + 0.2*j, 0.9 - 0.25*i])
            ax = pyplot.subplot()
            rect = patches.Rectangle(xy, 0.1, 0.05, color='#' + color_hex[k][2:])
            ax.add_patch(rect)
            if len(color_name_ch) > 0:
                pyplot.text(xy[0] + 0.02, xy[1] - 0.05, color_name_ch[k], fontdict={'family': 'SimHei', 'size':24})
            if len(color_name_en) > 0:
                pyplot.text(xy[0] + 0.02, xy[1] - 0.1, color_name_en[k], fontdict={'size':18})
            count = count + 1
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.savefig('colormap_' + out_file_name +'.png')
    # pyplot.show()


plot_color_map(color_rules_of_schp(len(color_name_en_lip)), [], color_name_en_lip, "schp_lip")
plot_color_map(color_rules_of_schp(len(color_name_en_atr)), [], color_name_en_atr, "schp_atr")
plot_color_map(color_rules_of_schp(len(color_name_en_pascal)), [], color_name_en_pascal, "schp_pascal")