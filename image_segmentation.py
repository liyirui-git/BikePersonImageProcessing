import os
import cv2

class_color_list = [[197, 215, 20],[132, 248, 207],[155, 244, 183],
                    [111, 71, 144],[71, 48, 128],[75, 158, 50],
                    [37, 169, 241],[51, 181, 222],[161, 104, 244],
                    [226, 133, 31],[7, 47, 204],[0, 252, 170],
                    [124, 166, 32],[97, 113, 122],[72, 229, 46],
                    [41, 163, 250],[55, 154, 149],[63, 170, 104],
                    [147, 227, 46],[197, 162, 123]]

color_name_en = ["Background", "Hat", "Hair", "Glove", "Sunglasses", "UpperClothes",
                "Dress","Coat","Socks","Pants","Jumpsuits","Scarf","Skirt","Face",
                "Left-arm","Right-arm","Left-leg","Right-leg","Left-shoe","Right-shoe"]

color_name_ch = ["背景", "帽子", "头发", "手套", "太阳镜", "上衣", "连衣裙", "外套", "袜子", "裤子",
                 "连身裤", "围巾", "短裙", "脸部", "左臂", "右臂", "左腿", "右腿", "左鞋", "右鞋"]

CLASS_NUM = 20

FOLDER_NAME = "./BikePersonDatasetProcess"


# 新增一个参数，background_color, 是一个长度为三的数组，分别表示，三个颜色通道
# 之前使用的是 isk，它对应的 background_color = [197, 215, 20]
# 后来使用的是 schp，它对应的 background_color = [0, 0, 0]
def seg_person_from_mask(folder_name, background_color):

    img_list = os.listdir(folder_name + "/img")

    seg_folder_name = folder_name + "/seg"
    seg_person_folder_name = seg_folder_name + "/segPerson"
    seg_other_folder_name = seg_folder_name + "/segOther"
    if not os.path.exists(seg_folder_name):
        os.mkdir(seg_folder_name)
    if not os.path.exists(seg_person_folder_name):
        os.mkdir(seg_person_folder_name)
    if not os.path.exists(seg_other_folder_name):
        os.mkdir(seg_other_folder_name)

    for img_name in img_list:
        img_name = img_name.split(".")[0]
        img = cv2.imread(folder_name + "/img/" + img_name + ".jpg")
        img2 = img.copy()
        mask = cv2.imread(folder_name + "/mask/" + img_name + ".png")
        shape = img.shape
        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                d1 = mask[i][j][0] - background_color[0]
                d2 = mask[i][j][1] - background_color[1]
                d3 = mask[i][j][2] - background_color[2]
                d = d1*d1 + d2*d2 + d3 * d3
                if d < 50: img[i][j] = [0, 0, 0]
                else: img2[i][j] = [0, 0, 0]

        cv2.imwrite(seg_person_folder_name + "/" + img_name + ".png", img)
        cv2.imwrite(seg_other_folder_name + "/" + img_name + ".png", img2)


def seg_upper_body_from_mask(folder_name):

    # 上半身所不涵盖的部分
    not_in_upper_body = ["背景", "左腿", "右腿", "左鞋",
                         "右鞋", "裤子", "袜子", "短裙"]

    img_list = os.listdir(folder_name + "/img")

    seg_folder_name = folder_name + "/segUpperBody"
    seg_upper_folder_name = seg_folder_name + "/segUpper"
    seg_other_folder_name = seg_folder_name + "/segOther"

    if not os.path.exists(seg_folder_name):
        os.mkdir(seg_folder_name)
    if not os.path.exists(seg_upper_folder_name):
        os.mkdir(seg_upper_folder_name)
    if not os.path.exists(seg_other_folder_name):
        os.mkdir(seg_other_folder_name)

    ct = 0
    for img_name in img_list:
        img = cv2.imread(folder_name + "/img/" + img_name)
        img2 = img.copy()
        mask = cv2.imread(folder_name + "/mask/" + img_name)
        shape = img.shape
        # 将人的上半部分所不涵盖的部分分割出来
        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                flag = False
                for k in range(0, CLASS_NUM):
                    # 只抠可能涉及到的部位
                    if color_name_ch[k] not in not_in_upper_body:
                        d1 = mask[i][j][0] - class_color_list[k][0]
                        d2 = mask[i][j][1] - class_color_list[k][1]
                        d3 = mask[i][j][2] - class_color_list[k][2]
                        d = d1 * d1 + d2 * d2 + d3 * d3
                        if d < 50:
                            flag = flag or True
                        # else:
                        #     img2[i][j] = [0, 0, 0]
                if not flag:
                    img[i][j] = [0, 0, 0]
                else:
                    img2[i][j] = [0, 0, 0]

        cv2.imwrite(seg_upper_folder_name + "/" + img_name, img)
        cv2.imwrite(seg_other_folder_name + "/" + img_name, img2)

        if ct % 10 == 0:
            print(ct)
        ct = ct+1


# seg_upper_body_from_mask(FOLDER_NAME)
