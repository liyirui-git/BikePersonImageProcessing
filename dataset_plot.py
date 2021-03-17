# 本程序统计所有的人骑车图像的长宽比
import os
import cv2
import utils
import time
import imagesize

# 之前在一部分数据集上做操作，这里的值是 "./selectPicture"
ROOT_PATH = "./BikePersonDatasetProcess"


# 统计分割后的图片，人体所占的比例，并绘制折线图
def plot_segment_area_ratio():
    picture_folder_name = "./path_to_predictions_epoch10"
    picture_list = os.listdir(picture_folder_name)

    value = []

    count = 0
    for picture in picture_list:
        # 计算分割结果的像素在所有像素中所占的比例
        img = cv2.imread(picture_folder_name + "/" + picture)
        shape = img.shape
        area = shape[0] * shape[1]
        # 图片的通道与数组下标的对应关系
        # img[x][y][0]: B, img[x][y][1]: G, img[x][y][2]: R
        ct = 0
        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                if img[i][j][0] == 198 and img[i][j][1] == 215 and img[i][j][2] == 20:
                    ct = ct + 1
        v = 1.0-ct/area
        value.append(v)
        count = count + 1
        if count % 100 == 0 : print(count)

    utils.plot_data(value, "statistic-segment-area-ratio.png")


# 统计人骑车图像的长宽比，并绘制折线图
def plot_length_width_ratio ():
    picture_folder_name = "./BikePersonDatasetNew"
    picture_list = os.listdir(picture_folder_name)

    value = []

    for picture in picture_list:
        # 如何得到picture的尺寸大小？
        img = cv2.imread(picture_folder_name + "/" + picture)
        shape = img.shape
        value.append(shape[0] / shape[1])

    utils.plot_data(value, "statistic-length-width-ratio.png")


# 统计左右半边，绘制分割结果所占的面积的比的折线图，并且将图片分类
# 这里写的不好，名字是一个绘制，里面还有一些其他的功能
def plot_left_right_area_ratio():
    # picture_folder_name = rootPath + "/side/img"
    mask_folder_name = ROOT_PATH + "/side/mask/"
    mask_img_list = os.listdir(mask_folder_name)

    folder_list = utils.makedir_from_name_list([ROOT_PATH + "/side/left/", ROOT_PATH + "/side/right/"])

    left_folder_name = folder_list[0]
    right_folder_name = folder_list[1]

    value = []

    count = 0
    for mask_img in mask_img_list:
        mask = cv2.imread(mask_folder_name + mask_img)
        shape = mask.shape
        # shape[0] 是长，shape[1] 是宽
        left_count = 0
        right_count = 1
        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                d1 = mask[i][j][0]
                d2 = mask[i][j][1]
                d3 = mask[i][j][2]
                if d1 > 10 or d2 > 10 or d3 > 10:
                    if j > shape[1] / 2:
                        right_count = right_count + 1
                    else:
                        left_count = left_count + 1

        left_right_ratio = left_count / right_count
        if left_right_ratio > 1:
            cv2.imwrite(left_folder_name + mask_img, mask)
        else:
            cv2.imwrite(right_folder_name + mask_img, mask)

        value.append(left_right_ratio)

        count = count + 1
        if count % 50 == 0: print(count)

    # utils.plot_data(value, "statistic-left-right-ratio.png")


# 绘制面部所占面积比例的折线图
def plot_face_area_ratio():
    picture_folder_name = ROOT_PATH + "/back_front/img/"
    mask_folder_name = ROOT_PATH + "/back_front/mask/"
    mask_img_list = os.listdir(mask_folder_name)

    folder_list = utils.makedir_from_name_list([ROOT_PATH + "/back_front/front/", ROOT_PATH + "/back_front/front_mask/",
                                                ROOT_PATH + "/back_front/back/", ROOT_PATH + "/back_front/back_mask/",
                                                ROOT_PATH + "/back_front/other/", ROOT_PATH + "/back_front/other_mask/"])

    front_folder_name = folder_list[0]
    front_mask_folder_name = folder_list[1]
    back_folder_name = folder_list[2]
    back_mask_folder_name = folder_list[3]
    other_folder_name = folder_list[4]
    other_mask_folder_name = folder_list[5]

    value = []

    count = 0
    for mask_img in mask_img_list:
        img = cv2.imread(picture_folder_name + mask_img)
        mask = cv2.imread(mask_folder_name + mask_img)
        shape = mask.shape
        face_count = 0
        body_count = 1
        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                # 图片的通道与数组下标的对应关系
                # img[x][y][0]: B, img[x][y][1]: G, img[x][y][2]: R
                d1 = mask[i][j][0]
                d2 = mask[i][j][1]
                d3 = mask[i][j][2]
                if d1 > 10 or d2 > 10 or d3 > 10:
                    body_count = body_count + 1
                # face BGR : 128 0 192
                if abs(d1-128) <= 5 and abs(d2-0) <= 5 and abs(d3-192) <= 5:
                    face_count = face_count + 1

        face_area_ratio = face_count / body_count

        if face_area_ratio >= 0.03:
            cv2.imwrite(front_folder_name + mask_img, img)
            cv2.imwrite(front_mask_folder_name + mask_img, mask)
        elif face_area_ratio <= 0.01:
            cv2.imwrite(back_folder_name + mask_img, img)
            cv2.imwrite(back_mask_folder_name + mask_img, mask)
        else:
            cv2.imwrite(other_folder_name + mask_img, img)
            cv2.imwrite(other_mask_folder_name + mask_img, mask)

        value.append(face_area_ratio)
        count = count + 1
        if count % 50 == 0: print(count)

    utils.plot_data(value, "statistic-face-area-ratio.png")


# 统计重新使用LIP分割，然后按照fast-reid的格式组织起来的图片，人体所占的比例，并绘制折线图
# 为了避免重复计算，将每个图片对应的值存在一个文件中
def plot_segment_area_ratio_and_restore(mask_pic_dir_path="BikePersonDatasetNew-mask-700", plot=True):

    log_file = open("txt/segment_area_ratio_log_" + str(int(time.time())) + ".txt", "w")
    picture_path_list = utils.get_all_file_path_in_reid_path_format(mask_pic_dir_path)

    value = []

    count = 0
    for picture_path in picture_path_list:
        # 计算分割结果的像素在所有像素中所占的比例
        log_file.write(picture_path+" ")
        img = cv2.imread(picture_path)
        shape = img.shape
        area = shape[0] * shape[1]
        # 图片的通道与数组下标的对应关系
        # img[x][y][0]: B, img[x][y][1]: G, img[x][y][2]: R
        ct = 0
        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                # 背景是黑色
                if img[i][j][0] == 0 and img[i][j][1] == 0 and img[i][j][2] == 0:
                    ct = ct + 1
        v = 1.0 - ct / area
        log_file.write(str(v)+"\n")
        value.append(v)
        count = count + 1
        utils.progress_bar(count, len(picture_path_list))

    if plot:
        utils.plot_data(value, "statistic-segment-area-ratio-full-LIP.png")


# 分视角绘制分割面积所占比例的折线图
def plot_segment_area_ratio_angle_from_file(file_path="txt/BP700_segment_area_ratio_log.txt"):
    log_file = open(file_path, "r")
    value_side, value_front_back = [], []
    log_file_lines = log_file.readlines()
    count = 0
    for line in log_file_lines:
        file_name, ratio = line.split(" ")[0], float(line.split(" ")[1])
        width, height = imagesize.get(file_name)
        # 如果是正后视图
        if height/width > 1.3:
            value_front_back.append(ratio)
        else:
            value_side.append(ratio)
        count = count + 1
        utils.progress_bar(count, len(log_file_lines))

    img_name = "segment_area_ratio_" + str(int(time.time()))
    utils.plot_data(value_side, img_name + "_side.png")
    utils.plot_data(value_front_back, img_name + "_front_back.png")



