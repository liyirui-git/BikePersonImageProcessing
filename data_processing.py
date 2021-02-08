# 本程序统计所有的人骑车图像的长宽比
import os
import cv2
import utils
import random
import shutil

# 之前在一部分数据集上做操作，这里的值是 "./selectPicture"
ROOT_PATH = "./BikePersonDataset"


# 将 BikePerson 数据集中的数据集集中到同一个文件夹下，并且防止重名的事情发生。
def concentrating_img_and_rename():
    source_folder_name = "C:\\Users\\11029\\Documents\\BUAAmaster\\GPdataset\\BikePerson Dataset"
    subfolder_list = ["cam_1_2", "cam_2_3", "cam_3_5", "cam_4_5", "cam_4_5", "cam_5_6", "cam_6_1"]
    subsubfolder_name = "Eletric"
    target_folder_name = "BikePersonDatasetNew"
    os.mkdir(target_folder_name)

    ct = 0

    for subfolder_name in subfolder_list:
        temp_path = source_folder_name + "\\" + subfolder_name + "\\" + subsubfolder_name
        sub3_folder_list = os.listdir(temp_path)
        for sub3_folderName in sub3_folder_list:
            subtemp_path = temp_path + "\\" + sub3_folderName
            pic_file_list = os.listdir(subtemp_path)
            for pic_file_name in pic_file_list:
                src_path = subtemp_path + "\\" + pic_file_name
                dst_path = target_folder_name + "\\" + subfolder_name + "-" + sub3_folderName + "-" + pic_file_name
                print(ct)
                shutil.copyfile(src_path, dst_path)
                ct = ct + 1


# 将 Bike Person 的数据整理成 DukeMTMC-reID 的数据格式
def img_rename_and_concentrating_as_dukemtmcreid():
    source_folder_name = "C:\\Users\\11029\\Documents\\BUAAmaster\\GPdataset\\BikePerson Dataset"
    subfolder_list = ["cam_1_2", "cam_2_3", "cam_3_5", "cam_4_5", "cam_4_5", "cam_5_6", "cam_6_1"]
    subsubfolder_name = "Eletric"
    target_folder_name = utils.makedir_from_name_list(["BikePersonDatasetNew"])[0]

    ct = 1
    for subfolder_name in subfolder_list:
        temp_path = source_folder_name + "\\" + subfolder_name + "\\" + subsubfolder_name
        sub3_folder_list = os.listdir(temp_path)
        for sub3_folder_name in sub3_folder_list:
            subtemp_path = temp_path + "\\" + sub3_folder_name
            pic_file_list = os.listdir(subtemp_path)
            for pic_file_name in pic_file_list:
                pic_file_name_split = pic_file_name.split('_')
                camera, vehicle, frame = pic_file_name_split[0], pic_file_name_split[1], pic_file_name_split[2]
                src_path = subtemp_path + "\\" + pic_file_name
                dst_path = target_folder_name + "\\" + str(ct).zfill(4) + "_c" + str(camera[3]) + "_" + vehicle + frame
                print(ct)
                shutil.copyfile(src_path, dst_path)
            ct = ct + 1


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


# 统计左右半边，分割结果所占的面积的比，并且将图片分类
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


# 获取一些前后视角和侧视角的图片
def select_view_angle_picture():
    lwRatio_front_back = 1.9
    lwRatio_side = 1.1

    pictureFolderName = ROOT_PATH + "/img"
    maskFolderName = ROOT_PATH + "/LIP"
    pictureList = os.listdir(pictureFolderName)

    back_front_folder_name = ROOT_PATH + "/back_front"
    side_folder_name = ROOT_PATH + "/side"
    if not os.path.exists(back_front_folder_name):
        os.mkdir(back_front_folder_name)
        os.mkdir(back_front_folder_name + "/img")
        os.mkdir(back_front_folder_name + "/mask")
    if not os.path.exists(side_folder_name):
        os.mkdir(side_folder_name)
        os.mkdir(side_folder_name + '/img')
        os.mkdir(side_folder_name + '/mask')

    count = 0
    for picture in pictureList:
        mask = cv2.imread(maskFolderName + "/" + picture.split(".")[0] + ".png")
        img = cv2.imread(pictureFolderName + "/" + picture)
        shape = img.shape
        lwRatio = shape[0] / shape[1]

        # 根据长宽比得到一些前后视角和侧视角的图片
        if lwRatio <= lwRatio_side:
            cv2.imwrite(side_folder_name + "/mask/" + picture, mask)
            cv2.imwrite(side_folder_name + "/img/" + picture, img)

        if lwRatio >= lwRatio_front_back:
            cv2.imwrite(back_front_folder_name + "/mask/" + picture, mask)
            cv2.imwrite(back_front_folder_name + "/img/" + picture, img)

        count = count + 1
        if count % 100 == 0: print(count)


# 展示当前得到的四种图片
# 这里其实应该将函数里面的这个 imgArr作为一个参数传进来
# 或者是把文件名放在一个数组里面传进来
def img_display_after_segment(img_name):
    img_name_list = [ROOT_PATH + "/img" + "/" + img_name + ".jpg",
                     ROOT_PATH + "/mask" + "/" + img_name + ".jpg",
                     ROOT_PATH + "/seg/segPerson" + "/" + img_name + ".jpg",
                     ROOT_PATH + "/seg/segOther" + "/" + img_name + ".jpg"]

    save_img_name = utils.get_origin_name(img_name)

    utils.img_display(img_name_list, save_img_name)


# 展示不同来源的分割结果的图像
def img_display_from_diff_source(img_name, fig_show=True):
    img_name_list = [ROOT_PATH + "/img" + "/" + img_name + ".jpg",
                     ROOT_PATH + "/mask" + "/" + img_name + ".jpg",
                     ROOT_PATH + "/LIP" + "/" + img_name + ".png",
                     ROOT_PATH + "/ATR" + "/" + img_name + ".png",
                     ROOT_PATH + "/PASCAL" + "/" + img_name + ".png"]
    # 加一个文件名的反映射
    save_img_name = "diff_source_" + utils.get_origin_name(img_name)
    # 图片名列表
    tag_list = ['origin', 'isk-LIP', 'schp-LIP', 'schp-ATR', 'schp-PASCAL']
    utils.img_display(img_name_list, save_img_name, tag_list, fig_show=fig_show)


def random_get_img(num, seed=0, fig_show=True):
    random.seed(seed)
    img_list = [str(random.randint(0, 6340)) for _ in range(num)]
    for img in img_list:
        img_display_from_diff_source(img, fig_show=fig_show)

# img_display_from_diff_source("3255")

# img_display_after_segment("15")

# select_proper_picture()

# plot_left_right_area_ratio()

# plot_face_area_ratio()


###### 慎用，调用之前要知道自己在做什么 本函数只用一次就好
# 用途：将文件夹下的图片名字导入到文本文件中，作为更名数字以后的对照
# def img_name_list_into_file():
#     imgList = os.listdir(root_path + "/img")
#     fileName = "pictureNameList_Full.txt"
#     if os.path.exists(fileName):
#         print(fileName + "is Exist, you make sure to rewrite it?\n")
#     else:
#         file = open(fileName, "w")
#         for imgName in imgList:
#             file.write(imgName)
#             file.write("\n")
#         file.close()

####### 慎用，调用之前要知道自己在做什么 本函数只用一次就好
# 用途：将rootPath文件夹下的冗长的文件名，统一一下
# def picture_name_process():
#     imgList = os.listdir(root_path + "/img")
#     ct = 1
#     for imgName in imgList:
#         os.rename(root_path + "/img" + "/" + imgName, root_path + "/img" + "/" + str(ct) + ".jpg")
#         os.rename(root_path + "/mask" + "/" + imgName, root_path + "/mask" + "/" + str(ct) + ".jpg")
#         os.rename(root_path + "/seg/segPerson" + "/" + imgName, root_path + "/seg/segPerson" + "/" + str(ct) + ".jpg")
#         os.rename(root_path + "/seg/segOther" + "/" + imgName, root_path + "/seg/segOther" + "/" + str(ct) + ".jpg")
#         ct = ct + 1
