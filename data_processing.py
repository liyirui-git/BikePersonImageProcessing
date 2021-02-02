# 本程序统计所有的人骑车图像的长宽比
import os
import cv2
from matplotlib import pyplot
import matplotlib.image as mpimg
import utils

# 之前在一部分数据集上做操作，这里的值是 "./selectPicture"
rootPath = "./BikePersonDataset"

img1Path = rootPath + "/img"
img2Path = rootPath + "/mask"
img3Path = rootPath + "/seg/segPerson"
img4Path = rootPath + "/seg/segOther"
schp_lip_path = rootPath + "/LIP"
schp_atr_path = rootPath + "/ATR"
schp_pascal_path = rootPath + "/PASCAL"


# 统计分割后的图片，人体所占的比例，并绘制折线图
def plot_segment_area_ratio():
    pictureFolderName = "./path_to_predictions_epoch10"
    pictureList = os.listdir(pictureFolderName)

    value = []

    count = 0
    for picture in pictureList:
        # 计算分割结果的像素在所有像素中所占的比例
        img = cv2.imread(pictureFolderName + "/" + picture)
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
    pictureFolderName = "./BikePersonDatasetNew"
    pictureList = os.listdir(pictureFolderName)

    value = []

    for picture in pictureList:
        # 如何得到picture的尺寸大小？
        img = cv2.imread(pictureFolderName + "/" + picture)
        shape = img.shape
        value.append(shape[0] / shape[1])

    utils.plot_data(value, "statistic-length-width-ratio.png")


# 统计左右半边，分割结果所占的面积的比，并且将图片分类
def plot_left_right_area_ratio():
    # picture_folder_name = rootPath + "/side/img"
    mask_folder_name = rootPath + "/side/mask/"
    mask_img_list = os.listdir(mask_folder_name)
    left_folder_name = rootPath + "/side/left/"
    right_folder_name = rootPath + "/side/right/"
    if not os.path.exists(left_folder_name):
        os.mkdir(left_folder_name)
    if not os.path.exists(right_folder_name):
        os.mkdir(right_folder_name)

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
    picture_folder_name = rootPath + "/back_front/img/"
    mask_folder_name = rootPath + "/back_front/mask/"
    mask_img_list = os.listdir(mask_folder_name)

    folder_list = utils.makedir([rootPath + "/back_front/front/", rootPath + "/back_front/front_mask/",
                                rootPath + "/back_front/back/", rootPath + "/back_front/back_mask/",
                                rootPath + "/back_front/other/", rootPath + "/back_front/other_mask/"])

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
def select_proper_picture():
    lwRatio_front_back = 1.9
    lwRatio_side = 1.1

    pictureFolderName = rootPath + "/img"
    maskFolderName = rootPath + "/LIP"
    pictureList = os.listdir(pictureFolderName)

    back_front_folder_name = rootPath + "/back_front"
    side_folder_name = rootPath + "/side"
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
def img_display(imgName):
    imgArr = []
    img1 = mpimg.imread(img1Path + "/" + imgName + ".jpg")
    img2 = mpimg.imread(img2Path + "/" + imgName + ".jpg")
    img3 = mpimg.imread(img3Path + "/" + imgName + ".jpg")
    img4 = mpimg.imread(img4Path + "/" + imgName + ".jpg")
    imgArr.append(img1)
    imgArr.append(img2)
    imgArr.append(img3)
    imgArr.append(img4)

    for i in range(0, 4):
        pyplot.subplot(1, 4, i + 1)
        pyplot.imshow(imgArr[i])
        pyplot.xticks([])
        pyplot.yticks([])

    # 加一个文件名的反映射
    nameList = []
    nameFile = open("pictureNameList_Full.txt", "r")
    for line in nameFile.readlines():
        nameList.append(line.split("\n")[0])

    imgNum = int(imgName.split(".")[0]) - 1
    pyplot.savefig(nameList[imgNum])
    pyplot.show()

# 该函数调用了imgDisplay可以展示多张图片
# def images_display():
#     img_display("799.jpg")



###### 慎用，调用之前要知道自己在做什么 本函数只用一次就好
# 用途：将文件夹下的图片名字导入到文本文件中，作为更名数字以后的对照
# def img_name_list_into_file():
#     imgList = os.listdir(img1Path)
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
#     imgList = os.listdir(img1Path)
#     ct = 1
#     for imgName in imgList:
#         os.rename(img1Path + "/" + imgName, img1Path + "/" + str(ct) + ".jpg")
#         os.rename(img2Path + "/" + imgName, img2Path + "/" + str(ct) + ".jpg")
#         os.rename(img3Path + "/" + imgName, img3Path + "/" + str(ct) + ".jpg")
#         os.rename(img4Path + "/" + imgName, img4Path + "/" + str(ct) + ".jpg")
#         ct = ct + 1


# 展示不同来源的分割结果的图像
def img_display_diff_source(imgName):
    pyplot.figure(figsize=(12, 9))
    imgArr = []
    img1 = mpimg.imread(img1Path + "/" + imgName + ".jpg")
    img2 = mpimg.imread(img2Path + "/" + imgName + ".jpg")
    img3 = mpimg.imread(schp_lip_path + "/" + imgName + ".png")
    img4 = mpimg.imread(schp_atr_path + "/" + imgName + ".png")
    img5 = mpimg.imread(schp_pascal_path + "/" + imgName + ".png")
    imgArr.append(img1)
    imgArr.append(img2)
    imgArr.append(img3)
    imgArr.append(img4)
    imgArr.append(img5)

    tag = ['origin', 'isk-LIP', 'schp-LIP', 'schp-ATR', 'schp-PASCAL']
    for i in range(0, len(imgArr)):
        pyplot.subplot(1, len(imgArr), i + 1)
        pyplot.text(20, -30, tag[i], fontdict={'size':20})
        pyplot.imshow(imgArr[i])
        pyplot.xticks([])
        pyplot.yticks([])

    # 加一个文件名的反映射
    nameList = []
    nameFile = open("pictureNameList_Full.txt", "r")
    for line in nameFile.readlines():
        nameList.append(line.split("\n")[0])

    imgNum = int(imgName.split(".")[0]) - 1
    pyplot.savefig("diff_source_" + nameList[imgNum])
    pyplot.show()


# img_display_diff_source("478")

# select_proper_picture()

# plot_left_right_area_ratio()

plot_face_area_ratio()