# 本程序统计所有的人骑车图像的长宽比
import os
import cv2
from matplotlib import pyplot
import matplotlib.image as mpimg

# 之前在一部分数据集上做操作，这里的值是 "./selectPicture"
rootPath = "./BikePersonDataset"

img1Path = rootPath + "/img"
img2Path = rootPath + "/mask"
img3Path = rootPath + "/seg/segPerson"
img4Path = rootPath + "/seg/segOther"


# 统计分割后的图片，人体所占的比例，并绘制折线图
def plot_segment_area_ratio():
    # 绘制折线图时区间采样的个数
    N = 200
    # 绘制图片的大小
    S = 5.12

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
        # img[x][y][0]: B
        # img[x][y][1]: G
        # img[x][y][2]: R
        ct = 0
        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                if img[i][j][0] == 198 and img[i][j][1] == 215 and img[i][j][2] == 20:
                    ct = ct + 1
        v = 1.0-ct/area
        value.append(v)
        count = count + 1
        if count % 100 == 0 : print(count)


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
    pyplot.savefig("statistic.png")


# 统计人骑车图像的长宽比，并绘制折线图
def plot_length_width_ratio ():
    # 绘制折线图时区间采样的个数
    N = 200
    # 绘制图片的大小
    S = 5.12

    pictureFolderName = "./BikePersonDatasetNew"
    pictureList = os.listdir(pictureFolderName)

    value = []

    for picture in pictureList:
        # 如何得到picture的尺寸大小？
        img = cv2.imread(pictureFolderName + "/" + picture)
        shape = img.shape
        value.append(shape[0] / shape[1])

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
    pyplot.savefig("statistic.png")

    ct = 0
    for d in dataCalcu:
        ct = ct + d

    print(ct)


# 获取一些高质量的图片
def select_proper_picture():

    areaRatio_min = 0.3
    areaRatio_max = 0.5
    lwRatio_delta = 0.1
    lwRatio_front = 1.0
    lwRatio_side = 2.0

    pictureFolderName = "./BikePersonDatasetNew"
    maskFolderName = "./path_to_predictions_epoch10"
    pictureList = os.listdir(pictureFolderName)

    targetFolderName = "./selectPicture"
    if not os.path.exists(targetFolderName):
        os.mkdir(targetFolderName)
    if not os.path.exists(targetFolderName + "/" + "img"):
        os.mkdir(targetFolderName + "/" + "img")
        os.mkdir(targetFolderName + "/" + "mask")
    
    for picture in pictureList:
        mask = cv2.imread(maskFolderName + "/" + picture)
        img = cv2.imread(pictureFolderName + "/" + picture)
        shape = img.shape
        area = shape[0] * shape[1]
        lwRatio = shape[0] / shape[1]
        # 先判断一波长宽比是否合适
        if not (lwRatio < lwRatio_front + lwRatio_delta and lwRatio > lwRatio_front - lwRatio_delta
                or lwRatio < lwRatio_side + lwRatio_delta and lwRatio > lwRatio_side - lwRatio_delta) :
            continue
        # 再判断面积的占比是否合适
        ct = 0
        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                if mask[i][j][0] == 198 and mask[i][j][1] == 215 and mask[i][j][2] == 20:
                    ct = ct + 1
        areaRatio = 1.0 - ct / area

        if areaRatio < areaRatio_min or areaRatio > areaRatio_max:
            continue

        cv2.imwrite(targetFolderName + "/mask/" + picture, mask)
        cv2.imwrite(targetFolderName + "/img/" + picture, img)


# 展示当前得到的四种图片
def img_display(imgName):
    imgArr = []
    img1 = mpimg.imread(img1Path + "/" + imgName)
    img2 = mpimg.imread(img2Path + "/" + imgName)
    img3 = mpimg.imread(img3Path + "/" + imgName)
    img4 = mpimg.imread(img4Path + "/" + imgName)
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
