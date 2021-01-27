import os
import cv2


def segPersonFromMask(folderName):

    imgList = os.listdir(folderName + "/img")

    segFolderName = folderName + "/seg"
    segPersonFolderName = segFolderName + "/segPerson"
    segOtherFolderName = segFolderName + "/segOther"
    if not os.path.exists(segFolderName):
        os.mkdir(segFolderName)
    if not os.path.exists(segPersonFolderName):
        os.mkdir(segPersonFolderName)
    if not os.path.exists(segOtherFolderName):
        os.mkdir(segOtherFolderName)

    for imgName in imgList:
        img = cv2.imread(folderName + "/img/" + imgName)
        img2 = img.copy()
        mask = cv2.imread(folderName + "/mask/" + imgName)

        shape = img.shape
        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                d1 = mask[i][j][0] - 198
                d2 = mask[i][j][1] - 215
                d3 = mask[i][j][2] - 20
                d = d1*d1 + d2*d2 + d3 * d3
                if d < 50: img[i][j] = [0, 0, 0]
                else: img2[i][j] = [0, 0, 0]

        cv2.imwrite(segPersonFolderName + "/" + imgName, img)
        cv2.imwrite(segOtherFolderName + "/" + imgName, img2)


segPersonFromMask("./BikePersonDataset")