# 本程序将 BikePerson 数据集中的数据集集中到同一个文件夹下，并且防止重名的事情发生。
import os
import shutil

sourceFolderName = "C:\\Users\\11029\\Documents\\BUAAmaster\\GPdataset\\BikePerson Dataset"
subfolderList = ["cam_1_2", "cam_2_3", "cam_3_5", "cam_4_5", "cam_4_5", "cam_5_6", "cam_6_1"]
subsubfloderName = "Eletric"
targetFolderName = "BikePersonDatasetNew"
os.mkdir(targetFolderName)

ct = 0

for subfolderName in subfolderList:
    tempPath = sourceFolderName + "\\" + subfolderName + "\\" + subsubfloderName
    sub3_folderList = os.listdir(tempPath)
    for sub3_folderName in sub3_folderList:
        subtempPath = tempPath + "\\" + sub3_folderName
        picFileList = os.listdir(subtempPath)
        for picFileName in picFileList:
            srcPath = subtempPath + "\\" + picFileName
            dstPath = targetFolderName + "\\" + subfolderName + "-" + sub3_folderName + "-" + picFileName
            print(ct)
            shutil.copyfile(srcPath, dstPath)
            ct = ct + 1