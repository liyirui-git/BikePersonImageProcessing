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

#### 将 BikePerson 数据集中的数据集集中到同一个文件夹下，并且防止重名的事情发生。
# def concentrating_img_and_rename():
#     source_folder_name = "C:\\Users\\11029\\Documents\\BUAAmaster\\GPdataset\\BikePerson Dataset"
#     subfolder_list = ["cam_1_2", "cam_2_3", "cam_3_5", "cam_4_5", "cam_4_5", "cam_5_6", "cam_6_1"]
#     subsubfolder_name = "Eletric"
#     target_folder_name = "BikePersonDatasetNew"
#     os.mkdir(target_folder_name)
#
#     ct = 0
#
#     for subfolder_name in subfolder_list:
#         temp_path = source_folder_name + "\\" + subfolder_name + "\\" + subsubfolder_name
#         sub3_folder_list = os.listdir(temp_path)
#         for sub3_folderName in sub3_folder_list:
#             subtemp_path = temp_path + "\\" + sub3_folderName
#             pic_file_list = os.listdir(subtemp_path)
#             for pic_file_name in pic_file_list:
#                 src_path = subtemp_path + "\\" + pic_file_name
#                 dst_path = target_folder_name + "\\" + subfolder_name + "-" + sub3_folderName + "-" + pic_file_name
#                 print(ct)
#                 shutil.copyfile(src_path, dst_path)
#                 ct = ct + 1
