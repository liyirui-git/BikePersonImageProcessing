import os, random, shutil, re, time
import cv2, imagesize
import utils

ROOT_PATH = "./BikePersonDatasetProcess"

origin_pic_dir_path = "BikePerson-full/BikePerson-full-origin"  # .jpg
mask_pic_dir_path = "BikePerson-full/BikePerson-full-mask"  # .png
seg_pic_dir_path = "BikePerson-full/BikePerson-full-seg"  # .png

sub_folder_name_list = ["origin", "mask", "seg"]
reid_folder_name_list = ["bounding_box_train", "bounding_box_test", "query"]


# 将 Bike Person 的数据整理成 DukeMTMC-reID 的数据格式
# 这里以编号为奇数的作为测试集，编号为偶数的为训练集
# copyfile == True 才向新的地址复制图片
def create_dataset_as_dukemtmcreid(copyfile=False):
    source_folder_name = "C:\\Users\\11029\\Documents\\BUAAmaster\\GPdataset\\BikePerson Dataset"
    # 出错的地方，将"cam_4_5"多写了一遍！
    # subfolder_list = ["cam_1_2", "cam_2_3", "cam_3_5", "cam_4_5", "cam_4_5", "cam_5_6", "cam_6_1"]
    subfolder_list = ["cam_1_2", "cam_2_3", "cam_3_5", "cam_4_5", "cam_5_6", "cam_6_1"]
    utils.makedir_from_name_list(["BikePersonDatasetNew"])
    target_folder_name_list = utils.makedir_from_name_list(["BikePersonDatasetNew\\bounding_box_train",
                                                            "BikePersonDatasetNew\\bounding_box_test"])

    num_to_name_txt = "txt/num_2_name.txt"
    if not os.path.exists(num_to_name_txt):
        num_to_name_file = open(num_to_name_txt, "w")
    else:
        print("[Error] There exist txt/num_2_name.txt, do some check please!")
        return

    id_count = 1
    img_count = 0
    subsubfolder_name = "Eletric"
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
                if id_count % 2 == 0:
                    target_folder_name = target_folder_name_list[0]
                else:
                    target_folder_name = target_folder_name_list[1]
                dst_path = target_folder_name + "\\" + str(id_count).zfill(4) + "_c" + str(camera[3]) + "_" + vehicle + frame
                if copyfile:
                    shutil.copyfile(src_path, dst_path)
                img_count = img_count+1
            utils.progress_bar(id_count, 4579)
            num_to_name_file.write(str(id_count) + " "+ subtemp_path + "\n")
            id_count = id_count + 1
    print("  Eletric id end:" + str(id_count-1))

    subsubfolder_name = "Bike"
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
                if id_count % 2 == 0:
                    target_folder_name = target_folder_name_list[0]
                else:
                    target_folder_name = target_folder_name_list[1]
                dst_path = target_folder_name + "\\" + str(id_count).zfill(4) + "_c" + str(camera[3]) + "_" + vehicle + frame
                if copyfile:
                    shutil.copyfile(src_path, dst_path)
                img_count = img_count + 1
            utils.progress_bar(id_count, 4579)
            num_to_name_file.write(str(id_count) + " "+ subtemp_path + "\n")
            id_count = id_count + 1
    print("  Bike id end:" + str(id_count-1))

    subsubfolder_name = "Motor"
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
                if id_count % 2 == 0:
                    target_folder_name = target_folder_name_list[0]
                else:
                    target_folder_name = target_folder_name_list[1]
                dst_path = target_folder_name + "\\" + str(id_count).zfill(4) + "_c" + str(camera[3]) + "_" + vehicle + frame
                if copyfile:
                    shutil.copyfile(src_path, dst_path)
                img_count = img_count + 1
            utils.progress_bar(id_count, 4579)
            num_to_name_file.write(str(id_count) + " "+ subtemp_path + "\n")
            id_count = id_count + 1
    print("  Motor id end:" + str(id_count-1))
    print("Total images: " + str(img_count))

# 从测试图片中，得到所需要的query图片
def create_query_from_test_images(dataset_path, seed=0):
    test_image_folder_name = os.path.join(dataset_path, "bounding_box_test")
    test_image_name_list = os.listdir(test_image_folder_name)
    query_image_folder_name = os.path.join(dataset_path, "query")

    if os.path.exists(query_image_folder_name):
        print("There exist query folder, are you sure there everything is ok? ")
        return
    else:
        utils.makedir_from_name_list([query_image_folder_name])

    # key: 人id与摄像头id组成的字符串
    # value：字符串数组，表示一组图片名
    image_group_dir = {}
    dir_key_arr = []

    # 图片命名方式： person-id_camera-id_xxxxx.jpg
    for image_name in test_image_name_list:
        person_id, camera_id, other_text = image_name.split('_')
        dir_key = person_id + camera_id
        if dir_key in image_group_dir:
            image_group_dir[dir_key].append(image_name)
        else:
            dir_key_arr.append(dir_key)
            temp_arr = [image_name]
            image_group_dir[dir_key] = temp_arr

    # 遍历 image_group_arr 提取出来想要的照片
    random.seed(seed)
    for key in dir_key_arr:
        image_name_arr = image_group_dir[key]
        # print(image_name_arr)
        query_image_name = image_name_arr[random.randint(0, len(image_name_arr)-1)]
        shutil.move(test_image_folder_name + "/" + query_image_name, query_image_folder_name + "/" + query_image_name)
        # print(query_image_name)


# 从完整的数据集中，随机得到一组训练集和测试集id数量都为 n 的图片库
def create_subdataset(amount=5, seed=0, total_id=4579):
    random.seed(seed)
    dataset_name = "BikePerson-" + str(amount)
    dst_folder_list = []

    # 创建对应的数据集的文件结构
    utils.makedir_from_name_list([dataset_name])
    for sub_folder_name in sub_folder_name_list:
        sub_folder_path = os.path.join(dataset_name, dataset_name + "-" + sub_folder_name)
        dst_folder_list.append(sub_folder_path)
        utils.makedir_from_name_list([sub_folder_path])
        utils.makedir_from_name_list([os.path.join(sub_folder_path, reid_folder_name_list[0]),
                                      os.path.join(sub_folder_path, reid_folder_name_list[1]),
                                      os.path.join(sub_folder_path, reid_folder_name_list[2])])

    # 根据编号，先得到每个id下的文件的名字，存入字典 picture_id_name_map 中
    picture_name_list = utils.get_all_file_name_in_reid_path_format(origin_pic_dir_path)
    picture_id_name_map = {}
    for picture_name in picture_name_list:
        id = int(picture_name.split('_')[0])
        if id in picture_id_name_map:
            picture_id_name_map[id].append(picture_name)
        else:
            temp = [picture_name]
            picture_id_name_map[id] = temp

    # 产生一个只有奇数的id列表，从而保证训练集和测试集中的id数量严格相等
    odd_id_list = []
    i = 1
    while i < total_id:
        odd_id_list.append(i)
        i = i + 2
    odd_id_list = random.sample(odd_id_list, k=amount)

    # 从 odd_id_list 中选取编号，然后去之前的字典中找到对应的图片，复制到对应的文件夹下
    count = 0
    for id in odd_id_list:
        # 奇数对应的是测试集，偶数对应的是训练集
        id_test_picture_name_list = picture_id_name_map[id]
        id_train_picture_name_list = picture_id_name_map[id+1]
        dataset_path_list = [origin_pic_dir_path, mask_pic_dir_path, seg_pic_dir_path]
        for i in range(0, len(dataset_path_list)):
            dataset_path = dataset_path_list[i]
            # 测试集
            for id_picture_name in id_test_picture_name_list:
                id_picture_name = id_picture_name.split(".")[0]
                # 查找是否存在.png或者.jpg
                # 查找bounding_box_test和query文件夹下，得到源文件所在的路径
                test_folder_path = os.path.join(dataset_path, reid_folder_name_list[1])
                query_folder_path = os.path.join(dataset_path, reid_folder_name_list[2])
                src_path = ""
                in_query = False
                if os.path.exists(os.path.join(test_folder_path, id_picture_name + ".jpg")):
                    src_path = os.path.join(test_folder_path, id_picture_name + ".jpg")
                elif os.path.exists(os.path.join(test_folder_path, id_picture_name + ".png")):
                    src_path = os.path.join(test_folder_path, id_picture_name + ".png")
                elif os.path.exists(os.path.join(query_folder_path, id_picture_name + ".jpg")):
                    src_path = os.path.join(query_folder_path, id_picture_name + ".jpg")
                    in_query = True
                elif os.path.exists(os.path.join(query_folder_path, id_picture_name + ".png")):
                    src_path = os.path.join(query_folder_path, id_picture_name + ".png")
                    in_query = True
                else:
                    print("[Error]: in dataset_processing.py in create_subdataset() test")
                # 将源文件复制到目的文件的地址
                dst_path = ""
                if in_query:
                    dst_path = os.path.join(dst_folder_list[i], reid_folder_name_list[2])
                else:
                    dst_path = os.path.join(dst_folder_list[i], reid_folder_name_list[1])
                dst_path = os.path.join(dst_path, id_picture_name + ".png")
                shutil.copyfile(src_path, dst_path)
            # 训练集
            for id_picture_name in id_train_picture_name_list:
                id_picture_name = id_picture_name.split(".")[0]
                # 查找是否存在.png或者.jpg
                # 查找bounding_box_test和query文件夹下，得到源文件所在的路径
                train_folder_path = os.path.join(dataset_path, reid_folder_name_list[0])
                src_path = ""
                if os.path.exists(os.path.join(train_folder_path, id_picture_name + ".jpg")):
                    src_path = os.path.join(train_folder_path, id_picture_name + ".jpg")
                elif os.path.exists(os.path.join(train_folder_path, id_picture_name + ".png")):
                    src_path = os.path.join(train_folder_path, id_picture_name + ".png")
                else:
                    print("[Error]: in dataset_processing.py in create_subdataset() train")

                dst_path = ""
                dst_path = os.path.join(dst_folder_list[i], reid_folder_name_list[0])
                dst_path = os.path.join(dst_path, id_picture_name + ".png")
                shutil.copyfile(src_path, dst_path)
        count = count + 1
        utils.progress_bar(count, len(odd_id_list))

# 构造一个分割效果好的数据集
# 这里的问题是，如何排布训练集和测试集，因为在一个id下，可能会因为分割质量的问题，丢失一部分图片
# threshold1: 这是侧视图对应的threshold
# threshold2: 这是正/后视图对应的threshold
# separate_ratio: 这是用来区分侧视图和正/后视图的长宽比
# def create_high_quality_seg_dataset(threshold1, threshold2, separate_ratio=1.3, dataset_prefix="700"):
#     txt_path = os.path.join("txt", "BP" + dataset_prefix + "_segment_area_ratio_log.txt")
#     txt_file = open(txt_path, "r")
#     for line in txt_file.readlines():
#         file_name, area_ratio = line.split(" ")[0], float(line.split(" ")[1])
#         width, height = imagesize.get(file_name)
#         hw_ratio = height/width
#         path_part1, path_part2 = file_name.split("mask")[0], file_name.split("mask")[1]
#         seg_path = path_part1 + "seg" + path_part2
#         # 如果是正后视图
#         if hw_ratio < 1.3 and area_ratio > threshold1:
#             # 将这些放到新的数据集的文件夹下
#
#         elif hw_ratio > 1.3 and area_ratio > threshold2:


# 构造一个掺杂着分割前和分割后图片的数据集
# threshold1: 这是侧视图对应的threshold
# threshold2: 这是正/后视图对应的threshold
# separate_num: 这是用来区分侧视图和正/后视图的长宽比
# mixed_ratio: 这是用来确定分割后的图片占原图片的比例
def create_mixed_dataset(threshold1=0.1, threshold2=0.2, separate_num=1.3, mixed_ratio=0.4, seed=0, dataset_num="700"):
    time_begin = time.time()
    # 先不管图片质量，直接随机，确定那些图片使用分割后的
    # 然后真正取那个分割后的图片的时候，再判断一下它的质量，如果质量不好，就不取；质量好，则取该图片的分割
    txt_path = os.path.join("txt", "BP" + dataset_num + "_segment_area_ratio_log.txt")
    txt_file = open(txt_path, "r")
    txt_lines = txt_file.readlines()

    # 建立新的文件夹
    folder_name = "BikePerson-" + dataset_num
    mixed_ratio_str = str(mixed_ratio).split(".")[0] + "_" + str(mixed_ratio).split(".")[1]
    sub_folder_path = os.path.join(folder_name, folder_name+"-mixed-"+mixed_ratio_str)
    utils.makedir_from_name_list([sub_folder_path])
    utils.makedir_from_name_list([os.path.join(sub_folder_path, reid_folder_name_list[0]),
                                  os.path.join(sub_folder_path, reid_folder_name_list[1]),
                                  os.path.join(sub_folder_path, reid_folder_name_list[2])])

    # 按照比例随机确定那些图片需要被替换
    random.seed(seed)
    line_number_list = []
    num = 0
    while num < len(txt_lines):
        line_number_list.append(num)
        num = num + 1
    seg_number_list = random.sample(line_number_list, k=int(mixed_ratio*len(txt_lines)))
    seg_number_map = {}
    for number in seg_number_list:
        seg_number_map[number] = 0

    ct_total = 0
    ct_seg_approx = 0
    ct_seg_exact = 0
    for i in range(0, len(txt_lines)):
        file_name, area_ratio = txt_lines[i].split(" ")[0], float(txt_lines[i].split(" ")[1])
        width, height = imagesize.get(file_name)
        hw_ratio = height / width
        path_part1, path_part2 = file_name.split("mask")[0], file_name.split("mask")[1]
        seg_path = path_part1 + "seg" + path_part2
        origin_path = path_part1 + "origin" + path_part2
        # '[/\\\]' 使用正则表达式，将路径名中的斜杠和反斜杠都作为分割字符
        path_split_list = re.split('[/\\\]', origin_path)
        target_path = os.path.join(sub_folder_path, path_split_list[2], path_split_list[3])
        # 如果图片需要被替换成分割后的图片
        replace_seg_flag = False
        if i in seg_number_map:
            ct_seg_approx = ct_seg_approx + 1
            # 如果图片的质量高
            if (hw_ratio < separate_num and area_ratio > threshold1) \
                    or (hw_ratio > separate_num and area_ratio > threshold2):
                replace_seg_flag = True
                ct_seg_exact = ct_seg_exact + 1
        if replace_seg_flag:
            shutil.copyfile(seg_path, target_path)
        else:
            shutil.copyfile(origin_path, target_path)
        ct_total = ct_total + 1
        utils.progress_bar(i+1, len(txt_lines))

    print("total:       " + str(ct_total))
    print("seg approx:  " + str(ct_seg_approx))
    print("seg exact:   " + str(ct_seg_exact))
    time_end = time.time()
    print("total time:  %.2fs" % (time_end-time_begin))


# 计算BikePerson最原始的数据集中有多少张图片
def count_initial_dataset_img_number():
    source_folder_name = "C:\\Users\\11029\\Documents\\BUAAmaster\\GPdataset\\BikePerson Dataset"
    subfolder_list = ["cam_1_2", "cam_2_3", "cam_3_5", "cam_4_5", "cam_5_6", "cam_6_1"]

    count = 0
    for subfolder_name in subfolder_list:
        sub_path = os.path.join(source_folder_name, subfolder_name)
        for subsubfoler_name in os.listdir(sub_path):
            subsub_path = os.path.join(sub_path, subsubfoler_name)
            for sub3foler_name in os.listdir(subsub_path):
                count = count + len(os.listdir(os.path.join(subsub_path, sub3foler_name)))

    print(count)


# 获取一些前后视角和侧视角的图片
def select_view_angle_picture():
    lw_ratio_front_back = 1.9
    lw_ratio_side = 1.1

    picture_folder_name = ROOT_PATH + "/img"
    mask_folder_name = ROOT_PATH + "/LIP"
    picture_list = os.listdir(picture_folder_name)

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
    for picture in picture_list:
        mask = cv2.imread(mask_folder_name + "/" + picture.split(".")[0] + ".png")
        img = cv2.imread(picture_folder_name + "/" + picture)
        shape = img.shape
        lw_ratio = shape[0] / shape[1]

        # 根据长宽比得到一些前后视角和侧视角的图片
        if lw_ratio <= lw_ratio_side:
            cv2.imwrite(side_folder_name + "/mask/" + picture, mask)
            cv2.imwrite(side_folder_name + "/img/" + picture, img)

        if lw_ratio >= lw_ratio_front_back:
            cv2.imwrite(back_front_folder_name + "/mask/" + picture, mask)
            cv2.imwrite(back_front_folder_name + "/img/" + picture, img)

        count = count + 1
        utils.progress_bar(count, len(picture_list))
