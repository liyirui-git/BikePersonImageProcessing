# 本代码作为一个补丁，主要是提供一些函数处理之前的数据集将cam_4_5多选入了一遍的 bug
import os
import utils
import shutil


def print_map_table(dataset_name="BikePersonDatasetNew-without-seg"):
    sub_dir_list = ["bounding_box_test", "bounding_box_train", "query"]
    total_dir_name = os.path.join(dataset_name, "total")
    name_num_dir = {}
    count = 1
    if not os.path.exists(total_dir_name):
        utils.makedir_from_name_list([total_dir_name])

        for sub_dir in sub_dir_list:
            for pic_name in os.listdir(os.path.join(dataset_name, sub_dir)):
                num = int(pic_name.split("_")[0])
                if num >= 188 and num <= 316 or num >= 2135 and num <= 3443 or num >= 5683 and num <= 6092:
                    continue
                src_path = os.path.join(dataset_name, os.path.join(sub_dir, pic_name))
                dst_path = os.path.join(total_dir_name, pic_name)
                shutil.copyfile(src_path, dst_path)

    for pic_name in os.listdir(total_dir_name):
        num = int(pic_name.split("_")[0])
        if num not in name_num_dir:
            name_num_dir[num] = count
            count = count + 1

    for key in name_num_dir:
        print(str(key) + " " + str(name_num_dir[key]))
    print(len(name_num_dir))


def dataset_patch(dataset_name):
    sub_dir_list = ["bounding_box_test", "bounding_box_train", "query"]
    new_test_dir_path = os.path.join(dataset_name, "test_dir")
    new_train_dir_path = os.path.join(dataset_name, "train_dir")

    if not os.path.exists(new_test_dir_path):
        utils.makedir_from_name_list([new_test_dir_path, new_train_dir_path])
        f_map = open("txt/old_num_2_new_num.txt", "r")
        name_map = {}
        for line in f_map.readlines():
            num1, num2 = int(line.split()[0]), int(line.split()[1])
            name_map[utils.four_bit_num(num1)] = utils.four_bit_num(num2)
        count = 0
        for sub_dir in sub_dir_list:
            for pic_name in os.listdir(os.path.join(dataset_name, sub_dir)):
                num_str = pic_name.split("_")[0]
                if num_str in name_map:
                    pic_name_new = name_map[num_str] + pic_name[4:]
                    src_path = os.path.join(dataset_name, os.path.join(sub_dir, pic_name))
                    dst_path = ""
                    if int(name_map[num_str]) % 2 == 0:
                        dst_path = os.path.join(new_train_dir_path, pic_name_new)
                    else:
                        dst_path = os.path.join(new_test_dir_path, pic_name_new)
                    count = count + 1
                    shutil.copyfile(src_path, dst_path)
        print("total: " + str(count))


# 将合并到test中的原本属于query的图片再提取出来
def rebuild_query_images():
    # 先去到origin下，看看query中的图片是哪些
    origin_dataset_path_dir = "BikePerson-700/BikePerson-700-origin"
    recolor_dataset_path_dir = "BikePerson-700/BikePerson-700-recolor"

    utils.makedir_from_name_list([os.path.join(recolor_dataset_path_dir, "query")])

    query_image_dir = {}
    origin_query_image_path = os.path.join(origin_dataset_path_dir, "query")
    ct = 0
    for name in os.listdir(origin_query_image_path):
        name = name.split(".png")[0]
        num = int(name.split("_")[0])
        if num not in query_image_dir:
            query_image_dir[num] = []
        query_image_dir[num].append(name)
        ct = ct + 1
        utils.progress_bar(ct, len(os.listdir(origin_query_image_path)))

    recolor_test_image_path = os.path.join(recolor_dataset_path_dir, "bounding_box_test")
    ct = 0
    for name in os.listdir(recolor_test_image_path):
        num = int(name.split("_")[0])
        if name.find(query_image_dir[num][0]) != -1 or name.find(query_image_dir[num][1]) != -1:
            src = os.path.join(recolor_test_image_path, name)
            dst = os.path.join(recolor_dataset_path_dir, "query", name)
            shutil.move(src, dst)
        ct = ct + 1
        utils.progress_bar(ct, len(os.listdir(recolor_test_image_path)))
