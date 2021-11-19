'''
Author: Li, Yirui
Date: 2021-07-26
Description: This code is about some result analysis of this project.
FilePath: /liyirui/PycharmProjects/BikePersonImageProcessing/result_analyse.py
'''
import os
import utils
import glob

class AngleInfluenceAnalysis:
    '''
    This class is used to analys the influence of angle of different view in 
    ReID task in BikePerson-700 dataset.
    input:
        dataset_path: str, path of ISP-reID's result. something like 
            "ISP-reID/log/ISP-bikeperson-700-same-id-diff-view-0_1-test-backup/results/"
            you can get this result by using code in ISP-reID tools/result_visualize.py
        lable_path: str, path of labeled point view images.
    output:
        print the result in the terminal
    '''
    def __init__(self, dataset_path, label_path):
        self.dataset_path = dataset_path
        self.label_path = label_path
        self.image_name_label_dir = {}
        self.image_name_ap_dir = {}
        label_image_paths = glob.glob(self.label_path + "*/*.png")
        result_image_paths = glob.glob(self.dataset_path + "*.png")
        print("The number of image with label: ", len(label_image_paths))
        print("The number of image in result: ", len(result_image_paths))
        for path in label_image_paths:
            image_name = path.split("/")[-1]
            if path.find("back") != -1 or path.find("front") != -1:
                self.image_name_label_dir[image_name] = "back-front"
            else:
                self.image_name_label_dir[image_name] = "side"
        for path in result_image_paths:
            image_name = path.split("/")[-1]
            ap = float(image_name[:7])
            image_name = image_name[8:-4]
            self.image_name_ap_dir[image_name] = ap

    def run(self):
        id_angle_dir, id_ap_dir = {}, {}
        for image_name in self.image_name_ap_dir.keys():
            id = image_name.split("_")[0]
            ap = self.image_name_ap_dir[image_name]
            angle = self.image_name_label_dir[image_name]
            if id in id_angle_dir:
                if id_angle_dir[id] == angle:
                    id_angle_dir[id] = "same"
                else:
                    id_angle_dir[id] = "diff"
                mean_ap = (ap + id_ap_dir[id]) / 2
                id_ap_dir[id] = mean_ap
            else: 
                id_angle_dir[id] = angle
                id_ap_dir[id] = ap
        
        same_ap, same_ct, diff_ap, diff_ct = 0, 0, 0, 0 
        for id in id_angle_dir.keys():
            if id_angle_dir[id] == "same":
                same_ap = same_ap + id_ap_dir[id]
                same_ct = same_ct + 1
            else:
                diff_ap = diff_ap + id_ap_dir[id]
                diff_ct = diff_ct + 1

        print("There are " + str(same_ct) + " have same angle.")
        print("The mean ap of identity between same angle is :", same_ap / same_ct)
        print("There are " + str(diff_ct) + " have different angle.")
        print("The mean ap of identity between different angle is :", diff_ap/diff_ct)

if __name__ == "__main__":
    dataset_path = "/home/liyirui/PycharmProjects/ISP-reID/log/ISP-BikePerson-700-train-140-7-angle-new-55-new-test/results/"
    label_path = "/home/liyirui/PycharmProjects/dataset/standard_angle_bikeperson/"
    aia = AngleInfluenceAnalysis(dataset_path, label_path)
    aia.run()


'''
This code maybe about analyse the the two different kinds of 
BikePerson-700 dataset's AP result of each query images.
Something like compare BikePerson-700-orgin with BikePerson-700-seg,
or compare BikePerson-700-origin with BikePerson-700-mixed_0_4.
So maybe can use this class by give two dataset path when initializing.

This code was used when I want to analyse how much influence of changing 
color of bike in BikePerson re-id task.

some old code in main.py:
# This ResultAnalyse here is the old name of APChangeAnalysis.
# dir_map = {"origin": "BP-700-origin-20210317",
#            "seg": "BP-700-seg-20210317",
#            "0_3": "BP-700-mixed_0_3-20210318",
#            "0_4": "BP-700-mixed_0_4-20210317",
#            "0_6": "BP-700-mixed_0_6-20210317"}
#
# result_analyse = ResultAnalyse(dir_map["origin"], dir_map["0_4"])
# result_analyse.ap_result_analyse(do_display=True)

'''
class APChangeAnalysis:
    result_path1 = ""
    result_path2 = ""

    def __init__(self, path1, path2):
        self.result_path1 = path1
        self.result_path2 = path2

    def ap_result_analyse(self, do_display=False):
        result_txt_path1 = os.path.join(self.result_path1, "num_map.txt")
        result_txt_path2 = os.path.join(self.result_path2, "num_map.txt")
        result1 = open(result_txt_path1, "r")
        result2 = open(result_txt_path2, "r")

        simplify_name1 = self.result_path1.split("-")[2]
        simplify_name2 = self.result_path2.split("-")[2]

        output_file = utils.open_file("txt/ap_result_analyse_" + simplify_name1 + "_" + simplify_name2, ".txt", "w")

        # name_ap_map_delta 中存放的是图片名对应的两个ap的差值
        name_ap_map1, name_ap_map_delta, name_num_map1, name_num_map2 = {}, {}, {}, {}
        for line in result1.readlines():
            num, img_name, ap = int(line.split(" ")[0]), line.split(" ")[1], float(line.split(" ")[2])
            name_ap_map1[img_name] = ap
            name_num_map1[img_name] = num

        delta_array = []
        for line in result2.readlines():
            num, img_name, ap = int(line.split(" ")[0]), line.split(" ")[1], float(line.split(" ")[2])
            name_ap_map_delta[img_name] = name_ap_map1[img_name] - ap
            delta_array.append(name_ap_map_delta[img_name])
            name_num_map2[img_name] = num

        utils.plot_data(delta_array, "ap_result_" + simplify_name1 + "_" + simplify_name2 + ".png")
        # 对ap的差值进行排序
        sorted_name_ap_map_delta = sorted(name_ap_map_delta.items(), key=lambda item: item[1], reverse=True)

        # 创建一个文件夹，存放输出的图片
        if do_display:
            output_img_folder_path = os.path.join("plot", "ap_result_analyse_" + simplify_name1 + "_" + simplify_name2)
            utils.makedir_from_name_list([output_img_folder_path])

        count = 1
        count_better = 0
        count_worse = 0
        for item in sorted_name_ap_map_delta:
            print_str = item[0] + " " + str(item[1]) + " " + str(name_num_map1[item[0]]) + " " \
                        + str(name_num_map2[item[0]])
            if float(item[1]) >= 0.0:
                count_worse = count_worse + 1
            else:
                count_better = count_better + 1
            output_file.write(print_str + "\n")

            # 导出对比图片
            if do_display:
                display_img_list = [os.path.join(self.result_path1, str(name_num_map1[item[0]]) + ".jpg"),
                                    os.path.join(self.result_path2, str(name_num_map2[item[0]]) + ".jpg")]
                output_img_name = '{:.6f}_{}'.format(float(item[1]), item[0])
                utils.img_display(display_img_list, output_img_name, display_img_list, False, False, (12, 12), output_img_folder_path + "/")

            utils.progress_bar(count, len(sorted_name_ap_map_delta))
            count = count + 1

        print("worse picture: " + str(count_worse))
        print("better picture: " + str(count_better))