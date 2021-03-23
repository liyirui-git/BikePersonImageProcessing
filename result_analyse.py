import os
import utils


class ResultAnalyse:
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