import os
import random

import utils
import cv2
import imagesize
import json
import numpy as np


class KeyPointDetection:
    dataset_folder_path = ""
    pose_img_folder_path = ""
    origin_img_folder_path = ""
    all_img_name_list = []
    all_img_path_list = []
    annotations_list = []
    img_name_path_dir = {}
    img_name_pose_dir = {}
    subfolder_list = ["bounding_box_test", "bounding_box_train", "query"]
    keypoint_name_table_alphapose = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                                     "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                                     "left_wrist", "right_wrist", "left_hip", "right_hip",
                                     "left_knee", "right_knee", "left_ankle", "right_ankle", "neck"]
    keypoint_name_table_openpose = ["nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
                                    "left_shoulder", "left_elbow", "left_wrist",
                                    "right_hip", "right_knee", "right_ankle",
                                    "left_hip", "left_knee", "left_ankle",
                                    "right_eye", "left_eye", "right_ear", "left_ear"]

    def __init__(self, dataset_folder_path):
        self.dataset_folder_path = dataset_folder_path
        self.pose_img_folder_path = os.path.join(self.dataset_folder_path,
                                                 self.dataset_folder_path + "-pose")
        self.origin_img_folder_path = os.path.join(self.dataset_folder_path,
                                              self.dataset_folder_path + "-origin")
        self.json_path_list = ["alphapose-query-results.json", "alphapose-test-results.json", "alphapose-train-results.json"]
        self.all_img_name_list = utils.get_all_file_name_in_reid_path_format(self.origin_img_folder_path)
        self.all_img_path_list = utils.get_all_file_path_in_reid_path_format(self.origin_img_folder_path)
        for json_path in self.json_path_list:
            full_json_path = os.path.join(self.pose_img_folder_path, json_path)
            self.annotations_list = self.annotations_list + utils.read_annotations_list_from_json(full_json_path)
        for i in range(len(self.all_img_name_list)):
            self.img_name_path_dir[self.all_img_name_list[i]] = self.all_img_path_list[i]
        for annotation in self.annotations_list:
            if annotation["image_id"] in self.img_name_pose_dir:
                self.img_name_pose_dir[annotation["image_id"]].append(annotation)
            else:
                self.img_name_pose_dir[annotation["image_id"]] = [annotation]

    def is_image_name_detect_annotation(self, image_name):
        return image_name in self.img_name_pose_dir

    # 返回值是一个list
    def get_annotation_from_image_name(self, image_name):
        return self.img_name_pose_dir[image_name]

    def show_keypoint_on_image(self):
        for ct in range(len(self.annotations_list)):
            image_name = self.annotations_list[ct]["image_id"]
            img = cv2.imread(self.img_name_path_dir[image_name])

            keypoint_list = self.annotations_list[ct]["keypoints"]
            for i in range(17):
                cv2.circle(img, (int(keypoint_list[i * 3]), int(keypoint_list[i * 3 + 1])), 3, color=(255, 255, 255))
                cv2.imshow("img", img)
                cv2.waitKey(0)

    def paint_pose_on_images(self, image_name, cv_img=None, threshold=0, thickness=6):
        thinkness = min(thickness, 6)

        # l_pair p_color line_color 来自 AlphaPose/alphapose/utils/vis.py 的 vis_frame 函数
        l_pair = [(0, 1), (0, 2), (1, 3), (2, 4),  # Head
                  (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                  (17, 11), (17, 12),  # Body
                  (11, 13), (12, 14), (13, 15), (14, 16)]
        p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),
                   # Nose, LEye, REye, LEar, REar
                   (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),
                   # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                   (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),
                   (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
        line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                      (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                      (77, 222, 255), (255, 156, 127),
                      (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]

        # 如果没本身这张图片没检测到人体，直接返回
        if not self.is_image_name_detect_annotation(image_name):
            return cv_img

        image_path = self.img_name_path_dir[image_name]
        [image_width, image_height] = imagesize.get(image_path)

        # 初始化大小，默认初始化为黑色的背景
        if cv_img is None:
            cv_img = np.zeros((image_height, image_width, 3), np.uint8)
        # 这里需要给keypoints增加一个脖子，它的位置是左右肩膀的连线中点
        annotation_list = self.get_annotation_from_image_name(image_name)
        for annotation in annotation_list:
            body_keypoints = annotation["keypoints"]

            # 加一个有关置信度阈值的判断
            flag = True
            for i in range(17):
                if body_keypoints[i * 3 + 2] < threshold:
                    flag = False
                    break
            if not flag:
                return

            body_keypoints.append((body_keypoints[5 * 3] + body_keypoints[6 * 3]) / 2)
            body_keypoints.append((body_keypoints[5 * 3 + 1] + body_keypoints[6 * 3 + 1]) / 2)
            body_keypoints.append(min(body_keypoints[5 * 3 + 2], body_keypoints[5 * 3 + 2]))  # 置信度

            # 绘制关键点
            for i in range(18):
                r = 1 + int(thickness/3)
                cv2.circle(cv_img, (int(body_keypoints[i * 3]), int(body_keypoints[i * 3 + 1])), r, color=p_color[i],
                           thickness=-1)
            # 绘制链接线
            for i in range(15):
                point1 = (int(body_keypoints[l_pair[i][0] * 3]), int(body_keypoints[l_pair[i][0] * 3 + 1]))
                point2 = (int(body_keypoints[l_pair[i][1] * 3]), int(body_keypoints[l_pair[i][1] * 3 + 1]))
                cv2.line(cv_img, point1, point2, color=line_color[i], thickness=thickness)

        return cv_img

    # 当前函数缺少置信度的阈值，所以结果可能有点乱
    def get_black_background_picture(self, threshold=0.35, out_img_size=(128, 128)):

        black_bg_folder_path = utils.makedir_from_name_list([os.path.join(self.dataset_folder_path,
                                                                          "BikePerson-700-pose-black-bg")])
        ct = 0
        image_map = {}
        for anno in self.annotations_list:
            image_name = anno["image_id"]
            img = self.paint_pose_on_images(image_name, threshold=threshold)
            # 调整一下图片的尺寸
            img = cv2.resize(img, out_img_size)
            # 保存图片
            if image_name in image_map:
                image_map[image_name] = image_map[image_name] + 1
            else:
                image_map[image_name] = 1
            image_name = image_name.split(".png")[0] + "#" + str(image_map[image_name]) + ".png"
            cv2.imwrite(os.path.join(black_bg_folder_path[0], image_name), img)
            ct = ct + 1
            utils.progress_bar(ct, len(self.annotations_list))

    def trans_to_coco_format(self):
        skeleton_list = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                         [8, 10],
                         [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
        keypoint_name_list = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
                              "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                              "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
                              "right_ankle"]
        # 得到 images 和 annotations
        coco_query_images_list = []
        coco_test_images_list = []
        coco_train_images_list = []
        coco_query_annotations_list = []
        coco_test_annotations_list = []
        coco_train_annotations_list = []
        image_name_to_id = {}
        img_id = 0
        anno_id = 0
        for i in range(len(self.annotations_list)):
            int_keypoints = []
            image_name = self.annotations_list[i]["image_id"]
            image_path = self.img_name_path_dir[image_name]
            image_size = imagesize.get(image_path)

            # 那这里得判断一下这个图片是不是新出现的
            new_image_flag = False
            if image_name not in image_name_to_id:
                image_name_to_id[image_name] = img_id
                img_id = img_id + 1
                new_image_flag = True
            image_id = image_name_to_id[image_name]

            for k in self.annotations_list[i]["keypoints"]:
                int_keypoints.append(int(round(k, 0)))

            annotation = {"image_id": image_id,
                          "area": 1,
                          "num_keypoints": len(self.annotations_list[i]["keypoints"]),
                          "iscrowd": 0,
                          "id": anno_id,
                          "category_id": 1,
                          "keypoints": int_keypoints,
                          'segmentation': [[]],
                          'bbox': [0, 0, 2, 2]
                          }

            # 这里的id不应该是单独的img_id
            image = {"id": image_id,
                     "file_name": image_name,
                     "height": image_size[1],
                     "width": image_size[0]
                     }

            if image_path.find("query") != -1:
                coco_query_annotations_list.append(annotation)
                if new_image_flag:
                    coco_query_images_list.append(image)
            elif image_path.find("bounding_box_test") != -1:
                coco_test_annotations_list.append(annotation)
                if new_image_flag:
                    coco_test_images_list.append(image)
            else:
                coco_train_annotations_list.append(annotation)
                if new_image_flag:
                    coco_train_images_list.append(image)
            anno_id = anno_id + 1
            utils.progress_bar(anno_id, len(self.annotations_list))

        print("total images: " + str(img_id))
        print("total annotations: " + str(anno_id))

        # 得到 categories
        coco_categories = [{"supercategory": "person",
                           "id": 1,
                           "name": "person",
                           "keypoints": keypoint_name_list,
                           "skeleton": skeleton_list
                           }]
        coco_images_list = [coco_query_images_list, coco_test_images_list, coco_train_images_list]
        coco_annotations_list = [coco_query_annotations_list, coco_test_annotations_list, coco_train_annotations_list]
        subfolder_name_list = ["query", "test", "train"]
        for i in range(3):
            coco_format_data = {"images": coco_images_list[i],
                                "annotations": coco_annotations_list[i],
                                "categories": coco_categories}
            json_str = json.dumps(coco_format_data)
            json_file_name = "coco_format_" + subfolder_name_list[i] + ".json"
            json_file = open(os.path.join(self.pose_img_folder_path, json_file_name), "w")
            json_file.write(json_str)

    # 关于AlphaPose的结果中的score的含义：
    # score is the confidence score for the whole person, computed by our parametric pose NMS.
    def plot_scores_in_annotations(self):
        values = []
        ct = 0
        for annotation in self.annotations_list:
            values.append(annotation["score"])
            ct = ct + 1
            utils.progress_bar(ct, len(self.annotations_list))
        utils.plot_data(values, "annotation_scores.png")

    def count_image_which_detect_human(self):
        image_dic = {}
        ct = 0
        for annotation in self.annotations_list:
            image_id = annotation["image_id"]
            if image_id not in image_dic:
                image_dic[image_id] = image_id
            ct = ct + 1
            utils.progress_bar(ct, len(self.annotations_list))
        print("total image number: " + str(len(self.all_img_name_list)))
        print("detected human image number: " + str(len(image_dic)))
        print("detected ratio: " + str(len(image_dic) / len(self.all_img_name_list)))

    def count_human_with_threshold(self, threshold=0.5):
        ct = 0
        human_counter = 0
        for annotation in self.annotations_list:
            flag = True
            for i in range(17):
                if annotation["keypoints"][i*3+2] < threshold:
                    flag = False
                    break
            if flag:
                human_counter = human_counter + 1
            ct = ct + 1
            utils.progress_bar(ct, len(self.annotations_list))
        print("detected prefect human: " + str(human_counter))
        return human_counter

    def get_human_pose_with_threshold(self, threshold):
        ct = 0
        human_poses = []
        human_counter = 0
        for annotation in self.annotations_list:
            flag = True
            for i in range(17):
                if annotation["keypoints"][i * 3 + 2] < threshold:
                    flag = False
                    break
            if flag:
                human_poses.append(annotation)
                human_counter = human_counter + 1
            ct = ct + 1
            utils.progress_bar(ct, len(self.annotations_list))
        print("detected prefect human: " + str(human_counter))
        return human_poses

    def get_image_path_with_pose_threshold(self, threshold):
        ct = 0
        image_path_list = []
        human_counter = 0
        for annotation in self.annotations_list:
            flag = True
            for i in range(17):
                if annotation["keypoints"][i * 3 + 2] < threshold:
                    flag = False
                    break
            if flag:
                image_path_list.append(self.img_name_path_dir[annotation["image_id"]])
                human_counter = human_counter + 1
            ct = ct + 1
            utils.progress_bar(ct, len(self.annotations_list))
        print("detected prefect human: " + str(human_counter))
        return image_path_list

    def count_human_upper_body_with_threshold(self, threshold=0.5):
        ct = 0
        human_counter = 0
        upper_body_part_number = [5, 6, 7, 8, 11, 12]
        for annotation in self.annotations_list:
            flag = True
            for i in upper_body_part_number:
                if annotation["keypoints"][i * 3 + 2] < threshold:
                    flag = False
                    break
            if flag:
                human_counter = human_counter + 1
            ct = ct + 1
            utils.progress_bar(ct, len(self.annotations_list))
        print("detected prefect human: " + str(human_counter))
        return

    def plot_human_counter_with_different_threshold(self, sample_number=200):
        x, dy, y = [], [], []
        for i in range(0, sample_number):
            print(str(i) + ": ")
            y.append(self.count_human_with_threshold(i/sample_number))
            if i == 0:
                dy.append(32897-y[i])
            else:
                dy.append(y[i-1]-y[i])
            x.append(i/sample_number)
        utils.plot_x_y_line_chart(x, dy, "plot_human_counter_with_different_threshold_new.png")

    def plot_human_upper_body_counter_with_different_threshold(self, sample_number=200):
        x, dy, y = [], [], []
        for i in range(0, sample_number):
            print(str(i) + ": ")
            y.append(self.count_human_upper_body_with_threshold(i/sample_number))
            if i == 0:
                dy.append(32897-y[i])
            else:
                dy.append(y[i-1]-y[i])
            x.append(i/sample_number)
        utils.plot_x_y_line_chart(x, dy, "plot_human_counter_with_different_threshold_new.png")

    def trans_alphapose_2_xinggan_input(self, threshold=0.35):
        # 先生成query对应的annotation，并且把query中的图片复制过来，命名改为jpg，并调整长宽比以及keypoint所在的位置
        find_person_image_dir = {}
        query_image_path = "BikePerson-700/BikePerson-700-xinggan/test"
        utils.makedir_from_name_list([query_image_path])
        annotations_file = open("BikePerson-700/BikePerson-700-xinggan/market-annotation-test.csv", "w")
        annotations_file.write("name:keypoints_y:keypoints_x\n")
        for annotation in self.annotations_list:
            image_name = annotation["image_id"]
            image_path = self.img_name_path_dir[image_name]
            if image_path.find("query") != -1:
                find_person_image_dir[image_name] = 1
                # 将query中的图片调整尺寸并复制过来
                cv_image = cv2.imread(image_path)
                # print(cv_image.shape)  # (289, 127, 3)
                image_shape = cv_image.shape
                image_shape_new = (64, 128)
                cv_image = cv2.resize(cv_image, image_shape_new)
                cv2.imwrite(os.path.join(query_image_path, image_name), cv_image)

                annotations_file.write(image_name + ": ")
                body_keypoints = annotation["keypoints"]
                # 更新keypoints的位置，需要将老的图片坐标系中的图片映射到新的坐标系上
                for i in range(17):
                    body_keypoints[i*3] = body_keypoints[i*3] * 64 / image_shape[1]
                    body_keypoints[i*3 + 1] = body_keypoints[i*3 + 1] * 128 / image_shape[0]
                # 可视化
                # for i in range(17):
                #     cv2.circle(cv_image, (int(body_keypoints[i * 3]), int(body_keypoints[i * 3 + 1])), 3,
                #                color=(0, 0, 255))
                #     cv2.imshow("img", cv_image)
                #     cv2.waitKey(0)
                # 增加neck
                body_keypoints.append((body_keypoints[5 * 3] + body_keypoints[6 * 3]) / 2)
                body_keypoints.append((body_keypoints[5 * 3 + 1] + body_keypoints[6 * 3 + 1]) / 2)
                body_keypoints.append(min(body_keypoints[5 * 3 + 2], body_keypoints[5 * 3 + 2]))  # 置信度
                point_dir = {}
                for i in range(18):
                    confi = body_keypoints[i*3+2]
                    if confi > threshold:
                        px, py = int(body_keypoints[i*3]), int(body_keypoints[i*3+1])
                    else:
                        px, py = -1, -1
                    point_dir[self.keypoint_name_table_alphapose[i]] = {"x": px, "y": py}
                keypoint_x_list, keypoint_y_list = [], []
                for part in self.keypoint_name_table_openpose:
                    point = point_dir[part]
                    keypoint_x_list.append(point["x"])
                    keypoint_y_list.append(point["y"])
                # 可能是横纵坐标反了，这里调换一下
                annotations_file.write(str(keypoint_y_list) + ": ")
                annotations_file.write(str(keypoint_x_list) + "\n")

        # 再生成query对应的pair
        origin_query_image_path = os.path.join(self.origin_img_folder_path, "query")
        pairs_file = open("BikePerson-700/BikePerson-700-xinggan/market-pairs-test.csv", "w")
        pairs_file.write("from,to\n")
        query_image_name_list = os.listdir(origin_query_image_path)
        ct = 0
        for i in range(len(query_image_name_list)):
            if i % 2 == 0:
                if query_image_name_list[i] in find_person_image_dir \
                        and query_image_name_list[i + 1] in find_person_image_dir:
                    pairs_file.write(query_image_name_list[i] + "," + query_image_name_list[i + 1] + "\n")
            else:
                if query_image_name_list[i] in find_person_image_dir \
                        and query_image_name_list[i - 1] in find_person_image_dir:
                    pairs_file.write(query_image_name_list[i - 1] + "," + query_image_name_list[i] + "\n")
            ct = ct + 1
            utils.progress_bar(ct, len(query_image_name_list))

    def random_get_images(self, num, image_path_list=None, seed=0):
        # 如果输入的image_path_list为空的话，就去取全部的图片进来，如果不为空，按照要求随机
        if image_path_list is None:
            image_path_list = []
            for subfolder_name in self.subfolder_list:
                for image_name in os.listdir(os.path.join(self.origin_img_folder_path, subfolder_name)):
                    image_path_list.append(os.path.join(self.origin_img_folder_path, subfolder_name, image_name))
            print(len(image_path_list))
        random.seed(seed)
        sample_image_path_list = random.sample(image_path_list, num)
        sample_image_path_file = open("txt/market1501_sample_image_path_" + str(num) + "file.txt", "w")
        ct = 0
        for sample_image_path in sample_image_path_list:
            sample_image_path_file.write(sample_image_path + "\n")
            ct = ct + 1
            utils.progress_bar(ct, num)
        return image_path_list

'''
    def random_create_xinggan_input_from_images_poses(self, pose_list, sample_num=12000, seed=0):
        output_path = os.path.join(self.dataset_folder_path, "Market1501-sample" + str(sample_num))
        # 对于随机得到的图片，它不一定有Pose，所以先从所有的图片中，筛选得到一部分有Pose的图片
        image_path_list = self.get_image_path_with_pose_threshold(0.23)
        sample_image_path_list = []

        if not os.path.exists("txt/market1501_sample_image_path_" + str(sample_num) + "file.txt"):
            sample_image_path_list = self.random_get_images(sample_num, image_path_list, seed)
        else:
            sample_image_path_file = open("txt/market1501_sample_image_path_" + str(sample_num) + "file.txt", "r")
            for line in sample_image_path_file:
                sample_image_path_list.append(line.split("\n")[0])

        # 选好的Pose放在pose_list中，随机得到pose的顺序，存放在pose_number_list中
        pose_number_list = []
        random.seed(seed)
        for i in range(sample_num):
            pose_number_list.append(random.randint(0, len(pose_list)-1))

        # 创建一个对应的文件目录
        output_image_dir = os.path.join(output_path, "test")
        utils.makedir_from_name_list([output_path, output_image_dir])

        # 然后给每个图片，与其将要生成的pose对应起来
        # 先文件移动过来
        # 标注的文件中，需要有哪些文件的标注？
        # 旧的图片的标注，和新生成的图片的标注
        # 旧的标注好说，直接放进去，新生成的标注怎么办？
        # 所以需要一个旧的图片名与新生成的图片名对应的映射(在文件名之前加1)，这个图片可以是全黑的
        black_image = cv2.imread("plot/black.png")


        annotations_file = open(os.path.join(output_path, "market-annotation-test.csv"), "w")
        annotations_file.write("name:keypoints_y:keypoints_x\n")

        for pose_number in pose_number_list:
            annotation = pose_list[pose_number]
            image_name = annotation["image_id"]
            image_path = self.img_name_path_dir[image_name]
            # 将query中的图片调整尺寸并复制过来
            cv_image = cv2.imread(image_path)
            # print(cv_image.shape)  # (289, 127, 3)
            image_shape = cv_image.shape
            image_shape_new = (64, 128)
            cv_image = cv2.resize(cv_image, image_shape_new)
            cv2.imwrite(os.path.join(output_image_dir, image_name), cv_image)

            annotations_file.write(image_name + ": ")
            body_keypoints = annotation["keypoints"]
            # 更新keypoints的位置，需要将老的图片坐标系中的图片映射到新的坐标系上
            for i in range(17):
                body_keypoints[i * 3] = body_keypoints[i * 3] * 64 / image_shape[1]
                body_keypoints[i * 3 + 1] = body_keypoints[i * 3 + 1] * 128 / image_shape[0]

            body_keypoints.append((body_keypoints[5 * 3] + body_keypoints[6 * 3]) / 2)
            body_keypoints.append((body_keypoints[5 * 3 + 1] + body_keypoints[6 * 3 + 1]) / 2)
            body_keypoints.append(min(body_keypoints[5 * 3 + 2], body_keypoints[5 * 3 + 2]))  # 置信度
            point_dir = {}
            for i in range(18):
                px, py = int(body_keypoints[i * 3]), int(body_keypoints[i * 3 + 1])
                point_dir[self.keypoint_name_table_alphapose[i]] = {"x": px, "y": py}
            keypoint_x_list, keypoint_y_list = [], []
            for part in self.keypoint_name_table_openpose:
                point = point_dir[part]
                keypoint_x_list.append(point["x"])
                keypoint_y_list.append(point["y"])
            # 可能是横纵坐标反了，这里调换一下
            annotations_file.write(str(keypoint_y_list) + ": ")
            annotations_file.write(str(keypoint_x_list) + "\n")

        # 再生成query对应的pair
        origin_query_image_path = os.path.join(self.origin_img_folder_path, "query")
        pairs_file = open("BikePerson-700/BikePerson-700-xinggan/market-pairs-test.csv", "w")
        pairs_file.write("from,to\n")
        query_image_name_list = os.listdir(origin_query_image_path)
        ct = 0
        for i in range(len(query_image_name_list)):
            if i % 2 == 0:
                if query_image_name_list[i] in find_person_image_dir \
                        and query_image_name_list[i + 1] in find_person_image_dir:
                    pairs_file.write(query_image_name_list[i] + "," + query_image_name_list[i + 1] + "\n")
            else:
                if query_image_name_list[i] in find_person_image_dir \
                        and query_image_name_list[i - 1] in find_person_image_dir:
                    pairs_file.write(query_image_name_list[i - 1] + "," + query_image_name_list[i] + "\n")
            ct = ct + 1
            utils.progress_bar(ct, len(query_image_name_list))
'''