import os
import utils
import cv2
import imagesize
import json
import numpy as np


class KeyPointDetection:
    dataset_num_str = ""
    dataset_folder_path = ""
    pose_img_folder_path = ""
    origin_img_folder_path = ""
    all_img_name_list = []
    all_img_path_list = []
    annotations_list = []
    img_name_path_dir = {}

    def __init__(self, dataset_num=700):
        self.dataset_num_str = str(dataset_num)
        self.dataset_folder_path = "BikePerson-" + self.dataset_num_str
        self.pose_img_folder_path = os.path.join(self.dataset_folder_path,
                                                 "BikePerson-" + self.dataset_num_str + "-pose")
        self.origin_img_folder_path = os.path.join(self.dataset_folder_path,
                                              "BikePerson-" + self.dataset_num_str + "-origin")
        self.json_path_list = ["alphapose-query-results.json", "alphapose-test-results.json", "alphapose-train-results.json"]
        self.all_img_name_list = utils.get_all_file_name_in_reid_path_format(self.origin_img_folder_path)
        self.all_img_path_list = utils.get_all_file_path_in_reid_path_format(self.origin_img_folder_path)
        self.annotations_list = []
        for json_path in self.json_path_list:
            full_json_path = os.path.join(self.pose_img_folder_path, json_path)
            self.annotations_list = self.annotations_list + utils.read_annotations_list_from_json(full_json_path)
        self.img_name_path_dir = {}
        for i in range(len(self.all_img_name_list)):
            self.img_name_path_dir[self.all_img_name_list[i]] = self.all_img_path_list[i]

    def show_keypoint_on_image(self):
        for ct in range(len(self.annotations_list)):
            image_name = self.annotations_list[ct]["image_id"]
            img = cv2.imread(self.img_name_path_dir[image_name])

            keypoint_list = self.annotations_list[ct]["keypoints"]
            for i in range(17):
                cv2.circle(img, (int(keypoint_list[i * 3]), int(keypoint_list[i * 3 + 1])), 3, color=(255, 255, 255))
                cv2.imshow("img", img)
                cv2.waitKey(0)

    # 当前函数缺少置信度的阈值，所以结果可能有点乱
    def get_black_background_picture(self, threshold=0.5, out_img_size=(128, 128)):
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

        black_bg_folder_path = utils.makedir_from_name_list([os.path.join(self.dataset_folder_path,
                                                                          "BikePerson-700-pose-black-bg")])
        ct = 0
        image_map = {}
        for anno in self.annotations_list:
            image_name = anno["image_id"]
            image_path = self.img_name_path_dir[image_name]
            [image_width, image_height] = imagesize.get(image_path)

            # 初始化大小，默认初始化为黑色的背景
            img = np.zeros((image_height, image_width, 3), np.uint8)
            # 这里需要给keypoints增加一个脖子，它的位置是左右肩膀的连线中点
            body_keypoints = anno["keypoints"]
            body_keypoints.append((body_keypoints[5*3] + body_keypoints[6*3])/2)
            body_keypoints.append((body_keypoints[5 * 3 + 1] + body_keypoints[6 * 3 + 1]) / 2)
            body_keypoints.append(min(body_keypoints[5 * 3 + 2], body_keypoints[5 * 3 + 2]))    # 置信度

            # 绘制关键点
            for i in range(18):
                cv2.circle(img, (int(body_keypoints[i*3]), int(body_keypoints[i*3 + 1])), 2, color=p_color[i], thickness=-1)
            # 绘制链接线
            for i in range(15):
                point1 = (int(body_keypoints[l_pair[i][0]*3]), int(body_keypoints[l_pair[i][0]*3 + 1]))
                point2 = (int(body_keypoints[l_pair[i][1]*3]), int(body_keypoints[l_pair[i][1]*3 + 1]))
                cv2.line(img, point1, point2, color=line_color[i], thickness=3)
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