'''
Author: Li, Yirui
Date: 2021-08-08
Description: transform human pose like OpenPose format or AlphaPose format to angle size
FilePath: /liyirui/PycharmProjects/BikePersonImageProcessing/keypoint_to_angle.py
'''

from multiprocessing import set_forkserver_preload
import os
import glob
import json
import re
import shutil
import numpy as np
import math
import imagesize  # https://github.com/shibukawa/imagesize_py

import utils

DEBUG = True
glb_dataset_path = "/home/liyirui/PycharmProjects/dataset"
glb_dataset_name = "BikePerson-700"
glb_output_path = "/home/liyirui/PycharmProjects/dataset/keypoint2angle"

class Keypoint2Angle:
    pose_dir_list = ["bounding_box_pose_train", "bounding_box_pose_test", "query_pose"]
    reid_dir_list = ["bounding_box_train", "bounding_box_test", "query"]
    openpose_kp_dir = {"nose":1, "neck":2, "right_shoulder":3, "right_elbow":4, 
                       "right_wrist":5, "left_shoulder":6, "left_elbow":7, "left_wrist":8, 
                       "right_hip":9, "right_knee":10, "right_ankle":11, "left_hip":12, 
                       "left_knee":13, "left_ankle":14, "right_eye":15, "left_eye":16, 
                       "right_ear":17, "left_ear": 18}
    alphapose_kp_list = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                         "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                         "left_wrist", "right_wrist", "left_hip", "right_hip",
                         "left_knee", "right_knee", "left_ankle", "right_ankle", "neck"]
    
    def __init__(self, dataset_path, dataset_name, output_path, format='openpose'):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.format = format
        self.output_path = output_path
        if os.path.exists(self.output_path):
            os.remove(self.output_path + ".zip")
        else:
            os.mkdir(self.output_path)
        

    def openpose_dataloader(self):
        '''
        output:
            image_name = []
            pose_data_dir = [{'json_path': str,
                              'image_path': str,
                              'image_size': {'width': int, 'height': int},
                              'pose': [float []] },
                            ]
        '''
        '''
        Example of openpose json
        {"version":1.3,
         "people":[{"person_id":[-1],
                    "pose_keypoints_2d":[63.5233,36.5017,0.778086,...],
                    "face_keypoints_2d":[],
                    "hand_left_keypoints_2d":[],
                    "hand_right_keypoints_2d":[],
                    "pose_keypoints_3d":[],
                    "face_keypoints_3d":[],
                    "hand_left_keypoints_3d":[],
                    "hand_right_keypoints_3d":[]
                    }]
        }
        '''
        pose_data_dir = {}
        image_name_list = []
        for index in range(3):
            # 在debug模式下，只处理query中的图片
            if DEBUG:
                index = 2
            pose_folder_path = os.path.join(self.dataset_path, self.dataset_name, self.pose_dir_list[index])
            pose_json_files = glob.glob(pose_folder_path + "/*_keypoints.json")
            total_len = len(pose_json_files)
            partition = 0
            for pose_json_file_path in pose_json_files:
                image_name_pattern = re.compile(r'\/([0-9]+_c[0-9]+_[a-z]+[0-9]+)_keypoints.json')
                image_name = image_name_pattern.search(pose_json_file_path).groups()[0]
                image_folder_path = os.path.join(self.dataset_path, self.dataset_name, self.reid_dir_list[index])
                image_path = os.path.join(image_folder_path, image_name + ".png")
                image_width, image_height = imagesize.get(image_path)
                pose_json_file = open(pose_json_file_path, "r")
                pose = json.load(pose_json_file)
                pose_json_file.close()
                image_name_list.append(image_name)
                pose_data_dir[image_name] = {'json_path': pose_json_file_path, 
                                             'image_path': image_path,
                                             'image_size': {'width': image_width,
                                                            'height': image_height},
                                             'pose': []
                                            }
                for i in range(0, len(pose['people'])):
                    pose_data_dir[image_name]['pose'].append(pose['people'][i]['pose_keypoints_2d'])
                partition = partition + 1
                utils.progress_bar(partition, total_len)
            
            if DEBUG:
                break
        return image_name_list, pose_data_dir

    def calculator(self, pose, image_size, part_name="shoulder", 
                    confi_threshold=0.7, hw_ratio_threshold=1.3):
        '''
        input:
            pose -- a float array
            image_size -- a dic {'width': int, 'height': int}
        return:
            theta -- a float varible
            confi -- a float varible
        '''
        # 先用长宽比判断，低于阈值，则判断为侧视图，直接返回
        if image_size['height'] / image_size['width'] < hw_ratio_threshold:
            return 180, 0
        
        l_index = self.openpose_kp_dir["left_"+part_name]-1
        r_index = self.openpose_kp_dir["right_"+part_name]-1
        l_point = (pose[3*l_index], pose[3*l_index+1])
        l_confi = pose[3*l_index+2]
        r_point = (pose[3*r_index], pose[3*r_index+1])
        r_confi = pose[3*r_index+2]

        v_i = np.asarray([r_point[0] - l_point[0], 
                            r_point[1] - l_point[1]])
        v_vetical = np.asarray([0, 1])

        # 置信度如果低于阈值，直接根据向量的x坐标的正负返回结果
        confi = min(l_confi, r_confi)
        if confi < confi_threshold:
            if v_i[0] > 0: return 90, 0
            else: return 270, 0

        # 置信度和长宽比都高于阈值
        if v_i[0] == 0 and v_i[1] > 0: theta = 0.0
        elif v_i[0] == 0 and v_i[1] < 0: theta = 180.0
        else:
            # 弧度制
            theta = math.acos( np.dot(v_i, v_vetical) 
                                / (np.linalg.norm(v_i, ord=2) * np.linalg.norm(v_vetical, ord=2)))
            # 角度制
            theta = theta * 180 / np.pi
            if v_i[0] > 0: theta
            else: theta = 360.0-theta

        return theta, confi

    def run(self, threshold=15):
        image_name_list, pose_data_dir = self.openpose_dataloader()
        for image_name in image_name_list:
            '''
            pose_data = { 'json_path': str,
                          'image_path': str,
                          'image_size': {'width': int, 'height': int},
                          'pose': [float []]
                        }
            '''
            pose_data = pose_data_dir[image_name]
            [front_dir, back_dir, side_dir] = utils.makedir_from_name_list([os.path.join(self.output_path, "front"),
                                                        os.path.join(self.output_path, "back"),
                                                        os.path.join(self.output_path, "side")])

            # 暂时仅处理图片中检测到一个行人的
            if len(pose_data['pose']) == 1:
                theta, confi = self.calculator(pose_data['pose'][0], pose_data['image_size'])
                dst_image_name = "c" + str(confi) + "_t" + str(theta) + "_" + image_name + ".png"
                if theta >= 270-threshold and theta <= 270+threshold:
                    dst_path = os.path.join(front_dir, dst_image_name)
                elif theta >= 90-threshold and theta <= 90+threshold:
                    dst_path = os.path.join(back_dir, dst_image_name)
                else:
                    dst_path = os.path.join(side_dir, dst_image_name)
                shutil.copyfile(pose_data['image_path'], dst_path)
                

if __name__ == "__main__":
    kp2a = Keypoint2Angle(glb_dataset_path, glb_dataset_name, glb_output_path)
    kp2a.run()
    os.system("zip -qr " + glb_output_path + ".zip " + glb_output_path)
    