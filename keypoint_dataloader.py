'''
Author: Li, Yirui
Date: 2021-09-12
Description: 
FilePath: /liyirui/PycharmProjects/BikePersonImageProcessing/keypoint_dataloader.py
'''

# public lib
import os
import imagesize # https://github.com/shibukawa/imagesize_py
import json
import glob
import re

# my lib
import utils

pose_dir_list = ["bounding_box_pose_train", "bounding_box_pose_test", "query_pose"]
reid_dir_list = ["bounding_box_train", "bounding_box_test", "query"]

def openpose_file_loader(image_name, pose_folder_path, image_folder_path):
    image_path = os.path.join(image_folder_path, image_name + ".png")
    rendered_image_path = os.path.join(pose_folder_path, image_name + "_rendered.png")
    heatmaps_path = os.path.join(pose_folder_path, image_name + "_pose_heatmaps.png")
    pose_json_file_path = os.path.join(pose_folder_path, image_name + "_keypoints.json")
    if not os.path.exists(image_path):
        image_path = os.path.join(image_folder_path, image_name + ".jpg")
    image_width, image_height = imagesize.get(image_path)
    pose_json_file = open(pose_json_file_path, "r")
    pose = json.load(pose_json_file)
    pose_json_file.close()
    pose_list = []

    for i in range(0, len(pose['people'])):
        pose_list.append(pose['people'][i]['pose_keypoints_2d'])

    return {'json_path': pose_json_file_path, 
            'image_name': image_name,
            'image_path': image_path,
            'image_size': {'width': image_width,
                            'height': image_height},
            'rendered_image_path': rendered_image_path,
            'heatmaps_path': heatmaps_path,
            'pose': pose_list
            }

class KeypointDataLoader:
    
    def __init__(self, dataset_path, dataset_name, pose_dir_list=pose_dir_list, reid_dir_list=reid_dir_list):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.pose_dir_list = pose_dir_list
        self.reid_dir_list = reid_dir_list
    
    def openpose_dataloader(self):
        '''
        detail:
            read openpose format data from file
        return:
            image_name = []
            pose_data_dir = {'image_name': {'json_path': str,
                                            'image_name': str,
                                            'image_path': str,
                                            'image_size': {'width': int, 'height': int},
                                            'rendered_image_path': str,
                                            'heatmaps_path': str,
                                            'pose': [float []]} }
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
        for i in range(len(self.pose_dir_list)):
            pose_folder_path = os.path.join(self.dataset_path, self.dataset_name, self.pose_dir_list[i])
            pose_json_files = glob.glob(pose_folder_path + "/*_keypoints.json")
            image_folder_path = os.path.join(self.dataset_path, self.dataset_name, self.reid_dir_list[i])
            total_len = len(pose_json_files)
            partition = 0
            for pose_json_file_path in pose_json_files:
                image_name_pattern = re.compile(r'\/([0-9]+_c[0-9]+_[a-z,0-9]+)_keypoints.json')
                image_name = image_name_pattern.search(pose_json_file_path).groups()[0]
                image_name_list.append(image_name)                
                pose_data_dir[image_name] = openpose_file_loader(image_name, pose_folder_path, image_folder_path)
                
                partition = partition + 1
                utils.progress_bar(partition, total_len)

        return image_name_list, pose_data_dir