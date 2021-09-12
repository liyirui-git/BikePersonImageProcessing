'''
Author: Li, Yirui
Date: 2021-09-12
Description: Some util functions of keypoints.
FilePath: /liyirui/PycharmProjects/BikePersonImageProcessing/keypoint_utils.py
'''

import math
import cv2

def select_main_character(pose_data):
    '''
    detail:
        Select the main character from persons which were detected from the image.
    input:
        pose_data: {'json_path': str,
                    'image_name': str,
                    'image_path': str,
                    'image_size': {'width': int, 'height': int},
                    'rendered_image_path': str,
                    'heatmaps_path': str,
                    'pose': [float []]}
    return: 
        mc_pose_data: {'json_path': str,
                       'image_name': str,
                       'image_path': str,
                       'image_size': {'width': int, 'height': int},
                       'rendered_image_path': str,
                       'heatmaps_path': str,
                       'pose': [float []]}
    '''
    pose_list = pose_data['pose']
    image_size = pose_data['image_size']
    mid_p = (image_size['width']/2, image_size['height']/2)
    nearest_pose = []
    nearest_dist = math.sqrt(mid_p[0]*mid_p[0] + mid_p[1]*mid_p[1])
    for pose in pose_list:
        avg_p = [0, 0]
        counter = 0
        for i in range(18):
            if pose[i*3 + 2]:
                avg_p[0] = avg_p[0] + pose[i*3]
                avg_p[1] = avg_p[1] + pose[i*3 + 1]
                counter = counter + 1
        avg_p = (avg_p[0] / counter, avg_p[1] / counter)
        d_vec = (avg_p[0] - mid_p[0], avg_p[1] - mid_p[1])
        dist = math.sqrt(d_vec[0]*d_vec[0] + d_vec[1]*d_vec[1])
        if dist < nearest_dist:
            nearest_pose = pose
            nearest_dist = dist

    pose_data['pose'] = [nearest_pose]
    return pose_data


def paint_keypoints_on_image(pose_list, image, color=(255, 255, 255)):
    '''
    detail:
        paint body keypoints in the image
    input:
        pose_list: float []
        image: cv2 format image
    return:
        image: cv2 format image
    '''
    for pose in pose_list:
        for i in range(18):
            cv2.circle(image, (int(pose[i * 3]), int(pose[i * 3 + 1])), 3, color=color)
    return image