'''
Author: Li, Yirui
Date: 2021-09-12
Description: Some util functions of keypoints.
FilePath: /liyirui/PycharmProjects/BikePersonImageProcessing/keypoint_utils.py
'''

import math
import numpy
import cv2
import imagesize

openpose_kp_dir = {"nose":0, "neck":1, "right_shoulder":2, "right_elbow":3, 
                       "right_wrist":4, "left_shoulder":5, "left_elbow":6, "left_wrist":7, 
                       "right_hip":8, "right_knee":9, "right_ankle":10, "left_hip":11, 
                       "left_knee":12, "left_ankle":13, "right_eye":14, "left_eye":15, 
                       "right_ear":16, "left_ear": 17}

# reference from this link: 
# https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/.github/media/keypoints_pose_18.png
openpose_line_pair = [(0, 1), (0, 14), (0, 15), (1, 2), (1, 5), (1, 8), (1, 11), (2, 3), (3, 4), (5, 6), (6, 7), 
                      (8, 9), (9, 10), (11, 12), (12, 13), (14, 16), (15, 17)]

'''
[pose data example]
pose_data: {'json_path': str,
            'image_name': str,
            'image_path': str,
            'image_size': {'width': int, 'height': int},
            'rendered_image_path': str,
            'heatmaps_path': str,
            'pose': [float []]}
'''

numpy.set_printoptions(threshold=numpy.inf)

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

def get_heatmaps_matrix(pose_data, visualize=False, ostu=True, binary_threshold=5):
    '''
    note:
        About heatmaps size get from openpose, if set the net resolution is 384x384, 
        you will get a 21504x384 size heatmap, which is (56x384)x384.
        56 means: 18 keypoints images + 19x2 parts images.
    detail:
        Get heatmap matrix from a heatmap of image of penpose's result.
    input:
        pose_date: refer to pose_data example in this file. 
        visualize: bool, if True write the visualizing result of this function
        ostu: bool, if True use ostu threshold as result
        binary_threshold: int, it is a threshold of binarylize the image if not use ostu.
    '''
    heatmaps_path = pose_data["heatmaps_path"]
    img_width, img_height = pose_data["image_size"]['width'], pose_data["image_size"]['height']
    hm_width, hm_height = imagesize.get(heatmaps_path)
    hm_img_width, hm_img_height = int(hm_width/56), hm_height
    heatmaps_total_image = cv2.imread(heatmaps_path, cv2.IMREAD_GRAYSCALE)
    heatmap_image = numpy.zeros((hm_img_height, hm_img_width),dtype=numpy.uint8)
    
    # conbine 19x2 parts of images together
    for i in range(19):
        # 比如 当i==0的时候，对应的是18和19张图像（从0开始编号）
        i1 = 18 + (i*2)
        i2 = 18 + (i*2 + 1)
        for j in range(hm_img_height):
            for k in range(hm_img_width):
                k1 = k + i1*hm_img_width
                k2 = k + i2*hm_img_width
                heatmap_image[j][k] = max(heatmap_image[j][k], 
                                          max(heatmaps_total_image[j][k1], heatmaps_total_image[j][k2]))
    
    # resize image
    heatmap_image = cv2.resize(heatmap_image, (img_width, img_height))
    heatmap_color = cv2.applyColorMap(heatmap_image, cv2.COLORMAP_JET)
    
    # binarylize image
    if not ostu or visualize:
        heatmap_binary = heatmap_image.copy()
        for i in range(img_height):
            for j in range(img_width):
                if heatmap_image[i][j] > 129 + binary_threshold:
                    heatmap_binary[i][j] = 255
                else: heatmap_binary[i][j] = 0
                
    # ostu threshold
    _, heatmap_otsu = cv2.threshold(heatmap_image, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    if visualize:
        heatmap_image_rgb = cv2.cvtColor(heatmap_image, cv2.COLOR_GRAY2RGB)
        heatmap_binary_rgb = cv2.cvtColor(heatmap_binary, cv2.COLOR_GRAY2RGB)
        heatmap_otsu_rgb = cv2.cvtColor(heatmap_otsu, cv2.COLOR_GRAY2RGB)
        origin_image = cv2.imread(pose_data['image_path'])
        pose_image = cv2.imread(pose_data['rendered_image_path'])

        heatmap_total = numpy.hstack((origin_image, pose_image, heatmap_image_rgb, heatmap_color, heatmap_binary_rgb, heatmap_otsu_rgb))
        print("result path: " + heatmaps_path.split(".")[0] + "_convert.png")
        cv2.imwrite(heatmaps_path.split(".")[0] + "_convert.png", heatmap_total)
    
    if ostu:
        # cv2.imshow("heatmap", heatmap_otsu)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return heatmap_otsu
    else: 
        return heatmap_binary

def get_pose_skeleton_matrix(pose_data, debug=False):
    '''
    detail:
        This method is used to get pose's line pass pixel in a image.
    input:
        pose_data: refer to pose_data example in this file. 
    return:
        matrix: a numpy format matrix
    '''
    height, width = pose_data['image_size']['height'], pose_data['image_size']['width']
    matrix = numpy.zeros((height, width))
    # let line pass pixel in numpy set to 1
    if len(pose_data['pose']) == 0: return matrix
    for line in openpose_line_pair:
        index1, index2 = line[0], line[1]
        for pose in pose_data['pose']:
            p1 = {'y': pose[3*index1 + 0], 'x': pose[3*index1 + 1], 'confi': pose[3*index1 + 2]}
            p2 = {'y': pose[3*index2 + 0], 'x': pose[3*index2 + 1], 'confi': pose[3*index2 + 2]}
            if p1['confi'] == 0 or p2['confi'] == 0: continue
            if p1['x'] == p2['x']:
                for y in range(int(min(p1['y'], p2['y'])), int(max(p1['y'], p2['y'])+1)):
                    matrix[int(p1['x'])][y] = 1
            else:
                if p1['x'] > p2['x']:
                    temp = p1
                    p1 = p2
                    p2 = temp
                k = (p2['y'] - p1['y']) / (p2['x'] - p1['x'])
                for i in range(0, int(p2['x']) - int(p1['x']) + 1):
                    x = i+int(p1['x'])
                    y = int(p1['y'] + int(i*k))
                    if x >= 0 and x < height and y >= 0 and y < width:
                        matrix[x][y] = 1   
    if debug:
        # 存成一张图像
        image = numpy.zeros((height, width),dtype=numpy.uint8) 
        for i in range(height):
            for j in range(width):
                if matrix[i][j] == 1:
                    image[i][j] = 255
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        cv2.imwrite("test_skeleton.png", image)
        
    return matrix