'''
Author: Li, Yirui
Date: 2021-08-08
Description: transform human pose like OpenPose format to angle size
FilePath: /liyirui/PycharmProjects/BikePersonImageProcessing/keypoint_to_angle.py
'''
# public lib
import os
import glob
import json
import re
import shutil
from sys import float_repr_style
from types import LambdaType
import numpy as np
import math
import imagesize  # https://github.com/shibukawa/imagesize_py
import cv2
import random

# my lib
import utils
from mylog import MyLog


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


class Keypoint2Angle:
    pose_dir_list = ["bounding_box_pose_train", "bounding_box_pose_test", "query_pose"]
    reid_dir_list = ["bounding_box_train", "bounding_box_test", "query"]
    openpose_kp_dir = {"nose":1, "neck":2, "right_shoulder":3, "right_elbow":4, 
                       "right_wrist":5, "left_shoulder":6, "left_elbow":7, "left_wrist":8, 
                       "right_hip":9, "right_knee":10, "right_ankle":11, "left_hip":12, 
                       "left_knee":13, "left_ankle":14, "right_eye":15, "left_eye":16, 
                       "right_ear":17, "left_ear": 18}
    
    def __init__(self, dataset_path, dataset_name, label_path, output_folder, format='openpose'):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.label_path = label_path
        self.format = format
        self.output_folder = output_folder
        self.output_path = os.path.join(dataset_path, output_folder)
        self.query_angle_label_dir = {}
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        if os.path.exists(self.output_path + ".zip"):
            os.remove(self.output_path + ".zip")
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        
        # initialize the query angle view label
        angle_view_folder_list = ["back_view", "front_view", "left_view", "right_view"]
        for folder_name in angle_view_folder_list:
            folder_path = os.path.join(self.label_path, folder_name)
            for image_name in os.listdir(folder_path):
                self.query_angle_label_dir[image_name] = folder_name

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
        for index in range(3):
            if DEBUG:   # In debug mode，only processing image in 'query'
                index = 2
            pose_folder_path = os.path.join(self.dataset_path, self.dataset_name, self.pose_dir_list[index])
            pose_json_files = glob.glob(pose_folder_path + "/*_keypoints.json")
            total_len = len(pose_json_files)
            partition = 0
            for pose_json_file_path in pose_json_files:
                image_name_pattern = re.compile(r'\/([0-9]+_c[0-9]+_[a-z,0-9]+)_keypoints.json')
                image_name = image_name_pattern.search(pose_json_file_path).groups()[0]
                rendered_image_path = os.path.join(pose_folder_path, image_name + "_rendered.png")
                heatmaps_path = os.path.join(pose_folder_path, image_name + "_pose_heatmaps.png")
                image_folder_path = os.path.join(self.dataset_path, self.dataset_name, self.reid_dir_list[index])
                image_path = os.path.join(image_folder_path, image_name + ".png")
                if not os.path.exists(image_path):
                    image_path = os.path.join(image_folder_path, image_name + ".jpg")
                image_width, image_height = imagesize.get(image_path)
                pose_json_file = open(pose_json_file_path, "r")
                pose = json.load(pose_json_file)
                pose_json_file.close()
                image_name_list.append(image_name)

                pose_data_dir[image_name] = {'json_path': pose_json_file_path, 
                                             'image_name': image_name,
                                             'image_path': image_path,
                                             'image_size': {'width': image_width,
                                                            'height': image_height},
                                             'rendered_image_path': rendered_image_path,
                                             'heatmaps_path': heatmaps_path,
                                             'pose': []
                                            }
                for i in range(0, len(pose['people'])):
                    pose_data_dir[image_name]['pose'].append(pose['people'][i]['pose_keypoints_2d'])
                
                partition = partition + 1
                utils.progress_bar(partition, total_len)
            
            if DEBUG:
                break
        return image_name_list, pose_data_dir

    def get_standard_label_for_debug_mode(self, image_name):
        return self.query_angle_label_dir[image_name + ".png"]

    def vector_included_angle(self, v1, v2):
        '''
        detail:
            calculate vector included angle of v1 and v2
        args:
            v1, v2 = [x1,x2], [y1, y2]
        return:
            theta
        '''
        if v1[0] == 0 and v1[1] > 0: theta = 0.0
        elif v1[0] == 0 and v1[1] < 0: theta = 180.0
        else:
            # 弧度制theta
            theta = math.acos( np.dot(v1, v2) 
                                / (np.linalg.norm(v1, ord=2) * np.linalg.norm(v2, ord=2)))
            # 角度制theta
            theta = theta * 180 / np.pi
            if v1[0] > 0: theta
            else: theta = 360.0-theta
        return theta

    def get_keypoint_position(self, pose, keypoint_name):
        '''
        detail:
            get keypoint position and confidence from given pose array
        args:
            pose -- float[], a array of pose info
            keypoint_name -- str, a name of keypoint which is a key in self.openpose_kp_dir 
        return:
            pnt -- (float, float), point position 
            confi -- float, confidence of this keypoint 
        '''
        pnt = (pose[3*(self.openpose_kp_dir[keypoint_name]-1)], 
               pose[3*(self.openpose_kp_dir[keypoint_name]-1) + 1])
        confi = pose[3*(self.openpose_kp_dir[keypoint_name]-1) + 2]
        
        return pnt, confi

    def get_np_vector_from_pnt(self, p1, p2):
        '''
        detail:
            return a numpy array format vector, point from point2 to point1
        input:
            v1, v2 -- (float, float)
        return:
            numpy array
        '''
        return np.asarray([p2[0]-p1[0], p2[1]-p1[1]])

    def shoulder_hip_calculator(self, pose):
        '''
        detail:
            Calculating the angle between the vector which conect left and right shoulder,
            and the vector which in the middle of two vetors, first connect neck and left hip, 
            second connect neck and right hip.
        input:
            pose -- a float array
        return:
            theta -- a float varible
            confi -- a float varible 
        '''
        l_shu_pnt, l_shu_confi = self.get_keypoint_position(pose, "left_shoulder")
        r_shu_pnt, r_shu_confi = self.get_keypoint_position(pose, "right_shoulder")
        l_hip_pnt, l_hip_confi = self.get_keypoint_position(pose, "left_hip")
        r_hip_pnt, r_hip_confi = self.get_keypoint_position(pose, "right_hip")
        neck_pnt, neck_confi = self.get_keypoint_position(pose, "neck")
        
        confi = min(l_shu_confi, r_shu_confi, l_hip_confi, r_hip_confi, neck_confi)

        v_i = self.get_np_vector_from_pnt(l_shu_pnt, r_shu_pnt)
        v_left_down = self.get_np_vector_from_pnt(l_hip_pnt, neck_pnt)
        v_right_down = self.get_np_vector_from_pnt(r_hip_pnt, neck_pnt)
        v_down = v_left_down + v_right_down

        return self.vector_included_angle(v_i, v_down), confi

    def shoulder_calculator(self, pose):
        '''
        detail:
            Calculating the angle between the vector which conecct left and right shoulder, 
            and the vector which from top to buttom.
        input:
            pose -- a float array
        return:
            theta -- a float varible
            confi -- a float varible
        '''
        
        l_point, l_confi = self.get_keypoint_position(pose, "left_shoulder")
        r_point, r_confi = self.get_keypoint_position(pose, "right_shoulder")
        confi = min(l_confi, r_confi)

        v_i = self.get_np_vector_from_pnt(l_point, r_point)
        v_vetical = np.asarray([0, 1])

        return self.vector_included_angle(v_i, v_vetical), confi
    
    def towards_of_side_view(self, pose):
        '''
        detail:
            Result is Left or Right View.
            Distinguish which specific view of person in side view in BikePerson dataset.
            Use left arm, right arm, left leg, right leg this four parts.
            Choose which is the most confident, and it's direction determine this image's point view.
        input:
            pose -- float[]
        output:
            "left_view" or "right_view" -- a string
        '''
        part_confi_dir, part_vec_dir = {}, {}

        l_wrist_pnt, l_wrist_confi = self.get_keypoint_position(pose, "left_wrist")
        l_shu_pnt, l_shu_confi = self.get_keypoint_position(pose, "left_shoulder")
        l_arm_vec = self.get_np_vector_from_pnt(l_wrist_pnt, l_shu_pnt)
        part_vec_dir["left_arm"], part_confi_dir["left_arm"] = l_arm_vec, min(l_shu_confi, l_wrist_confi)

        r_wrist_pnt, r_wrist_confi = self.get_keypoint_position(pose, "right_wrist")
        r_shu_pnt, r_shu_confi = self.get_keypoint_position(pose, "right_shoulder")
        r_arm_vec = self.get_np_vector_from_pnt(r_wrist_pnt, r_shu_pnt)
        part_vec_dir["right_arm"], part_confi_dir["right_arm"] = r_arm_vec, min(r_shu_confi, r_wrist_confi)

        l_knee_pnt, l_knee_confi = self.get_keypoint_position(pose, "left_knee")
        l_hip_pnt, l_hip_confi = self.get_keypoint_position(pose, "left_hip")
        l_leg_vec = self.get_np_vector_from_pnt(l_knee_pnt, l_hip_pnt)
        part_vec_dir["left_leg"], part_confi_dir["left_leg"] = l_leg_vec, min(l_knee_confi, l_hip_confi) 

        r_knee_pnt, r_knee_confi = self.get_keypoint_position(pose, "right_knee")
        r_hip_pnt, r_hip_confi = self.get_keypoint_position(pose, "right_hip")
        r_leg_vec = self.get_np_vector_from_pnt(r_knee_pnt, r_hip_pnt)
        part_vec_dir["right_leg"], part_confi_dir["right_leg"] = r_leg_vec, min(r_knee_confi, r_hip_confi)

        # choose the most confience part from two arms and two legs
        most_confient = sorted(part_confi_dir.items(), key=lambda item:item[1], reverse=True)[0][0]
        if part_vec_dir[most_confient][0] >= 0: 
            return "right_view"
        else: 
            return "left_view"


    def make_args(self, args):
        '''
        detail:
            make sure every needed key is exists, if not, give a default value
        '''
        if 'angle_threshold' not in args:
            args['angle_threshold'] = 15
        if 'hw_ratio_threshold' not in args: 
            args['hw_ratio_threshold']=1.3
        if 'confi_threshold' not in args:
            args['confi_threshold']=0.7
        if 'calculator' not in args:
            args['calculator'] = "shoulder_calculator"
        if 'mode' not in args:
            args['mode'] = MODE
        if 'zip' not in args:
            args['zip'] = False
        if 'output_label' not in args:
            args['output_label'] = False
        if 'copy_rendered_image' not in args:
            args['copy_rendered_image'] = False
        
        return args


    def run(self, args):
        # init log file
        ml = MyLog(self.output_path)
        
        # from args get arguments
        self.make_args(args)
        angle_threshold = args['angle_threshold']
        hw_ratio_threshold = args['hw_ratio_threshold']
        confi_threshold = args['confi_threshold']
        calculator = args['calculator']
        mode = args['mode']
        if_zip = args['zip']
        if_output_label = args['output_label']
        if_copy_rendered_image = args['copy_rendered_image']

        ml.write("---------------------- args ---------------------", color='g')
        args_str = utils.dic_to_string(args)
        ml.write(args_str)
        
        utils.color_print("---------------------- loading data ---------------------", color='g')

        # do keypoint to angle
        image_name_list, pose_data_dir = self.openpose_dataloader()
        ct_total, ct_0 = 0, 0
        ct_back, ct_front, ct_side, ct_left, ct_right = 0, 0, 0, 0, 0
        ct_back_correct, ct_front_correct, ct_side_correct, ct_left_correct, ct_right_correct = 0, 0, 0, 0, 0
        image_name_angle_view_dir = {}

        utils.color_print("---------------------- calculation ---------------------", color='g')
        for image_name in image_name_list:
            '''
            format of 'pose_data' can be found in openpose_dataloader()
            '''
            pose_data = pose_data_dir[image_name]
            front_dir, back_dir, side_dir, left_dir, right_dir = os.path.join(self.output_path, "front_view"), \
                                                                 os.path.join(self.output_path, "back_view"),  \
                                                                 os.path.join(self.output_path, "side_view"),  \
                                                                 os.path.join(self.output_path, "left_view"),  \
                                                                 os.path.join(self.output_path, "right_view")
            
            '''
            if TEST_GET_MAIN_CHARACTER:
                if len(pose_data['pose']) > 1:
                    pose_data = select_main_character(pose_data=pose_data)
                    image = cv2.imread(pose_data["rendered_image_path"])
                    image = paint_keypoints_on_image(pose_data['pose'], image)
                    cv2.imwrite(os.path.join("test", pose_data['image_name']+".png"), image)
                continue
            '''

            if len(pose_data['pose']) == 0: 
                ct_0 = ct_0 + 1
            # 如果图片中检测得到多个人，可能是多人骑一辆车，也可能是存在遮挡，则先选取主要的那个人
            if len(pose_data['pose']) > 1: 
                pose_data = select_main_character(pose_data=pose_data)

            # 1. 先用长宽比判断, 长宽比小于阈值的图片直接作为侧视图
            #                  长宽比大于阈值的图片在后续的判断中再决定是那种视图
            image_size = pose_data['image_size']
            if image_size['height'] / image_size['width'] < hw_ratio_threshold:
                theta, confi = 180, 1

            # 2. 对长宽比粗分完以后，没有被归为侧视图的图像，使用shoulder_hip_calculator来判断属于那种视图
            # 先处理检测到人的图像
            elif len(pose_data['pose']) > 0:
                if calculator == "shoulder_hip_calculator":
                    theta, confi = self.shoulder_hip_calculator(pose_data['pose'][0])
                else:
                    theta, confi = self.shoulder_calculator(pose_data['pose'][0])
                # 3. 根据置信度分类（前提是长宽比这个阈值存在），
                #    如果置信度如果低于阈值，直接根据判断角度是否是大于180度，划分视角
                if hw_ratio_threshold > 0 and confi < confi_threshold:
                    if theta > 180: 
                        theta, confi = 270, 0
                    else: 
                        theta, confi = 90, 0
            else:
                # 对应，没有检测到人的图像，则随机分配前后视角
                if random.randint(1,2) == 1:
                    theta, confi = 270, 0
                else: 
                    theta, confi = 90, 0
                
            # 3. 对图片按照之前计算得到的视角进行分类
            dst_image_name = "t" + str(theta) + "_c" + str(confi) + "_" + image_name + ".png"
            # front_view
            if theta >= 270-angle_threshold and theta <= 270+angle_threshold:
                dst_path = os.path.join(front_dir, dst_image_name)
                ct_front = ct_front + 1
                if DEBUG and self.get_standard_label_for_debug_mode(image_name) == "front_view":
                    ct_front_correct = ct_front_correct + 1
                image_name_angle_view_dir[image_name] = "front_view"
            # back_view
            elif theta >= 90-angle_threshold and theta <= 90+angle_threshold:
                dst_path = os.path.join(back_dir, dst_image_name)
                ct_back = ct_back + 1
                if DEBUG and self.get_standard_label_for_debug_mode(image_name) == "back_view":
                    ct_back_correct = ct_back_correct + 1
                image_name_angle_view_dir[image_name] = "back_view"
            # side_view
            else:
                # 如果是bikeperson数据集，则判断人的左右朝向
                if mode == 'bikeperson_4_view':
                    angle_view = ""
                    # 如果不存在人，则随机分配左右视角
                    if len(pose_data['pose']) == 0:
                        if random.randint(1,2) == 1:
                            angle_view = "left_view"
                        else:
                            angle_view = "right_view"
                    else:
                        angle_view = self.towards_of_side_view(pose_data['pose'][0])

                    if angle_view == "left_view":
                        dst_path = os.path.join(left_dir, dst_image_name)
                        ct_left = ct_left + 1
                        if DEBUG and self.get_standard_label_for_debug_mode(image_name) == "left_view":
                            ct_left_correct = ct_left_correct + 1
                        image_name_angle_view_dir[image_name] = "left_view"
                    else:
                        dst_path = os.path.join(right_dir, dst_image_name)
                        ct_right = ct_right + 1
                        if DEBUG and self.get_standard_label_for_debug_mode(image_name) == "right_view":
                            ct_right_correct = ct_right_correct + 1
                        image_name_angle_view_dir[image_name] = "right_view"
                else:
                    dst_path = os.path.join(side_dir, dst_image_name)
                    ct_side = ct_side + 1
                    if DEBUG:
                        view = self.get_standard_label_for_debug_mode(image_name)
                        if view == "left_view" or view == "right_view":
                            ct_side_correct = ct_side_correct + 1
                    image_name_angle_view_dir[image_name] = "side_view"
            
            if if_copy_rendered_image:
                utils.makedir_from_name_list([front_dir, back_dir, side_dir, left_dir, right_dir])
                # display pose rendered image rather than original image.
                shutil.copyfile(pose_data['rendered_image_path'], dst_path)

            ct_total = ct_total + 1
            utils.progress_bar(ct_total, len(image_name_list))
        
        ml.write("---------------------- statistic ---------------------", color='g')
        ml.write("image which not detected person number: \t" + str(ct_0) + "/" + str(ct_total) + "=" + str(round(ct_0/ct_total, 4)))
        ml.write("---------------------- result ---------------------", color='g')
        ml.write("back view: " + str(ct_back))
        if DEBUG:
            ml.write("correct rate: " + str(ct_back_correct) + "/" + str(ct_back) + "=" + str(round(ct_back_correct/ct_back)) + "\n")
        ml.write("front view: " + str(ct_front))
        if DEBUG:
            ml.write("correct rate: " + str(ct_front_correct) + "/" + str(ct_front) + "=" + str(round(ct_front_correct/ct_front)) +"\n")
        if mode == "bikeperson_4_view":
            ml.write("left view: " + str(ct_left))
            if DEBUG:
                ml.write("correct rate: " + str(ct_left_correct) + "/" + str(ct_left) + "=" + str(round(ct_left_correct/ct_left)) +"\n")
            ml.write("right view: " + str(ct_right))
            if DEBUG:
                ml.write("correct rate: " + str(ct_right_correct) + "/" + str(ct_right) + "=" + str(round(ct_right_correct/ct_right)) +"\n")
        elif mode == "bikeperson_3_view":
            ml.write("side view: " + str(ct_side))
            if DEBUG:
                ml.write("correct rate: " + str(ct_side_correct) + "/" + str(ct_side) + "=" + str(round(ct_side_correct/ct_side)) +"\n")
        ml.write("total: " + str(ct_side + ct_left + ct_right + ct_back + ct_front))
        
        if if_zip:
            ml.write("---------------------- zip ---------------------", color="g")
            os.system("cd " + self.dataset_path + "&& zip -qr " + self.output_folder + ".zip " + self.output_folder + "/")
            ml.write("ZIP finished.\nResult in " + os.path.join(self.dataset_path, self.output_folder+".zip"))

        if if_output_label:
            ml.write("---------------------- output label ---------------------", color="g")
            json_format_dic = json.dumps(image_name_angle_view_dir)
            output_file = open(os.path.join(self.output_path, "angle_label.json"), "w")
            output_file.write(json_format_dic)
            output_file.close()
            ml.write("Output label finished.\nResult in " + os.path.join(self.output_path, "angle_label.json"))
        
        ml.close()


# some global varible
DEBUG = True   # If DEBUG == True, it means only calculate query images and return accuracy
TEST_GET_MAIN_CHARACTER = False  # If this True, it will output GET_MAIN_CHARACTER method's result
MODE = 'bikeperson_3_view'  # There is two options of MODE: 'bikeperson_3_view', 'bikeperson_4_view'
'''
bikeperson_3_view: back, front, side
bikeperson_4_view: back, front, left, right
'''

if __name__ == "__main__":
    # init different varible for different mode
    if MODE == 'bikeperson_3_view' or MODE == "bikeperson_4_view":
        dataset_path = "/home/liyirui/PycharmProjects/dataset"
        dataset_name = "BikePerson-700-origin"
        output_folder = "keypoint2angle_" + MODE
        if DEBUG:
            output_folder = output_folder + "_DEBUG"
        label_path = os.path.join(dataset_path, "standard_angle_bikeperson")
        # Can refer to method: make_args() to get other arguments and their default value.
        args = {'mode': MODE, 'output_label': True}
    else:
        utils.color_print("[Error]: Uncorrect MODE value.")
        exit()
    
    kp2a = Keypoint2Angle(dataset_path, dataset_name, label_path, output_folder)
    kp2a.run(args)
    