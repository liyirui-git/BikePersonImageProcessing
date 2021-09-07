'''
Author: Li, Yirui
Date: 2021-08-08
Description: transform human pose like OpenPose format to angle size
FilePath: /liyirui/PycharmProjects/BikePersonImageProcessing/keypoint_to_angle.py
'''
# public lib
from multiprocessing import set_forkserver_preload
import os
import glob
import json
import re
import shutil
from types import LambdaType
import numpy as np
import math
import imagesize  # https://github.com/shibukawa/imagesize_py
import cv2

# my lib
import utils
from mylog import MyLog

# some global varible
DEBUG = True
TEST = True
MODE = 'bikeperson'  # 'bikeperson', 'dml_ped'


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
    
    def __init__(self, dataset_path, dataset_name, output_path, format='openpose'):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.format = format
        self.output_path = output_path
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        if os.path.exists(self.output_path + ".zip"):
            os.remove(self.output_path + ".zip")
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        
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

        ml.write("---------------------- args ---------------------", color='g')
        args_str = utils.dic_to_string(args)
        ml.write(args_str)
        
        image_name_list, pose_data_dir = self.openpose_dataloader()
        ct_0, ct_1, ct_n = 0, 0, 0
        ct_back, ct_front, ct_side, ct_left, ct_right = 0, 0, 0, 0, 0
        for image_name in image_name_list:
            '''
            format of 'pose_data' can be found in openpose_dataloader()
            '''
            pose_data = pose_data_dir[image_name]
            [front_dir, back_dir, side_dir, left_dir, right_dir] \
                        = utils.makedir_from_name_list([os.path.join(self.output_path, "front_view"),
                                                        os.path.join(self.output_path, "back_view"),
                                                        os.path.join(self.output_path, "side_view"),
                                                        os.path.join(self.output_path, "left_view"),
                                                        os.path.join(self.output_path, "right_view")])

            if TEST:
                if len(pose_data['pose']) > 1:
                    pose_data = select_main_character(pose_data=pose_data)
                    image = cv2.imread(pose_data["rendered_image_path"])
                    image = paint_keypoints_on_image(pose_data['pose'], image)
                    cv2.imwrite(os.path.join("test", pose_data['image_name']+".png"), image)
                continue

            if len(pose_data['pose']) == 0:
                ct_0 = ct_0 + 1
            # 暂时仅处理图片中检测到一个行人的
            elif len(pose_data['pose']) == 1:
                # 1. 先用长宽比判断, 长宽比小于阈值的图片直接作为侧视图
                #                  长宽比大于阈值的图片在后续的判断中再决定是那种视图
                image_size = pose_data['image_size']
                if image_size['height'] / image_size['width'] < hw_ratio_threshold:
                    theta, confi = 180, 1
                else:
                    if calculator == "shoulder_hip_calculator":
                        theta, confi = self.shoulder_hip_calculator(pose_data['pose'][0])
                    else:
                        theta, confi = self.shoulder_calculator(pose_data['pose'][0])
                    # 2. 根据置信度分类（前提是长宽比这个阈值存在），如果置信度如果低于阈值，直接根据判断角度是否是大于
                    if hw_ratio_threshold > 0 and confi < confi_threshold:
                        if theta > 180: 
                            theta, confi = 270, 0
                        else: 
                            theta, confi = 90, 0
                
                # 对图片按照视角分类
                dst_image_name = "t" + str(theta) + "_c" + str(confi) + "_" + image_name + ".png"
                if theta >= 270-angle_threshold and theta <= 270+angle_threshold:
                    dst_path = os.path.join(front_dir, dst_image_name)
                    ct_front = ct_front + 1
                elif theta >= 90-angle_threshold and theta <= 90+angle_threshold:
                    dst_path = os.path.join(back_dir, dst_image_name)
                    ct_back = ct_back + 1
                else:
                    # 如果是bikeperson数据集，则判断人的左右朝向
                    if MODE == 'bikeperson':
                        if self.towards_of_side_view(pose_data['pose'][0]) == "left_view":
                            dst_path = os.path.join(left_dir, dst_image_name)
                            ct_left = ct_left + 1
                        else:
                            dst_path = os.path.join(right_dir, dst_image_name)
                            ct_right = ct_right + 1
                    else:
                        dst_path = os.path.join(side_dir, dst_image_name)
                        ct_side = ct_side + 1
                # display pose rendered image rather than original image.
                shutil.copyfile(pose_data['rendered_image_path'], dst_path)
                ct_1 = ct_1 + 1
                
            else: ct_n = ct_n + 1
        
        ml.write("---------------------- statistic ---------------------", color='g')
        ml.write("image which not detected person number: \t" + str(ct_0))
        ml.write("image which detected 1 person number: \t\t" + str(ct_1))
        ml.write("image which detected >=2 person number: \t" + str(ct_n))
        ml.write("---------------------- result ---------------------", color='g')
        ml.write("back view: " + str(ct_back) + "\nfront view: " + str(ct_front))
        if MODE == "bikeperson":
            ml.write("left view: " + str(ct_left) + "\nright view: " + str(ct_right))
        elif MODE == "dml_ped":
            ml.write("side view: " + str(ct_side))
        ml.close()

if __name__ == "__main__":
    # init different varible for different mode
    if MODE == 'bikeperson':
        dataset_path = "/home/liyirui/PycharmProjects/dataset"
        dataset_name = "BikePerson-700-origin"
        output_folder = "keypoint2angle_bikeperson"
        output_path = os.path.join(dataset_path, output_folder)
        args = {}
    elif MODE == 'dml_ped':
        dataset_path = "/home/liyirui/PycharmProjects/dataset"
        dataset_name = "dml_ped12_market"
        output_folder = "keypoint2angle_dml_ped12"
        output_path = os.path.join(dataset_path, output_folder)
        args = {'hw_ratio_threshold': 0, 
                'confi_threshold': 0.5,
                'calculator': 'shoulder_hip_calculator'}
    else:
        utils.color_print("[Error]: Uncorrect MODE value.")
        exit()
    
    kp2a = Keypoint2Angle(dataset_path, dataset_name, output_path)
    kp2a.run(args)
    utils.color_print("---------------------- zip ---------------------", color="g")
    os.system("cd " + dataset_path + "&& zip -qr " + output_folder + ".zip " + output_folder + "/")
    print("ZIP finished.\nResult in " + os.path.join(dataset_path, output_folder+".zip"))
    