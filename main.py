'''
Author: Li, Yirui
Date: 2021-07-26
Description: 
FilePath: /liyirui/PycharmProjects/BikePersonImageProcessing/main.py
'''
import random

import dataset_plot
import dataset_patch
import dataset_processing
import superpixel_segment
import keypoint_detection
from superpixel_segment import SuperPixelSegmentation
from keypoint_to_angle import Keypoint2Angle
from result_analyse import ResultAnalyse
import utils

# 在两个数据集上进行对比分析
# dir_map = {"origin": "BP-700-origin-20210317",
#            "seg": "BP-700-seg-20210317",
#            "0_3": "BP-700-mixed_0_3-20210318",
#            "0_4": "BP-700-mixed_0_4-20210317",
#            "0_6": "BP-700-mixed_0_6-20210317"}
#
# result_analyse = ResultAnalyse(dir_map["origin"], dir_map["0_4"])
# result_analyse.ap_result_analyse(do_display=True)

# 创建一个有700个训练id的数据集
# dataset_processing.create_subdataset(amount=700)

# 计算BikePerson-700中图片分割后的部分所占原图的比例
# dataset_plot.plot_segment_area_ratio_and_restore("BikePerson-700/BikePerson-700-mask")
# dataset_plot.plot_segment_area_ratio_and_restore("BikePerson-full/BikePerson-full-mask")

# 关键点检测相关程序
# kpd_bikeperson = keypoint_detection.KeyPointDetection("BikePerson-700")
# kpd_bikeperson.get_human_pose_with_threshold(threshold=0.2)
# kpd_bikeperson.count_human_upper_body_with_threshold(threshold=0.2)
# kpd_bikeperson.get_image_path_with_pose_threshold(threshold=0.2)
# kpd_market1501 = keypoint_detection.KeyPointDetection("market1501")
# kpd_market1501.random_create_xinggan_input_from_images_poses(kpd_market1501.get_human_pose_with_threshold(0.46))

