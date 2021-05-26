import random

import dataset_plot
import dataset_patch
import dataset_processing
import superpixel_segment
import keypoint_detection
import market1501
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

#
# kpd_bikeperson = keypoint_detection.KeyPointDetection("BikePerson-700")
# kpd_market1501 = keypoint_detection.KeyPointDetection("market1501")
# kpd_market1501.random_create_xinggan_input_from_images_poses(kpd_market1501.get_human_pose_with_threshold(0.46))

superpixel_segment.superpixel_slic("BikePerson-700/BikePerson-700-origin/query/0001_c2_eletric0007.png",
                                  show=True, region_size=10)
superpixel_segment.superpixel_slic("BikePerson-700/BikePerson-700-origin/query/0001_c2_eletric0007.png",
                                  show=True, region_size=30)

# superpixel_segment.display_superpixel("BikePerson-700/BikePerson-700-origin/query")
