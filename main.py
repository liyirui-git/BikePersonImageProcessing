import dataset_plot
import dataset_processing
from result_analyse import ResultAnalyse

dir_map = {"origin": "BP-700-origin-20210317",
           "seg": "BP-700-seg-20210317",
           "0_3": "BP-700-mixed_0_3-20210318",
           "0_4": "BP-700-mixed_0_4-20210317",
           "0_6": "BP-700-mixed_0_6-20210317"}

result_analyse = ResultAnalyse(dir_map["origin"], dir_map["0_4"])
result_analyse.ap_result_analyse(do_display=True)

# result_analyse.ap_result_analyse("BP-700-origin-20210317", "BP-700-mixed_0_3-20210318")
# result_analyse.ap_result_analyse("BP-700-origin-20210317", "BP-700-mixed_0_6-20210317")

# 创建一个有700个训练id的数据集
# dataset_processing.create_subdataset(amount=700)

# 计算BikePerson-700中图片分割后的部分所占原图的比例
# dataset_plot.plot_segment_area_ratio_and_restore("BikePerson-700/BikePerson-700-mask")
# dataset_plot.plot_segment_area_ratio_and_restore("BikePerson-full/BikePerson-full-mask")

# 查看不同视角下的折线图
# dataset_plot.plot_segment_area_ratio_angle_from_file("txt/BPfull_segment_area_ratio_log.txt")

# 在BikePerson-700的基础上，创建一个混合了分割前和分割后图片的数据集
# dataset_processing.create_mixed_dataset(mixed_ratio=0.3, dataset_num="700")