import dataset_plot
import dataset_processing

# 创建一个有700个训练id的数据集
# dataset_processing.create_subdataset(amount=700)

# 计算BikePerson-700中图片分割后的部分所占原图的比例
# dataset_plot.plot_segment_area_ratio_and_restore("BikePerson-700/BikePerson-700-mask")
# dataset_plot.plot_segment_area_ratio_and_restore("BikePerson-full/BikePerson-full-mask")

# 在BikePerson-700的基础上，创建一个混合了分割前和分割后图片的数据集
dataset_processing.create_mixed_dataset(mixed_ratio=0.6, dataset_num="700")