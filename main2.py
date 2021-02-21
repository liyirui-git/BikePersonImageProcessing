import data_processing
import segmentation

BLACK = [0, 0, 0]

# data_processing.get_query_from_test_images()
print("train seg begin")
segmentation.seg_person_from_mask("BikePersonDatasetNew/train-seg", BLACK)
print("end!")