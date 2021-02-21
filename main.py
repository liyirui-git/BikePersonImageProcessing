import data_processing
import segmentation

BLACK = [0, 0, 0]

# data_processing.get_query_from_test_images()
print("query seg begin")
segmentation.seg_person_from_mask("BikePersonDatasetNew/query-seg", BLACK)
print("test seg begin")
segmentation.seg_person_from_mask("BikePersonDatasetNew/test-seg", BLACK)
print("end!")

