import data_processing
import segmentation

BLACK = [0, 0, 0]

# data_processing.get_query_from_test_images()

segmentation.seg_person_from_mask("BikePersonDatasetNew/query-seg-test", BLACK)
