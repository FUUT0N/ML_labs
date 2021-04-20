import os
import numpy as np
from features import *

NUM_TREES = 200
TEST_SIZE = 0.2
SEED = 9
TRAIN_PATH = "data/train"
TEST_PATH = "data/test"
H5_DATA = 'output/data.h5'
H5_LABELS = 'output/labels.h5'
SCORING = "accuracy"

IMAGES_PER_CLASS = 80
FIXED_SIZE = tuple((500, 500))

TRAIN_LABELS = os.listdir(TRAIN_PATH)
TRAIN_LABELS.sort()


def get_feature(image):
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick = fd_haralick(image)
    fv_histogram = fd_histogram(image)
    global_feature = np.hstack([fv_histogram, fv_hu_moments, fv_haralick])
    return global_feature
