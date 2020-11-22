from utils.cafffe_compute import caffe_compute_all
from utils.dectection_functions import difference_general
from utils.pytorch_compute import pytorch_compute_all
from utils.tensorflow_compute import tensorflow_compute_all


def decide_dectection_function(DLFramework, DLFramework_Other, interface):
    # if DLFramework == 'caffe':
    #     return difference_caffe_other
    # if DLFramework_Other == 'caffe':
    #     return difference_other_caffe
    if interface == 'dense1':
        return None
    return difference_general


def decide_seed_corpus_target(DLFramework):
    if DLFramework == 'caffe':
        return caffe_compute_all
    elif DLFramework == 'tensorflow':
        return tensorflow_compute_all
    elif DLFramework == 'pytorch':
        return pytorch_compute_all


def decide_computer(DLFramework):
    if DLFramework == 'caffe':
        return caffe_compute_all
    elif DLFramework == 'tensorflow':
        return tensorflow_compute_all
    elif DLFramework == 'pytorch':
        return pytorch_compute_all
