import copy
import numpy as np

def make_clean_recalls_mask2d(data):
    """makes a clean mask without repetition and intrusion"""
    result = copy.deepcopy(data)
    for num, item in enumerate(data):
        seen = []
        for index, recall in enumerate(item):

            if recall > 0 and recall not in seen:
                result[num][index] = 1
                seen.append(recall)
            else:
                result[num][index] = 0
    return result


def make_mask_only_pli2d(data):
    """makes a mask with only pli as True aka 1, and 0 everywhere else"""
    result = copy.deepcopy(data)
    for num, item in enumerate(data):
        for index, recall in enumerate(item):
            if recall != 0 and recall != -1:
                result[num][index] = 1
            else:
                result[num][index] = 0
    return result



def make_mask_only_xli2d(data):
    """makes a mask with only xli as True aka 1, and 0 everywhere else"""
    result = copy.deepcopy(data)
    for num, item in enumerate(data):
        for index, recall in enumerate(item):
            if recall == -1:
                result[num][index] = 1
            else:
                result[num][index] = 0
    return result

def make_tomask_from_frommask(frommask):
    """makes a to_mask from from_mask"""
    tomask = []
    for num, list in enumerate(frommask):
        tomask.append(list[1:] + [0])
    return tomask


def make_blank_mask(data_matrix):
    """makes an all true mask"""
    return np.ones(data_matrix.shape)

def mask_data(data, mask):
    """id data is same shape as mask, returns values in data where mask is true"""
    result = copy.deepcopy(data)
    if len(data) != len(mask):
        raise Exception('data and mask need to have same shape')
    for index, item in enumerate(mask):
        for ind, num in enumerate(item):
            if num == 0:
                result[index][ind] = 0
    return result

def mask_nan(data):
    """converts nan from data to 0"""
    result = copy.deepcopy(data)
    for index, item in enumerate(result):
        for ind, num in enumerate(item):
            if np.isnan(num):
                result[index][ind] = 0
            else:
                result[index][ind] = 1
    return result

