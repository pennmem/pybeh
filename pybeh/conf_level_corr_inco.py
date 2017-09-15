from __future__ import division
import numpy as np

def conf_level_corr_inco(events, level):
    """
    INPUTS:
        events:     recog_events structure
        level:      scalar value of level of confidence level available
    OUTPUTS:
        conf_level: a matrix of confidence percentage with subj for rows and levels for columns
    """
    corr_result = [0] * level
    inco_result = [0] * level
    if len(events) < 1:
        raise Exception('events file is empty')
    for num in range(len(events)):
        if np.isnan(events[num].__dict__['recog_conf']) or (
                    events[num].__dict__['recog_conf'] < 1) or (events[num].__dict__['recog_conf'] > 5):
            continue

        if events[num].__dict__['recog_conf'] == 'nan':
            continue
        if events[num].__dict__['recog_rt'] == 0:
            continue
        if events[num].__dict__['recog_resp'] == 1 and events[num].__dict__['type'] == 'RECOG_TARGET':
            corr_result[int(events[num].__dict__['recog_conf']) - 1] += 1
        elif events[num].__dict__['recog_resp'] == 0 and events[num].__dict__['type'] == 'RECOG_LURE':
            corr_result[int(events[num].__dict__['recog_conf']) - 1] += 1
        elif events[num].__dict__['recog_resp'] == 1 and events[num].__dict__['type'] == 'RECOG_LURE':
            inco_result[int(events[num].__dict__['recog_conf']) - 1] += 1
        elif events[num].__dict__['recog_resp'] == 0 and events[num].__dict__['type'] == 'RECOG_TARGET':
            inco_result[int(events[num].__dict__['recog_conf']) - 1] += 1
    total = 0
    for item in corr_result:
        total += item
    for item in inco_result:
        total += item

    for num, item in enumerate(corr_result):
        corr_result[num] = item / float(total)
    for num, item in enumerate(inco_result):
        inco_result[num] = item / float(total)
    return corr_result, inco_result