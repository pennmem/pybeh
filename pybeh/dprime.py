from __future__ import division
import numpy as np
from scipy import stats


def dprime(events):
    """
    Input:
        events: recog_events structure

    Output:
        dprime: scalar drpime value
    """
    hit = 0
    miss = 0
    false_alarm = 0
    correct_rej = 0
    target = 0
    lure = 0
    if len(events) < 1:
        raise Exception('events file is empty')
    for num in range(len(events)):
        if np.isnan(events[num].__dict__['recog_conf']) or (
                    events[num].__dict__['recog_conf'] < 1) or (
                    events[num].__dict__['recog_conf'] > 5):
            continue

        if events[num].__dict__['recog_conf'] == 'nan':
            continue
        if events[num].__dict__['recog_rt'] == 0:
            continue
        if events[num].__dict__['recog_resp'] == 1 and events[num].__dict__['type'] == 'RECOG_TARGET':
            hit += 1
            target += 1
        elif events[num].__dict__['recog_resp'] == 0 and events[num].__dict__['type'] == 'RECOG_LURE':
            correct_rej += 1
            lure += 1
        elif events[num].__dict__['recog_resp'] == 1 and events[num].__dict__['type'] == 'RECOG_LURE':
            false_alarm += 1
            lure += 1
        elif events[num].__dict__['recog_resp'] == 0 and events[num].__dict__['type'] == 'RECOG_TARGET':
            miss += 1
            target += 1

    hit /= float(target)
    miss /= float(target)
    false_alarm /= float(lure)
    correct_rej /= float(lure)
    return (stats.norm.ppf(hit)- stats.norm.ppf(false_alarm))
