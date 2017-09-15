from __future__ import division
import numpy as np

def conf_level(events, level):
    """
    INPUTS:
        events:     recog_events structure
        level:      scalar value of level of confidence level available
    OUTPUTS:
        conf_level: a matrix of confidence percentage with subj for rows and levels for columns
    """

    result = [0] * level
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
        result[int(events[num].__dict__['recog_conf']) - 1] += 1

    total = 0
    for item in result:
        total += item

    for num, item in enumerate(result):
        result[num] = item / float(total)
    return result

    #If data is already in an int structure of rows for subjects and confidence values as columns, use code below


    # final = []
    # if len(events) < 1:
    #     raise Exception('events file is empty')
    # for item in events:
    #     result = [0] * level
    #     for thing in item:
    #         if np.isnan(thing) or thing < 1 or thing > level:
    #             continue
    #         if thing == 'nan':
    #             continue
    #         if thing == 0:
    #             continue
    #         result[thing - 1] += 1
    #     final.append(result)
    # total = [0]* len(events)
    # for ind, item in enumerate(final):
    #     for thing in item:
    #         total[ind] += thing
    # for num, item in enumerate(final):
    #     for ind, thing in enumerate(item):
    #         final[num][ind] = thing / float(total[num])
    # return final