from __future__ import division
import numpy as np

def dist_fact(recall_itemnos=None, pres_itemnos=None, subjects=None, dist_mat=None, listLength=None):
    """sanity check"""
    if recall_itemnos is None:
        raise Exception('You must pass a recall_itemnos matrix.')
    elif subjects is None:
        raise Exception('You must pass a subjects vector.')
    elif pres_itemnos is None:
        raise Exception('You must pass a pres_itemnos matrix.')
    elif dist_mat is None:
        raise Exception('You must pass a distance matrix')
    elif listLength is None:
        raise Exception('You must pass a list length.')
    elif len(recall_itemnos) != len(subjects):
        raise Exception('recalls matrix must have the same number of rows as subjects.')
    final_data = []
    subject = np.unique(subjects)
    for subject_index in range(len(subject)):
        total = 0
        count = 0
        for subj in range(len(subjects)):
            if subjects[subj] == subject[subject_index]:
                seen = []
            for serial_pos in range(len(recall_itemnos[0])):
                if recall_itemnos[subj][serial_pos] != 0 and recall_itemnos[subj][serial_pos] != -1:

                    seen.append(recall_itemnos[subj][serial_pos])
                    possibles = [dist_mat[recall_itemnos[subj][serial_pos] - 1][item - 1] for item in pres_itemnos[subj]
                                 if item not in seen]
                    if serial_pos + 1 < len(recall_itemnos[0]):
                        if dist_percentile_rank(dist_mat[recall_itemnos[subj][serial_pos] - 1][
                                                            recall_itemnos[subj][serial_pos + 1] - 1],
                                                possibles) is not None:
                            total += dist_percentile_rank(dist_mat[recall_itemnos[subj][serial_pos] - 1][
                                                              recall_itemnos[subj][serial_pos + 1] - 1], possibles)
                            count += 1
        if count != 0:
            final_data.append(total / count)
    return final_data


def dist_percentile_rank(actual, possible):
    possible = sorted(possible)
    temp = 0
    count = 0
    for index, item in enumerate(possible):
        if item == actual and len(possible) > 1:
            temp += (len(possible) - 1 - index) / (len(possible) - 1)
            count += 1
    if count != 0:
        return 1 - temp / count
    return None