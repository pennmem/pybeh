from __future__ import division
import numpy as np
import mask_maker as mask

def crp(recalls=None, subjects=None, listLength=None, lag_num=None, from_mask_rec=None, to_mask_rec=None):
    """
    CRP   Conditional response probability as a function of lag (lag-CRP).

            lag_crps = crp(recalls_matrix, subjects, list_length, lag_num)

    INPUTS:
        recalls_matrix:  a matrix whose elements are serial positions of recalled
                   items.  The rows of this matrix should represent recalls
                   made by a single subject on a single trial.

        subjects:  a column vector which indexes the rows of recalls_matrix
                   with a subject number (or other identifier).  That is,
                   the recall trials of subject S should be located in
                   recalls_matrix(find(subjects==S), :)

        list_length:  a scalar indicating the number of serial positions in the
                   presented lists.  serial positions are assumed to run
                   from 1:list_length.

         lag_num:  a scalar indicating the max number of lag to keep track


    OUTPUTS:
        lag_crps:  a matrix of lag-CRP values.  Each row contains the values
                   for one subject.  It has as many columns as there are
                   possible transitions (i.e., the length of
                   (-list_length + 1) : (list_length - 1) ).
                   The center column, corresponding to the "transition of
                   length 0," is guaranteed to be filled with NaNs.

                   For example, if list_length == 4, a row in lag_crps
                   has 7 columns, corresponding to the transitions from
                   -3 to +3:
                   lag-CRPs:     [ 0.1  0.2  0.3  NaN  0.3  0.1  0.0 ]
                   transitions:    -3   -2    -1   0    +1   +2   +3 """
    if recalls is None:
        raise Exception('You must pass a recalls matrix.')
    elif subjects is None:
        raise Exception('You must pass a subjects vector.')
    elif listLength is None:
        raise Exception('You must pass a list length.')
    elif len(recalls) != len(subjects):
        raise Exception('recalls matrix must have the same number of rows as subjects.')
    if from_mask_rec != None:
        recalls = mask.mask_data(recalls, from_mask_rec)
    if to_mask_rec != None:
        recalls = mask.mask_data(recalls, to_mask_rec)
    if not (lag_num > 0):
        raise ValueError('lag number needs to be positive')
    if lag_num > listLength:
        raise ValueError('Lag number too big')

    subject = np.unique(subjects)
    result = [[0] * (2 * lag_num + 1) for count in range(len(subject))]

    for subject_index in range(len(subject)):
        lag_bin_count = [0] * (2 * lag_num + 1)
        poss = [0] * (2 * lag_num + 1)
        for subj in range(len(subjects)):
            if subjects[subj] == subject[subject_index]:
                seen = []
                for serial_pos in range(len(recalls[0])):
                    if recalls[subj][serial_pos] > 0 and recalls[subj][
                        serial_pos] < 1 + listLength and serial_pos + 1 < len(recalls[0]):
                        seen.append(recalls[subj][serial_pos])
                        if recalls[subj][serial_pos + 1] > 0 and recalls[subj][serial_pos + 1] < 1 + listLength and \
                                        recalls[subj][serial_pos + 1] != recalls[subj][serial_pos]:
                            lag = recalls[subj][serial_pos + 1] - recalls[subj][serial_pos]

                            for x in range(1, listLength + 1):
                                if x not in seen:
                                    poss_lag = x - recalls[subj][serial_pos]
                                    if poss_lag >= -lag_num and poss_lag <= lag_num:
                                        poss[lag_num + poss_lag] += 1

                            if lag_num + lag >= 0 and lag_num + lag <= 2 * lag_num:
                                lag_bin_count[lag_num + lag] += 1

        for index in range(2 * lag_num + 1):
            if poss[index] == 0:
                result[subject_index][index] = float('nan')
            else:
                result[subject_index][index] = lag_bin_count[index] / np.long(poss[index])
    return result
