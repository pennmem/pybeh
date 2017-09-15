from __future__ import division
import numpy as np
def crl(recalls = None, times = None, subject = None, listLength = None, lag_num = None):
    """
    CRL  Inter-response time as a function of lag.

    Calculates the mean time it takes to move from one word position to another
    as a funciton of lag.  Returns lag-conditional response latency times.

    Please note: if i and j are consecutive words only real transitions are counted meaning
    1) neither i nor j are intrusions ( not == -1)
    2) words that have been recalled cannot be
      transitioned to nor cannot be transitioned from

    FUNCTION:
        crl = crl(recalls, times, subject, listLength, lag_num);

    INPUT ARGS:
        recalls    - recall positions
        times      - time associated with each recall
        subjects   - subject number associated with each trial
        listLength - number of words in the list
        lag_num    - lag number to output


    OUTPUT ARGS:
        crl - a matrix of average crl times by lag position for each subject
    """
    if recalls is None:
        raise Exception('You must pass a recalls matrix.')
    if times is None:
        raise Exception('You must pass a times vector.')
    if subject is None:
        raise Exception('You must pass a subject vector.')
    if listLength is None:
        raise Exception('You must pass a list length.')
    if lag_num is None:
        lag_num = listLength - 1
    if not (lag_num > 0):
        raise ValueError('lag number needs to be positive')
    if lag_num >= listLength:
        raise ValueError('Lag number too big')

    subjects = np.unique(subject)
    result = [[0] * (2 * lag_num + 1) for count in range(len(subjects))]
    for subject_index in range(len(subjects)):
        lag_bin_count = [0] * (2 * lag_num + 1)
        irt = [0] * (2 * lag_num + 1)

        for subj in range(len(subject)):
            if subject[subj] == subjects[subject_index]:
                seen = []
                for serial_pos in range(len(recalls[0])):
                    if recalls[subj][serial_pos] > 0 and recalls[subj][serial_pos] < 1 + listLength and serial_pos + 1 < len(recalls[0]) and recalls[subj][serial_pos] not in seen:
                        seen.append(recalls[subj][serial_pos])
                        if recalls[subj][serial_pos + 1] > 0 and recalls[subj][serial_pos + 1] < 1 + listLength and recalls[subj][serial_pos + 1] not in seen:
                            lag = recalls[subj][serial_pos + 1] - recalls[subj][serial_pos]
                            if lag_num + lag >= 0 and lag_num + lag <= 2 * lag_num:
                                lag_bin_count[lag_num + lag] += 1
                                irt[lag_num + lag] += (times[subj][serial_pos + 1] - times[subj][serial_pos])
        for index in range(2 * lag_num + 1):
            if lag_bin_count[index] == 0:
                lag_bin_count[index] = 1
            result[subject_index][index] = irt[index] / float(lag_bin_count[index])
    return result

