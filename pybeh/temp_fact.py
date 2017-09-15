from __future__ import division
import numpy as np

def temp_fact(recalls=None, subjects=None, listLength=None):
    """
    returns a Lag-based temporal clustering factor for each subject

    INPUTS:
        recalls:    a matrix whose elements are serial positions of recalled
                    items.  The rows of this matrix should represent recalls
                    made by a single subject on a single trial.

        subjects:   a column vector which indexes the rows of recalls
                    with a subject number (or other identifier).  That is,
                    the recall trials of subject S should be located in
                    recalls(find(subjects==S), :)

        list_length:a scalar indicating the number of serial positions in the
                    presented lists.  serial positions are assumed to run
                    from 1:list_length.
    OUTPUTS:
        temp_facts: a vector of temporal clustering factors, one
                    for each subject.
    """
    if recalls is None:
        raise Exception('You must pass a recall_itemnos matrix.')
    elif subjects is None:
        raise Exception('You must pass a subjects vector.')
    elif listLength is None:
        raise Exception('You must pass a list length.')
    elif len(recalls) != len(subjects):
        raise Exception('recalls matrix must have the same number of rows as subjects.')
    final_data = []
    subject = np.unique(subjects)
    for subject_index in range(len(subject)):
        total = 0
        count = 0
        for subj in range(len(subjects)):
            if subjects[subj] == subject[subject_index]:
                seen = []
            for serial_pos in range(len(recalls[0])):
                if recalls[subj][serial_pos] != 0 and recalls[subj][serial_pos] != -1 and recalls[subj][
                    serial_pos] not in seen:
                    seen.append(recalls[subj][serial_pos])
                    possibles = [abs(item - recalls[subj][serial_pos]) for item in range(1, listLength + 1) if
                                 item not in seen]
                    if serial_pos + 1 < len(recalls[0]) and recalls[subj][serial_pos + 1] != 0 and recalls[subj][
                                serial_pos + 1] != -1 and recalls[subj][serial_pos + 1] not in seen:
                        if temp_percentile_rank(abs(recalls[subj][serial_pos + 1] - recalls[subj][serial_pos]),
                                                possibles) is not None:
                            total += temp_percentile_rank(
                                abs(recalls[subj][serial_pos + 1] - recalls[subj][serial_pos]),
                                possibles)
                            count += 1

        if count != 0:
            final_data.append(total / count)
        else:
            final_data.append(0)
    return final_data

#Helper function to return the percentile rank of the input within the possible inputs.
#Gives out the mean of the values of input for ties

def temp_percentile_rank(actual, possible):
    possible = sorted(possible)
    temp = 0
    count = 0
    for index, item in enumerate(possible):
        if item == actual and len(possible) > 1:
            temp += (len(possible) - 1 - index) / (len(possible) - 1)
            count += 1
    if count != 0:
        return temp / count
    return None
