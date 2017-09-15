from __future__ import division
import numpy as np

def spc(recalls = None, subjects = None, listLength = None):
    """
    SPC   Serial position curve (recall probability by serial position).

    p_recall = spc(recalls, subjects, listLength)

    INPUTS:
        recalls:    a matrix whose elements are serial positions of recalled
                    items.  The rows of this matrix should represent recalls
                    made by a single subject on a single trial.

        subjects:   a column vector which indexes the rows of recalls_matrix
                    with a subject number (or other identifier).  That is,
                    the recall trials of subject S should be located in
                    recalls_matrix(find(subjects==S), :)

        listLength: a scalar indicating the number of serial positions in the
                    presented lists.  serial positions are assumed to run
                    from 1:list_length.


    OUTPUTS:
        p_recall:  a matrix of probablities.  Its columns are indexed by
                   serial position and its rows are indexed by subject.
                   """
    if recalls is None:
        raise Exception('You must pass a recalls matrix.')
    elif subjects is None:
        raise Exception('You must pass a subjects vector.')
    elif listLength is None:
        raise Exception('You must pass a list length.')
    elif len(recalls) != len(subjects):
        raise Exception('recalls matrix must have the same number of rows as subjects.')

    subject = np.unique(subjects)
    result = [[0] * listLength for _ in range(len(subject))]
    for subject_index in range(len(subject)):
        
        count = 0
        for subj in range(len(subjects)):
            if subjects[subj] == subject[subject_index]:
                seen = []
                for serial_pos in range(len(recalls[0])):
                    if recalls[subj][serial_pos] > 0 and recalls[subj][serial_pos] < 1 + listLength and recalls[subj][serial_pos] not in seen:
                        result[subject_index][recalls[subj][serial_pos] - 1] += 1
                        seen.append(recalls[subj][serial_pos])
                count += 1
        for index in range(listLength):
            result[subject_index][index] /= count
    return result
