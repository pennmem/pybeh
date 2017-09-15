import numpy as np

def xli(intrusions = None, subjects = None, rec_itemnos = None):
    """
    XLI   Number of prior list intrusions.

    xlis = xli(intrusions, subjects, rec_itemnos)

    INPUTS:
        intrusions:     matrix whose elements indicate PLI if
                        equal to positive integer, where the integer indicates
                        number of lists prior. Indicates XLI if
                        equal to -1.

        subjects:       column vector which indexes the rows of
                        recall_itemnos with a subject number (or other
                        identifier). That is, the recall trials of subject S
                        should be located in:
                        recall_itemnos(find(subjects==S), :)

        rec_itemnos:    matrix whose elements are indices of recalled
                        items. The rows of this matrix should represent
                        recalls made by a single subject on a single trial.

    OUTPUTS:
        xlis:           vector of total number of XLIs. Its rows are indexed
                        by subject.

    """
    if intrusions is None:
        raise Exception('You must pass a intrusions matrix.')
    if subjects is None:
        raise Exception('You must pass a subjects vector.')
    if len(intrusions) != len(subjects):
        raise Exception('recalls matrix must have the same number of rows as subjects.')
    if rec_itemnos is None:
        raise Exception('You must pass a recall_itemnos matrix.')

    subject = np.unique(subjects)
    result = [0] * len(subject)
    for subject_index in range(len(subject)):
        count = 0
        for subj in range(len(subjects)):
            if subjects[subj] == subject[subject_index]:
                for serial_pos in range(len(intrusions[0])):
                    if intrusions[subj][serial_pos] < 0:
                        count += 1
        result[subject_index] = count
    return result