from __future__ import division
import numpy as np

def serial_ratios(recalls=None, subject=None, min_recs=None):
    """
    ratios = serial_ratios(recalls_matrix, subjects, mins_recs)

    returns a the ratio of trials in which a string of consecutive recalls of
    length min_recs or greater occurred to trials in which it did not occur.


    INPUTS:
        recalls_matrix: a matrix whose elements are serial positions of recalled
                        items. The rows of this matrix should represent recalls
                        made by a single subject on a single trial.

        subjects:       a column vector which indexes the rows of recalls_matrix
                        with a subject number (or other identifier). That is,
                        the recall trials of subject S should be located in
                        recalls_matrix(find(subjects==S)

        min_recs:       the minimum number of serial recalls for a subject to
                        count as a serial recaller. For example, if min_recs = 5,
                        subjects who recall 5 words in a row that were presented
                        consecutively would count as a serial recaller.

    OUTPUTS:
        ratios:         The ratio, for each subject, of the number of trials on
                        which they recalled min_recs words serially.
    """
    if recalls is None:
        raise Exception('You must pass a recalls matrix.')
    elif subject is None:
        raise Exception('You must pass a subjects vector.')
    elif min_recs is None:
        raise Exception('You must pass a minimum recall length.')
    elif len(recalls) != len(subject):
        raise Exception('recalls matrix must have the same number of rows as subjects.')

    result = []
    subjects = np.unique(subject)

    for subj in subjects:
        yes = 0
        total = 0
        for subj_ind, subj_num in enumerate(subject):
            if subj == subj_num:
                total += 1

                for index, item in enumerate(recalls[subj_ind]):

                    if item > 0:
                        count = 1
                        while check_next(recalls[subj_ind], index):
                            count += 1
                            index += 1
                        if count >= min_recs:
                            yes += 1
                            break
        result.append(yes / float(total))
    return result

def check_next(list, index):
    """
    returns True if next index is serially greater than previous, False if not
    """
    if index + 1 < len(list):
        if list[index + 1] == list[index] + 1:
            return True
    return False

