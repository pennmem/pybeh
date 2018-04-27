from __future__ import division
import numpy as np
import pybeh.mask_maker as mask

def p_reject(rejects_matrix = None, subject = None, rec_mask = None, recalls = None):
    """
    P_REJECT Computes probability of rejecting recalled items.

    p_rejects = p_reject(reject_matrix, rec_mask)

    INPUTS:
        rejects_matrix: a matrix whose elements indicates whether recalled
                        items were rejected or accepted as correct items
                        in externalized free recall (EFR).
                        The rows of this matrix should
                        represent recalls made by a single subject on a single
                        trial. An element of the rejected matrix should be
                        equal to 1 if and only if that item was rejected.

        subjects:       a column vector which indexes the rows of recalls_matrix
                        with a subject number (or other identifier). That is,
                        the recall trials of subject S should be located in
                        recalls_matrix(find(subjects==S), :smile:

        rec_mask:       if given, a logical matrix of the same shape as
                        recalls_matrix, which is false at positions (i, j) where
                        the value at recalls_matrix(i, j) should be excluded from
                        the calculation of the probability of recall. If NOT
                        given, a standard clean recalls mask is used, which
                        excludes repeats, intrusions and empty cells


    OUTPUTS:
        p_reject:       a vector of probablities. Rows are indexed by subject.
    """
    if rejects_matrix is None:
        raise Exception('You must pass a rejects matrix.')
    elif subject is None:
        raise Exception('You must pass a subject.')
    elif rec_mask is None:
        rec_mask = mask.make_clean_recalls_mask2d(recalls)
    elif len(rejects_matrix) != len(subject):
        raise Exception('rejects matrix needs to be same length as subjects.')
    subjects = np.unique(subject)
    result = []
    for subj in subjects:

        for subj_ind, subj_num in enumerate(subject):
            if subj == subj_num:
                denom = 0
                num = 0
        for subj_ind, subj_num in enumerate(subject):
            if subj == subj_num:
                for index, item in enumerate(rec_mask[subj_ind]):
                    if item == 1:
                        denom += 1
                        if rejects_matrix[subj_ind][index] == 1:
                            num += 1
        print(subj, denom, num)
        if denom != 0:
            result.append(num / float(denom))
        else:
            result.append(0)
    return result