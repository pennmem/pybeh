from __future__ import division
import numpy as np
import pybeh.mask_maker as mask

def positional_crp(recalls = None, subject = None,  list_length = None, rec_mask = None):
    """
    POSITIONAL_CRP  Computes conditional response probabilities of recalling
                    an item in its correc position (i.e., its output
                    position and serial position should be equal), given a
                    matrix of recalled serial positions.

    [pos_crps,numer,denom] =    positional_crp(recalls_matrix, subjects, list_length, rec_mask)

    INPUTS:
        recalls_matrix: a matrix whose elements are serial positions of recalled
                        items. The rows of this matrix should represent recalls
                        made by a single subject on a single trial.

        subjects:       a column vector which indexes the rows of recalls_matrix
                        with a subject number (or other identifier). That is,
                        the recall trials of subject S should be located in
                        recalls_matrix(find(subjects==S), :smile:

        list_length:    a scalar indicating the number of serial positions in the
                        presented lists. serial positions are assumed to run
                        from 1:list_length.

        rec_mask:       if given, a logical matrix of the same shape as
                        recalls_matrix, which is false at positions (i, j) where
                        the transition FROM and TO recalls_matrix(i, j) to
                        recalls_matrix(i, j+1) should be excluded from
                        the calculation of the CRP.

    OUTPUTS:
        pos_crps:       a matrix of positional CRP values. Each row contains
                        the values for one subject. It has as many columns as
                        there are possible transitions (i.e., the length of
                        (-list_length + 1) : (list_length - 1) ).
                        Unlike in free recall, the center column, corresponding
                        to the "transition of length 0," will have a real value.
    """
    if recalls is None:
        raise Exception('You must pass a recalls matrix.')
    elif subject is None:
        raise Exception('You must pass a subjects vector.')
    elif list_length is None:
        raise Exception('You must pass a list length.')
    elif len(recalls) != len(subject):
        raise Exception('recalls matrix must have the same number of rows as subjects.')
    if rec_mask != None:
        if len(rec_mask) != len(subject):
            raise Exception('rec_mask must have same shape a recall matrix.')
        else:
            recalls = mask.mask_data(rec_mask, rec_mask)
    result = []
    subjects = np.unique(subject)

    for subj in subjects:
        crp = [0] * (2 * list_length - 1)
        count = 0
        for subj_ind, subj_num in enumerate(subject):
            if subj == subj_num:
                for index, item in enumerate(recalls[subj_ind]):
                    if item > 0 and item < list_length + 1:
                        crp[(index + 1) - item + list_length - 1] += 1
                        count += 1
        for num, item in enumerate(crp):
            crp[num]  = item / float(count)
        result.append(crp)

    return result