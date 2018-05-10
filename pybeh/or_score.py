from __future__ import division
import numpy as np
import pybeh.mask_maker as mask

def or_score(recalls=None, subjects = None, listLength = None, rec_mask= None):
    """
    OR_SCORE  Recall probability for either of a pair, conditional on their lag.

    Computes the OR scores between pairs of items as a function of
    lag, irrespective of their serial positions.

    or_scores = or_score(recalls_matrix, subjects,list_length,pres_mask)

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


        rec_mask:       if given, a logical matrix the same shape as
                        recalls_matrix, true at positions for items to be
                        counted. Note that this mask does NOT need to exclude
                        repeats and intrusions, but it should include repeated
                        items if we're only interested in OR scores of 1p items
                        in mixed lists.

    OUTPUTS:
        or_scores:      a matrix of OR scores, i.e. the probability of
                        recalling one item or the other from a pair. Its
                        columns are indexed by lag and its rows are indexed by
                        subject.

    NOTES:
                        Using the proper pres_mask is CRUCIAL for determining proper or scores.
                        For the typical serial position curve, items from the recency and primacy
                        portions should be excluded, which is typically determined manually.
    """
    if recalls is None:
        raise Exception('You must pass a recall matrix.')
    elif listLength is None:
        raise Exception('You must pass a list length.')
    elif subjects is None:
        raise Exception('You must pass a subject matrix.')
    elif rec_mask is None:
        rec_mask = mask.make_clean_recalls_mask2d(recalls)
        recalls = mask.mask_data(recalls, rec_mask)
    elif len(rec_mask) != len(recalls):
        raise Exception('rec_mask needs to be same shape as recalls.')
    result = []
    subject = np.unique(subjects)
    for subj in subject:
        orscore_subj = []

        for lag in range(1, listLength):

            track = [0] * (listLength - lag)
            n = 0

            for subj_ind, subj_num in enumerate(subjects):
                if subj == subj_num:
                    n += 1
                    for ind in range(len(track)):
                        if ind+1 in recalls[subj_ind] or ind+1 + lag in recalls[subj_ind]:
                            track[ind] += 1
            total = 0
            print(track)
            for val in track:
                total += val
            print(total)
            total = total / float(n * (listLength - lag))
            orscore_subj.append(total)
        result.append(orscore_subj)
    return result