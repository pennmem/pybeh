from __future__ import division
import numpy as np

def p_trans_op(subject, from_mask, to_mask):
    """
    P_TRANS_OP  Probability of transitioning recall types, by output position.
    Computes probability of transitioning from one recall type to another,
    first taking the mean at each output position and then averaging
    across output positions

    [p_tran,denoms] = p_trans(subjects,from_mask,to_mask)

    INPUTS:
        subjects:   a column vector which indexes the rows of recalls
                    with a subject number (or other identifier). That is,
                    the recall trials of subject S should be located in
                    recalls(find(subjects==S)

        from_mask:  recall type being transitioned from.
                    rows represent trial, columns represent output
                    position. a particular item i is true only if we are
                    determining the probability of transitions FROM items
                    of type *i*.

        to_mask:    recall type being transitioned to.
                    rows represent trial, columns represent output
                    position. a particular item i is true only if we are
                    determining the probability of transition TO items
                    of type *i+1*.



    OUTPUTS:
        p_trans:    a column vector of transition probabilities with rows
                    representing subjects.


    Notes about the masks:  these are logical matrices the SAME shape as a
                            recalls or intrusions matrix. Although neither of
                            these matrices are explicitly required by this
                            function, it is expected that these matrices
                            has the standard form as in other functions:
                            The rows of this matrix should represent recalls
                            made by a single subject on a single trial.
                            Each mask should be true for any item in the
                            condition of interest.

                            You can use the masks to analyze different types
                            of recalls (e.g., correct recalls, repetitions,
                            intrusions). There are masking functions that will
                            create these.

    To only look at correct responses:
    mask = mask.make_clean_recalls_mask2d(data)



    To look at intrusions, you can create a mask using an
    intrusions matrix (which must be the same size as
    recalls, where a positive integer indicates a prior
    list intrusions, and a -1 indicates an extra list
    intrusion)

    To only look at PLIs:
    mask = mask.make_mask_only_pli2d(data)

    To only look at XLIs:
    mask = mask.make_mask_only_xli2d(data)

    You can use the repetition mask, PLI mask, and XLI mask
    to create a mask for all incorrect recalls.
    """
    if to_mask is None:
        raise Exception('You must pass a to_mask.')
    elif from_mask is None:
        raise Exception('You must pass a from_mask.')
    elif subject is None:
        raise Exception('You must pass a subject.')
    elif len(to_mask) != len(from_mask):
        raise Exception('to_mask needs to be same shape as from_mask.')
    final = []
    subjects = np.unique(subject)
    for subj in subjects:
        result = []
        for subj_ind, subj_num in enumerate(subject):
            if subj == subj_num:
                denom = [0] * len(from_mask[subj_ind])
                num = [0] * len(from_mask[subj_ind])
        for subj_ind, subj_num in enumerate(subject):
            if subj == subj_num:
                for index, item in enumerate(from_mask[subj_ind]):
                    if item == 1:
                        denom[index] += 1
                        if to_mask[subj_ind][index] == 1:
                            num[index] += 1
        for index in range(len(denom)):
            if denom[index] != 0:
                result.append(num[index] / float(denom[index]))
            else:
                result.append(0)
        n = 0
        m = 0
        for item in result:
            n += 1
            m += item
        final.append(m / float(n))

    return final