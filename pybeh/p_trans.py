from __future__ import division
import numpy as np

def p_trans(subject, from_mask, to_mask):
    """
    P_TRANS  Probability of transitioning between recall types.

    Computes probability of transitioning from one recall type to another, ignoring output position

    p_trans(subjects,from_mask,to_mask)


    INPUTS:

        subjects:   a column vector which indexes the rows of recalls
                    with a subject number (or other identifier). That is,
                    the recall trials of subject S should be located in
                    recalls(find(subjects==S), :smile:


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

    EXAMPLES:

    recalls = [1 2 -1 3];
    intrusions = [0 0 1 0];

    To look at transitions from correct recalls to correct recalls only:
    The FROM mask is simply the clean recalls matrix, i.e.

    from_mask = mask.make_clean_recalls_mask2d(recalls)

    from_mask = 1 1 0 1

    We want the TO mask to be true for those items that are transitioning TO
    a correct item, so this is simply the from_mask shifted over by 1 to the
    left. Naturally, this leaves the last column empty. Set the
    last column as all false, since there can never be a transition from the
    last recalled item to another item.

    to_mask = mask.make_tomask_from_frommask(frommask)

    to_mask = 1 0 1 0

    We can see that of the three transitions FROM correct recalls, only one
    was a transition TO correct recalls.

    So, if we set
    >> subjects = [1;2];
    >> p_trans(subjects,from_mask,to_mask)
    >> ans =
     0.3333

    A similar technique can be used to generate the transitions from correct
    recalls to PLIs:
    1. generate the PLI to mask using
    >> from_mask_pli = make_mask_only_pli2d(intrusions);

    2. generate the to mask by shifting everything over by 1
    >> to_mask_pli = [from_mask_pli(:,2:end) false(size(from_mask_pli,1),1)];

    >> p_trans(subjects,from_mask,to_mask_pli)
    >> ans =
     0.3333 k"""
    if to_mask is None:
        raise Exception('You must pass a to_mask.')
    elif from_mask is None:
        raise Exception('You must pass a from_mask.')
    elif subject is None:
        raise Exception('You must pass a subject.')
    elif len(to_mask) != len(from_mask):
        raise Exception('to_mask needs to be same shape as from_mask.')
    result = []
    subjects = np.unique(subject)
    for subj in subjects:
        count = 0
        total = 0
        for subj_ind, subj_num in enumerate(subject):
            if subj == subj_num:
                for index, item in enumerate(from_mask[subj_ind]):
                    if item == 1:
                        total += 1
                        if to_mask[subj_ind][index] == 1:
                            count += 1
        result.append(count / float(total))
    return result