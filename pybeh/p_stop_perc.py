from __future__ import division
import numpy as np
import mask_maker as mask

def p_stop_perc(recalls = None, subject = None, time = None, record_time = None, exit_time_thresh = None, rec_mask = None):
    """
    P_STOP_PERC  Probability of stopping recall.

    [p_stops,denoms] = p_stop_op(recalls,time_mat,rec_length,exit_time_thresh,subjects,mask)

    INPUTS:
        recalls:            a matrix whose elements are serial positions of recalled
                            items. The rows of this matrix should represent recalls
                            made by a single subject on a single trial.

        time_mat:           a matrix whose elements are the millisecond times of the
                            recalled items. The rows of this matrix should represent
                            recalls made by a single subject on a single trial.

        rec_length:         a column vector of the length, in milliseconds, of the
                            recall period for each trial.

        exit_time_thresh:   a scalar, in ms, representing time required between
                            the final recall in a trial and the end of the recall
                            period. If the final recall occurred less than
                            exit_time_thresh away from the end of the recall period,
                            the trial will not be used in the analysis (the idea
                            being that perhaps the subject simply ran out of time
                            but was not done recalling). A trial will only be
                            included if the final recall is greater than
                            exit_time_thresh away from the end of the recall period
                            AND the time between the final recall and the end of
                            of the recall period is greater than all of the
                            inter-response times on that trial. NOTE: If you do not
                            want to exclude any trials based on these criteria, do not
                            pass values for time_mat, rec_length, and exit_time_thresh.

        subjects:           a column vector which indexes the rows of recalls
                            with a subject number (or other identifier). That is,
                            the recall trials of subject S should be located in
                            recalls(find(subjects==S)

        mask:               a logical matrix the same shape as recalls. The mask
                            should be true for any item in the condition of interest.
                            If NOT given, a clean recalls mask is used (i.e., only
                            correct recalls will be analyzed.)

    OUTPUTS:
        p_stops:            a column vector of stopping probabilities with rows
                            representing subjects.

        denoms:             a column vector of denominator values that went into the
                            probability calculations.

    Notes about the mask:   You can use the mask input to analyze different types
                            of recalls (e.g., correct recalls, repetitions,
                            intrusions). There are masking functions that will
                            create these.

    To only look at correct responses:  mask = make_clean_recalls_mask2d(recalls)

    To only look at repetitions:        mask = make_mask_only_reps2d(recalls);
    To look at intrusions, you can create a mask using an
    intrusions matrix (which must be the same size as
    recalls, where a positive integer indicates a prior
    list intrusions, and a -1 indicates an extra list
    intrusion)

    To only look at PLIs:
    mask = make_mask_only_pli2d(intrusions);

    To only look at XLIs:
    mask = make_mask_only_xli2d(intrusions);

    You can use the repetition mask, PLI mask, and XLI mask
    to create a mask for all incorrect recalls.

    EXAMPLE:
    [p_stops,denom] = p_stop_perc(recalls,time_mat,rec_length,12000,subjects,mask)
    """
    if recalls is None:
        raise Exception('You must pass a recalls matrix.')
    elif subject is None:
        raise Exception('You must pass a subjects vector.')

    elif len(recalls) != len(subject):
        raise Exception('recalls matrix must have the same number of rows as subjects.')
    if rec_mask == None:
        rec_mask = mask.make_clean_recalls_mask2d(recalls)
    if any([time != None, record_time != None, exit_time_thresh != None]) and not all([time != None, record_time != None, exit_time_thresh != None]):
        raise Exception('You must pass a time_mat, recall_length scalar, and an exit_time_thresh scalar, or all must be empty.')
    elif all([time != None, record_time != None, exit_time_thresh != None]):
        marker = True
        if len(time) != len(recalls):
            raise Exception('time matrix needs to be same shape as recalls')
    else:
        marker = False
    recalls = mask.mask_data(recalls, rec_mask)
    subjects = np.unique(subject)
    result = []
    for subj in subjects:
        stop = [0] * len(recalls[0])
        num = [0] * len(recalls[0])
        for subj_ind, subj_num in enumerate(subject):
            if subj == subj_num:
                if marker == True:
                    if last_nonzero(time[subj_ind]) == None:
                        continue
                    elif record_time - last_nonzero(time[subj_ind]) > exit_time_thresh and record_time - last_nonzero(time[subj_ind]) > max_irt(time[subj_ind]):
                        for n, rec in enumerate(recalls[subj_ind]):
                            if rec != 0:
                                num[n] += 1
                        for n, rec in enumerate(recalls[subj_ind][::-1]):
                            if rec != 0:
                                stop[len(recalls[0]) - n - 1] += 1
                                break
                    else:
                        continue
                else:
                    for n, rec in enumerate(recalls[subj_ind]):
                        if rec != 0:
                            num[n] += 1
                    for n, rec in enumerate(recalls[subj_ind][::-1]):
                        if rec != 0:
                            stop[len(recalls[0]) - n -1] += 1
                            break
        total_num = 0
        total_denom = 0
        for item in stop:
            total_num += item
        for item in num:
            total_denom += item
        result.append(total_num / float(total_denom))
    return result

def last_nonzero(list):
    for item in list[::-1]:
        if item != 0:
            return item
    return None

def max_irt(list):
    max = 0
    for ind, item in enumerate(list):
        if ind < len(list) - 1:
            if list[ind + 1] - item > max:
                max = list[ind + 1] - item
    return max