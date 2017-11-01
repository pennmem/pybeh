import numpy as np


def pli(intrusions, subjects, rec_items=None, exclude_reps=False, per_list=False):
    """
    PLI   Number of prior list intrusions.

    plis = pli(intrusions, subjects)

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

        rec_items:      A trials x recalls matrix of strings indicating which words
                        were recalled at which output positions on each trial. Only
                        required for excluding repetitions of the same intrusion, i.e.
                        if exclude_reps == True.

        exclude_reps:   If exclude_reps is True, each PLI word will only be counted
                        once per list. If False, every repeat of a given PLI word will
                        be counted. (Default == False)

        per_list:       Boolean indicating whether raw counts or average per-list
                        counts should be returned. Returns raw counts if False,
                        average count per list if True. (Default == False)

    OUTPUTS:
        plis:           vector of total number of PLIs. Its rows are indexed
                        by subject.

    """
    if intrusions is None:
        raise Exception('You must pass a intrusions matrix.')
    if subjects is None:
        raise Exception('You must pass a subjects vector.')
    if len(intrusions) != len(subjects):
        raise Exception('recalls matrix must have the same number of rows as subjects.')
    if not isinstance(per_list, bool):
        raise Exception('per_list must be True or False.')
    if not isinstance(exclude_reps, bool):
        raise Exception('exclude_reps must be True or False.')
    if exclude_reps and rec_items is None:
        raise Exception('rec_items must be provided in order to exclude repetitions.')

    subjects = np.array(subjects)
    # Get list of unique participants (or other trial identifier)
    usub = np.unique(subjects)
    # PLIs are any value greater than 0 in the intrusions matrix
    plis = np.array(intrusions) > 0

    if exclude_reps:
        rec_items = np.array(rec_items)
        result = np.zeros_like(usub, dtype=float)
        for i, subj in enumerate(usub):
            # Get PLI map from current subject
            cur_plis = plis[subjects == subj]
            cur_recs = rec_items[subjects == subj]
            for j, row in enumerate(cur_plis):
                # Identify the index of the first occurrence of each unique recall in a trial
                _, indx = np.unique(cur_recs[j], return_index=True)
                # Count the number of unique PLIs using these indices
                result[i] += np.sum(row[indx])
            # Convert raw counts to average PLIs per trial if desired
            if per_list:
                result[i] = result[i] / cur_plis.shape[0]
        result = result.tolist()
    else:
        # Count the PLIs from each subject
        result = [np.sum(plis[subjects == subj, :]) for subj in usub] if not per_list \
            else [np.sum(plis[subjects == subj, :]) / plis[subjects == subj].shape[0] for subj in usub]

    return result
