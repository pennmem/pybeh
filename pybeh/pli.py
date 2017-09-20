import numpy as np


def pli(intrusions=None, subjects=None, per_list=False, exclude_reps=True):
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

        per_list:       Boolean indicating whether raw counts or average per-list
                        counts should be returned. Returns raw counts if False,
                        average count per list if True. (Default == False)

        exclude_reps:   COMING SOON. If exclude_reps is True, each PLI word will
                        only be counted once per list. If False, every repeat of
                        a given PLI word will be counted. (MATLAB default was True)
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

    # Get list of unique participants (or other trial identifier)
    usub = np.unique(subjects)
    # PLIs are any value greater than 0 in the intrusions matrix
    plis = np.array(intrusions) > 0
    # Count the PLIs from each subject
    result = [np.sum(plis[subjects == subj, :]) for subj in usub] if not per_list \
        else [np.sum(plis[subjects == subj, :]) / plis[subjects == subj].shape[0] for subj in usub]

    return result
