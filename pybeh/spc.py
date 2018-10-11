from __future__ import division
import numpy as np


def spc(recalls=None, subjects=None, listLength=None):
    """
    SPC   Serial position curve (recall probability by serial position).

    p_recall = spc(recalls, subjects, listLength)

    INPUTS:
        recalls:    a matrix whose elements are serial positions of recalled
                    items. The rows of this matrix should represent recalls
                    made by a single subject on a single trial.

        subjects:   a 1D array which indexes the rows of the recalls matrix
                    with a subject number (or other identifier). That is,
                    the recall trials of subject S should be located in
                    recalls[subjects==S, :]

        listLength: a scalar indicating the number of serial positions in the
                    presented lists. serial positions are assumed to run
                    from [1, 2, ..., list_length].


    OUTPUTS:
        p_recall:  a matrix of probablities.  Its columns are indexed by
                   serial position and its rows are indexed by subject.
    """
    if recalls is None:
        raise Exception('You must pass a recalls matrix.')
    elif subjects is None:
        raise Exception('You must pass a subjects vector.')
    elif listLength is None:
        raise Exception('You must pass a list length.')
    elif len(recalls) != len(subjects):
        raise Exception('recalls matrix must have the same number of rows as subjects.')

    # Convert inputs to numpy arrays if they are not already
    recalls = np.array(recalls)
    subjects = np.array(subjects)
    # Get a list of unique subjects
    usub = np.unique(subjects)
    # Create list of all serial positions (starting with 1)
    positions = np.arange(1, listLength+1)
    # We will return one SPC for each unique subject
    result = np.zeros((len(usub), listLength))

    for i, subj in enumerate(usub):
        # Select only the trials from the current subject
        subj_recalls = recalls[subjects == subj]
        # Create a "recalled" matrix of ones and zeroes indicating whether each presented item was correctly recalled
        subj_recalled = np.array([np.isin(positions, trial_data) for trial_data in subj_recalls])
        # Calculate the subject's SPC as the fraction of trials on which they recalled each serial position's item
        result[i, :] = subj_recalled.mean(axis=0)

    return result
