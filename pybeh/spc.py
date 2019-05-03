from __future__ import division
import numpy as np


def spc(recalls, subjects, listLength, start_position=None):
    """
    Serial position curve (recall probability by serial position).

    :param recalls: A trials x items matrix whose elements are the serial positions of the items recalled on each trial.
        Item (i, j) should therefore be the serial position of the jth item recalled on trial i.
    :param subjects: A 1D array indicating which subject (or other identifier) produced the data from each row of the
        recalls matrix, i.e., recall trials from subject S should be located in recalls[subjects == S, :].
    :param listLength: A scalar indicating the length of presented lists. Serial positions are assumed to range from
        1 to listLength (inclusive).
    :param start_position: (Optional) If an integer, only include trials where recall was initiated from that serial
        position. If a list of integers is provided, only include trials where recall was initiated from any of the
        provided serial positions. For example, start_positions=1 will produce a serial position curve for trials where
        recall began from the first list item, and start_positions=[1, 2, 3] will produce an SPC for trials where recall
        began from any of the first three items.
    :return: A 2D numpy array where each row is the average SPC from one subject. Rows will match the subject order
        produced by np.unique(subjects).
    """
    if len(recalls) != len(subjects):
        raise Exception('The recalls matrix must have the same number of rows as subjects.')
    if isinstance(start_position, int):
        start_position = [start_position]

    # Convert inputs to numpy arrays if they are not already
    recalls = np.array(recalls)
    subjects = np.array(subjects)
    # Get list of unique subjects
    usub = np.unique(subjects)
    # Create list of all possible serial positions (from 1 through list length)
    positions = np.arange(1, listLength+1)

    # Calculate average SPC for each unique subject
    result = np.full((len(usub), listLength), np.nan)
    for i, subj in enumerate(usub):
        # Select only the trials from the current subject
        subj_recalls = recalls[(subjects == subj)]
        # If filtering by recall start position, select only trials that match
        if hasattr(start_position, __iter__):
            subj_recalls = subj_recalls[np.isin(subj_recalls[0], start_position), :]
        if len(subj_recalls) > 0:
            # Create a matrix of ones and zeroes indicating whether each presented item was correctly recalled
            subj_recalled = np.array([np.isin(positions, trial_data) for trial_data in subj_recalls])
            # Calculate the subject's SPC as the fraction of trials on which they recalled each serial position's item
            result[i, :] = subj_recalled.mean(axis=0)

    return result
