import numpy as np


def serial_ratios(recalls, subjects, min_recs):
    """
    Calculates the proportion of trials on which participants initiated recall by recalling the first min_recs items
    presented on that trial in serial order. For instance, if min_recs = 4, trials where recall began recall with items
    1->2->3->4 will be considered serial recall. This function can be used to help check whether participants in free
    recall are rehearsing and performing serial recall instead.

    :param recalls: A trials x items matrix whose elements are the serial positions of recalled items.
    :param subjects: An array of identifier values indicating which subject (or session) each row of the recalls matrix
        originates from.
    :param min_recs: The minimum number of serial recalls that must be made on a trial to be counted as serial recall.
    :return: An array containing the fraction of trials on which each participant performed serial recall. Participants'
        results are sorted by alphabetical order if subject identifiers were strings, or numerical order if identifiers
        were integers.
    """
    if len(recalls) != len(subjects):
        raise Exception('Recalls matrix must have the same number of rows as subjects.')

    usub = np.unique(subjects)
    results = np.zeros(len(usub))

    # If min_recs is greater than the maximum number of recalls anyone made, just return an array of zeros.
    if min_recs > recalls.shape[1]:
        return results

    # Determine whether the first min_recs recalls in each trial matched the first min_recs words presented, in order.
    # In other words, word 1 should be recalled at index 0, word 2 should be recalled at index 1, etc.
    serial = np.ones(recalls.shape[0], dtype=bool)
    for i in range(min_recs):
        serial = serial & (recalls[:, i] == i + 1)

    # Calculate the proportion of trials from each subject that demonstrate serial recall
    for i, subj in enumerate(usub):
        results[i] = np.mean(serial[subjects == subj])

    return results
