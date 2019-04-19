import numpy as np

def pnr(recalls, subjects, listLength, n=0):
    """
    PNR   Probability of nth recall.

    Computes probability of recall by serial position for the
    nth output position.
    [p_recalls] = pfr(recalls_matrix, subjects, list_length)

    INPUTS:
        recalls:    a matrix whose elements are serial positions of recalled
                    items.  The rows of this matrix should represent recalls
                    made by a single subject on a single trial.

        subjects:   a column vector which indexes the rows of recalls_matrix
                    with a subject number (or other identifier).  That is,
                    the recall trials of subject S should be located in
                    recalls_matrix(find(subjects==S), :)

        listLength: a scalar indicating the number of serial positions in the
                    presented lists.  serial positions are assumed to run
                    from 1:list_length.

        n:          the output position for which the probabilities of nth recall
                    will be calculated (Zero indexed, i.e. n=0 produces probability
                    of first recall, n=1 produces probability of second recall, etc.)


    OUTPUTS:
        p_recalls:  a matrix of probablities.  Its columns are indexed by
                    serial position and its rows are indexed by subject.
    """
    if len(recalls) != len(subjects):
        raise ValueError('The recalls matrix must have the same number of rows as subjects.')
    if n >= listLength:
        raise ValueError('N must be less than the list length.')

    usub = np.unique(subjects)
    result = np.zeros((len(usub), listLength))

    # Get the Nth recall from each trial
    nth_recs = np.array(recalls)[:, n]

    for i, subj in enumerate(usub):
        # Select only the trials from the current subject
        subj_recs = nth_recs[subjects == subj]
        # Count the number of times each serial position was recalled in output position N
        for rec in subj_recs[subj_recs > 0]:
            result[i, rec - 1] += 1
        # Divide by the number of trials the participant completed
        result[i] /= len(subj_recs)

    return result
