import numpy as np
from pybeh.pnr import pnr

def pfr(recalls, subjects, listLength):
    """"
    PFR   Probability of first recall.

    Computes probability of recall by serial position for the
    first output position.
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

    OUTPUTS:
        p_recalls:  a matrix of probablities.  Its columns are indexed by
                    serial position and its rows are indexed by subject.
    """
    return pnr(recalls, subjects, listLength, 0)