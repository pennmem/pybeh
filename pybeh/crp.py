from __future__ import division
import numpy as np
from pybeh.mask_maker import make_clean_recalls_mask2d


def crp(recalls=None, subjects=None, listLength=None, lag_num=None, skip_first_n=0):
    """
    %CRP   Conditional response probability as a function of lag (lag-CRP).
    %
    %  lag_crps = crp(recalls_matrix, subjects, list_length, lag_num)
    %
    %  INPUTS:
    %  recalls_matrix:  a matrix whose elements are serial positions of recalled
    %                   items.  The rows of this matrix should represent recalls
    %                   made by a single subject on a single trial.
    %
    %        subjects:  a column vector which indexes the rows of recalls_matrix
    %                   with a subject number (or other identifier).  That is,
    %                   the recall trials of subject S should be located in
    %                   recalls_matrix(find(subjects==S), :)
    %
    %     list_length:  a scalar indicating the number of serial positions in the
    %                   presented lists.  serial positions are assumed to run
    %                   from 1:list_length.
    %
    %         lag_num:  a scalar indicating the max number of lag to keep track
    %
    %    skip_first_n:  an integer indicating the number of recall transitions to
    %                   to ignore from the start of the recall period, for the
    %                   purposes of calculating the CRP. this can be useful to avoid
    %                   biasing your results, as the first 2-3 transitions are
    %                   almost always temporally clustered. note that the first
    %                   n recalls will still count as already recalled words for
    %                   the purposes of determining which transitions are possible.
    %                   (DEFAULT=0)
    %
    %
    %  OUTPUTS:
    %        lag_crps:  a matrix of lag-CRP values.  Each row contains the values
    %                   for one subject.  It has as many columns as there are
    %                   possible transitions (i.e., the length of
    %                   (-list_length + 1) : (list_length - 1) ).
    %                   The center column, corresponding to the "transition of
    %                   length 0," is guaranteed to be filled with NaNs.
    %
    %                   For example, if list_length == 4, a row in lag_crps
    %                   has 7 columns, corresponding to the transitions from
    %                   -3 to +3:
    %                   lag-CRPs:     [ 0.1  0.2  0.3  NaN  0.3  0.1  0.0 ]
    %                   transitions:    -3   -2    -1   0    +1   +2   +3
    """
    if recalls is None:
        raise Exception('You must pass a recalls matrix.')
    elif subjects is None:
        raise Exception('You must pass a subjects vector.')
    elif listLength is None:
        raise Exception('You must pass a list length.')
    elif len(recalls) != len(subjects):
        raise Exception('recalls matrix must have the same number of rows as subjects.')
    if lag_num is None:
        lag_num = listLength - 1
    elif lag_num < 1 or lag_num >= listLength or not isinstance(lag_num, int):
        raise ValueError('Lag number needs to be a positive integer that is less than the list length.')
    if not isinstance(skip_first_n, int):
        raise ValueError('skip_first_n must be an integer.')

    # Convert recalls and subjects to numpy arrays
    recalls = np.array(recalls)
    subjects = np.array(subjects)
    # Get a list of unique subjects -- we will calculate a CRP for each
    usub = np.unique(subjects)
    # Number of possible lags = (listLength - 1) * 2 + 1; e.g. a length-24 list can have lags -23 through +23
    num_lags = 2 * listLength - 1
    # Initialize array to store the CRP for each subject (or other unique identifier)
    result = np.zeros((usub.size, num_lags))
    # Initialize arrays to store transition counts
    actual = np.empty(num_lags)
    poss = np.empty(num_lags)

    # For each subject/unique identifier
    for i, subj in enumerate(usub):
        # Reset counts for each participant
        actual.fill(0)
        poss.fill(0)
        # Create trials x items matrix where item j, k indicates whether the kth recall on trial j was a correct recall
        clean_recalls_mask = np.array(make_clean_recalls_mask2d(recalls[subjects == subj]))
        # For each trial that matches that identifier
        for j, trial_recs in enumerate(recalls[subjects == subj]):
            seen = set()
            for k, rec in enumerate(trial_recs[:-1]):
                seen.add(rec)
                # Only increment transition counts if the current and next recall are BOTH correct recalls
                if clean_recalls_mask[j][k] and clean_recalls_mask[j][k + 1] and k >= skip_first_n:
                    next_rec = trial_recs[k + 1]
                    pt = np.array([trans for trans in range(1 - rec, listLength + 1 - rec) if rec + trans not in seen], dtype=int)
                    poss[pt + listLength - 1] += 1
                    trans = next_rec - rec
                    # Record the actual transition that was made
                    actual[trans + listLength - 1] += 1

        result[i, :] = [a/p if p!=0 else np.nan for a,p in zip(actual, poss)]

    result[:, listLength - 1] = np.nan

    return result[:, listLength - lag_num - 1:listLength + lag_num]
