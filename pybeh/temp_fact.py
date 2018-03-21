from __future__ import division
import numpy as np
from pybeh.mask_maker import make_clean_recalls_mask2d

def temp_fact(recalls=None, subjects=None, listLength=None, skip_first_n=0):
    """
    returns a Lag-based temporal clustering factor for each subject

    INPUTS:
        recalls:    a matrix whose elements are serial positions of recalled
                    items.  The rows of this matrix should represent recalls
                    made by a single subject on a single trial.

        subjects:   a column vector which indexes the rows of recalls
                    with a subject number (or other identifier).  That is,
                    the recall trials of subject S should be located in
                    recalls(find(subjects==S), :)

        list_length:a scalar indicating the number of serial positions in the
                    presented lists.  serial positions are assumed to run
                    from 1:list_length.

       skip_first_n:  an integer indicating the number of recall transitions to
                      to ignore from the start of the recall period, for the
                      purposes of calculating the temporal factor. this can be useful
                      to avoid biasing your results, as the first 2-3 transitions are
                      almost always temporally clustered. note that the first n recalls
                      will still count as already recalled words for the purposes of
                      determining which transitions are possible. (DEFAULT=0)

    OUTPUTS:
        temp_facts: a vector of temporal clustering factors, one
                    for each subject.
    """
    if recalls is None:
        raise Exception('You must pass a recalls matrix.')
    elif subjects is None:
        raise Exception('You must pass a subjects vector.')
    elif listLength is None:
        raise Exception('You must pass a list length.')
    elif len(recalls) != len(subjects):
        raise Exception('recalls matrix must have the same number of rows as subjects.')
    if not isinstance(skip_first_n, int):
        raise ValueError('skip_first_n must be an integer.')

    # Convert recalls and subjects to numpy arrays if they are not arrays already
    recalls = np.array(recalls)
    subjects = np.array(subjects)

    # Initialize range for possible next recalls, based on list length
    possibles_range = range(1, listLength + 1)

    # Initialize arrays to store each participant's results
    usub = np.unique(subjects)
    total = np.zeros_like(usub, dtype=float)
    count = np.zeros_like(usub, dtype=float)

    # Identify locations of all correct recalls (not PLI, ELI, or repetition)
    clean_recalls_mask = np.array(make_clean_recalls_mask2d(recalls))

    # Calculate temporal factor score for each trial
    for i, trial_data in enumerate(recalls):
        seen = []
        subj_ind = np.where(usub == subjects[i])[0][0]  # Identify the current subject's index in the total and count arrays
        # Loop over recalls on current trial
        for j, serialpos in enumerate(trial_data[:-1]):
            seen.append(serialpos)
            # Only count transition if both recalls involved are correct
            if clean_recalls_mask[i, j] and clean_recalls_mask[i, j+1] and j >= skip_first_n:
                # Identify possible transitions
                possibles = [abs(item - serialpos) for item in possibles_range if item not in seen]
                # Identify actual transition
                next_serialpos = trial_data[j + 1]
                actual = abs(next_serialpos - serialpos)
                # Find percentile rank of actual size of transition among possible transition sizes
                ptile_rank = temp_percentile_rank(actual, possibles)
                # Add transition to the appropriate participant's score
                if ptile_rank is not None:
                    total[subj_ind] += ptile_rank
                    count[subj_ind] += 1

    # Find temporal factor scores as the participants' average transition scores
    final_data = total / count

    return final_data

#Helper function to return the percentile rank of the input within the possible inputs.
#Gives out the mean of the values of input for ties

def temp_percentile_rank(actual, possible):
    possible = sorted(possible)
    temp = 0
    count = 0
    for index, item in enumerate(possible):
        if item == actual and len(possible) > 1:
            temp += (len(possible) - 1 - index) / (len(possible) - 1)
            count += 1
    if count != 0:
        return temp / count
    return None