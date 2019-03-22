import numpy as np
from pybeh.mask_maker import make_clean_recalls_mask2d


def temp_fact(recalls=None, subjects=None, listLength=None, skip_first_n=0):
    """
    Returns the lag-based temporal clustering factor for each subject (Polyn, Norman, & Kahana, 2009).

    :param recalls: A trials x recalls matrix containing the serial positions (between 1 and listLength) of words
        recalled on each trial. Intrusions should appear as -1, and the matrix should be padded with zeros if the number
        of recalls differs by trial.
    :param subjects: A list/array containing identifiers (e.g. subject number) indicating which subject completed each
        trial.
    :param listLength: A positive integer indicating the number of items presented on each trial.
    :param skip_first_n: An integer indicating the number of recall transitions to ignore from the start of each recall
        period, for the purposes of calculating the clustering factor. This can be useful to avoid biasing your results,
        as early transitions often differ from later transition in terms of their clustering. Note that the first n
        recalls will still count as already recalled words for the purposes of determining which transitions are
        possible. (DEFAULT=0)

    :return: An array containing the temporal clustering factor score for each subject (sorted by alphabetical order).
    """

    if recalls is None:
        raise Exception('You must pass a recalls matrix.')
    if subjects is None:
        raise Exception('You must pass a subjects vector.')
    if listLength is None:
        raise Exception('You must pass a list length.')
    if len(recalls) != len(subjects):
        raise Exception('The recalls matrix must have the same number of rows as the list of subjects.')
    if not isinstance(skip_first_n, int) or skip_first_n < 0:
        raise ValueError('skip_first_n must be a nonnegative integer.')

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
        seen = set()
        # Identify the current subject's index in usub to determine their position in the total and count arrays
        subj_ind = np.where(usub == subjects[i])[0][0]
        # Loop over the recalls on the current trial
        for j, serialpos in enumerate(trial_data[:-1]):
            seen.add(serialpos)
            # Only count transition if both the current and next recalls are valid
            if clean_recalls_mask[i, j] and clean_recalls_mask[i, j+1] and j >= skip_first_n:
                # Identify possible transitions
                possibles = np.array([abs(item - serialpos) for item in possibles_range if item not in seen])
                # Identify actual transition
                next_serialpos = trial_data[j + 1]
                actual = abs(next_serialpos - serialpos)
                # Find the proportion of transition lags that were larger than the actual transition
                ptile_rank = temp_percentile_rank(actual, possibles)
                # Add transition to the appropriate participant's score
                if ptile_rank is not None:
                    total[subj_ind] += ptile_rank
                    count[subj_ind] += 1

    # Find temporal factor scores as the participants' average transition scores
    count[count == 0] = np.nan
    final_data = total / count

    return final_data


def temp_percentile_rank(actual, possible):
    """
    Helper function to return the percentile rank of the actual transition within the list of possible transitions.

    :param actual: The distance of the actual transition that was made.
    :param possible: The list of all possible transition distances that could have been made.

    :return: The proportion of possible transitions that were more distant than the actual transition.
    """
    # If there were fewer than 2 possible transitions, we can't compute a meaningful percentile rank
    if len(possible) < 2:
        return None

    # Sort possible transitions from largest to smallest
    possible = sorted(possible)[::-1]

    # Get indices of the one or more possible transitions with the same distance as the actual transition
    matches = np.where(possible == actual)[0]

    if len(matches) > 0:
        # Get the number of possible transitions that were more distant than the actual transition
        # If there were multiple transitions with the same distance as the actual one, average across their ranks
        rank = np.mean(matches)
        # Convert rank to the proportion of possible transitions that were more distant than the actual transition
        ptile_rank = rank / (len(possible) - 1.)
    else:
        ptile_rank = None

    return ptile_rank
