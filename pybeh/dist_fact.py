import warnings
import numpy as np
from pybeh.mask_maker import make_clean_recalls_mask2d
from pybeh.make_recalls_matrix import make_recalls_matrix


def dist_fact(rec_itemnos=None, pres_itemnos=None, subjects=None, dist_mat=None, is_similarity=False, skip_first_n=0):
    """
    Returns a clustering factor score for each subject, based on the provided distance metric (Polyn, Norman, & Kahana,
    2009). Can also be used with a similarity matrix (e.g. LSA, word2vec) if is_similarity is set to True.

    :param rec_itemnos: A trials x recalls matrix containing the ID numbers (between 1 and N) of the items recalled on
        each trial. Extra-list intrusions should appear as -1, and the matrix should be padded with zeros if the number
        of recalls differs by trial.
    :param pres_itemnos: A trials x items matrix containing the ID numbers (between 1 and N) of the items presented on
        each trial.
    :param subjects: A list/array containing identifiers (e.g. subject number) indicating which subject completed each
        trial.
    :param dist_mat: An NxN matrix (where N is the number of words in the wordpool) defining either the distance or
        similarity between every pair of words in the wordpool. Whether dist_mat defines distance or similarity can be
        specified with the is_similarity parameter.
    :param is_similarity: If False, dist_mat is assumed to be a distance matrix. If True, dist_mat is instead treated as
        a similarity matrix (i.e. larger values correspond to smaller distances). (DEFAULT = False)
    :param skip_first_n: An integer indicating the number of recall transitions to ignore from the start of each recall
        period, for the purposes of calculating the clustering factor. This can be useful to avoid biasing your results,
        as early transitions often differ from later transition in terms of their clustering. Note that the first n
        recalls will still count as already recalled words for the purposes of determining which transitions are
        possible. (DEFAULT = 0)

    :return: An array containing the clustering factor score for each subject (sorted by alphabetical order).
    """

    if rec_itemnos is None:
        raise Exception('You must pass a recall_itemnos matrix.')
    if pres_itemnos is None:
        raise Exception('You must pass a pres_itemnos matrix.')
    if subjects is None:
        raise Exception('You must pass a subjects vector.')
    if dist_mat is None:
        raise Exception('You must pass either a similarity matrix or a distance matrix.')
    if len(rec_itemnos) != len(subjects) or len(pres_itemnos) != len(subjects):
        raise Exception('The rec_itemnos and pres_itemnos matrices must have the same number of rows as the list of'
                        'subjects.')
    if not isinstance(skip_first_n, int) or skip_first_n < 0:
        raise ValueError('skip_first_n must be a nonnegative integer.')

    # Convert inputs to numpy arrays if they are not arrays already
    rec_itemnos = np.array(rec_itemnos)
    pres_itemnos = np.array(pres_itemnos)
    subjects = np.array(subjects)
    dist_mat = np.array(dist_mat)

    # Provide a warning if the user inputs a dist_mat that looks like a similarity matrix (scores on diagonal are
    # large), but has left is_similarity as False
    if (not is_similarity) and np.nanmean(np.diagonal(dist_mat)) > np.nanmean(dist_mat):
        warnings.warn('It looks like you might be using a similarity matrix (e.g. LSA, word2vec) instead of a distance'
                      ' matrix, but you currently have is_similarity set to False. If you are using a similarity'
                      ' matrix, make sure to set is_similarity to True when running dist_fact().')

    # Initialize arrays to store each participant's results
    usub = np.unique(subjects)
    total = np.zeros_like(usub, dtype=float)
    count = np.zeros_like(usub, dtype=float)

    # Identify locations of all correct recalls (not PLI, ELI, or repetition)
    clean_recalls_mask = np.array(make_clean_recalls_mask2d(make_recalls_matrix(pres_itemnos, rec_itemnos)))

    # Calculate distance factor score for each trial
    for i, trial_data in enumerate(rec_itemnos):
        seen = set()
        # Identify the current subject's index in usub to determine their position in the total and count arrays
        subj_ind = np.where(usub == subjects[i])[0][0]
        # Loop over the recalls on the current trial
        for j, rec in enumerate(trial_data[:-1]):
            seen.add(rec)
            # Only count transition if both the current and next recalls are valid
            if clean_recalls_mask[i, j] and clean_recalls_mask[i, j+1] and j >= skip_first_n:
                # Identify the distance between the current recall and all valid recalls that could follow it
                possibles = np.array([dist_mat[rec - 1, poss_rec - 1] for poss_rec in pres_itemnos[i] if poss_rec not in seen])
                # Identify the distance between the current recall and the next
                actual = dist_mat[rec - 1, trial_data[j + 1] - 1]
                # Find the proportion of possible transitions that were larger than the actual transition
                ptile_rank = dist_percentile_rank(actual, possibles, is_similarity)
                # Add transition to the appropriate participant's score
                if ptile_rank is not None:
                    total[subj_ind] += ptile_rank
                    count[subj_ind] += 1

    # Find temporal factor scores as the participants' average transition scores
    count[count == 0] = np.nan
    final_data = total / count

    return final_data


def dist_percentile_rank(actual, possible, is_similarity=False):
    """
    Helper function to return the percentile rank of the actual transition within the list of possible transitions.

    :param actual: The distance of the actual transition that was made.
    :param possible: The list of all possible transition distances that could have been made.
    :is_similarity: If False, actual and possible values are assumed to be distances. If True, values are assumed to be
        similarity scores, where smaller values correspond to more distant transitions.

    :return: The proportion of possible transitions that were more distant than the actual transition.
    """
    # If there were fewer than 2 possible transitions, we can't compute a meaningful percentile rank
    if len(possible) < 2:
        return None

    # Sort possible transitions from largest to smallest distance (taking into account whether the values are
    # similarities or distances)
    possible = sorted(possible) if is_similarity else sorted(possible)[::-1]

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
