from __future__ import division
import numpy as np
from pybeh.mask_maker import make_clean_recalls_mask2d

def sem_crp(recalls = None, recalls_itemnos = None, pres_itemnos = None, subjects = None, sem_sims = None, n_bins = 10, listLength = None):
    """sanity check"""
    if recalls_itemnos is None:
        raise Exception('You must pass a recalls-by-item-numbers matrix.')
    elif pres_itemnos is None:
        raise Exception('You must pass a presentations-by-item-numbers matrix.')
    elif sem_sims is None:
        raise Exception('You must pass a semantic similarity matrix.')
    elif subjects is None:
        raise Exception('You must pass a subjects vector.')
    elif listLength is None:
        raise Exception('You must pass a listLength')
    elif len(recalls_itemnos) != len(subjects):
        raise Exception('recalls matrix must have the same number of rows as subjects.')

    # Make sure that all input arrays and matrices are numpy arrays
    recalls = np.array(recalls)
    recalls_itemnos = np.array(recalls_itemnos)
    pres_itemnos = np.array(pres_itemnos)
    subjects = np.array(subjects)
    sem_sims = np.array(sem_sims)

    # Set diagonal of the similarity matrix to nan
    np.fill_diagonal(sem_sims, np.nan)
    # Sort and split all similarities into equally sized bins
    all_sim = sem_sims.flatten()
    all_sim = np.sort(all_sim[~np.isnan(all_sim)])
    bins = np.array_split(all_sim, n_bins)
    bins = [b[0] for b in bins]
    # Convert the similarity matrix to bin numbers for easy bin lookup later
    bin_sims = np.digitize(sem_sims, bins) - 1

    # Convert recalled item numbers to the corresponding indices of the similarity matrix by subtracting 1
    recalls_itemnos -= 1

    usub = np.unique(subjects)
    bin_means = np.zeros((len(usub), n_bins))
    crp = np.zeros((len(usub), n_bins))
    # For each subject
    for i, subj in enumerate(usub):
        # Create a filter to select only the current subject's data
        subj_mask = subjects == subj
        subj_recalls = recalls[subj_mask]
        subj_rec_itemnos = recalls_itemnos[subj_mask]
        subj_pres_itemnos = pres_itemnos[subj_mask]

        # Create trials x items matrix where item j, k indicates whether the kth recall on trial j was a correct recall
        clean_recalls_mask = np.array(make_clean_recalls_mask2d(subj_recalls))

        # Setup counts for number of possible and actual transitions, as well as the sim value of actual transitions
        actual = np.zeros(n_bins)
        poss = np.zeros(n_bins)
        val = np.zeros(n_bins)

        # For each of the current subject's trials
        for j, trial_recs in enumerate(subj_recalls):
            seen = set()
            # For each recall on the current trial
            for k, rec in enumerate(trial_recs[:-1]):
                seen.add(rec)
                # Only increment transition counts if the current and next recall are BOTH correct recalls
                if clean_recalls_mask[j, k] and clean_recalls_mask[j, k+1]:
                    this_recno = subj_rec_itemnos[j, k]
                    next_recno = subj_rec_itemnos[j, k+1]
                    # Lookup semantic similarity and its bin between current recall and next recall
                    sim = sem_sims[this_recno, next_recno]
                    b = bin_sims[this_recno, next_recno]
                    actual[b] += 1
                    val[b] += sim

                    # Get a list of not-yet-recalled word numbers
                    poss_rec = [subj_pres_itemnos[j][x] for x in range(listLength) if x+1 not in seen]
                    # Lookup the similarity bins between the current recall and all possible correct recalls
                    poss_trans = np.unique([bin_sims[this_recno, itemno] for itemno in poss_rec])
                    for b in poss_trans:
                        poss[b] += 1

        crp[i, :] = actual / poss  # CRP is calculated as number of actual transitions / number of possible ones
        bin_means[i, :] = val / actual  # Bin means are defined as the average similarity of actual transitions per bin

    return bin_means, crp
