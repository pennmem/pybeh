from __future__ import division
import numpy as np

def sem_crl(recalls = None, times = None, recalls_itemnos = None, pres_itemnos = None, subjects = None, sem_sims = None, n_bins = None, listLength = None):
    """sanity check"""
    if recalls_itemnos is None:
        raise Exception('You must pass a recalls-by-item-numbers matrix.')
    elif pres_itemnos is None:
        raise Exception('You must pass a presentations-by-item-numbers matrix.')
    elif sem_sims is None:
        raise Exception('You must pass a semantic similarity matrix.')
    elif subjects is None:
        raise Exception('You must pass a subjects vector.')
    elif n_bins is None:
        n_bins = 10
    elif listLength is None:
        raise Exception('You must pass a listLength')
    elif len(recalls_itemnos) != len(subjects):
        raise Exception('recalls matrix must have the same number of rows as subjects.')

    bin_total = [0] * n_bins
    bin_times = [0] * n_bins
    bin_counts = [0] * n_bins
    bin_val = [0] * n_bins

    """find all values in similarity matrix"""
    all_val = [sem_sims[item1][item2] for item1 in range(len(sem_sims)) for item2 in range(len(sem_sims[0]))]
    all_val = np.sort(all_val)
    all_val = all_val[~np.isnan(all_val)]
    all_val = list(chunkIt(all_val, n_bins))

    for subj in range(len(recalls)):
        encounter = []
        for sp in range(len(recalls[0])):
            if recalls[subj][sp] > 0 and recalls[subj][sp] < listLength + 1 and recalls[subj][sp] not in encounter:
                encounter.append(recalls[subj][sp])
                if sp + 1 < len(recalls[0]):
                    if recalls[subj][sp + 1] > 0 and recalls[subj][sp + 1] < listLength + 1 and (
                        recalls[subj][sp + 1] != recalls[subj][sp]) and (recalls[subj][sp + 1] not in encounter):

                        if np.isnan(sem_sims[(int)(recalls_itemnos[subj][sp] - 1)][(int)(recalls_itemnos[subj][sp + 1] - 1)]) != True:
                            add_to_bin(sem_sims[int(recalls_itemnos[subj][sp] - 1)][int(recalls_itemnos[subj][sp + 1] - 1)],
                                       times[subj][sp+1] - times[subj][sp], all_val, bin_val, bin_times, bin_counts)
                            for i in poss_bin(encounter, sp, all_val, subj, sem_sims, pres_itemnos, recalls_itemnos, listLength):
                                bin_total[i] += 1

    bin_mean = [0] * n_bins
    crl = [0] * n_bins

    for index in range(n_bins):
        if bin_times[index] == 0:
            bin_mean[index] = 0
        else:
            bin_mean[index] = bin_val[index] / float(bin_counts[index])
        if bin_total[index] == 0:
            crl[index] = 0
        else:
            crl[index] = bin_times[index] / float(bin_total[index])
    return bin_mean, crl

"""helper function to chunk sequence into equally sized bins"""


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out


"""function that returns the index of bin the semantic similarity value belongs to"""
def find_bin(sem_sim_val, bin):
    for index in range(len(bin)):
        if sem_sim_val >= bin[index][0] and sem_sim_val <= bin[index][len(bin[index])-1]:
            return index

"""function adds semantic similarity value to total bin count"""


def add_to_bin(sem_sim_val, trans_time, bin, bin_val, bin_times, bin_counts):
    bin_val[find_bin(sem_sim_val, bin)] += sem_sim_val
    bin_counts[find_bin(sem_sim_val, bin)] += 1
    bin_times[find_bin(sem_sim_val, bin)] += trans_time


"""function that returns the possible bins for next item """


def poss_bin(encounter, sp, bin, subj, sem_sims, pres_itemnos, recalls_itemnos, listLength):
    temp = []
    for index in range(listLength):
        if index+1 not in encounter:
            if not np.isnan(sem_sims[int(recalls_itemnos[subj][sp])-1][int(pres_itemnos[subj][index])-1]):
                if find_bin(sem_sims[int(recalls_itemnos[subj][sp])-1][int(pres_itemnos[subj][index])-1], bin) not in temp:
                    temp.append(find_bin(sem_sims[int(recalls_itemnos[subj][sp]) -1][int(pres_itemnos[subj][index])-1], bin))
    return temp
