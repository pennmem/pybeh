import numpy as np


def reps(recalls, subjects, unique_reps=False, per_list=False):
    """
    Calculate's each partcipant's average number of repetitions per list.

    :param recalls: A trials x items matrix whose elements are the serial positions of recalled items. Intrusions
    :param subjects: A list of subject codes, indicating which subject produced each row of the intrusions matrix
    :param unique_reps: If True, counts the number of unique repetitions made. If False, counts all repetitions. For
    example, if a subject recalls the same word 3 times in a list, this counts as 2 repetitions if unique_reps is False,
    and only 1 repetition if unique_reps is True.
    :param per_list: If True, returns the average number of repetitions per list for each subject. If False, returns the
    total number of repetitions made by each subject.
    :return: An array where each entry is the total or average (per list) number of repetitions for a participant.
    """
    recalls = np.array(recalls)
    subjects = np.array(subjects)
    usub = np.unique(subjects)
    result = np.zeros_like(usub, dtype=float)

    for i, subj in enumerate(usub):
        cur_recs = recalls[subjects == subj]
        for trial in cur_recs:
            # Find the number of times each word was recalled during the current trial
            times_recalled = np.array([np.sum(trial == rec) for rec in np.unique(trial) if rec > 0])
            # Subtract 1 from the number of times each correct word was recalled in the list to give the number of
            # repetitions of each correct word
            repetitions = times_recalled - 1
            # Sum the number of repetitions made in the current trial (either unique or any)
            if unique_reps:
                result[i] += np.sum(repetitions > 0)
            else:
                result[i] += repetitions.sum()

        # If desired, convert the raw repetition counts to average repetitions per list
        if per_list:
            result[i] = result[i] / cur_recs.shape[0]

    return result