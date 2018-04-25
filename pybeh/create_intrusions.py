import copy

def intrusions(rec_itemnos= None, pres_itemnos= None, subjects= None, sessions= None):
    """
    CREATE_INTRUSIONS  Create a standard intrusions matrix.

    USAGE:
    create_intrusions(rec_itemnos, pres_itemnos, subjects, sesssions)

    INPUTS:
    rec_itemnos: a matrix whose elements are INDICES of recalled
    items. The rows of this matrix should represent recalls
    made by a single subject on a single trial.

    pres_itemnos: a matrix whose elements are INDICES of PRESENTED
    items. The rows of this matrix should represent the index of words
    shown to subjects during a trial.

    subjects: a column vector which indexes the rows of
    rec_itemnos with a subject number (or other identifier).
    That is, the recall trials of subject S should be located in
    rec_itemnos(find(subjects==S), :smile:

    sessions: a column vector which indexes the rows of
    rec_itemnos with a session number (or other identifier).
    That is, the recall trials of subject S and session R should be
    located in rec_itemnos(find(subjects==S & sessions==R), :smile:
    """
    if rec_itemnos is None:
        raise Exception('You must pass a rec_itemnos matrix.')
    elif pres_itemnos is None:
        raise Exception('You must pass a presentations-by-item-numbers matrix.')
    elif subjects is None:
        raise Exception('You must pass a subjects vector.')
    elif sessions is None:
        raise Exception('You must pass a sessions vector.')
    elif len(rec_itemnos) != len(subjects):
        raise Exception('rec_itemnos matrix must have the same number of rows as subjects.')
    elif len(sessions) != len(subjects):
        raise Exception('sessions vector must have the same length as subjects.')
    result = copy.deepcopy(rec_itemnos)
    for num, list in enumerate(rec_itemnos):
        for index, item in enumerate(list):
            # Extra-list intrusions
            if item < 0:
                result[num][index] = -1
                #make non-recalls 0
            # Non-recalls (coming from zero padding in recalls matrix)
            elif item == 0:
                result[num][index] = 0
            # Correct recalls
            elif item in pres_itemnos[num]:
                result[num][index] = 0
            # Prior-list intrusions; may also be an extra-list intrusion that happens to be in the wordpool
            else:
                # Start by assuming the word is an ELI that happens to be in the wordpool
                result[num][index] = -1

                # Loop backwards over previous lists from the current subject and session to determine whether the word
                # is a PLI. If it is, replace the -1 we just wrote with a positive integer indicating the number of
                # lists back that the PLI was made.
                count = 1
                while num - count >= 0:
                    if subjects[num] == subjects[num - count] and sessions[num] == sessions[num - count]:
                        # If the word was presented "count" lists back, then mark it as a PLI
                        if item in pres_itemnos[num - count]:
                            result[num][index] = count
                            break
                        # Otherwise, keep looping backwards through the session
                        else:
                            count += 1
                    # If we go back far enough that we reach a different session or subject, then the word is an ELI
                    # that happens to also be in the wordpool.
                    else:
                        break

    return result