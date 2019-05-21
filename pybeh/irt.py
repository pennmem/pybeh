def irt(times=None):
    """
    IRT Inter-response time.

    [irts] = irt(times)

    INPUTS:
        times:  a matrix whose elements are times of recalled
                items, relative to the start of a recall period during
                a trial. The rows of this matrix should
                represent times of recalls by a single subject
                on a single trial.

    OUTPUTS:
        irts:   a matrix whose rows contain mean inter-response times
                for each of the unique values in index
    """
    irt = times[:]
    for num, item in enumerate(times):
        for index,recall in enumerate(item):
            if index == len(item) - 1:
                irt[num][index] = 0
            elif recall == 0:
                irt[num][index-1] = 0
            elif index != 0:
                irt[num][index-1] = times[num][index] - times[num][index - 1]
    return irt
