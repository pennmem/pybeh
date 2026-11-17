import numpy as np


def irt(times=None):
    """
    IRT Inter-response time.

    [irts] = irt(times)

    INPUTS:
        times:  a matrix whose elements are times of recalled
                items, relative to the start of a recall period during
                a trial. The rows of this matrix should
                represent times of recalls by a single subject
                on a single trial. Pad shorter trials with 0.

    OUTPUTS:
        irts:   a matrix the same shape as times. irts[i, j] is the time
                from recall j to recall j + 1 on trial i, and 0 where
                there is no next recall (including the last column).
    """
    # Work on a copy so the caller's times are left alone
    times = np.array(times, dtype=float)
    irts = np.zeros_like(times)

    # Interval j runs from recall j to recall j + 1, so it only exists
    # when there is a recall at j + 1 (padding is 0 or NaN)
    next_times = times[:, 1:]
    has_next = next_times > 0
    irts[:, :-1] = np.where(has_next, next_times - times[:, :-1], 0)
    return irts
