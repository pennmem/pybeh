import numpy as np
from collections import Counter

def events2data(events=None, index=None, index_rows=None, empty_val=None, ignore_fields=None):

    if empty_val is None:
        empty_val = 0
    if events is None:
        raise Exception('You must pass an events structure')
    if index is None:
        raise Exception('You must pass a numeric index')
    if len(index) != len(events):
        raise Exception('index must be the same length as events')
    if any(np.isnan(index)):
        raise Exception('index must not contain any NaN values')
    if ignore_fields is None:
        ignore_fields = ['badEventChannel']

    unique_index = np.unique(index)

    # set the order of rows in the final matrices
    if not index_rows.any():
        index_rows = unique_index
    elif set(unique_index) - set(index_rows):
        raise Exception('index_rows must include every value in index')

    # get the row for each index in terms of all possible indices
    rows = np.in1d(index_rows,unique_index)

    # get the maximum row length
    max_row_length = np.max([Counter(index).most_common(1)[0][1], Counter(index_rows).most_common(1)[0][1], 1])

    data = dict()
    # convert each field to matrix format and add to the data structure
    for field in [fname for fname in events.dtype.names if not np.any([ignfields == fname for ignfields in ignore_fields])]:
    # for field in events.dtype.names:

        data[field] = np.empty([len(index_rows), max_row_length], dtype=events[field].dtype)

        if events[field].dtype.type is not np.string_:
            data[field].fill(empty_val)
        else:
            data[field].fill('')

        # and finally fill in values according to index
        for identifier in unique_index:
            row_indices = index==identifier
            data[field][index_rows==identifier,:sum(row_indices)] = events[field][row_indices]

    return data
