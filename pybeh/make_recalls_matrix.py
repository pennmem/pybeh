import numpy as np

def make_recalls_matrix(pres_itemnos=None, rec_itemnos=None):
    '''

    MAKE_RECALLS_MATRIX   Make a standard recalls matrix.

    Given presented and recalled item numbers, finds the position of
    recalled items in the presentation list. Creates a standard
    recalls matrix for use with many toolbox functions.

    recalls = make_recalls_matrix(pres_itemnos, rec_itemnos)

    INPUTS:
    pres_itemnos:  [trials X items] matrix of item numbers of
                 presented items. Must be positive.

    rec_itemnos:  [trials X recalls] matrix of item numbers of recalled
                  items. Must match pres_itemnos. Items not in the
                  stimulus pool (extra-list intrusions) should be
                  labeled with -1. Rows may be padded with zeros or
                  NaNs.

    OUTPUTS:
    recalls:  [trials X recalls] matrix. For recall(i,j), possible
             values are:
             >0   correct recall. Indicates the serial position in
                  which the recalled item was presented.
              0   used for padding rows. Corresponds to no recall.
             <0   intrusion of an item not presented on the list.

    :param pres_itemnos:
    :param rec_itemnos:
    :return:
    '''

    n_trials = np.shape(pres_itemnos)[0]
    n_items = np.shape(pres_itemnos)[1]

    n_recalls = np.shape(rec_itemnos)[1]

    recalls = np.empty([n_trials,n_recalls], dtype=int)
    recalls.fill(0)

    for trial in np.arange(n_trials):
        for recall in np.arange(n_recalls):

            if (rec_itemnos[trial,recall]) == 0 | (np.isnan(rec_itemnos[trial, recall])):
                continue

            elif rec_itemnos[trial,recall] > 0:

                serialpos = np.where(rec_itemnos[trial,recall] == pres_itemnos[trial,:])[0]+1

                if len(serialpos) > 1:
                    raise Exception('An item was presented more than once.')
                elif not any(serialpos):
                    recalls[trial, recall] = -1
                else:
                    recalls[trial, recall] = serialpos
            else:
                recalls[trial, recall] = -1

    return recalls