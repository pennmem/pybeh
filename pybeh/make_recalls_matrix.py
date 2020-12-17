import numpy as np
import pandas as pd


def get_itemno_matrices(evs, itemno_column='itemno', list_index=['subject', 'session', 'list'], fill_value=np.nan):
    """
    Transforms a pandas dataframe into a matrix of item id's with one row per trial, 
    as is expected by most behavioral toolbox functions.
    
    Expects as input a dataframe (df) for one subject
    
    INPUTS:
    evs:            The dataframe from which to extract itemnos. By default, each 
                    distinct set of values in the (subject, session, list) columns
                    denotes a different trial.
                    
    itemno_column:  The column of the dataframe where items are annotated with item numbers
    
    list_index:     Columns passed to pd.groupby that uniquely identify each trial
    
    fill_value:     The default value with which to pad missing data to align rows
    
    OUTPUTS:
    A matrix of item numbers with shape (trials, max_length), where trials is determined by the
    number of combinations of list_index coordinates in the data and max_length is determined
    by the trial with the greatest number of items.
    """
    evs.loc[:, itemno_column] = evs.loc[:, itemno_column].astype(int)
    evs['pos'] = evs.groupby(list_index).cumcount()
    itemnos_df = pd.pivot_table(evs, values=itemno_column, 
                                 index=list_index, 
                                 columns='pos', fill_value=fill_value)
    itemnos = itemnos_df.values
    return itemnos


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
