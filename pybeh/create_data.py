import numpy as np
from events2data import events2data
from make_recalls_matrix import make_recalls_matrix
from create_intrusions import intrusions

def create_data(events=None, trial_field=None):

    '''

    * 'subject' field with subject id string
    * List length is the same for all trials (because analysis
      scripts assume this, and data.listLength is expected to
      be a scalar)
    * events must contain a field with a scalar indicating the
      current list
    * events must contain a 'type' field
    * Presentation events correspond to type = 'WORD'
    * Recall events correspond to type = 'REC_WORD'
    * 'itemno' field in both presentation and recall events
    * No item is presented more than once in a session

    DATA FIELDS
    subject       The index of where each subject was in the
                  sorted list of subject ids
    subjid        Cell array of string ids for each subject
    session       Session number
    listLength    Number of items presented in each list
    pres_itemnos  Presented wordpool item numbers
    rec_itemnos   Recalled wordpool item numbers
    recalls       Serial position of each recall;
                   -1   Intrusion (see the intrusions field)
                    0   Used for padding; indicates no recall
                   >0   Correct recall, number gives serial position
    times         Time (in ms) of each recall
    intrusions    Gives information about intrusions:
                    0   Correct recall (or no recall)
                   -1   Extralist intrusion (XLI)
                   >0   Prior-list intrusion (PLI); number indicates
                        the number of lists back

    :param events: events structure (e.g., from ptsa BaseEventReader)
    :param trial_field: field within the events structure specifying
           which trial an event belongs to (e.g., list, trial)
    :return data: data structure as described above
    '''

    if events is None:
        raise Exception('Must pass events structure.')
    if trial_field is None:
        trial_field = 'trial'
    if not any(np.array(events.dtype.names)=='subject'):
        raise Exception('Events must have a subject field')
    if not any(np.array(events.dtype.names)=='session'):
        raise Exception('Events must have a session field')
    if not any(np.array(events.dtype.names)=='itemno'):
        raise Exception('Events must have an itemno field')
    if not any(np.array(events.dtype.names)=='type'):
        raise Exception('Events must have a type field')

    subjects = np.unique(events['subject'])

    for subject in subjects:

        subject_number = subject[findNumbers(subject)]
        # get free recall events for this subject
        subject_events = events[(events['subject']==subject) & ((events['type']=='WORD') | (events['type']=='REC_WORD'))]

        # remove vocalizations
        subject_events = subject_events[subject_events['item'] != 'VV']

        sessions = np.unique([subject_events.session])

        for session in sessions:

            sess_data = dict()
            sess_events = subject_events[subject_events['session']==session]

            trials = sess_events[trial_field]
            unique_trials = np.unique(trials)

            # presentation data
            item_pres = sess_events['type'] == 'WORD'

            if not any(item_pres):
                raise Exception('No presentation events in session '+str(session)+' for subject '+subject)

            pres_data = events2data(events=sess_events[item_pres], index=trials[item_pres], index_rows=unique_trials)
            n_trials = np.shape(pres_data['itemno'])[0]
            n_items = np.shape(pres_data['itemno'])[1]

            # recall data
            recalls = sess_events['type'] == 'REC_WORD'
            rec_data = events2data(events=sess_events[recalls], index=trials[recalls], index_rows=unique_trials)

            if not any(recalls):
                for field in events.dtype.names:
                    if events[field].dtype.type is np.string_:
                        rec_data[field] = np.empty(np.shape(rec_data[field]), dtype=events[field].dtype)
                        rec_data[field].fill('')

            # initialize session data
            sess_data['subject'] = np.empty([n_trials, 1])
            sess_data['subject'].fill(subject_number)

            sess_data['subjid'] = pres_data['subject']
            sess_data['session'] = pres_data['session']

            sess_data['pres_items'] = pres_data['item']
            sess_data['pres_itemnos'] = pres_data['itemno']

            sess_data['rec_items'] = rec_data['item']
            sess_data['rec_itemnos'] = rec_data['itemno']

            sess_data['recalls'] = make_recalls_matrix(sess_data['pres_itemnos'], sess_data['rec_itemnos'])

            if any(np.array(list(rec_data.keys()))=='rectime'):
                sess_data['times'] = rec_data['rectime']

            sess_data['session'] = sess_data['session'][:,0]

            sess_data['intrusions'] = intrusions(rec_itemnos=sess_data['rec_itemnos'], pres_itemnos=sess_data['pres_itemnos'],
                                            subjects=sess_data['subject'], sessions=sess_data['session'])

            sess_data['listLength'] = n_items
            # sess_data['pres'] = pres_data
            # sess_data['rec'] = rec_data

            if (session == sessions[0]) and (subject==subjects[0]):
                data = sess_data
            else:
                for key in list(sess_data.keys()):
                    if np.shape(sess_data[key]):
                        if len(np.shape(data[key])) != 1:
                            d_cols = np.shape(data[key])[1]
                            sd_cols = np.shape(sess_data[key])[1]
                            if d_cols > sd_cols:
                                sess_data[key] = np.pad(sess_data[key], ((0,0),(0, d_cols-sd_cols)), 'constant', constant_values=0)
                            elif d_cols < sd_cols:
                                data[key] = np.pad(data[key], ((0,0),(0, sd_cols-d_cols)), 'constant', constant_values=0)

                        data[key] = np.concatenate((data[key],sess_data[key]), axis=0)

    return data

def findNumbers(inputString):
    out = bool()
    for char in inputString:
        out += char.isdigit()

    return out