from collections import OrderedDict

def FRdata(events = None):
    """
    FRdata

    Makes a dictionary out of events structure with each values as lists"""
    dict = OrderedDict()
    if events == None:
        raise Exception('You must pass an events file.')

    if len(events['events']) < 1:
        raise Exception('Events file empty')
    for num in range(len(events['events'])):
        for item in events['events'][num].__dict__['_fieldnames']:
            if item not in dict:
                dict[item] = [events['events'][num].__dict__[item]]
            else:
                dict[item].append(events['events'][num].__dict__[item])
    return dict