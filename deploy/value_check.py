'''
Microservice using saved model
'''


def collect_model():
    '''

    :return:
    '''


    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)

    return pickle_model

