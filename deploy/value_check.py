'''
Microservice using saved model
'''

import boto3
import pickle
from configparser import SafeConfigParser
from sklearn.ensemble import GradientBoostingClassifier


def lambda_handler(event,context):
    '''
    Call lambda service
    :param event:
    :param context:
    :return:
    '''

    result = predict(event)
    return {'StatusCode': 200,
            'body': result[0]}


def predict(event):
    sample = event['body']
    model = get_model()
    result = model.predict(sample)
    return result


def get_model():
    '''
    Collect model
    :return: Predicted model
    '''

    parser.read('config.ini')
    pkl_filename = parser.get('FILE', 'model')
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)

    return pickle_model


if __name__ == '__main__':
    '''
    Test before senndinng to lambda service
    '''
    temp = {
        'a': '1',
        'b': 2
    }
    result = predict(temp)
    print(result)


