'''
Microservice using saved model
'''


import pickle
from configparser import SafeConfigParser
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hey, we have Flask in a Docker container!'


@app.route('/predict')
def api():
    '''
    Call API prediction

    :return:
    '''
    age = request.args.get('age')
    gender = request.args.get('gender')
    rel = request.args.get('rel')
    aver_ut = request.args.get('aver_ut')
    mth_inact = request.args.get('mth_inact')
    credit = request.args.get('credit')
    avg_rate = request.args.get('avg_rate')
    event = [age, gender, rel, aver_ut, mth_inact, credit, avg_rate]
    result = predict(event)

    return jsonify(
        status='Se OK sem risco, se RISCO com risco de churn',
        result=result
    )


def predict(event):
    '''

    :param event: Data from API
    :return: Results from model
    '''
    sample = np.array(event)
    sample = np.expand_dims(sample, axis=0)
    model = get_model()
    result = model.predict(sample)
    if int(result) == 0:
        return 'RISCO'
    else:
        return 'OK'


def get_model():
    '''
    Collect model
    :return: Predicted model
    '''

    parser = SafeConfigParser()
    parser.read('./app/config.ini')
    pkl_filename = parser.get('FILE', 'model')
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)


    return pickle_model


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9028)


