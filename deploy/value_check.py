'''
Microservice using saved model
'''


import pickle
from configparser import SafeConfigParser
from sklearn.ensemble import GradientBoostingClassifier
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def hello_world:
    return 'Hey, we have Flask in a Docker container!'

@app.route('/predict')
def api():
    '''
    Call lambda service

    :return:
    '''
    age = request.args.get('age')
    gender = request.args.get('gender')
    rel = request.args.get('rel')
    aver_ut = request.args.get('aver_ut')
    mth_inact = request.args.get('mth_inact')
    credit = request.args.get('credit')
    event = [age, gender, rel, aver_ut, mth_inact, credit]
    result = predict(event)

    return jsonify(
        status='Se 0 sem risco, se 1 com risco de saida no cartao',
        result=result
    )

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

    parser = SafeConfigParser()
    parser.read('config.ini')
    pkl_filename = parser.get('FILE', 'model')
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)

    return pickle_model


if __name__ == '__main__':
    app.run(port=9028)


