'''
Example for train model
'''

import pickle
from configparser import SafeConfigParser
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def clean_structure(df):
    '''
    Return dataframe after cleaning
    :param df: Eaw Dataframe
    :return: Cleaned Dataframe
    '''

    df = df.drop(['CLIENTNUM',
                  'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2',
                  'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1'],
                 axis=1)

    df_clean = df.drop(['Dependent_count',
                        'Months_on_book',
                        'Contacts_Count_12_mon',
                        'Total_Revolving_Bal',
                        'Avg_Open_To_Buy',
                        'Total_Amt_Chng_Q4_Q1',
                        'Total_Trans_Amt',
                        'Total_Trans_Amt',
                        'Total_Trans_Ct',
                        'Total_Ct_Chng_Q4_Q1',
                        'Card_Category',
                        'Income_Category',
                        'Marital_Status'], axis=1)
    return df_clean


def normalization_data(df_clean):
    '''

    :param df_clean:
    :return:
    '''
    df_clean.Attrition_Flag.replace({'Existing Customer': 1, 'Attrited Customer': 0}, inplace=True)
    df_clean.Gender.replace({'F': 0, 'M': 1}, inplace=True)
    df_clean.Education_Level.replace(
        {'Graduate': 0, 'High School': 1, 'Unknown': 2, 'Uneducated': 3, 'College': 4, 'Post-Graduate': 5,
         'Doctorate': 6}, inplace=True)

    return df_clean


def separate_data(df_norm):
    '''

    :param df_norm:
    :return:
    '''
    X_train,\
    X_test,\
    Y_train,\
    Y_test = train_test_split(X, Y, test_size=0.3, random_state=44, shuffle=True)

    return X_train, X_test, Y_train, Y_test


def train(X_train, Y_Train):
    '''

    :param X_train:
    :param Y_Train:
    :param Y_Test:
    :return:
    '''

    GBCModel = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=33)
    model = GBCModel.fit(X_train, Y_Train)
    return model

def save_model(model):
    '''
    Save to file in the current working directory
    :return:
    '''

    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)


def run_model(sample):
    '''
    Verify if the model run correctly
    :return:
    '''
    parser = SafeConfigParser()
    parser.read('config.ini')
    pkl_filename = parser.get('FILE', 'model')
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)

    result = pickle_model.predict(sample)
    print(result)


if __name__ == '__main__':
    test = 1
    if test == 0:
        parser = SafeConfigParser()
        parser.read('config.ini')
        pkl_filename = parser.get('FILE', 'model')
        df = pd.read_csv(parser.get('FILE', 'csv'))
        clean_df = clean_structure(df)
        norm_df = normalization_data(clean_df)
        X_train, X_test, Y_train, Y_test = separate_data(norm_df)
        model = train(X_train, Y_train)
        save_model(model)
    else:
        sample = [45, 1, 1, 5, 1, 12691.0, 0.061] ## exemplo sem risco
        sample = [60, 1, 6, 0, 0, 40000, 0.061] ## exemplo com risco
        sample = np.array(sample)
        print(sample)
        sample = np.expand_dims(sample, axis=0)
        print(sample)
        run_model(sample)
