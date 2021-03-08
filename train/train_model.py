'''
Example for train model
'''

import pickle
from configparser import SafeConfigParser
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd

#Applying GradientBoostingClassifier Model


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

    pkl_filename = "model_gen.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)

if __name__ == '__main__':
    parser = SafeConfigParser()
    parser.read('config.ini')
    df = pd.read_csv(parser.get('FILE', 'csv'))
