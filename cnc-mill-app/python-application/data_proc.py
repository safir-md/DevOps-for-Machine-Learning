import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

"""
Data Pre-processing | Feature Engineering/Selection
"""

def fetch_data():
    dataset = pd.read_csv('data/train.csv')
    frames = []
    for ctr in range(1,19):
        num = '0' + str(ctr) if ctr < 10 else str(ctr)
        frame = pd.read_csv("data/experiment_{}.csv".format(num))
        row = dataset[dataset['No'] == ctr]
        frame['clamp_pressure'] = row.iloc[0]['clamp_pressure']
        frame['tool_condition'] = row.iloc[0]['tool_condition']
        frames.append(frame)
    dataframe = pd.concat(frames, ignore_index = True)
    print("Data Fetched...!!!")
    return dataframe

def feat_engg(dataframe):
    dataframe = dataframe[(dataframe.Machining_Process == 'Layer 1 Up') | 
                        (dataframe.Machining_Process == 'Layer 1 Down') |
                        (dataframe.Machining_Process == 'Layer 2 Up') | 
                        (dataframe.Machining_Process == 'Layer 2 Down') |
                        (dataframe.Machining_Process == 'Layer 3 Up') | 
                        (dataframe.Machining_Process == 'Layer 3 Down')]

    dataframe['X1_DiffPosition'] = dataframe['X1_CommandPosition'] - dataframe['X1_ActualPosition']
    dataframe['X1_DiffVelocity'] = dataframe['X1_CommandVelocity'] - dataframe['X1_ActualVelocity']
    dataframe['X1_DiffAcceleration'] = dataframe['X1_CommandAcceleration'] - dataframe['X1_ActualAcceleration']
    dataframe['Y1_DiffPosition'] = dataframe['Y1_CommandPosition'] - dataframe['Y1_ActualPosition']
    dataframe['Y1_DiffVelocity'] = dataframe['Y1_CommandVelocity'] - dataframe['Y1_ActualVelocity']
    dataframe['Y1_DiffAcceleration'] = dataframe['Y1_CommandAcceleration'] - dataframe['Y1_ActualAcceleration']
    dataframe['Z1_DiffPosition'] = dataframe['Z1_CommandPosition'] - dataframe['Z1_ActualPosition']
    dataframe['Z1_DiffVelocity'] = dataframe['Z1_CommandVelocity'] - dataframe['Z1_ActualVelocity']
    dataframe['Z1_DiffAcceleration'] = dataframe['Z1_CommandAcceleration'] - dataframe['Z1_ActualAcceleration']
    dataframe['S1_DiffPosition'] = dataframe['S1_CommandPosition'] - dataframe['S1_ActualPosition']
    dataframe['S1_DiffVelocity'] = dataframe['S1_CommandVelocity'] - dataframe['S1_ActualVelocity']
    dataframe['S1_DiffAcceleration'] = dataframe['S1_CommandAcceleration'] - dataframe['S1_ActualAcceleration']

    drop_cols = ['X1_CommandPosition', 'X1_CommandVelocity',
                    'X1_CommandAcceleration', 'Y1_CommandPosition',
                    'Y1_CommandVelocity', 'Y1_CommandAcceleration',
                    'Z1_CommandPosition', 'Z1_CommandVelocity',
                    'Z1_CommandAcceleration', 'S1_CommandPosition',
                    'S1_CommandVelocity', 'S1_CommandAcceleration',
                    'Machining_Process']
    dataframe = dataframe.drop(drop_cols, axis=1)

    dummies_tc = pd.get_dummies(dataframe.tool_condition)
    dataframe = pd.concat([dataframe, dummies_tc], axis='columns')
    dataframe = dataframe.drop(['unworn', 'tool_condition'], axis=1)

    drop_cols = ['S1_SystemInertia', 'Z1_OutputVoltage', 'Z1_OutputCurrent',
                    'Z1_DCBusVoltage', 'Z1_CurrentFeedback']
    dataframe = dataframe.drop(drop_cols, axis=1)
    print("Feature Engineering Done...!!!")
    return dataframe

def split_data(dataframe):
    y = dataframe['worn']
    X = dataframe.drop(['worn'], axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    Y_train.to_csv('data/Y_train.csv')
    Y_test.to_csv('data/Y_test.csv')
    print("Data Splitted...!!!")
    return X, X_train, X_test, Y_train, Y_test

def feat_select(classifier, X, X_train, X_test):
    importances = list(classifier.feature_importances_)
    feature_list = list(X.columns)
    feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(feature_list, importances) if round(importance, 4)>=0.02]
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

    importances.sort(reverse=True)
    cumulative_importances = np.cumsum(importances)
    feature_count = np.where(cumulative_importances > 0.95)[0][0] + 1
    # Number of Features to keep
    feature_count = feature_count if feature_count<=len(feature_importances) else len(feature_importances)
    imp_feature_names = [feature[0] for feature in feature_importances[0:feature_count]]

    X_train_NEW = X_train.loc[:, imp_feature_names]
    X_test_NEW = X_test.loc[:, imp_feature_names]
    print("Features Selected...!!!")
    return X_train_NEW, X_test_NEW
