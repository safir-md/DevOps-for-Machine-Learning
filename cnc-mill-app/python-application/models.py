import mlflow

import numpy as np
import pandas as pd

from joblib import dump
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from autosklearn.classification import AutoSklearnClassifier

from data_proc import feat_select

"""
ML Models
"""

def auto_model():
    X_train = pd.read_csv('data/X_train.csv')
    X_test = pd.read_csv('data/X_test.csv')
    Y_train = pd.read_csv('data/Y_train.csv')

    """
    ML-Flow Tracking
    """
    mlflow.set_tracking_uri('./mlruns')
    auto_exp_id = mlflow.set_experiment("Auto Sklearn Experiment")
    print("Please wait for 3 Minutes...!!!")
    with mlflow.start_run(experiment_id=auto_exp_id):
        """
        Model Building and Training
        """
        classifier = AutoSklearnClassifier(
            time_left_for_this_task=3 * 60,
            per_run_time_limit=30,
            ensemble_size=1,
            initial_configurations_via_metalearning=0,
            n_jobs=-1,
        )
        classifier.fit(X_train.iloc[:,1:], Y_train.iloc[:,1:])
        """
        Log Parameters and Model
        """
        mlflow.sklearn.log_model(classifier, "Auto Sklearn Classifier")

    """
    Save X_Train and X_Test
    """
    X_train.to_csv('data/X_train_auto.csv')
    X_test.to_csv('data/X_test_auto.csv')
    print(classifier.sprint_statistics())
    print(classifier.show_models())

    """
    Dump the Model in Pickle Format
    """
    pkl_file = "models/auto_model.pkl"
    dump(classifier, pkl_file)
    print("Auto Sklearn Model Trained...!!!")

    """
    Return X_Test and Classifier
    """
    return X_test.iloc[:,1:], classifier

def dec_tr_model(X, X_train, X_test, Y_train):

    """
    Dummy Model
    """
    classifier_dt_1 = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
    classifier_dt_1.fit(X_train, Y_train)

    """
    Feature Selection
    """
    X_train_DT, X_test_DT = feat_select(classifier_dt_1, X, X_train, X_test)

    """
    Save X_Train and X_Test
    """
    X_train_DT.to_csv('data/X_train_dt.csv')
    X_test_DT.to_csv('data/X_test_dt.csv')

    """
    Hyperparameter Selection
    """
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    max_features = ['auto', 'sqrt', 'log2', None]
    min_samples_split = [2, 4, 6, 8, 10]
    min_samples_leaf = [1, 2, 4, 8, 12]
    criterion = ['gini', 'entropy']

    random_grid = {'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'criterion': criterion}

    base_dt_model = DecisionTreeClassifier()

    """
    ML-Flow Tracking
    """
    mlflow.set_tracking_uri('./mlruns')
    dec_exp_id = mlflow.set_experiment("Decision-Tree Experiment")

    with mlflow.start_run(experiment_id=dec_exp_id):
        """
        Model Building and Training
        """
        random_dt_model = RandomizedSearchCV(estimator=base_dt_model,
                                                param_distributions=random_grid,
                                                n_iter=200, cv=5, verbose=2,
                                                random_state=0, n_jobs=-1)
        classifier_dt_2 = random_dt_model.fit(X_train_DT, Y_train)

        """
        Log Parameters and Model
        """
        mlflow.log_params(random_grid)
        mlflow.sklearn.log_model(random_dt_model.best_estimator_, "DT Classifier")

    """
    Dump the Model in Pickle Format
    """
    pkl_file = "models/dt_model.pkl"
    dump(random_dt_model.best_estimator_, pkl_file)
    print("Decision Tree Model Trained...!!!")

    """
    Return X_Test and Classifier
    """
    return X_test_DT, random_dt_model.best_estimator_

def rf_model(X, X_train, X_test, Y_train):
    """
    Dummy Model
    """
    classifier_rf_1 = RandomForestClassifier(n_estimators = 200, criterion = 'gini', random_state = 0)
    classifier_rf_1.fit(X_train, Y_train)

    """
    Feature Selection
    """
    X_train_RF, X_test_RF = feat_select(classifier_rf_1, X, X_train, X_test)

    """
    Save X_Train and X_Test
    """
    X_train_RF.to_csv('data/X_train_rf.csv')
    X_test_RF.to_csv('data/X_test_rf.csv')

    """
    Hyperparameter Selection
    """
    n_estimators = [int(x) for x in np.linspace(200, 2000, num = 10)]
    max_depth = [int(x) for x in np.linspace(10, 100, num = 10)]
    max_depth.append(None)
    max_features = ['auto', 'sqrt', None]
    min_samples_split = [2, 4, 6, 8]
    min_samples_leaf = [1, 2, 4, 8]
    criterion = ['gini', 'entropy']
    bootstrap = [True, False]

    random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'criterion': criterion,
                    'bootstrap': bootstrap}

    base_rf_model = RandomForestClassifier()

    """
    ML-Flow Tracking
    """
    mlflow.set_tracking_uri('./mlruns')
    raf_exp_id = mlflow.set_experiment("Random-Forest Experiment")
    with mlflow.start_run(experiment_id=raf_exp_id):
        """
        Model Building and Training
        """
        random_rf_model = RandomizedSearchCV(estimator=base_rf_model,
                                                param_distributions=random_grid,
                                                n_iter=10, cv=3, verbose=2,
                                                random_state=0, n_jobs=-1)
        classifier_rf_2 = random_rf_model.fit(X_train_RF, Y_train)

        """
        Log Parameters and Model
        """
        mlflow.log_params(random_grid)
        mlflow.sklearn.log_model(random_rf_model.best_estimator_, "RF Classifier")

    """
    Dump the Model in Pickle Format
    """
    pkl_file = "models/rf_model.pkl"
    dump(random_rf_model.best_estimator_, pkl_file)
    print("Random Forest Model Trained...!!!")

    """
    Return X_Test and Classifier
    """
    return X_test_RF, random_rf_model.best_estimator_