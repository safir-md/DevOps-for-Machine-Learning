import os
import mlflow
import mlflow.sklearn

from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, accuracy_score

from models import dec_tr_model, rf_model, auto_model
from data_proc import fetch_data, feat_engg, split_data

mlflow.set_tracking_uri('./mlruns')

def compare_models(acc_dct):
    max_key = max(acc_dct, key=acc_dct.get)
    os.rename('models/'+max_key+'_model.pkl','models/classifier.pkl')
    os.rename('data/X_train_'+max_key+'.csv', 'data/X_train.csv')
    os.rename('data/X_test_'+max_key+'.csv', 'data/X_test.csv')
    print(acc_dct)
    print("Final Model :-- ",max_key)

def model_stats(classifier, X_test, Y_test):
    mlflow.set_tracking_uri('./mlruns')
    fin_exp_id = mlflow.set_experiment("Final Model Experiment")
    with mlflow.start_run(experiment_id=fin_exp_id):
        Y_pred = classifier.predict(X_test)
        cm = confusion_matrix(Y_test, Y_pred)
        acc = round(accuracy_score(Y_test, Y_pred), ndigits=4)
        prc = round(precision_score(Y_test, Y_pred), ndigits=4)
        rcl = round(recall_score(Y_test, Y_pred), ndigits=4)
        f1 = round(f1_score(Y_test, Y_pred), ndigits=4)
        corcoeff = round(matthews_corrcoef(Y_test, Y_pred), ndigits=4)

        mlflow.log_metrics({'Accuracy Score': acc, 'Precision Score': prc, 'Recall Score':rcl, 'F1 Score':f1, 'Correlation Coefficient': corcoeff})
        mlflow.sklearn.log_model(classifier, "Final Classifier")

    return cm, acc, prc, rcl, f1, corcoeff

def get_data():
    dataframe = fetch_data()
    dataframe = feat_engg(dataframe)
    X, X_train, X_test, Y_train, Y_test = split_data(dataframe)

    return X, X_train, X_test, Y_train, Y_test

def retrain_model():
    
    X, X_train, X_test, Y_train, Y_test = get_data()
    X_test, classifier = dec_tr_model(X, X_train, X_test, Y_train)
    _, acc_dt, _, _, _, _ = model_stats(classifier, X_test, Y_test)
    
    X, X_train, X_test, Y_train, Y_test = get_data()
    X_test, classifier = rf_model(X, X_train, X_test, Y_train)
    _, acc_rf, _, _, _, _ = model_stats(classifier, X_test, Y_test)
    
    _, _, _, _, Y_test = get_data()
    X_test, classifier = auto_model()
    _, acc_auto, _, _, _, _ = model_stats(classifier, X_test, Y_test)

    acc_dct = {}
    acc_dct['dt'] = acc_dt
    acc_dct['rf'] = acc_rf
    acc_dct['auto'] = acc_auto

    compare_models(acc_dct)
