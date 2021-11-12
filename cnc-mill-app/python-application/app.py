import time
import prometheus_client

import pandas as pd
import flask_monitoringdashboard as dashboard

from joblib import load
from werkzeug.serving import run_simple
from flask_prometheus_metrics import register_metrics
from prometheus_client import make_wsgi_app, Counter, Histogram
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from flask import Flask, render_template, redirect, Response, Blueprint

from graphs import *
from pipeline_api import model_stats, retrain_model

CONFIG = {"version": "v3.0", "config": "staging"}
MAIN = Blueprint("main", __name__)

_INF = float("inf")
graphs = {}

graphs['model_ctr'] = Counter('model_run_count_total', 'Total number of times Model was run')
graphs['retrain_ctr'] = Counter('model_retrain_count_total', 'Total number of times Model was Re-Trained')
graphs['ctr'] = Counter('flask_request_operations_total', 'The total number of processed requests')
graphs['model_hst'] = Histogram('model_run_duration_seconds', 'Histogram for the prediction duration in seconds', buckets=(100, 150, 200, 250, 300, _INF))
graphs['retrain_hst'] = Histogram('model_retrain_duration_seconds', 'Histogram for the Retraining duration in seconds', buckets=(200, 300, 400, 500, 600, _INF))

classifier = load("models/classifier.pkl")
X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv')
Y_train = pd.read_csv('data/Y_train.csv')
Y_test = pd.read_csv('data/Y_test.csv')

@MAIN.route("/")
def disp_home():
    
    print("AHAD")
    graphs['ctr'].inc()
    return render_template('index.html')

@MAIN.route("/data_distribution")
def data_distribution():

    graphs['ctr'].inc()
    data_distr(data=X_train.iloc[:,1:], figsizes=(15, 40), cols=6)
    corr_matrix(X_train.iloc[:,1:])

    return render_template('data_dist.html')

@MAIN.route("/model_behaviour")
def model_behaviour():

    graphs['ctr'].inc()
    graphs['model_ctr'].inc()
    start = time.time()
    cm, acc, prcs, rcl, f1, corcoeff = model_stats(classifier, X_test.iloc[:,1:], Y_test.iloc[:,1])
    anch, prd, prc, cvr = anchor_expl(X_train.iloc[:,1:], X_test.iloc[:,1:], classifier)
    prec_rec(classifier, X_test.iloc[:,1:], Y_test.iloc[:,1])
    plot_roc(classifier, X_test.iloc[:,1:], Y_test.iloc[:,1:])
    lime_expl(X_train.iloc[:,1:], X_test.iloc[:,1:], Y_test.iloc[:,1:], classifier)
    tree_expl(classifier, X_train.iloc[:,1:])
    shap_expl(X_train.iloc[:,1:], X_test.iloc[:,1:], classifier)
    end = time.time()
    graphs['model_hst'].observe(end - start)


    return render_template('model_stat.html', cm=cm, acc=round(acc*100,2), prcs=prcs, rcl=rcl, f1=f1, mcc=corcoeff, anch=anch, prd=prd, prc=prc, cvr=cvr)

@MAIN.route("/retrain")
def retrain():
    
    graphs['ctr'].inc()
    graphs['retrain_ctr'].inc()
    start = time.time()
    retrain_model()
    end = time.time()
    graphs['retrain_hst'].observe(end - start)
    return render_template('retrain.html')

@MAIN.route("/mlflow_ui")
def model_monitoring():
    
    graphs['ctr'].inc()
    return redirect('http://0.0.0.0:81/')

@MAIN.route("/prom")
def prom():
    
    graphs['ctr'].inc()
    return redirect('http://0.0.0.0:82/')

@MAIN.route("/graf")
def graf():
    
    graphs['ctr'].inc()
    return redirect('http://0.0.0.0:83/')

@MAIN.route("/metrics")
def requests_count():
    res = []
    for key,val in graphs.items():
        res.append(prometheus_client.generate_latest(val))
    return Response(res, mimetype="text/plain")

def register_blueprints(app):
    app.register_blueprint(MAIN)

def create_app(config):
    app = Flask(__name__)
    register_blueprints(app)
    register_metrics(app, app_version=config["version"], app_config=config["config"])
    dashboard.config.init_from(file='flask_mon_config/config.cfg')
    dashboard.bind(app)
    return app

def create_dispatcher() -> DispatcherMiddleware:
    main_app = create_app(config=CONFIG)
    return DispatcherMiddleware(main_app.wsgi_app, {"/metrics": make_wsgi_app()})

if __name__ == "__main__":
    run_simple(
        "0.0.0.0",
        5000,
        create_dispatcher(),
        use_reloader=True,
        use_debugger=True,
        use_evalex=True,
    )
