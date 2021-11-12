import lime
import shap

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from anchor import anchor_tabular
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.metrics import average_precision_score, roc_curve, auc, plot_precision_recall_curve

"""
Visualization | Graphs | Charts
"""

def prec_rec(classifier, X_test, Y_test):
    Y_test_score = classifier.predict_proba(X_test)[:, 1]
    average_precision = average_precision_score(Y_test, Y_test_score)
    disp = plot_precision_recall_curve(classifier, X_test, Y_test)
    disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))
    plt.savefig('static/prec_rec.svg', bbox_inches='tight')

def plot_roc(classifier, X_test, y_test):
    y_test_score = classifier.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_test_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.savefig('static/roc.svg', bbox_inches='tight')

def data_distr(data, figsizes, cols, shareys=True, colors='green'):
    ref = 0 
    rows = round(len(data.columns)/cols)
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsizes, sharey=shareys, squeeze=True)

    for n in range(cols):
        for x in range(rows):
            if ref < data.shape[1]:
                axs[x, n].set_title(data.columns[ref])
                axs[x, n].hist(data[data.columns[ref]], color=colors)
                ref += 1
           
    plt.savefig('static/dat_dist.svg', bbox_inches='tight')
    #return plt.show()

def corr_matrix(dataframe):
    corr_dataframe = dataframe.corr()
    dissimilarity = 1 - abs(corr_dataframe)
    Z = linkage(squareform(dissimilarity), 'complete')

    # Clusterize the data
    threshold = 0.4
    labels = fcluster(Z, threshold, criterion='distance')

    # Keep the indices to sort labels
    labels_order = np.argsort(labels)

    # Build a new dataframe with the sorted columns
    for idx, i in enumerate(dataframe.columns[labels_order]):
        if idx == 0:
            clustered = pd.DataFrame(dataframe[i])
        else:
            df_to_append = pd.DataFrame(dataframe[i])
            clustered = pd.concat([clustered, df_to_append], axis=1)

    # Plot
    plt.figure(figsize=(20,20))
    correlations = clustered.corr()
    sns.heatmap(round(correlations,2),
                cmap=sns.diverging_palette(20, 220, n=200), 
                square=True, 
                annot=False, 
                linewidths=.5,
                vmin=-1, vmax=1, center= 0,
                cbar_kws={"shrink": .5})
    plt.title("Clusterized Correlation Matrix")
    plt.yticks(rotation=0)
    plt.savefig('static/cor_mat.svg', bbox_inches='tight')

def lime_expl(X_train, X_test, Y_test, classifier):
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.to_numpy(), 
        feature_names=X_train.columns.tolist()
    )

    i = np.random.randint(0, X_test.shape[0])
    explanation = lime_explainer.explain_instance(
        X_test.iloc[i], 
        classifier.predict_proba,
        num_features=5,
        top_labels=1
    )
    explanation.save_to_file('static/lime_expl.html')

def anchor_expl(X_train, X_test, classifier):
    anchor_explainer = anchor_tabular.AnchorTabularExplainer(
        class_names=["0", "1"],
        feature_names=X_train.columns.tolist(),
        train_data=X_train.to_numpy(),
    )
    i = np.random.randint(0, X_test.shape[0])
    sample = X_test.iloc[i].to_numpy().reshape(1, -1)
    prediction = classifier.predict(sample)[0]
    anchor_explanation = anchor_explainer.explain_instance(
        X_train.iloc[i].to_numpy(), 
        classifier.predict,
        threshold=0.95
        )
    exp = (" AND ".join(anchor_explanation.names()))
    return exp, anchor_explainer.class_names[prediction], anchor_explanation.precision(), anchor_explanation.coverage()

def shap_expl(X_train, X_test, classifier):
    shap.initjs()
    shap_explainer = shap.KernelExplainer(classifier.predict_proba, X_train[:100])
    shap_values = shap_explainer.shap_values(X_test.iloc[0,:])
    shap.force_plot(shap_explainer.expected_value[0], shap_values[0], X_test.iloc[0,:], show=False, matplotlib=True)
    plt.savefig('static/shap_exp_1.svg', bbox_inches='tight')

    """
    shap_values = shap_explainer.shap_values(X_test.iloc[0:10,:])
    shap.force_plot(shap_explainer.expected_value[0], shap_values[0], X_test.iloc[0:10,:], show=False)
    plt.savefig('static/shap_exp_2.png', bbox_inches='tight')
    """

def tree_expl(classifier, X_train):
    train_prediction = classifier.predict(X_train)

    gs_explainer = DecisionTreeClassifier(max_depth=3)
    gs_explainer.fit(X_train, train_prediction)
    plt.figure(figsize=(25, 20))
    plot_tree(gs_explainer, feature_names=X_train.columns, class_names=['unworn', 'worn'], fontsize=12)

    plt.savefig('static/tree_expl.svg', bbox_inches='tight')
