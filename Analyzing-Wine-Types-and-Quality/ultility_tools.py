# -*- coding: utf-8 -*-
"""
Created on May 15 2020
@author: Daniel Nguyen
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix



sns.set(color_codes=True)
#####################################################################################################


def plot_boxvio_one_att(df,num_att,cat_att):
    """
    - make box-plot and violin-plot for a numerical attribute grouped by a categorical attribute
    :param df: dataframe that contains the data
    :param num_att: name of the numerical attribute
    :param cat_att: name of the categorical attribute
    :return: the plots
    """
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,5))
    sns.boxplot(x=cat_att,y=num_att,data=df,hue=cat_att,ax=ax1)
    sns.violinplot(x=cat_att,y=num_att,data=df,hue=cat_att,ax=ax2)
    ax1.set_title(f'Box-Plot: {num_att}', fontsize= 18)
    ax2.set_title(f'Violin-Plot: {num_att}', fontsize= 18)
    ax1.legend_.remove()
    ax2.legend_.remove()

    plt.show()
    return
#####################################################################################################


def plot_hist_attributes(df, num_att, cat_att, n_bin=50):
    """
    - make histogram plots for a numerical attribute grouped by a categorical attribute
    :param df: dataframe that contains the data
    :param num_att: name of the numerical attribute
    :param cat_att: name of the categorical attribute
    :param n_bin: number of bins
    :return: the plots
    """
    # get all classes in the categorical attribute
    class_list = df[cat_att].unique()

    # number of classes in the categorical attribute
    num_class = len(class_list)

    # colors
    colors = plt.cm.rainbow(np.linspace(0, 1, num_class))
    # make hist plots
    fig, axs = plt.subplots(1, num_class, figsize=(12, 5))
    fig.suptitle(f"Histograms: {num_att}", fontsize=18)

    for j, class_name in enumerate(class_list):
        ax = axs[j]
        df1 = df[df[cat_att] == class_name][num_att]
        ax.hist(df1, bins=n_bin, color=colors[j], edgecolor="black")
        ax.set_title(class_name, fontsize=16)
        ax.text(1.0, 1.0, f'$\mu$= {round(df1.mean(), 2)}', fontsize=11, transform=ax.transAxes)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # to avoid the overlap
    plt.show()
    return


def get_confusion_matrix(y_true, y_pred, labels,title='Confusion matrix', cmap=plt.cm.Reds):

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize= 18)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.ylabel('True label',fontsize =14)
    plt.xlabel('Predicted label',fontsize =14)
    plt.grid(False)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    plt.plot()
    return cm


def display_model_performance_metrics(y_true, y_pred, labels):

    print()
    print('='*30)
    print('Model Classification report:')
    print('='*30)
    print(classification_report(y_true,y_pred,labels=labels))

    print()
    print('='*30)
    print('Prediction Confusion Matrix:')
    cm = get_confusion_matrix(y_true, y_pred, labels)
    print('='*30)

    return cm


def train_predict_model(classifier, X_train, y_train, X_test):
    # build model
    classifier.fit(X_train, y_train)
    # predict using model
    y_pred = classifier.predict(X_test)
    return y_pred