# -*- coding: utf-8 -*-
"""
Created on May 15 2020
@author: Daniel Nguyen
"""
import warnings
from sklearn.exceptions import FitFailedWarning,ConvergenceWarning

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.base import TransformerMixin,BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
sns.set(color_codes=True)

import itertools
from sklearn.metrics import confusion_matrix,roc_curve,auc,classification_report
from sklearn.preprocessing import label_binarize
from sklearn.calibration import CalibratedClassifierCV

#####################################################################################################


class CustomCategoricalImputer(TransformerMixin):
    """
    Custom Category imputer
    """
    def __init__(self,cols=None):
        self.cols = cols

    def transform(self,X):
        X_copied = X.copy()
        for col in self.cols:
            X_copied[col].fillna(X_copied[col].value_counts().index[0], inplace = True)
        return X_copied

    def fit(self,X,y=None):
        return self

#####################################################################################################


class CustomNumericalImputer(TransformerMixin):
    """
    Custom quantitative imputer
    """
    def __init__(self,cols=None,strategy='median'):
        self.cols = cols
        self.strategy = strategy

    def transform(self,X):
        X_copied = X.copy()
        imputer = SimpleImputer(strategy=self.strategy)
        for col in self.cols:
            X_copied[col] = imputer.fit_transform(X_copied[[col]])
        return X_copied

    def fit(self,X,y=None):
        return self

#####################################################################################################


class CustomDummyEncoder(TransformerMixin):
    """
    Custom Encoder, using dummy variables
    """
    def __init__(self,cols=None):
        self.cols = cols

    def transform(self,X):
        return pd.get_dummies(X,columns=self.cols)

    def fit(self,X,y=None):
        return self

#####################################################################################################


class CustomOrdinalEncoder(TransformerMixin):
    """
    Custom Encoder for ONE ordinal column, using map()
    """
    def __init__(self,col,ordering=None):
        self.col = col
        self.ordering = ordering

    def transform(self,X):
        X_copied = X.copy()
        X_copied[self.col] = X_copied[self.col].apply(lambda x: self.ordering.index(x))
        return X_copied

    def fit(self,X,y=None):
        return self

#####################################################################################################


class CustomQuantitativeEncoder(TransformerMixin):
    """
    Custom Encoder for ONE numerical (quantitative) column, using cut(bins)
    """
    def __init__(self,col,bins,labels = False):
        self.col = col
        self.bins = bins
        self.labels = labels

    def transform(self,X):
        X_copied = X.copy()
        X_copied[self.col] = pd.cut(X_copied[self.col],bins=self.bins,labels=self.labels)
        return X_copied

    def fit(self,X,y=None):
        return self


#####################################################################################################

def get_corr(df, annot=True,figsize=(15,8)):
    corr = df.corr()
    fig,ax = plt.subplots(figsize=figsize)
    sns.heatmap(round(corr,2),annot=annot,ax=ax, vmin= -1.0,vmax= 1.0,
                 cmap="coolwarm",linewidths=0.5,square=True)
    fig.subplots_adjust(top=0.9)
    fig.suptitle('Attribute Correlation Heatmap',fontsize=18)
    plt.show()
    return corr


#####################################################################################################

def get_best_for_model(model, params, X, y, cv=None, verbose = 1, off_warning = 1):
    if off_warning:
        warnings.simplefilter(action="ignore", category=UserWarning)
        warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
        warnings.filterwarnings(action='ignore', category=FitFailedWarning)

    grid = GridSearchCV(model, # the model to grid search
                        params, # the parameter set to try
                        cv=cv,
                        error_score=0.) # if a parameter set raises an error, continue and set the performance as a big, fat 0
    grid.fit(X, y)
    if verbose:
        print("************************************************")
        print(f"Best Accuracy: {grid.best_score_}")
        print(f"Best Parameters: {grid.best_params_}")
        print(f"Average Time to Fit: {round(grid.cv_results_['mean_fit_time'].mean(), 3)} (s)")
        print(f"Average Time to Score: {round(grid.cv_results_['mean_score_time'].mean(), 3)} (s)")
        print("************************************************")
    #return grid.best_score_,grid.best_params_,grid.cv_results_['mean_fit_time'].mean(),grid.cv_results_['mean_score_time'].mean()
    return grid

#####################################################################################################


def get_all_best_models(models, X, y):
    num_models = len(models["model_name"])
    grids = [0]*num_models
    best_score = [0]*num_models
    best_params = ['']*num_models
    fit_time = [0]*num_models
    score_time = [0]*num_models
    for i in range(num_models):
        grid = get_best_for_model(models["model_command"][i],models["model_params"][i], X, y,verbose = 0)
        grids[i] = grid
        best_score[i] = grid.best_score_
        best_params[i] = grid.best_params_
        fit_time[i] = grid.cv_results_['mean_fit_time'].mean()
        score_time[i] = grid.cv_results_['mean_score_time'].mean()

    max_value = max(best_score)
    max_index = best_score.index(max_value)
    best_grid = grids[max_index]
    df = pd.DataFrame(list(zip(models["model_name"],best_score,best_params,fit_time,score_time)),
                        columns=["model_name","best_score","best_params","fit_time","score_time"])
    return best_grid,df


#####################################################################################################


class CustomCorrelationChooser(TransformerMixin, BaseEstimator):
    """
    Custom Correlation:
    - The fit logic will select columns from the features matrix that are higher than a specified threshold
    - The transform logic will subset any future datasets to only include those columns that were deemed important
    """
    def __init__(self, response, cols_to_keep=[], threshold=None):
        # store the response series
        self.response = response
        # store the threshold that we wish to keep
        self.threshold = threshold
        # initialize a variable that will eventually
        # hold the names of the features that we wish to keep
        self.cols_to_keep = cols_to_keep

    def transform(self, X):
        # the transform method simply selects the appropriate columns from the original dataset
        return X[self.cols_to_keep]

    def fit(self, X, *_):
        # create a new dataframe that holds both features and response
        df = pd.concat([X, self.response], axis=1)

        # get the absolute values of the correlations between the features and the response variable
        corr = df.corr()[df.columns[-1]].abs()
        # store names of columns that meet correlation threshold
        self.cols_to_keep = df.columns[(corr > self.threshold) & (corr < 1.0)]
        return self

######################################################################################################
#


def fix_column_names(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')\
        .str.replace('(', '').str.replace(')', '').str.replace('.', '_')
    return df


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


def plot_hist_attributes(df, num_att, cat_att, bins=50):
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
        ax.hist(df1, bins=bins, color=colors[j], edgecolor="black")
        ax.set_title(class_name, fontsize=16)
        ax.text(1.0, 1.0, f'$\mu$= {round(df1.mean(), 2)}', fontsize=11, transform=ax.transAxes)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # to avoid the overlap
    plt.show()
    return
#####################################################################################################


def plot_in_groups(x, y, hue,title="Scatter Plot", x_label="X", y_label="Y"):
    """
    scatter plot (x,y) grouped by "hue" list (like seaborn)
    :param x:
    :param y:
    :param hue:
    :param title:
    :param x_label:
    :param y_label:
    :return:
    """
    colors  =["blue", "red", "green","orange", "yellow","purple", "black"]
    groups = list(set(hue))
    for i, group in enumerate(groups):
        plt.scatter(x.real[hue == group],
                    y.real[hue == group],
                    color=colors[i],alpha=0.5,label=group)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    leg = plt.legend(loc='upper left', bbox_to_anchor=(1., 0.9))
    leg.get_frame().set_alpha(0.5)
    plt.title(title,fontsize= 16)
    plt.show()
    return
#####################################################################################################
"""

def get_confusion_matrix(y_test, y_predict, labels,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):

    cm = confusion_matrix(y_test, y_predict, labels=labels)
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
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.plot()
    return cm
"""
#####################################################################################################


def get_roc(model,X,y):
    """
    Make the Receiver operator characteristic (ROC) graph, which is an  useful tool for selecting models for classification
     based on their performance with respect to the false positive and true positive rates
    :param model:
    :param X:
    :param y:
    :return:
    """
    # covert labels to numbers
    labels = model.classes_
    y_test_num = label_binarize(y,classes=labels)
    n_classes = y_test_num.shape[1]

    # get probability estimates for the X_test
    if str(type(model)) == "<class 'sklearn.linear_model._stochastic_gradient.SGDClassifier'>":
        model = CalibratedClassifierCV(model, cv='prefit')
    y_predict_proba = model.predict_proba(X)

    # Compute ROC curve and ROC area for each class
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    plt.figure(figsize=(10, 6))

    # plot the ROC for each class
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_num[:, i], y_predict_proba[:, i])
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr,tpr, lw=1,label=f'ROC of class {i} (area = {roc_auc:0.3f})')

    # plot the "random guessing" ROC
    plt.plot([0, 1], [0, 1], linestyle='--', color=(0.6, 0.6, 0.6),label='random guessing')

    # plot the mean ROC
    mean_tpr /= n_classes
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',label=f'mean ROC (area = {mean_auc:0.3f})')

    plt.title('Receiver Operator Characteristic', fontsize=18)
    plt.xlabel('False Positive Rate',fontsize =14)
    plt.ylabel('True Positive Rate',fontsize =14)
    plt.legend()
    plt.show()
    return
#####################################################################################################


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

#####################################################################################################
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