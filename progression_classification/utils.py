import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm, tnrange
from sklearn.metrics import accuracy_score, roc_curve, auc, plot_confusion_matrix, classification_report, precision_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, label_binarize

def encode(df):
    s = (df.dtypes == 'object')
    object_cols = list(s[s].index)
    OH_encoder = OneHotEncoder()
    OH_cols = pd.DataFrame(OH_encoder.fit_transform(df[object_cols]).toarray(), columns=OH_encoder.get_feature_names(object_cols))
    num_data = df.drop(object_cols, axis=1)
    data_new = pd.concat([num_data, OH_cols], axis=1)

    return data_new


def cv(model, cv_train_list, cv_val_list, precision=False):
    assert len(cv_train_list) == len(cv_val_list), "Length of training set cv list and validation set cv list must be the same"

    scores = []
    for i in range(len(cv_train_list)):
        X_train = cv_train_list[i][0]
        y_train = cv_train_list[i][1]
        X_val = cv_val_list[i][0]
        y_val = cv_val_list[i][1]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        if precision:
            acc = precision_score(y_val, y_pred, average='macro', zero_division=0)
        else:
            acc = accuracy_score(y_val, y_pred)
        scores.append(acc)

    return scores

def train_and_test(model, X_train, y_train, X_test, y_test, precision=False):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if precision:
        acc = precision_score(y_test, y_pred, average='macro', zero_division=0)
    else:
        acc = accuracy_score(y_test, y_pred)
    return acc, y_pred


def produce_roc(model_pred_probs):
    y_test_bin = label_binarize(y_test, classes=np.arange(3))
    fpr = {}
    tpr = {}
    roc_auc = {}
    plt.figure(figsize=(10, 8))
    colors = ['darkorange', 'darkviolet', 'navy']
    for i, color in zip(range(3), colors):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], model_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i],
                color=color,
                lw=2,
                label='ROC curve of {0} class (area = {1:0.2f})'
                ''.format(class_dict[i], roc_auc[i]))
        plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.legend(loc="lower right")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid()
    plt.show()


def print_metrics(model, X_train, y_train, y_pred, X_test, y_test, class_names):
    model.fit(X_train, y_train)
    print(classification_report(y_test, y_pred))
    fig, axs = plt.subplots(figsize=(10, 8))
    disp = plot_confusion_matrix(model, X_test, y_test,
                                display_labels=class_names,
                                cmap=plt.cm.Blues,
                                normalize=None,
                                ax=axs)
    axs.set_title('Normalized confusion matrix', fontsize=20)
    axs.set_xlabel('Predicted label', fontsize=16)
    axs.set_ylabel('True label', fontsize=16)
    print('Normalized confusion matrix:')
    print(disp.confusion_matrix)
    plt.show()
