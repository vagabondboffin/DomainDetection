import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error, roc_curve, classification_report,auc)
from sklearn.tree import DecisionTreeClassifier

def chooseClassifier(Xtrain, Ytrain, Xtest, Ytest, classifier):
    #sc = StandardScaler()
    #Xtrain = sc.fit_transform(Xtrain)
    #Xtest = sc.transform(Xtest)

    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)

    Xtest = np.array(Xtest)
    Ytest = np.array(Ytest)

    if classifier == 'logistic regression':
        model = LogisticRegression()

    elif classifier == 'Naive Bayes':
        model = GaussianNB()

    elif classifier == 'Decision Tree':
        model = DecisionTreeClassifier()

    elif classifier == 'AdaBoost':
        model = AdaBoostClassifier()

    elif classifier == 'Random Forest':
        model = RandomForestClassifier(n_estimators = 100)

    model.fit(Xtrain, Ytrain)
    expected = Ytest
    predicted = model.predict(Xtest)
    accuracy = accuracy_score(expected, predicted)
    recall = recall_score(expected, predicted, average = 'macro')
    precision = precision_score(expected, predicted, average = 'macro')
    f1 = f1_score(expected, predicted, average = 'macro')
    cm = metrics.confusion_matrix(expected, predicted)
    #print(cm)

    print("----------------------------------------------")
    print("accuracy")
    print("%.3f" %accuracy)
    print("precision")
    print("%.3f" %precision)
    print("racall")
    print("%.3f" %recall)
    print("f1score")
    print("%.3f" %f1)

def SVMClassifier(Xtrain, Ytrain, Xtest, Ytest):
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']

    for score in scores:
        clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                           scoring='%s_macro' % score)
        clf.fit(Xtrain, Ytrain)
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
            y_true, y_pred = Ytest, clf.predict(Xtest)
            predicted = y_pred
            expected = y_true
            accuracy = accuracy_score(expected, predicted)
            recall = recall_score(expected, predicted, average='macro')
            precision = precision_score(expected, predicted, average='macro')
            f1 = f1_score(expected, predicted, average='macro')
            # cm = metrics.confusion_matrix(expected, predicted)
            # print(cm)
            print("----------------------------------------------")
            print("accuracy")
            print("%.3f" % accuracy)
            print("precision")
            print("%.3f" % precision)
            print("racall")
            print("%.3f" % recall)
            print("f1score")
            print("%.3f" % f1)