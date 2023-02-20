import pandas as pd
import seaborn
import Classifiers
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

def extraTree(X, Y, n):
    model = ExtraTreesClassifier()
    model.fit(X, Y)
    print(model.feature_importances_)
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_selected = feat_importances.nlargest(n).keys()
    return feat_selected

def RecursiveFeatureElimination(X, Y, n):
    estimator = DecisionTreeClassifier()
    rfe = RFE(estimator, n_features_to_select = n)
    rfe = rfe.fit(X, Y)
    feat_selected = rfe.get_support(indices = True)
    return rfe, feat_selected

def PrincipalComponentAnalysis(X, n):
    pca = PCA(n_components = n)
    pca.fit(X)
    feat_selected = pca.get_feature_names_out()
    return pca, feat_selected

def RecursiveFeatureEliminationCV(X, Y):
    rfecv = RFECV(
        estimator = DecisionTreeClassifier(),
        min_features_to_select = 3,
        step = 5,
        n_jobs = -1,
        scoring = "r2",
        cv = 5,
    )
    _ = rfecv.fit(X, Y)
    feat_selected = rfecv.get_support(indices = True)
    return rfecv, feat_selected