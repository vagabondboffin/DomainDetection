# This is a sample Python script.

import ipaddress
import Clustering
import featureSelection
import Classifiers

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import seaborn
import Classifiers
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from matplotlib import pyplot as plt

def preprocessing_():
    train = pd.read_csv('Project2_training.csv')
    validation = pd.read_csv('project2_validation.csv')
    Xtrain = train.iloc[:,0:-1]
    Ytrain = train.iloc[:,-1]
    Xvalid= validation.iloc[:,0:-1]
    Yvalid = validation.iloc[:,-1]
    Xtrain['c_ip'] = Xtrain['c_ip'].apply(lambda x: int(ipaddress.ip_address(x)))
    Xvalid['c_ip'] = Xvalid['c_ip'].apply(lambda x: int(ipaddress.ip_address(x)))
    le = preprocessing.LabelEncoder()
    le.fit(Ytrain)
    Ytrain = le.transform(Ytrain)
    Yvalid = le.transform(Yvalid)

    return Xtrain, Ytrain, Xvalid, Yvalid, le

def EDA(X, Y):
    head = X.head()
    desc = X.describe()
    pd.DataFrame.to_csv(desc)
    print(desc)
    print(head)
    seaborn.heatmap(X.corr())
    ifnull_ = X.isnull().sum()  # no null attributes

    # feature selection
    model = ExtraTreesClassifier()
    model.fit(X, Y)
    print(model.feature_importances_)
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(5).plot.bar()
    plt.show()

    feat_selected = feat_importances.nlargest(5).keys()
    Xnew = X[feat_selected]
    seaborn.heatmap(Xnew.corr())
    headnew = Xnew.head()
    descnew = Xnew.describe()
    with pd.ExcelWriter('out_res.xlsx') as writer:
        headnew.to_excel(writer, sheet_name='Sheet1')
        descnew.to_excel(writer, sheet_name='Sheet2')
        desc.to_excel(writer, sheet_name='Sheet3')

    print(headnew)
    print(descnew)

if __name__ == '__main__':
    print("task 1")
    ## part 1: preprocessing the dataset and get an insight into it
    Xtrain, Ytrain, Xvalid, Yvalid, labelEncoder = preprocessing_()
    #EDA(Xtrain, Ytrain)

    print("task 2")
    ## part 2: choose a supervised algorithm
    ## using method chooseClassifier(), you can select a classifier (Logistic Regression, Random Forest,
    ## Decision Tree, Naive Bayes, or AdaBoost)
    #sc = StandardScaler()
    #Xtrain_scaled = sc.fit_transform(Xtrain)
    #Xtest_scaled = sc.transform(Xvalid)
    #classifier = 'Random Forest'
    #Classifiers.chooseClassifier(Xtrain_scaled, Ytrain, Xtest_scaled, Yvalid, classifier)

    ## SVM: before using svm, extraTreeClassifier is used to reduce the number of features
    #sc = StandardScaler()
    #Xtrain_scaled = pd.DataFrame(sc.fit_transform(Xtrain))
    #Xtest_scaled = pd.DataFrame(sc.transform(Xvalid))
    #selectedFeatures = featureSelection.extraTree(Xtrain_scaled, Ytrain, 5)
    #Xtrain_new = Xtrain_scaled[selectedFeatures]
    #Xvalid_new = Xtest_scaled[selectedFeatures]
    #Classifiers.SVMClassifier(Xtrain_new, Ytrain, Xvalid_new, Yvalid)

    print("task 3")
    ## part 3: Feature Selection methods
    #sc = StandardScaler()
    # = pd.DataFrame(sc.fit_transform(Xtrain))
    #Xtest_scaled = pd.DataFrame(sc.transform(Xvalid))
    ## method1
    #rfe, selectedFeatures1 = featureSelection.RecursiveFeatureElimination(Xtrain_scaled, Ytrain, 5)
    #rfecv, selectedFeatures2 = featureSelection.RecursiveFeatureEliminationCV(Xtrain_scaled, Ytrain)
    #classifier = 'Random Forest'
    #print("result for ref")
    #Classifiers.chooseClassifier(Xtrain_scaled[selectedFeatures1], Ytrain, Xtest_scaled[selectedFeatures1], Yvalid, classifier)
    #print("result for refcv")
    #Classifiers.chooseClassifier(Xtrain_scaled[selectedFeatures2], Ytrain, Xtest_scaled[selectedFeatures2], Yvalid, classifier)

    ## method2
    #pca, feat_selected = featureSelection.PrincipalComponentAnalysis(Xtrain_scaled, 5)

    ## method3
    #selectedFeatures3 = featureSelection.extraTree(Xtrain_scaled, Ytrain, 5)



    print('task 4')
    from sklearn.cluster import KMeans
    from matplotlib import pyplot as plt
    ## part 4: clustering
    sc = StandardScaler()
    Xtrain_scaled = sc.fit_transform(Xtrain)
    rfe, selectedFeatures1 = featureSelection.RecursiveFeatureElimination(Xtrain_scaled, Ytrain, 5)
    Xtrain_selected = Xtrain_scaled[selectedFeatures1]
    #Clustering.ElbowKMeans(Xtrain_selected)
    #Y_pred = Clustering.Kmeans_(Xtrain_selected, 6)
    kmeans = KMeans(n_clusters=6, init='k-means++', max_iter=300, n_init=10, random_state=0)


