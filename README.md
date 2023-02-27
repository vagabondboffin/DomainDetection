# DomainDetection
# Exploring different ML concept in a Domain Name Detection problem

## 1.	Exploratory Data Analysis
Codes for this part are mainly in main.py file. Preprocessing_() and EDA() functions describe actions in this part. 

a.	Preprocessing_()
In our dataset, there is a feature describing ip addresses. To handle this feature, we use ipaddress library to convert ip addresses to numbers. 
After that, we have to deal with the labels. Since labels are categorical, domain names” we use LabelEncoder() to convert labels to encoded versions.  

b.	EDA()
We use head() and describe() functions to get a view and sense of the data. We also use heatmap from seaborn library to map the correlations between features. 
But since the number of features is large, these information is not much readable. 
We use a feature selection method called “extra tree classifier” to first compute feature importance. 
We select the five most important features and repeat the head(), describe(), and correlation computing tasks. 
 
## 2.	Supervised Classification
Codes for this part are in the choosingClassifier() method in Classifier.py file. But the call for the module is in the main() method of main.py. 
Set the classifier in main(). Five different supervised classifiers are implemented. After running main.py with the desired classifier, the results are automatically printed. 
Also if you want to train a SVM, you can call the SVMClassifier() method. Using a grid search, an optimal set of parameters are obtained to train the SVM.

## 3.	Feature Selection
Three different feature selection methods are implemented. 

a.	Extra Tree Classifier
Implemented in ExtraTree()

b.	RFE
Implemented in RecursiveFeatureElimination(). 
SVM is used as the estimator.

c.	PCA
Implemented in PrincipalComponentAnalysis().
It is recommended to use a scalar before using PCA.

All these methods are implemented in the featureSelecetion.py file. Before using PCA in main.py, a scalar is used. 
Method RecursiveFeatureEliminationCV(), finds the optimal number of features (task 3-1)

## 4.	Clustering
Before clustering, a classifier (SVR was used but you can change it in featureSelection.py) was used to determine the best features. After feature selection, we continued with 5 features. 
We used KMeans as the clustering algorithm. To estimate the optimal number of clusters, we used Elbow method with Within Cluster Sum of Squares (WCSS) metric. According to the elbow method, we select the number of clusters where the change in WCSS begins to level off. Considering the below image a good number of clusters is 2 or 6. 
We continue clustering with 6 clusters. 
