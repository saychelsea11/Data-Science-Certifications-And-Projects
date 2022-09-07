
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.2** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-machine-learning/resources/bANLa) course resource._
# 
# ---

# # Assignment 3 - Evaluation
# 
# In this assignment you will train several models and evaluate how effectively they predict instances of fraud using data based on [this dataset from Kaggle](https://www.kaggle.com/dalpozz/creditcardfraud).
#  
# Each row in `fraud_data.csv` corresponds to a credit card transaction. Features include confidential variables `V1` through `V28` as well as `Amount` which is the amount of the transaction. 
#  
# The target is stored in the `class` column, where a value of 1 corresponds to an instance of fraud and 0 corresponds to an instance of not fraud.

# In[2]:

import numpy as np
import pandas as pd


# ### Question 1
# Import the data from `fraud_data.csv`. What percentage of the observations in the dataset are instances of fraud?
# 
# *This function should return a float between 0 and 1.* 

# In[3]:

def answer_one():
    from collections import Counter
    df = pd.read_csv("fraud_data.csv")
    bins = Counter(df["Class"])
    frauds_percent = bins[1]/len(df)
    
    return frauds_percent

answer_one()


# In[5]:

# Use X_train, X_test, y_train, y_test for all of the following questions
from sklearn.model_selection import train_test_split

df = pd.read_csv('fraud_data.csv')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# ### Question 2
# 
# Using `X_train`, `X_test`, `y_train`, and `y_test` (as defined above), train a dummy classifier that classifies everything as the majority class of the training data. What is the accuracy of this classifier? What is the recall?
# 
# *This function should a return a tuple with two floats, i.e. `(accuracy score, recall score)`.*

# In[6]:

def answer_two():
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import recall_score,accuracy_score
    from sklearn.metrics import confusion_matrix
    from adspy_shared_utilities import plot_class_regions_for_classifier
    
    dummy_fit = DummyClassifier(strategy="most_frequent").fit(X_train,y_train)
    accuracy = accuracy_score(y_test,dummy_fit.predict(X_test))
    recall = recall_score(y_test,dummy_fit.predict(X_test))
    cm = confusion_matrix(y_test,dummy_fit.predict(X_test))
    #print (dummy_fit.predict(X_test))
    
    return accuracy,recall

answer_two()


# ### Question 3
# 
# Using X_train, X_test, y_train, y_test (as defined above), train a SVC classifer using the default parameters. What is the accuracy, recall, and precision of this classifier?
# 
# *This function should a return a tuple with three floats, i.e. `(accuracy score, recall score, precision score)`.*

# In[31]:

def answer_three():
    from sklearn.metrics import recall_score, precision_score, accuracy_score
    from sklearn.svm import SVC

    svm = SVC().fit(X_train,y_train)
    y_predict = svm.predict(X_test)
    accuracy = accuracy_score(y_test,y_predict)
    recall = recall_score(y_test,y_predict)
    precision = precision_score(y_test,y_predict)
    
    return accuracy,recall,precision

answer_three()


# ### Question 4
# 
# Using the SVC classifier with parameters `{'C': 1e9, 'gamma': 1e-07}`, what is the confusion matrix when using a threshold of -220 on the decision function. Use X_test and y_test.
# 
# *This function should return a confusion matrix, a 2x2 numpy array with 4 integers.*

# In[52]:

def answer_four():
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC
    #import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    svm = SVC(gamma=1e-07,C=1e9).fit(X_train,y_train)
    thresholds = svm.decision_function(X_test)
    
    #plt.figure()
    #plt.plot(thresholds[thresholds>-10000])
    thresholds[thresholds>(-220)] = 1
    thresholds[thresholds<(-220)] = 0
    cm = confusion_matrix(y_test,thresholds)
    #plt.figure()
    #plt.plot(svm.predict(X_test))
    #plt.show()
    
    
    return cm

answer_four()


# ### Question 5
# 
# Train a logisitic regression classifier with default parameters using X_train and y_train.
# 
# For the logisitic regression classifier, create a precision recall curve and a roc curve using y_test and the probability estimates for X_test (probability it is fraud).
# 
# Looking at the precision recall curve, what is the recall when the precision is `0.75`?
# 
# Looking at the roc curve, what is the true positive rate when the false positive rate is `0.16`?
# 
# *This function should return a tuple with two floats, i.e. `(recall, true positive rate)`.*

# In[61]:

def answer_five():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import roc_curve,auc
    #import matplotlib.pyplot as plt
    
    logreg = LogisticRegression().fit(X_train,y_train)
    y_predict = logreg.predict(X_test)
    decisions = logreg.decision_function(X_test)
    proba = logreg.predict_proba(X_test)
    precision, recall, threshold = precision_recall_curve(y_test,decisions)
    fpr,tpr,_ = roc_curve(y_test,decisions)
    
    #plt.figure()
    #plt.plot(precision,recall)
    
    #plt.figure()
    #plt.plot(fpr,tpr)
    
    #plt.figure()
    #plt.plot(fpr1,tpr1)
    #plt.show()
    
    return 0.82,0.94

answer_five()


# ### Question 6
# 
# Perform a grid search over the parameters listed below for a Logisitic Regression classifier, using recall for scoring and the default 3-fold cross validation.
# 
# `'penalty': ['l1', 'l2']`
# 
# `'C':[0.01, 0.1, 1, 10, 100]`
# 
# From `.cv_results_`, create an array of the mean test scores of each parameter combination. i.e.
# 
# |      	| `l1` 	| `l2` 	|
# |:----:	|----	|----	|
# | **`0.01`** 	|    ?	|   ? 	|
# | **`0.1`**  	|    ?	|   ? 	|
# | **`1`**    	|    ?	|   ? 	|
# | **`10`**   	|    ?	|   ? 	|
# | **`100`**   	|    ?	|   ? 	|
# 
# <br>
# 
# *This function should return a 5 by 2 numpy array with 10 floats.* 
# 
# *Note: do not return a DataFrame, just the values denoted by '?' above in a numpy array. You might need to reshape your raw result to meet the format we are looking for.*

# In[26]:

def answer_six():    
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    params = {'penalty':['l1','l2'],'C':[0.01,0.1,1,10,100]}
    logreg = LogisticRegression()
    grid_fit = GridSearchCV(logreg,param_grid=params,scoring="recall",cv=3)
    grid_fit.fit(X_train,y_train)
    mean_scores = grid_fit.cv_results_['mean_test_score']
    mean_scores = np.array(mean_scores.reshape(5,2))
    
    return mean_scores

answer_six()


# In[27]:

# Use the following function to help visualize results from the grid search
def GridSearch_Heatmap(scores):
    get_ipython().magic('matplotlib notebook')
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure()
    sns.heatmap(scores.reshape(5,2), xticklabels=['l1','l2'], yticklabels=[0.01, 0.1, 1, 10, 100])
    plt.yticks(rotation=0);

#GridSearch_Heatmap(answer_six())


# In[ ]:



