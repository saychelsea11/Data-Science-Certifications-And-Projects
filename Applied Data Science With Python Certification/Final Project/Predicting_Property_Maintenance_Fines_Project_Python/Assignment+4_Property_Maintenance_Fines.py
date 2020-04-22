
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-machine-learning/resources/bANLa) course resource._
# 
# ---

# ## Assignment 4 - Understanding and Predicting Property Maintenance Fines
# 
# This assignment is based on a data challenge from the Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)). 
# 
# The Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)) and the Michigan Student Symposium for Interdisciplinary Statistical Sciences ([MSSISS](https://sites.lsa.umich.edu/mssiss/)) have partnered with the City of Detroit to help solve one of the most pressing problems facing Detroit - blight. [Blight violations](http://www.detroitmi.gov/How-Do-I/Report/Blight-Complaint-FAQs) are issued by the city to individuals who allow their properties to remain in a deteriorated condition. Every year, the city of Detroit issues millions of dollars in fines to residents and every year, many of these fines remain unpaid. Enforcing unpaid blight fines is a costly and tedious process, so the city wants to know: how can we increase blight ticket compliance?
# 
# The first step in answering this question is understanding when and why a resident might fail to comply with a blight ticket. This is where predictive modeling comes in. For this assignment, your task is to predict whether a given blight ticket will be paid on time.
# 
# All data for this assignment has been provided to us through the [Detroit Open Data Portal](https://data.detroitmi.gov/). **Only the data already included in your Coursera directory can be used for training the model for this assignment.** Nonetheless, we encourage you to look into data from other Detroit datasets to help inform feature creation and model selection. We recommend taking a look at the following related datasets:
# 
# * [Building Permits](https://data.detroitmi.gov/Property-Parcels/Building-Permits/xw2a-a7tf)
# * [Trades Permits](https://data.detroitmi.gov/Property-Parcels/Trades-Permits/635b-dsgv)
# * [Improve Detroit: Submitted Issues](https://data.detroitmi.gov/Government/Improve-Detroit-Submitted-Issues/fwz3-w3yn)
# * [DPD: Citizen Complaints](https://data.detroitmi.gov/Public-Safety/DPD-Citizen-Complaints-2016/kahe-efs3)
# * [Parcel Map](https://data.detroitmi.gov/Property-Parcels/Parcel-Map/fxkw-udwf)
# 
# ___
# 
# We provide you with two data files for use in training and validating your models: train.csv and test.csv. Each row in these two files corresponds to a single blight ticket, and includes information about when, why, and to whom each ticket was issued. The target variable is compliance, which is True if the ticket was paid early, on time, or within one month of the hearing data, False if the ticket was paid after the hearing date or not at all, and Null if the violator was found not responsible. Compliance, as well as a handful of other variables that will not be available at test-time, are only included in train.csv.
# 
# Note: All tickets where the violators were found not responsible are not considered during evaluation. They are included in the training set as an additional source of data for visualization, and to enable unsupervised and semi-supervised approaches. However, they are not included in the test set.
# 
# <br>
# 
# **File descriptions** (Use only this data for training your model!)
# 
#     train.csv - the training set (all tickets issued 2004-2011)
#     test.csv - the test set (all tickets issued 2012-2016)
#     addresses.csv & latlons.csv - mapping from ticket id to addresses, and from addresses to lat/lon coordinates. 
#      Note: misspelled addresses may be incorrectly geolocated.
# 
# <br>
# 
# **Data fields**
# 
# train.csv & test.csv
# 
#     ticket_id - unique identifier for tickets
#     agency_name - Agency that issued the ticket
#     inspector_name - Name of inspector that issued the ticket
#     violator_name - Name of the person/organization that the ticket was issued to
#     violation_street_number, violation_street_name, violation_zip_code - Address where the violation occurred
#     mailing_address_str_number, mailing_address_str_name, city, state, zip_code, non_us_str_code, country - Mailing address of the violator
#     ticket_issued_date - Date and time the ticket was issued
#     hearing_date - Date and time the violator's hearing was scheduled
#     violation_code, violation_description - Type of violation
#     disposition - Judgment and judgement type
#     fine_amount - Violation fine amount, excluding fees
#     admin_fee - $20 fee assigned to responsible judgments
# state_fee - $10 fee assigned to responsible judgments
#     late_fee - 10% fee assigned to responsible judgments
#     discount_amount - discount applied, if any
#     clean_up_cost - DPW clean-up or graffiti removal cost
#     judgment_amount - Sum of all fines and fees
#     grafitti_status - Flag for graffiti violations
#     
# train.csv only
# 
#     payment_amount - Amount paid, if any
#     payment_date - Date payment was made, if it was received
#     payment_status - Current payment status as of Feb 1 2017
#     balance_due - Fines and fees still owed
#     collection_status - Flag for payments in collections
#     compliance [target variable for prediction] 
#      Null = Not responsible
#      0 = Responsible, non-compliant
#      1 = Responsible, compliant
#     compliance_detail - More information on why each ticket was marked compliant or non-compliant
# 
# 
# ___
# 
# ## Evaluation
# 
# Your predictions will be given as the probability that the corresponding blight ticket will be paid on time.
# 
# The evaluation metric for this assignment is the Area Under the ROC Curve (AUC). 
# 
# Your grade will be based on the AUC score computed for your classifier. A model which with an AUROC of 0.7 passes this assignment, over 0.75 will recieve full points.
# ___
# 
# For this assignment, create a function that trains a model to predict blight ticket compliance in Detroit using `train.csv`. Using this model, return a series of length 61001 with the data being the probability that each corresponding ticket from `test.csv` will be paid, and the index being the ticket_id.
# 
# Example:
# 
#     ticket_id
#        284932    0.531842
#        285362    0.401958
#        285361    0.105928
#        285338    0.018572
#                  ...
#        376499    0.208567
#        376500    0.818759
#        369851    0.018528
#        Name: compliance, dtype: float32
#        
# ### Hints
# 
# * Make sure your code is working before submitting it to the autograder.
# 
# * Print out your result to see whether there is anything weird (e.g., all probabilities are the same).
# 
# * Generally the total runtime should be less than 10 mins. You should NOT use Neural Network related classifiers (e.g., MLPClassifier) in this question. 
# 
# * Try to avoid global variables. If you have other functions besides blight_model, you should move those functions inside the scope of blight_model.
# 
# * Refer to the pinned threads in Week 4's discussion forum when there is something you could not figure it out.

# In[5]:

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from pandas import ExcelWriter
from sklearn.metrics import accuracy_score, recall_score, auc, roc_curve, precision_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier

def ret_type(a):
    return type(a)

def blight_model():
    
    train_df = pd.read_csv("train.csv",encoding = 'ISO-8859-1')
    test_df = pd.read_csv("test.csv")
    address_df = pd.read_csv("addresses.csv")
    
    #Label encoding categorical variables
    le = LabelEncoder().fit(train_df["violation_street_name"].append(test_df["violation_street_name"]))
    train_df["violation_street_name"] = le.transform(train_df["violation_street_name"])
    test_df["violation_street_name"] = le.transform(test_df["violation_street_name"])
    
    
    train_df["mailing_address_str_name"] = train_df["mailing_address_str_name"].map(str)
    test_df["mailing_address_str_name"] = test_df["mailing_address_str_name"].map(str)
    le = LabelEncoder().fit(train_df["mailing_address_str_name"].append(test_df["mailing_address_str_name"]))
    train_df["mailing_address_str_name"] = le.transform(train_df["mailing_address_str_name"])
    test_df["mailing_address_str_name"] = le.transform(test_df["mailing_address_str_name"])
    
    
    le = LabelEncoder().fit(train_df["violation_description"].append(test_df["violation_description"]))
    train_df["violation_description"] = le.transform(train_df["violation_description"])
    test_df["violation_description"] = le.transform(test_df["violation_description"])
    
    le = LabelEncoder().fit(train_df["disposition"].append(test_df["disposition"]))
    train_df["disposition"] = le.transform(train_df["disposition"])
    test_df["disposition"] = le.transform(test_df["disposition"])
    '''
    #Remove instances in "mailing_address_str_number" that start with "P.O. Bo"
    test_df["mailing_address_str_number"] = test_df["mailing_address_str_number"].replace('P.O. Bo',0)
    test_df["mailing_address_str_number"] = test_df["mailing_address_str_number"].replace('P.O. BO',0)
    test_df["mailing_address_str_number"] = test_df["mailing_address_str_number"].replace('P.O. Box',0)
    test_df["mailing_address_str_number"] = test_df["mailing_address_str_number"].replace('P.O. BOX',0)
    test_df["mailing_address_str_number"] = test_df["mailing_address_str_number"].replace('PO BOX')
    test_df["mailing_address_str_number"] = test_df["mailing_address_str_number"].replace('P.O.BOX')
    test_df["mailing_address_str_number"] = test_df["mailing_address_str_number"].replace('P.O.')
    test_df["mailing_address_str_number"] = test_df["mailing_address_str_number"].replace('215-B')
    test_df["mailing_address_str_number"] = test_df["mailing_address_str_number"].replace('47-42')
    test_df["mailing_address_str_number"] = test_df["mailing_address_str_number"].replace('141-19')
    test_df["mailing_address_str_number"] = test_df["mailing_address_str_number"].replace('p.o. bo')
    test_df["mailing_address_str_number"] = test_df["mailing_address_str_number"].replace('98-8184')
    test_df["mailing_address_str_number"] = test_df["mailing_address_str_number"].replace('ONE')
    test_df["mailing_address_str_number"] = test_df["mailing_address_str_number"].replace('242-12')
    test_df["mailing_address_str_number"] = test_df["mailing_address_str_number"].replace('P. O. B')
    test_df["mailing_address_str_number"] = test_df["mailing_address_str_number"].replace('94-28')
    test_df["mailing_address_str_number"] = test_df["mailing_address_str_number"].replace('P O BOX')
    test_df["mailing_address_str_number"] = test_df["mailing_address_str_number"].replace('48-2')
    '''
    #Removing the features that don't provide much useful information regarding the compliance - feature selection
    #Also removing features that may directly relate to compliance and whereby cause data leakage
    train_df = train_df.drop(["violation_zip_code","clean_up_cost","grafitti_status","discount_amount","non_us_str_code"],axis=1)
    train_df = train_df.drop(["violator_name","balance_due","payment_status","payment_date","collection_status","compliance_detail"],axis=1)
    train_df = train_df.drop(["admin_fee","state_fee","payment_amount","state","agency_name","inspector_name","country","violation_code"],axis=1)
    train_df = train_df.drop(["zip_code","mailing_address_str_number"],axis=1)
    test_df = test_df.drop(["violation_zip_code","clean_up_cost","grafitti_status","discount_amount","non_us_str_code"],axis=1)
    test_df = test_df.drop(["violator_name","admin_fee","state_fee","state","agency_name","inspector_name","country","violation_code"],axis=1)
    test_df = test_df.drop(["zip_code","mailing_address_str_number"],axis=1)
    
    #Changing all city entries to upper case to make it simpler
    train_df["city"] = train_df["city"].apply(str.upper)
    
    #Filtering tickets only for the city of Detroit and then dropping the 'city' variable 
    #since all instances would be for Detroit
    train_df = train_df[train_df["city"]=="DETROIT"]
    train_df = train_df.drop(["city"],axis=1)
    test_df = test_df.drop(["city"],axis=1)
    
    #Removing all 'not found responsible' compliance entries
    train_df = train_df[(train_df["compliance"]==1.0) | (train_df["compliance"]==0.0)]
    
    #Turning the late_fee variable to binary values with 0 indicating no late fee and 1 indicating a late fee
    train_df["late_fee_bin"] = (train_df["late_fee"]>0).map(int)
    train_df = train_df.drop(["late_fee"],axis=1)
    test_df["late_fee_bin"] = (test_df["late_fee"]>0).map(int)
    test_df = test_df.drop(["late_fee"],axis=1)
    
    #Dropping all the 'Nan' instances from the training set
    train_df = train_df.dropna()
    
    #Creating a new variable for storing the time difference between the ticket issued date and hearing date
    #And converting the difference into days for easier calculation
    train_df["ticket_issued_date"] = train_df.apply(lambda x: pd.to_datetime(x[6]),axis=1)
    train_df["hearing_date"] = train_df.apply(lambda x: pd.to_datetime(x[7]),axis=1)
    test_df["ticket_issued_date"] = test_df.apply(lambda x: pd.to_datetime(x[6]),axis=1)
    test_df["hearing_date"] = test_df.apply(lambda x: pd.to_datetime(x[7]),axis=1)
    train_df["time_to_hearing"] = train_df["hearing_date"] - train_df["ticket_issued_date"]
    test_df["time_to_hearing"] = test_df["hearing_date"] - test_df["ticket_issued_date"]
    
    test_df["time_to_hearing"] = test_df["time_to_hearing"].map(str)
    test_df["time_to_hearing"] = test_df["time_to_hearing"].replace('NaT',str(pd.Timestamp('1/1/18')-pd.Timestamp('1/1/18')))
    test_df["time_to_hearing"] = test_df["time_to_hearing"].map(pd.Timedelta)
    
    train_df["time_to_hearing"] = train_df["time_to_hearing"].apply(lambda x: x.days)
    test_df["time_to_hearing"] = test_df["time_to_hearing"].apply(lambda x: x.days)
    
    #Removing original time variables
    train_df = train_df.drop(["ticket_issued_date","hearing_date"],axis=1)
    test_df = test_df.drop(["ticket_issued_date","hearing_date"],axis=1)
    
    #Separating the 'ticket_id' column
    ticket_id_train = train_df["ticket_id"]
    ticket_id_test = test_df["ticket_id"]
    train_df = train_df.drop(["ticket_id"],axis=1)
    test_df = test_df.drop(["ticket_id"],axis=1)
    
    
#################################################### End of Pre-processing #####################################################
    
    #Splitting the training dataset into features and target variables
    y = train_df["compliance"]
    x = train_df.drop(["compliance"],axis=1)
    
    #Train/Test Split
    X_train,X_test,y_train,y_test = train_test_split(x,y,random_state=0)
    
    #Normalization using MinMax scaler
    #scaler = MinMaxScaler()
    #X_train_scaled = scaler.fit_transform(X_train)
    #X_test_scaled = scaler.transform(X_test)
    
    #for i in [0.1]:
    #    for j in [25,30,35,40,45,50]:
    #        for k in [5,6,7,8,9,10]:
            
    #Fitting the model
    #logreg = LogisticRegression(C=0.1,random_state=0)
    #gbc = GradientBoostingClassifier(random_state=0,learning_rate=i,n_estimators=j,max_depth=k)
    gbc = GradientBoostingClassifier(random_state=0,learning_rate=0.1,n_estimators=45,max_depth=5)
    #model = logreg.fit(X_train,y_train)
    #model = logreg.fit(X_train_scaled,y_train)
    model = gbc.fit(X_train,y_train)

#################################################### End of Training/Fitting ###################################################

    #Determining the decision function
    #y_score_lr = model.decision_function(X_test_scaled)
    y_score_eval = model.decision_function(X_test)
    y_proba_eval = model.predict_proba(X_test)
    y_score = model.decision_function(test_df)
    y_proba = model.predict_proba(test_df)

    #print (y_score_lr[:20])
    #print (y_proba[:20,1])
    #print (y_proba.shape)
    #print (ticket_id_test.shape)
    #print (test_df.iloc[:20,:])
    #print (y_test[:20])

    #Determining ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_score_eval)
    roc_auc = auc(fpr, tpr)

    #Prediction
    #y_predict = model.predict(X_test_scaled)
    #y_predict_test_df = model.predict(test_df)
    #y_predict = model.predict(X_test)

    #Classification report
    #print (classification_report(y_test,y_predict))

    #print ("Learning rate: ",i)
    #print ("Estimators: ",j)
    #print ("Max depth: ",k)
    #print ("")
    #print ("Area under the curve (AUC) score ",roc_auc)
    #print ("Accuracy ",accuracy_score(y_test,y_predict))
    #print ("")
    #print (test_df.columns)
    
#################################################### End of evaluation #########################################################
    
    #grid_values = {'C': [0.001, 0.01, 0.05, 0.1, 1, 10, 100]}
    #grid_values = {'learning_rate':[0.001,0.01,0.1,1,10,100],'max_depth':[1,2,3,4,5],'n_estimators':[1,2,5,10]}
    
    #grid_clf = GridSearchCV(gbc, param_grid = grid_values, scoring='roc_auc',cv=5)
    #grid_clf.fit(X_train_scaled, y_train)
    #grid_clf.fit(X_train, y_train)
    #print (grid_clf.cv_results_)
    #print ("")
    #print ("Max AUC C parameter ",grid_clf.best_params_)
    #print ("Best AUC score ",grid_clf.best_score_)
    
    #y_decision_fn_scores = grid_clf.decision_function(X_test) 
    
################################################### End of Optimization ########################################################

    #Creating the final compliance probability outcomes with ticket ID as index
    compliance = pd.DataFrame()
    compliance["ticket_id"] = ticket_id_test
    compliance["compliance"] = y_proba[:,1]
    compliance = compliance.set_index("ticket_id")
    
    #plt.scatter(range(len(y_proba_lr[:20,1])),y_proba_lr[:20,1])
    #plt.scatter(range(len(y_proba_lr[:20,0])),y_proba_lr[:20,0])
    #plt.show()
    
    return compliance["compliance"]

blight_model()


# In[ ]:




# In[ ]:



