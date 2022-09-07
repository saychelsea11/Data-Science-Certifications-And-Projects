
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 3
# 
# In this assignment you will explore text message data and create models to predict if a message is spam or not. 

# In[1]:

import pandas as pd
import numpy as np

spam_data = pd.read_csv('spam.csv')

spam_data['target'] = np.where(spam_data['target']=='spam',1,0)
spam_data.head(10)


# In[2]:

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(spam_data['text'], 
                                                    spam_data['target'], 
                                                    random_state=0)


# ### Question 1
# What percentage of the documents in `spam_data` are spam?
# 
# *This function should return a float, the percent value (i.e. $ratio * 100$).*

# In[5]:

def answer_one():
    
    spam_data_spam_count = spam_data[spam_data["target"]==1]["target"].count()

    return (spam_data_spam_count/(spam_data["target"].count()))*100


# In[6]:

answer_one()


# ### Question 2
# 
# Fit the training data `X_train` using a Count Vectorizer with default parameters.
# 
# What is the longest token in the vocabulary?
# 
# *This function should return a string.*

# In[6]:

from sklearn.feature_extraction.text import CountVectorizer

def answer_two():
    
    # Fit the CountVectorizer to the training data
    vect = CountVectorizer().fit(X_train)
    
    df = pd.DataFrame()
    df["features"] = vect.get_feature_names()
    df["length"] = df["features"].map(len)
    df = df.sort_values(by="length",ascending=False)
    
    return df.iloc[0,0]#df["features"][0]


# In[10]:

answer_two()


# ### Question 3
# 
# Fit and transform the training data `X_train` using a Count Vectorizer with default parameters.
# 
# Next, fit a fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1`. Find the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[7]:

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

def answer_three():
    
    vect = CountVectorizer().fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    multi_NB = MultinomialNB(alpha=0.1).fit(X_train_vectorized,y_train)
    y_predict = multi_NB.predict(vect.transform(X_test))
    auc = roc_auc_score(y_test,y_predict)
    
    return auc


# In[31]:

answer_three()


# ### Question 4
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer with default parameters.
# 
# What 20 features have the smallest tf-idf and what 20 have the largest tf-idf?
# 
# Put these features in a two series where each series is sorted by tf-idf value and then alphabetically by feature name. The index of the series should be the feature name, and the data should be the tf-idf.
# 
# The series of 20 features with smallest tf-idfs should be sorted smallest tfidf first, the list of 20 features with largest tf-idfs should be sorted largest first. 
# 
# *This function should return a tuple of two series
# `(smallest tf-idfs series, largest tf-idfs series)`.*

# In[8]:

from sklearn.feature_extraction.text import TfidfVectorizer

def answer_four():
    
    vect = TfidfVectorizer().fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    
    # get the feature names as numpy array
    feature_names = np.array(vect.get_feature_names())
    
    sorted_tfidf_index = X_train_vectorized.max(0).toarray()[0].argsort()
    
    df = pd.DataFrame()
    df["smallest tfdif feature"] = feature_names[sorted_tfidf_index[:20]]
    df["largest tfdif feature"] = feature_names[sorted_tfidf_index[:-21:-1]]
    df["smallest tfdif value"] = sorted(X_train_vectorized.max(0).toarray()[0])[:20]
    df["largest tfdif value"] = sorted(X_train_vectorized.max(0).toarray()[0])[:-21:-1]
    
    small_tfdif = pd.Series(list(df["smallest tfdif value"]),index=df["smallest tfdif feature"])
    large_tfdif = pd.Series(list(df["largest tfdif value"]),index=df["largest tfdif feature"])
    
    return small_tfdif,large_tfdif


# In[4]:

answer_four()


# ### Question 5
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **3**.
# 
# Then fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1` and compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[84]:

def answer_five():
    
    vect = TfidfVectorizer(min_df=3).fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    multi_NB = MultinomialNB(alpha=0.1).fit(X_train_vectorized,y_train)
    y_predict = multi_NB.predict(vect.transform(X_test))
    auc = roc_auc_score(y_test,y_predict)
    
    return auc


# In[85]:

answer_five()


# ### Question 6
# 
# What is the average length of documents (number of characters) for not spam and spam documents?
# 
# *This function should return a tuple (average length not spam, average length spam).*

# In[91]:

def answer_six():
    
    spam = spam_data[spam_data["target"]==1]
    not_spam = spam_data[spam_data["target"]==0]
    
    return not_spam.apply(lambda x: len(x[0]),axis=1).mean(),spam.apply(lambda x: len(x[0]),axis=1).mean()


# In[ ]:

answer_six()


# <br>
# <br>
# The following function has been provided to help you combine new features into the training data:

# In[3]:

def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


# ### Question 7
# 
# Fit and transform the training data X_train using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5**.
# 
# Using this document-term matrix and an additional feature, **the length of document (number of characters)**, fit a Support Vector Classification model with regularization `C=10000`. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[11]:

from sklearn.svm import SVC

def answer_seven():

    vect = TfidfVectorizer(min_df=5).fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    
    doc_len = X_train.map(len)
    X_train_vectorized_new = add_feature(X_train_vectorized, doc_len)
    svc = SVC(C=10000).fit(X_train_vectorized_new,y_train)
    y_predict = svc.predict(add_feature(vect.transform(X_test),X_test.map(len)))
    auc = roc_auc_score(y_test,y_predict)
    
    return auc


# In[12]:

answer_seven()


# ### Question 8
# 
# What is the average number of digits per document for not spam and spam documents?
# 
# *This function should return a tuple (average # digits not spam, average # digits spam).*

# In[14]:

import re

def split_dig(x):
    length = 0
    for i in x:
        length = len(i) + length
    return length

def answer_eight():
    spam = spam_data[spam_data["target"]==1]
    not_spam = spam_data[spam_data["target"]==0]
    spam_dig = spam.apply(lambda x: re.findall("[0-9]+",x[0]),axis=1)
    not_spam_dig = not_spam.apply(lambda x: re.findall("[0-9]+",x[0]),axis=1)
    
    spam_dig = spam_dig.map(split_dig)
    not_spam_dig = not_spam_dig.map(split_dig)
    
    return sum(not_spam_dig)/len(not_spam_dig), sum(spam_dig)/len(spam_dig)


# In[15]:

answer_eight()


# ### Question 9
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **word n-grams from n=1 to n=3** (unigrams, bigrams, and trigrams).
# 
# Using this document-term matrix and the following additional features:
# * the length of document (number of characters)
# * **number of digits per document**
# 
# fit a Logistic Regression model with regularization `C=100`. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[26]:

from sklearn.linear_model import LogisticRegression

def answer_nine():
    
    #doc_len = X_train.apply(lambda x: re.findall("[a-zA-Z]+",x))
    doc_len = X_train.map(len)
    doc_dig = X_train.apply(lambda x: re.findall("[0-9]+",x))
    doc_dig = doc_dig.map(split_dig)
    doc_dig_X_test = X_test.apply(lambda x: re.findall("[0-9]+",x))
    doc_dig_X_test = doc_dig_X_test.map(split_dig)

    vect = TfidfVectorizer(min_df=5,ngram_range=(1,3)).fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    
    X_train_vectorized_new = add_feature(X_train_vectorized, [doc_len,doc_dig])
    svc = LogisticRegression(C=100).fit(X_train_vectorized_new,y_train)
    y_predict = svc.predict(add_feature(vect.transform(X_test),[X_test.map(len),doc_dig_X_test]))
    auc = roc_auc_score(y_test,y_predict)
    
    return auc


# In[81]:

answer_nine()


# ### Question 10
# 
# What is the average number of non-word characters (anything other than a letter, digit or underscore) per document for not spam and spam documents?
# 
# *Hint: Use `\w` and `\W` character classes*
# 
# *This function should return a tuple (average # non-word characters not spam, average # non-word characters spam).*

# In[33]:

def answer_ten():
    
    spam = spam_data[spam_data["target"]==1]
    not_spam = spam_data[spam_data["target"]==0]
    #spam = spam.apply(lambda x: re.findall("[^A-Za-z0-9_]",x[0]),axis=1)
    spam = spam.apply(lambda x: re.findall("\W",x[0]),axis=1)
    spam_len = spam.map(len)
    not_spam = not_spam.apply(lambda x: re.findall("\W",x[0]),axis=1)
    not_spam_len = not_spam.map(len)
    return sum(not_spam_len)/len(not_spam_len),sum(spam_len)/len(spam_len)


# In[34]:

answer_ten()


# ### Question 11
# 
# Fit and transform the training data X_train using a Count Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **character n-grams from n=2 to n=5.**
# 
# To tell Count Vectorizer to use character n-grams pass in `analyzer='char_wb'` which creates character n-grams only from text inside word boundaries. This should make the model more robust to spelling mistakes.
# 
# Using this document-term matrix and the following additional features:
# * the length of document (number of characters)
# * number of digits per document
# * **number of non-word characters (anything other than a letter, digit or underscore.)**
# 
# fit a Logistic Regression model with regularization C=100. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# Also **find the 10 smallest and 10 largest coefficients from the model** and return them along with the AUC score in a tuple.
# 
# The list of 10 smallest coefficients should be sorted smallest first, the list of 10 largest coefficients should be sorted largest first.
# 
# The three features that were added to the document term matrix should have the following names should they appear in the list of coefficients:
# ['length_of_doc', 'digit_count', 'non_word_char_count']
# 
# *This function should return a tuple `(AUC score as a float, smallest coefs list, largest coefs list)`.*

# In[29]:

def answer_eleven():
    
    vect = CountVectorizer(min_df=5,ngram_range=(2,5),analyzer='char_wb').fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    
    #Creating new features
    doc_len = X_train.map(len)
    doc_dig = X_train.apply(lambda x: re.findall("[0-9]+",x))
    doc_dig = doc_dig.map(split_dig)
    doc_dig_X_test = X_test.apply(lambda x: re.findall("[0-9]+",x))
    doc_dig_X_test = doc_dig_X_test.map(split_dig)
    doc_non_word = X_train.apply(lambda x: re.findall("[\W]",x))
    doc_non_word = doc_non_word.map(split_dig)
    doc_non_word_X_test = X_test.apply(lambda x: re.findall("[\W]",x))
    doc_non_word_X_test = doc_non_word_X_test.map(split_dig)
    
    #Adding new features to the document-term matrix
    X_train_vectorized_new = add_feature(X_train_vectorized, [doc_len,doc_dig,doc_non_word])
    
    #Fitting a Logistic Regression Model
    logreg = LogisticRegression(C=100).fit(X_train_vectorized_new,y_train)
    y_predict = logreg.predict(add_feature(vect.transform(X_test),[X_test.map(len),doc_dig_X_test,doc_non_word_X_test]))
    auc = roc_auc_score(y_test,y_predict)
    
    # get the feature names as numpy array and add the 3 new features
    feature_names = list(vect.get_feature_names())
    feature_names.extend(['length_of_doc', 'digit_count', 'non_word_char_count'])
    
    df = pd.DataFrame()
    df["coef"] = logreg.coef_[0]
    df["feature"] = feature_names 
    
    lowest_coef = df.sort_values(by="coef",ascending=True)
    largest_coef = df.sort_values(by="coef",ascending=False)
    
    return (auc,lowest_coef.iloc[:10,1],largest_coef.iloc[:10,1])


# In[30]:

answer_eleven()


# In[ ]:



