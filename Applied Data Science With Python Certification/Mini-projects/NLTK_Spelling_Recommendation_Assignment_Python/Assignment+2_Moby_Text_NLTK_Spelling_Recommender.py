
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 2 - Introduction to NLTK
# 
# In part 1 of this assignment you will use nltk to explore the Herman Melville novel Moby Dick. Then in part 2 you will create a spelling recommender function that uses nltk to find words similar to the misspelling. 

# ## Part 1 - Analyzing Moby Dick

# In[9]:

import nltk
import pandas as pd
import numpy as np

# If you would like to work with the raw text you can use 'moby_raw'
with open('moby.txt', 'r') as f:
    moby_raw = f.read()
    
# If you would like to work with the novel in nltk.Text format you can use 'text1'
moby_tokens = nltk.word_tokenize(moby_raw)
text1 = nltk.Text(moby_tokens)


# ### Example 1
# 
# How many tokens (words and punctuation symbols) are in text1?
# 
# *This function should return an integer.*

# In[ ]:

def example_one():
    
    return len(nltk.word_tokenize(moby_raw)) # or alternatively len(text1)

example_one()


# ### Example 2
# 
# How many unique tokens (unique words and punctuation) does text1 have?
# 
# *This function should return an integer.*

# In[ ]:

def example_two():
    
    return len(set(nltk.word_tokenize(moby_raw))) # or alternatively len(set(text1))

example_two()


# ### Example 3
# 
# After lemmatizing the verbs, how many unique tokens does text1 have?
# 
# *This function should return an integer.*

# In[ ]:

from nltk.stem import WordNetLemmatizer

def example_three():

    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w,'v') for w in text1]

    return len(set(lemmatized))

example_three()


# ### Question 1
# 
# What is the lexical diversity of the given text input? (i.e. ratio of unique tokens to the total number of tokens)
# 
# *This function should return a float.*

# In[ ]:

def answer_one():
    
    return len(set(moby_tokens))/len(moby_tokens)

answer_one()


# ### Question 2
# 
# What percentage of tokens is 'whale'or 'Whale'?
# 
# *This function should return a float.*

# In[12]:

def answer_two():
    
    whale = [x for x in moby_tokens if (x=='whale') or (x=='Whale')]
    return (len(whale)/len(moby_tokens))*100

answer_two()


# ### Question 3
# 
# What are the 20 most frequently occurring (unique) tokens in the text? What is their frequency?
# 
# *This function should return a list of 20 tuples where each tuple is of the form `(token, frequency)`. The list should be sorted in descending order of frequency.*

# In[13]:

def answer_three():
    
    token_freq = nltk.FreqDist(moby_tokens)
    token_df = pd.DataFrame()
    token_df["key"] = token_freq.keys()
    token_df["value"] = token_freq.values()
    token_df = token_df.sort_values(by="value",ascending=False)
    token_df = token_df.iloc[:20,:]
    
    return list(zip(token_df["key"],token_df["value"]))

answer_three()


# ### Question 4
# 
# What tokens have a length of greater than 5 and frequency of more than 150?
# 
# *This function should return a sorted list of the tokens that match the above constraints. To sort your list, use `sorted()`*

# In[ ]:

def answer_four():
    
    token_freq = nltk.FreqDist(moby_tokens)
    token_len =  [len(x) for x in moby_tokens]
    token_df = pd.DataFrame()
    token_df["key"] = token_freq.keys()
    token_df["value"] = token_freq.values()
    token_df["len"] = token_df["key"].map(len)
    token_df = token_df[(token_df["value"] > 150) & (token_df["len"]>5)]
    return sorted(token_df["key"])

answer_four()


# ### Question 5
# 
# Find the longest word in text1 and that word's length.
# 
# *This function should return a tuple `(longest_word, length)`.*

# In[ ]:

def answer_five():
    
    df = pd.DataFrame()
    df["word"] = moby_tokens
    df["length"] = df["word"].map(len)
    df = df.sort_values(by="length",ascending=False)
    
    return (df.iloc[0,0],df.iloc[0,1])

answer_five()


# ### Question 6
# 
# What unique words have a frequency of more than 2000? What is their frequency?
# 
# "Hint:  you may want to use `isalpha()` to check if the token is a word and not punctuation."
# 
# *This function should return a list of tuples of the form `(frequency, word)` sorted in descending order of frequency.*

# In[8]:

def answer_six():
    
    token_freq = nltk.FreqDist(moby_tokens)
    token_df = pd.DataFrame()
    token_df["key"] = token_freq.keys()
    token_df["value"] = token_freq.values()
    token_df = token_df[token_df["value"]>2000]
    token_df = token_df[token_df.apply(lambda x: x[0].isalpha(),axis=1)]
    token_df = token_df.sort_values(by="value",ascending=False)
    return list(zip(token_df["value"],token_df["key"]))

answer_six()


# ### Question 7
# 
# What is the average number of tokens per sentence?
# 
# *This function should return a float.*

# In[ ]:

def answer_seven():
    
    sentences = nltk.sent_tokenize(moby_raw)
    tokens_per_sentence = [len(nltk.word_tokenize(x)) for x in sentences]
    return np.mean(tokens_per_sentence)

answer_seven()


# ### Question 8
# 
# What are the 5 most frequent parts of speech in this text? What is their frequency?
# 
# *This function should return a list of tuples of the form `(part_of_speech, frequency)` sorted in descending order of frequency.*

# In[ ]:

def answer_eight():
    
    pos = nltk.pos_tag(moby_tokens)
    df = pd.DataFrame()
    df["token"] = list(zip(*pos))[0]
    df["pos"] = list(zip(*pos))[1]
    
    df_freq = pd.DataFrame()
    df_freq["pos"] = nltk.FreqDist(df["pos"]).keys()
    df_freq["freq"] = nltk.FreqDist(df["pos"]).values()
    df_freq = df_freq.sort_values(by="freq",ascending=False)
    return list(zip(df_freq.iloc[:5,0],df_freq.iloc[:5,1]))

answer_eight()


# ## Part 2 - Spelling Recommender
# 
# For this part of the assignment you will create three different spelling recommenders, that each take a list of misspelled words and recommends a correctly spelled word for every word in the list.
# 
# For every misspelled word, the recommender should find find the word in `correct_spellings` that has the shortest distance*, and starts with the same letter as the misspelled word, and return that word as a recommendation.
# 
# *Each of the three different recommenders will use a different distance measure (outlined below).
# 
# Each of the recommenders should provide recommendations for the three default words provided: `['cormulent', 'incendenece', 'validrate']`.

# In[10]:

from nltk.corpus import words

correct_spellings = words.words()


# ### Question 9
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the trigrams of the two words.**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[6]:

def answer_nine(entries=['cormulent', 'incendenece', 'validrate']):
    
    rec1 = [nltk.distance.jaccard_distance(set(nltk.ngrams(entries[0],n=3)),set(nltk.ngrams(x,n=3))) for x in correct_spellings]
    rec2 = [nltk.distance.jaccard_distance(set(nltk.ngrams(entries[1],n=3)),set(nltk.ngrams(x,n=3))) for x in correct_spellings]
    rec3 = [nltk.distance.jaccard_distance(set(nltk.ngrams(entries[2],n=3)),set(nltk.ngrams(x,n=3))) for x in correct_spellings]
    df = pd.DataFrame()
    df["rec1"] = rec1
    df["rec2"] = rec2
    df["rec3"] = rec3
    df["correct"] = correct_spellings
    rec3 = df.sort_values(by="rec3",ascending=True)["correct"].iloc[0]
    rec2 = df.sort_values(by="rec2",ascending=True)
    rec2 = rec2[rec2.apply(lambda x: x[3].startswith('i'),axis=1)]["correct"].iloc[0]
    rec1 = df.sort_values(by="rec1",ascending=True)
    rec1 = rec1[rec1.apply(lambda x: x[3].startswith('c'),axis=1)]["correct"].iloc[0]
    
    return [rec1,rec2,rec3]
    
answer_nine()


# ### Question 10
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the 4-grams of the two words.**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[13]:

def answer_ten(entries=['cormulent', 'incendenece', 'validrate']):
    
    rec1 = [nltk.distance.jaccard_distance(set(nltk.ngrams(entries[0],n=4)),set(nltk.ngrams(x,n=4))) for x in correct_spellings]
    rec2 = [nltk.distance.jaccard_distance(set(nltk.ngrams(entries[1],n=4)),set(nltk.ngrams(x,n=4))) for x in correct_spellings]
    rec3 = [nltk.distance.jaccard_distance(set(nltk.ngrams(entries[2],n=4)),set(nltk.ngrams(x,n=4))) for x in correct_spellings]
    df = pd.DataFrame()
    df["rec1"] = rec1
    df["rec2"] = rec2
    df["rec3"] = rec3
    df["correct"] = correct_spellings
    rec3 = df.sort_values(by="rec3",ascending=True)
    rec3 = rec3[rec3.apply(lambda x: x[3].startswith('v'),axis=1)]["correct"].iloc[0]
    rec2 = df.sort_values(by="rec2",ascending=True)
    rec2 = rec2[rec2.apply(lambda x: x[3].startswith('i'),axis=1)]["correct"].iloc[0]
    rec1 = df.sort_values(by="rec1",ascending=True)
    rec1 = rec1[rec1.apply(lambda x: x[3].startswith('c'),axis=1)]["correct"].iloc[0]
    
    return [rec1,rec2,rec3]
    
answer_ten()


# ### Question 11
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Edit distance on the two words with transpositions.](https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance)**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[11]:

def answer_eleven(entries=['cormulent', 'incendenece', 'validrate']):
    
    rec1 = [nltk.distance.edit_distance(entries[0],x) for x in correct_spellings]
    rec2 = [nltk.distance.edit_distance(entries[1],x) for x in correct_spellings]
    rec3 = [nltk.distance.edit_distance(entries[2],x) for x in correct_spellings]
    df = pd.DataFrame()
    df["rec1"] = rec1
    df["rec2"] = rec2
    df["rec3"] = rec3
    df["correct"] = correct_spellings
    rec3 = df.sort_values(by="rec3",ascending=True)["correct"].iloc[0]
    rec2 = df.sort_values(by="rec2",ascending=True)
    rec2 = rec2[rec2.apply(lambda x: x[3].startswith('i'),axis=1)]["correct"].iloc[0]
    rec1 = df.sort_values(by="rec1",ascending=True)
    rec1 = rec1[rec1.apply(lambda x: x[3].startswith('c'),axis=1)]["correct"].iloc[0]
    
    return [rec1,rec2,rec3]
    
answer_eleven()


# In[ ]:




# In[ ]:



