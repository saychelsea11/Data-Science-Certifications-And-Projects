
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 1
# 
# In this assignment, you'll be working with messy medical data and using regex to extract relevant infromation from the data. 
# 
# Each line of the `dates.txt` file corresponds to a medical note. Each note has a date that needs to be extracted, but each date is encoded in one of many formats.
# 
# The goal of this assignment is to correctly identify all of the different date variants encoded in this dataset and to properly normalize and sort the dates. 
# 
# Here is a list of some of the variants you might encounter in this dataset:
# * 04/20/2009; 04/20/09; 4/20/09; 4/3/09
# * Mar-20-2009; Mar 20, 2009; March 20, 2009;  Mar. 20, 2009; Mar 20 2009;
# * 20 Mar 2009; 20 March 2009; 20 Mar. 2009; 20 March, 2009
# * Mar 20th, 2009; Mar 21st, 2009; Mar 22nd, 2009
# * Feb 2009; Sep 2009; Oct 2010
# * 6/2008; 12/2009
# * 2009; 2010
# 
# Once you have extracted these date patterns from the text, the next step is to sort them in ascending chronological order accoring to the following rules:
# * Assume all dates in xx/xx/xx format are mm/dd/yy
# * Assume all dates where year is encoded in only two digits are years from the 1900's (e.g. 1/5/89 is January 5th, 1989)
# * If the day is missing (e.g. 9/2009), assume it is the first day of the month (e.g. September 1, 2009).
# * If the month is missing (e.g. 2010), assume it is the first of January of that year (e.g. January 1, 2010).
# * Watch out for potential typos as this is a raw, real-life derived dataset.
# 
# With these rules in mind, find the correct date in each note and return a pandas Series in chronological order of the original Series' indices.
# 
# For example if the original series was this:
# 
#     0    1999
#     1    2010
#     2    1978
#     3    2015
#     4    1985
# 
# Your function should return this:
# 
#     0    2
#     1    4
#     2    0
#     3    1
#     4    3
# 
# Your score will be calculated using [Kendall's tau](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient), a correlation measure for ordinal data.
# 
# *This function should return a Series of length 500 and dtype int.*

# In[1]:

import pandas as pd

doc = []
with open('dates.txt') as file:
    for line in file:
        doc.append(line)

df = pd.Series(doc)
df.head(10)
df.to_csv('dates_csv.csv')


# In[21]:

def bool_flip(x):
    if x==True:
        x=False
    else:
        x=True
    return x

def date_sorter():
    import re
    index = []
    temp = pd.DataFrame()
    temp["text"] = df
    
    #Extracting just the years in the format YYYY by first filtering out dates which contain a '\' before the year
    filt = temp.iloc[455:,0].str.extract(r'[^/](\d\d\d\d)').dropna().map(int)
    filt = filt[(filt<2025) & (filt>=1900)].map(str) #Filtering out dates before 1900 and after 2025
    #Removing entries that contain any months as alphabets
    filt_bool = filt.str.contains('(Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)')
    filt = filt[filt_bool.map(bool_flip)].map(pd.Timestamp)
    
    #Extracting dates with the formats mm/dd/yyyy, mm/yyyy and other variations
    filt3 = temp["text"].str.extract(r'(\s\d?\d/\d?\d?\d\d\s)').dropna().apply(lambda x: x.strip()).map(pd.Timestamp) #11/1990 format with space on both sides
    filt4 = temp["text"].str.extract(r'[(](\d?\d/\d?\d?\d\d)[)]').dropna().apply(lambda x: x.strip()).map(pd.Timestamp) #11/1990 format with paranthesis
    filt5 = temp["text"].str.extract(r'[(](\d?\d/\d?\d/\d?\d)[)]').dropna().map(pd.Timestamp) #11/11/90 format with paranthesis on both sides
    filt6 = temp["text"].str.extract(r'(\s\d?\d/\d?\d/\d?\d\s)').dropna().map(pd.Timestamp) #11/11/90 format with white spaces on both sides
    filt8 = temp["text"].str.extract(r'(\s\d?\d/\d?\d/\d\d\d\d\s)').dropna().map(pd.Timestamp) #11/11/1990 format with white spaces on both sides
    filt15 = temp["text"].str.extract(r'^(\d?\d/\d?\d/\d?\d?\d\d)').dropna().map(pd.Timestamp) #11/11/1990 format start
    filt20 = temp["text"].str.extract(r'^(\d?\d/\d\d\d\d)').dropna().map(pd.Timestamp) #11/1990 format start
    filt21 = temp["text"].str.extract(r'\s(\d?\d/\d\d\d\d)[.]').dropna().map(pd.Timestamp) #11/1990 format with space before and '.' after
    filt22 = temp["text"].str.extract(r'\s(\d?\d/\d\d\d\d)[)]').dropna().map(pd.Timestamp) #11/1990 format with space before and ')' after
    
    #Extracting dates with the month specified as alphabets with several variations
    filt9 = temp["text"].str.extract(r'(Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)(\W?\W?\W\d?\d\W?\W?\W\d?\d?\d\d)').dropna()
    filt9 = filt9.apply(lambda x: x[0]+x[1],axis=1).map(pd.Timestamp)
    filt10 = temp["text"].str.extract(r'(\W\d?\d\W?\W?\W)(Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)(\W?\W?\W\d?\d?\d\d)').dropna()
    filt10 = filt10.apply(lambda x: x[0]+x[1]+x[2],axis=1).map(pd.Timestamp)
    filt17 = temp["text"].str.extract(r'^(\d?\d\W?\W?\W)(Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)(\W?\W?\W\d?\d?\d\d)').dropna()
    filt17 = filt17.apply(lambda x: x[0]+x[1]+x[2],axis=1).map(pd.Timestamp)
    filt12 = temp["text"].str.extract(r'(\D\D)(Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)(\W?\W?\W?\d\d\d\d)').dropna()
    filt12 = filt12.apply(lambda x: x[1]+x[2],axis=1).map(pd.Timestamp)
    filt11 = temp["text"].str.extract(r'(\W?\d?\d-\d?\d-\d?\d?\d\d\W?)').dropna().map(pd.Timestamp)
    filt13 = temp["text"].str.extract(r'(\s\d?\d/\d?\d/\d?\d)[.]').dropna().map(pd.Timestamp)
    
    #Extracting only years at the start of the sentence followed by a whitespace
    filt1 = (temp["text"].str.extract(r'(^\d\d\d\d\W)')).dropna().map(int) 
    filt1 = filt1[filt1<2025].map(str).map(pd.Timestamp)
    index.extend(list(filt1.index))
    
    #Custom mining
    filt23 = temp["text"].str.extract(r'[0-9](\d\d/\d?\d/\d?\d?\d\d)').dropna().map(pd.Timestamp) #11/1990 format with an integer just before
    filt24 = temp["text"].str.extract(r'(5/04/74)').dropna().map(pd.Timestamp) #custom row
    
    #April 2009/09 format with a letter or number directly preceeding it
    filt25 = temp["text"].str.extract(r'[a-zA-z0-9](Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)(\W?\W?\W\d?\d?\d\d)').dropna()
    filt25 = filt25.apply(lambda x: x[0]+x[1],axis=1).map(pd.Timestamp)
    
    ##April 2009/09 format with a " directly preceeding it
    filt26 = temp["text"].str.extract(r'["](Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)(\W?\W?\W\d?\d?\d\d)').dropna()
    filt26 = filt26.apply(lambda x: x[0]+x[1],axis=1).map(pd.Timestamp)
    
    #April 2009 format at the start of the sentence
    filt27 = temp["text"].str.extract(r'^(Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)(\W?\W?\W\d\d\d\d)').dropna()
    filt27 = filt27.apply(lambda x: x[0]+x[1],axis=1).map(pd.Timestamp)
    
    #Custom formatting mispelled months
    filt28 = temp["text"].str.extract(r'(Janaury 1993)').dropna()
    filt28 = filt28.replace('Janaury 1993','1993').map(pd.Timestamp)
    filt29 = temp["text"].str.extract(r'(Decemeber 1978)').dropna()
    filt29 = filt29.replace('Decemeber 1978','Dec 1978').map(pd.Timestamp)
    
    #April 2009 format with a . at the beginning
    filt30 = temp["text"].str.extract(r'[.](Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)(\W?\W?\W?\d\d\d\d)').dropna()
    filt30 = filt30.apply(lambda x: x[0]+x[1],axis=1).map(pd.Timestamp)
    
    #09/2009 format with letter or '~' directly preceeding it
    filt31 = temp["text"].str.extract(r'[A-z~](\d?\d/\d\d\d\d)').dropna().map(pd.Timestamp)
    
    #09/2009 format with ',:' or alphabets or numbers following it
    filt32 = temp["text"].str.extract(r'(\d?\d/\d\d\d\d)[,:-^A-z^0-9]').dropna().map(pd.Timestamp)    
    
    #Combining all the filters together
    all_dates = pd.DataFrame()
    all_dates["dates"] = ((filt3).append(filt4).append(filt5).append(filt6).append(filt8).
                            append(filt15).append(filt20).append(filt21).append(filt22).
                            append(filt9).append(filt10).append(filt17).append(filt12).append(filt11).append(filt13).
                            append(filt1).append(filt).append(filt23).append(filt24).append(filt25).append(filt26).
                            append(filt27).append(filt28).append(filt29).append(filt30).append(filt31).append(filt32))
    
    #Sorting the dates
    all_dates["sorted"] = all_dates.index
    all_dates = all_dates.sort_values(by=["dates"],ascending=True)
    all_dates["id"] = range(len(all_dates))
    all_dates = all_dates.set_index("id")
    
    return all_dates["sorted"]

date_sorter()


# In[ ]:




# In[ ]:



