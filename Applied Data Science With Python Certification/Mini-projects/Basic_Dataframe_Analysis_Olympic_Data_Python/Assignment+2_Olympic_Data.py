
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.2** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource._
# 
# ---

# # Assignment 2 - Pandas Introduction
# All questions are weighted the same in this assignment.
# ## Part 1
# The following code loads the olympics dataset (olympics.csv), which was derrived from the Wikipedia entry on [All Time Olympic Games Medals](https://en.wikipedia.org/wiki/All-time_Olympic_Games_medal_table), and does some basic data cleaning. 
# 
# The columns are organized as # of Summer games, Summer medals, # of Winter games, Winter medals, total # number of games, total # of medals. Use this dataset to answer the questions below.

# In[3]:

import pandas as pd

df = pd.read_csv('olympics.csv', index_col=0, skiprows=1)

for col in df.columns:
    if col[:2]=='01':
        df.rename(columns={col:'Gold'+col[4:]}, inplace=True)
    if col[:2]=='02':
        df.rename(columns={col:'Silver'+col[4:]}, inplace=True)
    if col[:2]=='03':
        df.rename(columns={col:'Bronze'+col[4:]}, inplace=True)
    if col[:1]=='№':
        df.rename(columns={col:'#'+col[1:]}, inplace=True)

names_ids = df.index.str.split('\s\(') # split the index by '('

df.index = names_ids.str[0] # the [0] element is the country name (new index) 
df['ID'] = names_ids.str[1].str[:3] # the [1] element is the abbreviation or ID (take first 3 characters from that)

df = df.drop('Totals')
df.head()


# ### Question 0 (Example)
# 
# What is the first country in df?
# 
# *This function should return a Series.*

# In[2]:

# You should write your whole answer within the function provided. The autograder will call
# this function and compare the return value against the correct solution value
def answer_zero():
    # This function returns the row for Afghanistan, which is a Series object. The assignment
    # question description will tell you the general format the autograder is expecting
    return df.iloc[0]

# You can examine what your function returns by calling it in the cell. If you have questions
# about the assignment formats, check out the discussion forums for any FAQs
answer_zero() 


# ### Question 1
# Which country has won the most gold medals in summer games?
# 
# *This function should return a single string value.*

# In[27]:

def answer_one():
    temp = df.sort_values(by = "Gold")
    return (temp.index[-1])

answer_one()


# ### Question 2
# Which country had the biggest difference between their summer and winter gold medal counts?
# 
# *This function should return a single string value.*

# In[26]:

def answer_two():
    temp = abs(df["Gold"] - df["Gold.1"])
    df["Difference"] = temp
    return df.sort_values(by = "Difference").index[-1]

answer_two()


# ### Question 3
# Which country has the biggest difference between their summer gold medal counts and winter gold medal counts relative to their total gold medal count? 
# 
# $$\frac{Summer~Gold - Winter~Gold}{Total~Gold}$$
# 
# Only include countries that have won at least 1 gold in both summer and winter.
# 
# *This function should return a single string value.*

# In[25]:

def answer_three():
    temp = df[df["Gold"] & df["Gold.1"] != 0]
    relative = (temp["Gold"] - temp["Gold.1"])/temp["Gold.2"]
    temp["difference"] = relative  
    return temp.sort_values(by="difference").index[-1]
 
answer_three()


# ### Question 4
# Write a function that creates a Series called "Points" which is a weighted value where each gold medal (`Gold.2`) counts for 3 points, silver medals (`Silver.2`) for 2 points, and bronze medals (`Bronze.2`) for 1 point. The function should return only the column (a Series object) which you created, with the country names as indices.
# 
# *This function should return a Series named `Points` of length 146*

# In[24]:

def answer_four():
    Points = df["Gold.2"]*3 + df["Silver.2"]*2 + df["Bronze.2"]
    df["Points"] = Points
    return df["Points"]

answer_four()


# ## Part 2
# For the next set of questions, we will be using census data from the [United States Census Bureau](http://www.census.gov). Counties are political and geographic subdivisions of states in the United States. This dataset contains population data for counties and states in the US from 2010 to 2015. [See this document](https://www2.census.gov/programs-surveys/popest/technical-documentation/file-layouts/2010-2015/co-est2015-alldata.pdf) for a description of the variable names.
# 
# The census dataset (census.csv) should be loaded as census_df. Answer questions using this as appropriate.
# 
# ### Question 5
# Which state has the most counties in it? (hint: consider the sumlevel key carefully! You'll need this for future questions too...)
# 
# *This function should return a single string value.*

# In[4]:

census_df = pd.read_csv('census.csv')
census_df.head()


# In[41]:

def answer_five():
    max_count = 0
    current = 0
    state = ""
    temp = census_df["STNAME"].unique()
    for i in temp:
        current = census_df[census_df["STNAME"]==i]["STNAME"].count()
        if current > max_count:
            max_count = current
            state = i
    return state

answer_five()


# ### Question 6
# **Only looking at the three most populous counties for each state**, what are the three most populous states (in order of highest population to lowest population)? Use `CENSUS2010POP`.
# 
# *This function should return a list of string values.*

# In[40]:

def answer_six():
    current = 0
    highest = 0
    temp = pd.DataFrame(columns=["STATE","COUNTY","POP"])
    temp2 = pd.DataFrame(columns=["STATE","POP"])
    census_df_drop = census_df[census_df["SUMLEV"] == 50]
    census_df_drop = census_df_drop[census_df_drop["STNAME"] != "District of Columbia"]
    states = census_df_drop["STNAME"].unique()
    #print (type(census_df_drop["STNAME"]))
    
    for i in states: 
        m = max(census_df_drop[census_df_drop["STNAME"] == i]["CENSUS2010POP"])
        ID = census_df_drop[census_df_drop["STNAME"] == i]["CENSUS2010POP"].idxmax()
        cty = census_df_drop[census_df_drop.index == ID]["CTYNAME"]
        #print (m,cty.iloc[0],i)
        
        max_2 = census_df_drop[census_df_drop["CTYNAME"] != cty.iloc[0]]
        #print (max_2)
        #print (max_2[max_2["STNAME"]==i],max_2[max_2["STNAME"] == i]["CENSUS2010POP"])
        m2 = max(max_2[max_2["STNAME"] == i]["CENSUS2010POP"])
        #print (m2)
        ID_2 = max_2[max_2["STNAME"] == i]["CENSUS2010POP"].idxmax()
        cty2 = max_2[max_2.index == ID_2]["CTYNAME"]
        
        max_3 = max_2[max_2["CTYNAME"] != cty2.iloc[0]]
        #max_4 = max_3[max_3["CTYNAME" != cty2.iloc[0]]]
        m3 = max(max_3[max_3["STNAME"] == i]["CENSUS2010POP"])
        ID_3 = max_3[max_3["STNAME"] == i]["CENSUS2010POP"].idxmax()
        cty3 = max_3[max_3.index == ID_3]["CTYNAME"]
        
        temp = temp.append(pd.DataFrame([[i,cty.iloc[0],m]],columns=["STATE","COUNTY","POP"])) 
        temp = temp.append(pd.DataFrame([[i,cty2.iloc[0],m2]],columns=["STATE","COUNTY","POP"])) 
        temp = temp.append(pd.DataFrame([[i,cty3.iloc[0],m3]],columns=["STATE","COUNTY","POP"])) 
    #print (temp["STATE"][-1])
    
    for i in states:
        current = sum(temp[temp["STATE"]==i]["POP"])
        temp2 = temp2.append(pd.DataFrame([[i,current]],columns=["STATE","POP"]))
    
    temp2 = temp2.set_index("STATE")
    
    counties = []
    counties.extend([temp2.sort_values(by="POP").index[-1],temp2.sort_values(by="POP").index[-2],temp2.sort_values(by="POP").index[-3]])
    
    return counties 

answer_six()


# ### Question 7
# Which county has had the largest absolute change in population within the period 2010-2015? (Hint: population values are stored in columns POPESTIMATE2010 through POPESTIMATE2015, you need to consider all six columns.)
# 
# e.g. If County Population in the 5 year period is 100, 120, 80, 105, 100, 130, then its largest change in the period would be |130-80| = 50.
# 
# *This function should return a single string value.*

# In[41]:

def answer_seven():
    temp = census_df[census_df["SUMLEV"] == 50]
    temp = temp.iloc[:,9:15]
    temp["CTYNAME"] = census_df["CTYNAME"]
    min_vals = []
    max_vals = []
    diff = []
    
    #print (temp)
    for i in range(len(temp["POPESTIMATE2010"])):
        max_vals.append(max(temp.iloc[i,0:6]))
        min_vals.append(min(temp.iloc[i,0:6]))
        diff.append(abs(max_vals[i] - min_vals[i]))

    temp["Change"] = diff
    temp = temp.set_index("CTYNAME")
    temp = temp.sort_values(by="Change")
    return temp.index[-1] 

answer_seven()


# ### Question 8
# In this datafile, the United States is broken up into four regions using the "REGION" column. 
# 
# Create a query that finds the counties that belong to regions 1 or 2, whose name starts with 'Washington', and whose POPESTIMATE2015 was greater than their POPESTIMATE 2014.
# 
# *This function should return a 5x2 DataFrame with the columns = ['STNAME', 'CTYNAME'] and the same index ID as the census_df (sorted ascending by index).*

# In[18]:

def answer_eight():
    temp = census_df[census_df["SUMLEV"] == 50]
    temp = temp[((temp["REGION"] == 1) | (temp["REGION"] == 2)) & (temp["POPESTIMATE2015"] > temp["POPESTIMATE2014"]) & temp["CTYNAME"].str.contains('Washington')].loc[:,["STNAME","CTYNAME"]] #First condition
    return temp

answer_eight()


# In[ ]:



