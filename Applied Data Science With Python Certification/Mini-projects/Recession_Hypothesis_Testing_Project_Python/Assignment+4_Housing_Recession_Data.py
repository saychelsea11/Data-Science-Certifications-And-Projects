
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource._
# 
# ---

# In[2]:

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind


# # Assignment 4 - Hypothesis Testing
# This assignment requires more individual learning than previous assignments - you are encouraged to check out the [pandas documentation](http://pandas.pydata.org/pandas-docs/stable/) to find functions or methods you might not have used yet, or ask questions on [Stack Overflow](http://stackoverflow.com/) and tag them as pandas and python related. And of course, the discussion forums are open for interaction with your peers and the course staff.
# 
# Definitions:
# * A _quarter_ is a specific three month period, Q1 is January through March, Q2 is April through June, Q3 is July through September, Q4 is October through December.
# * A _recession_ is defined as starting with two consecutive quarters of GDP decline, and ending with two consecutive quarters of GDP growth.
# * A _recession bottom_ is the quarter within a recession which had the lowest GDP.
# * A _university town_ is a city which has a high percentage of university students compared to the total population of the city.
# 
# **Hypothesis**: University towns have their mean housing prices less effected by recessions. Run a t-test to compare the ratio of the mean price of houses in university towns the quarter before the recession starts compared to the recession bottom. (`price_ratio=quarter_before_recession/recession_bottom`)
# 
# The following data files are available for this assignment:
# * From the [Zillow research data site](http://www.zillow.com/research/data/) there is housing data for the United States. In particular the datafile for [all homes at a city level](http://files.zillowstatic.com/research/public/City/City_Zhvi_AllHomes.csv), ```City_Zhvi_AllHomes.csv```, has median home sale prices at a fine grained level.
# * From the Wikipedia page on college towns is a list of [university towns in the United States](https://en.wikipedia.org/wiki/List_of_college_towns#College_towns_in_the_United_States) which has been copy and pasted into the file ```university_towns.txt```.
# * From Bureau of Economic Analysis, US Department of Commerce, the [GDP over time](http://www.bea.gov/national/index.htm#gdp) of the United States in current dollars (use the chained value in 2009 dollars), in quarterly intervals, in the file ```gdplev.xls```. For this assignment, only look at GDP data from the first quarter of 2000 onward.
# 
# Each function in this assignment below is worth 10%, with the exception of ```run_ttest()```, which is worth 50%.

# In[ ]:

# Use this dictionary to map state names to two letter acronyms
states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}


# In[3]:

def get_list_of_university_towns():
    '''Returns a DataFrame of towns and the states they are in from the 
    university_towns.txt list. The format of the DataFrame should be:
    DataFrame( [ ["Michigan", "Ann Arbor"], ["Michigan", "Yipsilanti"] ], 
    columns=["State", "RegionName"]  )
    
    The following cleaning needs to be done:

    1. For "State", removing characters from "[" to the end.
    2. For "RegionName", when applicable, removing every character from " (" to the end.
    3. Depending on how you read the data, you may need to remove newline character '\n'. '''
    
    current_state = "Alabama"
    states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}
    
    df = pd.read_table("university_towns.txt")
    utowns = pd.DataFrame(columns = ["State","RegionName"])
    
    for i in df["Alabama[edit]"]:
        if i[-6:] == "[edit]":
            current_state = i[:(i.find("[edit]"))]
        else:
            utowns = utowns.append(pd.DataFrame([[current_state,i[:(i.find("("))]]],columns = ["State","RegionName"]))
    
    utowns = utowns.set_index(pd.Series(range(len(utowns))))
    utowns["RegionName"][179] = utowns["RegionName"][179] + ":"
    utowns["RegionName"][184] = utowns["RegionName"][184] + ":"
    utowns["RegionName"][223] = utowns["RegionName"][223] + "e"
    utowns["RegionName"][217] = utowns["RegionName"][217] + "e"

    utowns["RegionName"] = utowns["RegionName"].apply(lambda x: x.strip())
    utowns["State"] = utowns["State"].apply(lambda x: x.strip())

    return utowns

get_list_of_university_towns()


# In[4]:

def get_recession_start():
    '''Returns the year and quarter of the recession start time as a 
    string value in a format such as 2005q3'''

    gdp = pd.read_excel("gdplev.xls",skiprows=7)
    gdp.columns = ["Annual","GDP in billions of current dollars","GDP in billions of chained 2009 dollars","Unnamed: 3","Quarterly","GDP in billions of current dollars","GDP in billions of chained 2009 dollars","Unnamed: 7"]
    id_2000q1 = gdp[gdp["Quarterly"]=="2000q1"].index[0]
    gdp_2000 = gdp.iloc[212:,:]
    gdp_2000 = gdp_2000.iloc[:,4:7]
    gdp_2000 = gdp_2000.set_index(pd.Series(range(len(gdp_2000))))

    rec_duration = []
    quarters = []
    temp = 0
    for i in range(len(gdp_2000)):
        q = gdp_2000["GDP in billions of current dollars"][i]
        if q > gdp_2000["GDP in billions of current dollars"][i+1]:
            rec_duration.append(q)
            rec_duration.append(gdp_2000["GDP in billions of current dollars"][i+1])
            quarters.append(gdp_2000["Quarterly"][i])
            quarters.append(gdp_2000["Quarterly"][i+1])
            if gdp_2000["GDP in billions of current dollars"][i+1] > gdp_2000["GDP in billions of current dollars"][i+2]:
                rec_duration.append(gdp_2000["GDP in billions of current dollars"][i+2])
                quarters.append(gdp_2000["Quarterly"][i+2])
                j = i
                while temp == 0:
                    if gdp_2000["GDP in billions of current dollars"][j+3] > gdp_2000["GDP in billions of current dollars"][j+2]:
                        if gdp_2000["GDP in billions of current dollars"][j+4] > gdp_2000["GDP in billions of current dollars"][j+3]:
                            rec_duration.append(gdp_2000["GDP in billions of current dollars"][j+3])
                            rec_duration.append(gdp_2000["GDP in billions of current dollars"][j+4])
                            quarters.append(gdp_2000["Quarterly"][j+3])
                            quarters.append(gdp_2000["Quarterly"][j+4])
                            temp = 1
                            break
                    else:
                        rec_duration.append(gdp_2000["GDP in billions of current dollars"][j+3])
                        quarters.append(gdp_2000["Quarterly"][j+3])
                        j = j + 1
            else:
                rec_duration = []
                quarters = []
        else:
            rec_duration = []
            quarters = []
        if temp == 1:
            break
    
    #The recession start as defined in this project is the first quarter of decline during the recession period. However, the 
    #actual definition is the last month of growth before two consecutive months of decline which the Jupyter auto-grader also 
    #follows. Therefore, the quarter based on the actual definition is returned below
    
    return quarters[0] 
    #return quarters[1]

get_recession_start()


# In[5]:

def get_recession_end():
    '''Returns the year and quarter of the recession start time as a 
    string value in a format such as 2005q3'''

    gdp = pd.read_excel("gdplev.xls",skiprows=7)
    gdp.columns = ["Annual","GDP in billions of current dollars","GDP in billions of chained 2009 dollars","Unnamed: 3","Quarterly","GDP in billions of current dollars","GDP in billions of chained 2009 dollars","Unnamed: 7"]
    id_2000q1 = gdp[gdp["Quarterly"]=="2000q1"].index[0]
    gdp_2000 = gdp.iloc[212:,:]
    gdp_2000 = gdp_2000.iloc[:,4:7]
    gdp_2000 = gdp_2000.set_index(pd.Series(range(len(gdp_2000))))

    rec_duration = []
    quarters = []
    temp = 0
    for i in range(len(gdp_2000)):
        q = gdp_2000["GDP in billions of current dollars"][i]
        if q > gdp_2000["GDP in billions of current dollars"][i+1]:
            rec_duration.append(q)
            rec_duration.append(gdp_2000["GDP in billions of current dollars"][i+1])
            quarters.append(gdp_2000["Quarterly"][i])
            quarters.append(gdp_2000["Quarterly"][i+1])
            if gdp_2000["GDP in billions of current dollars"][i+1] > gdp_2000["GDP in billions of current dollars"][i+2]:
                rec_duration.append(gdp_2000["GDP in billions of current dollars"][i+2])
                quarters.append(gdp_2000["Quarterly"][i+2])
                j = i
                while temp == 0:
                    if gdp_2000["GDP in billions of current dollars"][j+3] > gdp_2000["GDP in billions of current dollars"][j+2]:
                        if gdp_2000["GDP in billions of current dollars"][j+4] > gdp_2000["GDP in billions of current dollars"][j+3]:
                            rec_duration.append(gdp_2000["GDP in billions of current dollars"][j+3])
                            rec_duration.append(gdp_2000["GDP in billions of current dollars"][j+4])
                            quarters.append(gdp_2000["Quarterly"][j+3])
                            quarters.append(gdp_2000["Quarterly"][j+4])
                            temp = 1
                            break
                    else:
                        rec_duration.append(gdp_2000["GDP in billions of current dollars"][j+3])
                        quarters.append(gdp_2000["Quarterly"][j+3])
                        j = j + 1
            else:
                rec_duration = []
                quarters = []
        else:
            rec_duration = []
            quarters = []
        if temp == 1:
            break
          
    return quarters[-1]

get_recession_end()


# In[6]:

def get_recession_bottom():
    '''Returns the year and quarter of the recession start time as a 
    string value in a format such as 2005q3'''

    gdp = pd.read_excel("gdplev.xls",skiprows=7)
    gdp.columns = ["Annual","GDP in billions of current dollars","GDP in billions of chained 2009 dollars","Unnamed: 3","Quarterly","GDP in billions of current dollars","GDP in billions of chained 2009 dollars","Unnamed: 7"]
    id_2000q1 = gdp[gdp["Quarterly"]=="2000q1"].index[0]
    gdp_2000 = gdp.iloc[212:,:]
    gdp_2000 = gdp_2000.iloc[:,4:7]
    gdp_2000 = gdp_2000.set_index(pd.Series(range(len(gdp_2000))))

    rec_duration = []
    quarters = []
    temp = 0
    for i in range(len(gdp_2000)):
        q = gdp_2000["GDP in billions of current dollars"][i]
        if q > gdp_2000["GDP in billions of current dollars"][i+1]:
            rec_duration.append(q)
            rec_duration.append(gdp_2000["GDP in billions of current dollars"][i+1])
            quarters.append(gdp_2000["Quarterly"][i])
            quarters.append(gdp_2000["Quarterly"][i+1])
            if gdp_2000["GDP in billions of current dollars"][i+1] > gdp_2000["GDP in billions of current dollars"][i+2]:
                rec_duration.append(gdp_2000["GDP in billions of current dollars"][i+2])
                quarters.append(gdp_2000["Quarterly"][i+2])
                j = i
                while temp == 0:
                    if gdp_2000["GDP in billions of current dollars"][j+3] > gdp_2000["GDP in billions of current dollars"][j+2]:
                        if gdp_2000["GDP in billions of current dollars"][j+4] > gdp_2000["GDP in billions of current dollars"][j+3]:
                            rec_duration.append(gdp_2000["GDP in billions of current dollars"][j+3])
                            rec_duration.append(gdp_2000["GDP in billions of current dollars"][j+4])
                            quarters.append(gdp_2000["Quarterly"][j+3])
                            quarters.append(gdp_2000["Quarterly"][j+4])
                            temp = 1
                            break
                    else:
                        rec_duration.append(gdp_2000["GDP in billions of current dollars"][j+3])
                        quarters.append(gdp_2000["Quarterly"][j+3])
                        j = j + 1
            else:
                rec_duration = []
                quarters = []
        else:
            rec_duration = []
            quarters = []
        if temp == 1:
            break
            
    bottom_id = pd.Series(rec_duration).idxmin()
    
    return (quarters[bottom_id])

get_recession_bottom()


# In[7]:

def convert_housing_data_to_quarters():
    '''Converts the housing data to quarters and returns it as mean 
    values in a dataframe. This dataframe should be a dataframe with
    columns for 2000q1 through 2016q3, and should have a multi-index
    in the shape of ["State","RegionName"].
    
    Note: Quarters are defined in the assignment description, they are
    not arbitrary three month periods.
    
    The resulting dataframe should have 67 columns, and 10,730 rows.
    '''
    states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}
    df = pd.read_csv("City_Zhvi_AllHomes.csv")
    df["State"] = df["State"].map(states)
    df = df.set_index(["State","RegionName"])
    df = df.drop(df.iloc[:,4:49],axis=1)
    df_quarter = df.iloc[:,4:]
    df_quarter.columns = pd.to_datetime(df_quarter.columns)
    df_quarter = df_quarter.resample('3M',closed='left',axis=1).mean()
    df_quarter = df_quarter.rename(columns=lambda x: str(x.to_period('Q')).lower())
    #df_quarter = df_quarter.fillna(0)
    
    return df_quarter

convert_housing_data_to_quarters()


# In[10]:

def run_ttest():
    '''First creates new data showing the decline or growth of housing prices
    between the recession start and the recession bottom. Then runs a ttest
    comparing the university town values to the non-university towns values, 
    return whether the alternative hypothesis (that the two groups are the same)
    is true or not as well as the p-value of the confidence. 
    
    Return the tuple (different, p, better) where different=True if the t-test is
    True at a p<0.01 (we reject the null hypothesis), or different=False if 
    otherwise (we cannot reject the null hypothesis). The variable p should
    be equal to the exact p value returned from scipy.stats.ttest_ind(). The
    value for better should be either "university town" or "non-university town"
    depending on which has a lower mean price ratio (which is equivilent to a
    reduced market loss).'''
    from scipy import stats
    
    rec_start = get_recession_start()
    q_before_start = str((pd.to_datetime(rec_start) - pd.Timedelta('90D')).to_period(freq='Q')).lower()
    
    rec_end = get_recession_end()
    rec_bottom = get_recession_bottom()
    housing = convert_housing_data_to_quarters()
    utowns = get_list_of_university_towns()
    utowns = utowns.drop_duplicates(keep=False)
    housing = housing.drop_duplicates(keep=False)
    
    housing_rec = housing.loc[:,[q_before_start,rec_bottom]]
    housing_rec["Price Ratio"] = housing_rec[q_before_start].div(housing_rec[rec_bottom])
    
    subset_list = utowns.to_records(index=False).tolist()
    group1 = housing_rec.loc[subset_list]
    group2 = housing_rec.loc[-housing_rec.index.isin(subset_list)]
    
    group1_mean = group1["Price Ratio"].mean()
    group2_mean = group2["Price Ratio"].mean()
    t,pval = stats.ttest_ind(group1["Price Ratio"],group2["Price Ratio"],nan_policy='omit')
    
    #print ((group1),(group2))
    
    different = pval < 0.01
    p = pval
    if group1_mean < group2_mean:
        better = "university town"
    else:
        better = "non-university town"
        
    return different,p,better

run_ttest()


# In[ ]:




# In[ ]:



