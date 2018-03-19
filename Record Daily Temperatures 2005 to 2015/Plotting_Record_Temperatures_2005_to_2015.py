
# coding: utf-8

# # Assignment 2
# 
# Before working on this assignment please read these instructions fully. In the submission area, you will notice that you can click the link to **Preview the Grading** for each step of the assignment. This is the criteria that will be used for peer grading. Please familiarize yourself with the criteria before beginning the assignment.
# 
# An NOAA dataset has been stored in the file `data/C2A2_data/BinnedCsvs_d400/3c2bda238ac9a3a7135f3585148e72a741314401221c42a4fe887c92.csv`. The data for this assignment comes from a subset of The National Centers for Environmental Information (NCEI) [Daily Global Historical Climatology Network](https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/readme.txt) (GHCN-Daily). The GHCN-Daily is comprised of daily climate records from thousands of land surface stations across the globe.
# 
# Each row in the assignment datafile corresponds to a single observation.
# 
# The following variables are provided to you:
# 
# * **id** : station identification code
# * **date** : date in YYYY-MM-DD format (e.g. 2012-01-24 = January 24, 2012)
# * **element** : indicator of element type
#     * TMAX : Maximum temperature (tenths of degrees C)
#     * TMIN : Minimum temperature (tenths of degrees C)
# * **value** : data value for element (tenths of degrees C)
# 
# For this assignment, you must:
# 
# 1. Read the documentation and familiarize yourself with the dataset, then write some python code which returns a line graph of the record high and record low temperatures by day of the year over the period 2005-2014. The area between the record high and record low temperatures for each day should be shaded.
# 2. Overlay a scatter of the 2015 data for any points (highs and lows) for which the ten year record (2005-2014) record high or record low was broken in 2015.
# 3. Watch out for leap days (i.e. February 29th), it is reasonable to remove these points from the dataset for the purpose of this visualization.
# 4. Make the visual nice! Leverage principles from the first module in this course when developing your solution. Consider issues such as legends, labels, and chart junk.
# 
# The data you have been given is near **Greensboro, North Carolina, United States**, and the stations the data comes from are shown on the map below.

# In[1]:

import matplotlib.pyplot as plt
import mplleaflet
import pandas as pd

def leaflet_plot_stations(binsize, hashid):

    df = pd.read_csv('data/C2A2_data/BinSize_d{}.csv'.format(binsize))

    station_locations_by_hash = df[df['hash'] == hashid]

    lons = station_locations_by_hash['LONGITUDE'].tolist()
    lats = station_locations_by_hash['LATITUDE'].tolist()

    plt.figure(figsize=(8,8))

    plt.scatter(lons, lats, c='r', alpha=0.7, s=200)

    return mplleaflet.display()

leaflet_plot_stations(400,'3c2bda238ac9a3a7135f3585148e72a741314401221c42a4fe887c92')


# In[167]:

file ='test.csv'
get_ipython().system('cp "$file" .')
from IPython.display import HTML
link = '<a href="{0}" download>Click here to download {0}</a>'
HTML(link.format(file.split('/')[-1]))


# In[11]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Reading the dataset and sorting the dataframe by date
df = pd.read_csv("3c2bda238ac9a3a7135f3585148e72a741314401221c42a4fe887c92.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date"])

#Creating new columns for year,month, day of the month and day of the year from the "Date" variable
df["Year"] = list(map(lambda x: x.year,df["Date"]))
df["Month"] = list(map(lambda x: x.month,df["Date"]))
df["Month Day"] = list(map(lambda x: x.day,df["Date"]))
df["Day"] = list(map(lambda x: int((x - pd.Timestamp(str(x.year)) + pd.Timedelta('1D')).days),df["Date"]))

#Removing any instances of Feb 29 - 2008 and 2012 leap years
df = df[~((df["Month"] == 2) & (df["Month Day"]==29))]
df.iloc[21807:27953,7] = list(df["Day"].iloc[21807:27953] - 1)
df.iloc[54963:62017,7] = list(df["Day"].iloc[54963:62017] - 1)

#Splitting the original dataset into 2015 records and 2005-2014 records
df_2015 = df[df["Year"] == 2015]
df = df[df["Date"] < '01/01/2015']

#Finding the max and min records from both datasets and determining points 
#where the 2015 values broke the 2005-2014 records
day_max = df.groupby("Day")["Data_Value"].max().div(10)
day_min = df.groupby("Day")["Data_Value"].min().div(10)
day_max_2015 = df_2015.groupby("Day")["Data_Value"].max().div(10)
day_min_2015 = df_2015.groupby("Day")["Data_Value"].min().div(10)
day_max_diff = day_max_2015[day_max_2015 > day_max]
day_min_diff = day_min_2015[day_min_2015 < day_min]

#Plotting the 365-days max and min temperatures across 2005-2014
plt.plot(day_min,label="2005-2014 Min",alpha=0.5,color="blue")
plt.plot(day_max,label="2005-2014 Max",alpha=0.5,color="red")

#Overlaying a scatter plot of any times the max and min from 2015 breaks the 2005-2014 records
plt.scatter(day_max_diff.index,day_max_diff,color="green",label="2015 Record Max")
plt.scatter(day_min_diff.index,day_min_diff,color="orange",label="2015 Record Min")

#Setting the other plotting attributes
plt.gca().fill_between(range(1,len(day_min)+1),day_min,day_max,facecolor="green",alpha=0.2)
plt.legend(loc=8)
plt.title("Record Day Temperatures from 2005-2015")
plt.xlabel("Day of the year")
plt.ylabel("Temperature (Celsius)")
plt.show()


# In[ ]:




# In[ ]:




# In[ ]:



