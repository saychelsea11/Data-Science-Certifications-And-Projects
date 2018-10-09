
# coding: utf-8

# # Assignment 4
# 
# Before working on this assignment please read these instructions fully. In the submission area, you will notice that you can click the link to **Preview the Grading** for each step of the assignment. This is the criteria that will be used for peer grading. Please familiarize yourself with the criteria before beginning the assignment.
# 
# This assignment requires that you to find **at least** two datasets on the web which are related, and that you visualize these datasets to answer a question with the broad topic of **religious events or traditions** (see below) for the region of **Greensboro, North Carolina, United States**, or **United States** more broadly.
# 
# You can merge these datasets with data from different regions if you like! For instance, you might want to compare **Greensboro, North Carolina, United States** to Ann Arbor, USA. In that case at least one source file must be about **Greensboro, North Carolina, United States**.
# 
# You are welcome to choose datasets at your discretion, but keep in mind **they will be shared with your peers**, so choose appropriate datasets. Sensitive, confidential, illicit, and proprietary materials are not good choices for datasets for this assignment. You are welcome to upload datasets of your own as well, and link to them using a third party repository such as github, bitbucket, pastebin, etc. Please be aware of the Coursera terms of service with respect to intellectual property.
# 
# Also, you are welcome to preserve data in its original language, but for the purposes of grading you should provide english translations. You are welcome to provide multiple visuals in different languages if you would like!
# 
# As this assignment is for the whole course, you must incorporate principles discussed in the first week, such as having as high data-ink ratio (Tufte) and aligning with Cairo’s principles of truth, beauty, function, and insight.
# 
# Here are the assignment instructions:
# 
#  * State the region and the domain category that your data sets are about (e.g., **Greensboro, North Carolina, United States** and **religious events or traditions**).
#  * You must state a question about the domain category and region that you identified as being interesting.
#  * You must provide at least two links to available datasets. These could be links to files such as CSV or Excel files, or links to websites which might have data in tabular form, such as Wikipedia pages.
#  * You must upload an image which addresses the research question you stated. In addition to addressing the question, this visual should follow Cairo's principles of truthfulness, functionality, beauty, and insightfulness.
#  * You must contribute a short (1-2 paragraph) written justification of how your visualization addresses your stated research question.
# 
# What do we mean by **religious events or traditions**?  For this category you might consider calendar events, demographic data about religion in the region and neighboring regions, participation in religious events, or how religious events relate to political events, social movements, or historical events.
# 
# ## Tips
# * Wikipedia is an excellent source of data, and I strongly encourage you to explore it for new data sources.
# * Many governments run open data initiatives at the city, region, and country levels, and these are wonderful resources for localized data sources.
# * Several international agencies, such as the [United Nations](http://data.un.org/), the [World Bank](http://data.worldbank.org/), the [Global Open Data Index](http://index.okfn.org/place/) are other great places to look for data.
# * This assignment requires you to convert and clean datafiles. Check out the discussion forums for tips on how to do this from various sources, and share your successes with your fellow students!
# 
# ## Example
# Looking for an example? Here's what our course assistant put together for the **Ann Arbor, MI, USA** area using **sports and athletics** as the topic. [Example Solution File](./readonly/Assignment4_example.pdf)

# In[4]:

import pandas as pd
import matplotlib.pyplot as plt

religiosity = []

#Reading in the two datasets
df_religion = pd.read_excel("State_Religion_Dataset_2010.xlsx")
df_rank = pd.read_excel("State_Religion_Rankings.xlsx",skiprows=1)

#Setting the column names for the 2nd dataset
df_rank.columns = ["STNAME","Rank Pop","Percent Stating Religion is Important","Religiosity Rank","Percent Religious 2014"]

#Removing any spaces from the state names in the 2nd dataset
df_rank["STNAME"] = df_rank.apply(lambda x: x[0].strip(),axis=1)

#Removing American Samoa, Northern Mariana Islands, US Virgin Islands, Guam and Puerto Rico from the df_rank dataset
#for simplifying the dataset to just the 50 states and District of Columbia
df_rank = df_rank[(df_rank["STNAME"]!="American Samoa") & (df_rank["STNAME"]!="Northern Mariana Islands") & (df_rank["STNAME"]!="US Virgin Islands") & (df_rank["STNAME"]!="Guam") & (df_rank["STNAME"]!="Puerto Rico")]
df_religion = df_religion.set_index("STNAME")
df_rank = df_rank.set_index("STNAME")

#Merging the two datasets
df_religion = pd.merge(df_rank,df_religion,left_index=True,right_index=True)

#Sorting the merged dataset based on the percentage of people who stated that religion is important
df_religion = df_religion.sort_values("Percent Stating Religion is Important",ascending=False)

#df_religion["Jewish Cong"] = df_religion["UMGCCNG"]
df_religion["Hindu Adherence"] = df_religion["HNIADH"] + df_religion["HNPRADH"] + df_religion["HNRADH"] + df_religion["HNTTADH"]
df_religion["Buddhist Adherence"] = df_religion["BUDTADH"] + df_religion["BUDMADH"] + df_religion["BUDVADH"]
df_religion["African Adherence"] = df_religion["AMEADH"] + df_religion["AMEZADH"]

#Extracting the top 5 and lowest 5 religious states as well as North Carolina
df_highest = df_religion.iloc[:5,:]
df_lowest = df_religion.iloc[-5:,:]
df_nc = df_religion.loc["North Carolina",:]

#Combining the top 5 and bottom 5 religious states as well as North Carolina for each religion
muslim_adh = df_highest["MSLMADH"].append(df_lowest["MSLMADH"],ignore_index=True).append(pd.Series(df_nc["MSLMADH"]),ignore_index=True)
hindu_adh = df_highest["Hindu Adherence"].append(df_lowest["Hindu Adherence"],ignore_index=True).append(pd.Series(df_nc["Hindu Adherence"]),ignore_index=True)
buddhist_adh = df_highest["Buddhist Adherence"].append(df_lowest["Buddhist Adherence"],ignore_index=True).append(pd.Series(df_nc["Buddhist Adherence"]),ignore_index=True)
african_adh = df_highest["African Adherence"].append(df_lowest["African Adherence"],ignore_index=True).append(pd.Series(df_nc["African Adherence"]),ignore_index=True)

#Concatenating all the required states for representation in the xaxis of the plot
nc_id = pd.DataFrame(["North Carolina"]).set_index(0)
states = df_highest.index.append(df_lowest.index).append(nc_id.index)

#Determining the religiosity or how religious a state is in terms of percentage from the second dataset
religiosity = df_highest["Percent Stating Religion is Important"].append(df_lowest["Percent Stating Religion is Important"],ignore_index=True).append(pd.DataFrame(pd.Series(df_nc["Percent Stating Religion is Important"])),ignore_index=True)

#Plotting the stacked bar charts for each state
fig,ax1 = plt.subplots()
ax1.bar(range(len(muslim_adh)),muslim_adh,label="Muslim",alpha=0.5)
ax1.bar(range(len(hindu_adh)),hindu_adh,bottom=muslim_adh,color="red",label="Hindu",alpha=0.5)
ax1.bar(range(len(buddhist_adh)),buddhist_adh,bottom=muslim_adh+hindu_adh,color="orange",label="Buddhist",alpha=0.5)
ax1.bar(range(len(african_adh)),african_adh,bottom=buddhist_adh+muslim_adh+hindu_adh,color="purple",label="African",alpha=0.5)
plt.xticks(range(len(muslim_adh)+1),states.map(str),rotation=45,size=10)
ax1.legend(bbox_to_anchor=(1.15, 1.02))
plt.xlabel("States",size=12)
plt.ylabel("Number of adherents",size=12)
plt.title("Distribution of minority religions in most and least \nreligious US states compared to NC")

#Plotting the religiosity % as a line plot
ax2 = ax1.twinx()
ax2.plot(religiosity,marker='o',color='black',alpha=0.8)
ax2.set_ylabel("Religiosity (%)",size=12)
fig.savefig("nc_religion_plot")

plt.show()


# In[ ]:



