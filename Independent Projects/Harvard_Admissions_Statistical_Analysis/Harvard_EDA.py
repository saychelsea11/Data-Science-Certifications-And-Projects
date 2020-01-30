
# coding: utf-8

# In[1]:



import types
import pandas as pd
from botocore.client import Config
import ibm_boto3
import matplotlib.pyplot as plt
import numpy as np
import re
import scipy.stats as st

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share your notebook.
client_dc8d30d252dd4735afdd8adc664a3b5f = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='Rzgf3RqvobPCmV_AnOIjeY3F37j3M1aiK0CHHIDVGWtD',
    ibm_auth_endpoint="https://iam.bluemix.net/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3-api.us-geo.objectstorage.service.networklayer.com')

body = client_dc8d30d252dd4735afdd8adc664a3b5f.get_object(Bucket='collegeseda-donotdelete-pr-jo1jxglzjujgax',Key='Most-Recent-Cohorts-All-Data-Elements.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df = pd.read_csv(body)
df.head()



# #### It took a few seconds to load up the data. Considering that the file is over 200MB in size that's not too bad
# 
# #### Now let's summarize the resulting dataframe

# In[2]:


df.shape


# #### We have 7058 rows and 1977 columns. Once we are done overviewing the dataset, we might want to reduce the number of columns by getting rid of the ones that we don't need.

# In[3]:


df.describe()


# #### The describe function is not too useful at this point since the dataset is massive and we don't yet know which columns to focus on. Going through the summary stats of 631 numerical variables will be an impossible task. 
# 
# #### To narrow this dataset down to a handful of important variables the best method would be to refer to the *College Scorecard* website as well as the *Data Dictionary* provided which describes each variable.
# 
# #### However, the first thing to do would be to reduce the sample size of this dataset to include universities that are comparable to Harvard in terms of quality, diversity and enrollment. For this we can filter out universities based on their average SAT scores. The average of reading and math scores can used for the filtering. 

# In[4]:


#Determining the combined math and reading score for each university
df_sat = df[["UNITID","INSTNM","SATVRMID","SATMTMID","UGDS","UGDS_ASIAN"]]
df_sat["SAT_combined"] = df_sat["SATVRMID"]+df_sat["SATMTMID"]

#Adding the combined SAT scores variable to the main dataset
df["SAT_combined"] = df_sat["SAT_combined"]

#Sorting universities in descending order based on the combined SAT scores
df_sat = df_sat.sort_values("SAT_combined",ascending=False)
df_sat.index = range(1,len(df_sat)+1)
df_sat.index.name = "Rank"
df_sat.columns = ["ID","Institution name","SAT reading median score","SAT math median score","Total undergraduate enrollment","Asian enrollment percentage","Combined SAT score"]
df_sat.head(30)


# #### We can see the filtererd dataset above, sorted by highest combined SAT score. The table below shows just the college name and the SAT scores to get a clearer idea of the ranks. 

# In[5]:


sat_scores = pd.Series(df_sat["Combined SAT score"]).dropna()
print ("Number of universities who have valid combined SAT scores: ",len(sat_scores))
print ("Harvard's SAT score percentile: ",(st.percentileofscore(sat_scores,df_sat["Combined SAT score"][5])))
df_sat.loc[:,["Institution name","Combined SAT score"]].head(10)


# In[6]:


plt.figure(figsize=(15,12))
barlist = plt.barh(range(len(df_sat["Institution name"][:30])),df_sat["Combined SAT score"][:30],alpha=0.7)
barlist[4].set_color('r')
plt.xlabel("Combined SAT score",size=14)
plt.ylabel("University",size=14)
plt.yticks(range(len(df_sat["Institution name"][:30])),df_sat["Institution name"],size=12)
plt.xticks(size=12)
plt.xlim(1400,1600)
plt.title("Bar plot of combined SAT scores of the top 30 universities",size=15)
plt.show()


# #### From the list and barplot above, you can see that when sorted based on combined SAT scores, some of the biggest universities make the cut with *California Institute of Technology* at the very top. *Harvard* sits at fifth spot while other renowned colleges such as *MIT* and *Yale* also make the top 10 in the list. Harvard is marked in Red in the graph. 
# 
# #### Next we can filter out colleges based on a SAT score cutoff point. For this we need to look at some more statistics such as how many colleges have higher scores at different thresholds. 

# In[7]:


print ("Number of universities with a combined SAT score of above 1200: "+str(len(df_sat[df_sat["Combined SAT score"]>1200])))
print ("Number of universities with a combined SAT score of above 1100: "+str(len(df_sat[df_sat["Combined SAT score"]>1100])))
print ("Number of universities with a combined SAT score of above 1000: "+str(len(df_sat[df_sat["Combined SAT score"]>1000])))


# #### There are only 256 universities with a combined score of above 1200 for reading and math which is a rather small sample size. This number increases to 604 for a threshold of 1100 and skyrockets to 1025 for a score of 1000. For this first filtering step let's go with the 1200 threshold, which leaves us with 256 universities which is a good sample size for further analysis. 
# 
# #### Below we remove the unwanted instances using the threshold

# In[8]:


df_sat = df_sat[df_sat["Combined SAT score"]>1200]
df_sat.shape


# #### Next we apply the 1025 filtered institutions to the larger dataset using the unique __UNIT IDs__ of each institution

# In[9]:


df_filter = df.loc[df["UNITID"].isin(df_sat["ID"])]
df_filter.shape


# #### Let's make sure that we have everything in the dataset

# In[10]:


df_filter.head()


# In[11]:


#Looking at highest undergraduate enrollments at each college
df_ugds = df.sort_values("UGDS",ascending=False)[["INSTNM","UGDS"]]

#Determining percentage of Asian students and sorting the colleges in descending order of percentage
df_asian = df_filter.sort_values("UGDS_ASIAN",ascending=False)[["INSTNM","UGDS_ASIAN","SAT_combined","ADM_RATE"]]

df_asian["Total enrollment"] = df_ugds["UGDS"]
df_asian["Number of Asian students"] = round(df_asian["UGDS_ASIAN"]*df_asian["Total enrollment"])
df_asian = df_asian.sort_values("SAT_combined",ascending=False)
df_asian.index = range(1,len(df_asian)+1)

#Converting the SAT,total enrollment and Asian enrollment values to integers for better reading
df_asian["Number of Asian students"] = list(map(int,df_asian["Number of Asian students"]))
df_asian["Total enrollment"] = list(map(int,df_asian["Total enrollment"]))
df_asian["SAT_combined"] = list(map(int,df_asian["SAT_combined"]))

#Finding the row for Harvard by simply looking for 'Harvard' under the institution name
df_asian_harvard = df_asian[df_asian.apply(lambda x: 'Harvard' in x[0],axis=1)]
df_asian.columns = ["Institution name","Asian enrollment percentage","Combined SAT score","Admission Rate","Total enrollment","Asian students enrollment"]
df_asian = df_asian.sort_values("Asian enrollment percentage",ascending=False)
df_asian.index = range(1,len(df_asian)+1)
df_asian.head()


# In[12]:


df_asian["Asian enrollment percentage"] = df_asian["Asian enrollment percentage"]*100
df_asian.loc[:,["Institution name","Asian enrollment percentage","Combined SAT score"]].head(10)


# #### We can now exactly see what the percentage of Asian students as well as the total number of Asian students at Harvard is. Now we can explore how these numbers compare with the other top colleges. 

# In[13]:


#Colleges with higher number of Asian students
print ("Colleges with higher number of Asian students than Harvard ",sum(df_asian["Asian students enrollment"]>int(df_asian_harvard["Number of Asian students"])))
print ("Colleges with lesser number of Asian students than Harvard ",sum(df_asian["Asian students enrollment"]<int(df_asian_harvard["Number of Asian students"])))
print ("Harvard lies in the "+str(round(((sum(df_asian["Asian students enrollment"]<int(df_asian_harvard["Number of Asian students"])))/df_asian.shape[0])*100))+"th percentile when it comes to highest enrollment of Asian students among the top 256 colleges")
df_asian.shape


# #### Harvard ranks at 54 when it comes to Asian students enrollment based purely on total numbers. To get a better idea of the situation, we can repeat the same exercise but this time with the percentage out of the total enrollment at the college. The percentage makes more sense in this case since admission to Harvard tends to be extremely selective therefore rating it based on total numbers would be unfair. 

# In[14]:


#Colleges with higher number of Asian students
print ("Colleges with higher percentage of Asian students than Harvard ",sum(df_asian["Asian enrollment percentage"]>float(df_asian_harvard["UGDS_ASIAN"]*100)))
print ("Colleges with lesser percentage of Asian students than Harvard ",sum(df_asian["Asian enrollment percentage"]<float(df_asian_harvard["UGDS_ASIAN"]*100)))
print ("Harvard lies in the "+str(round(((sum(df_asian["Asian enrollment percentage"]<float(df_asian_harvard["UGDS_ASIAN"]*100)))/df_asian.shape[0])*100))+"th percentile when it comes to highest enrollment percentage of Asian students among the top 256 colleges")
df_asian.shape


# In[15]:


print ("Harvard's Asian enrollment percentage percentile: ",st.percentileofscore(df_asian["Asian enrollment percentage"],float(df_asian_harvard["UGDS_ASIAN"]*100)))


# #### When the Asian enrollment percentage is taken into account, it provides better reading for Harvard as their percentile goes up from 79 to a respectable 86. Only 34 schools have a higher Asian percentage than Harvard. Some might argue that this number is too high especially considering the fact that Harvard is not only one of the top institutions of the US but also of the world. 
# 
# #### Let's take a look at some of the schools that made the cut above Harvard when it comes to Asian enrollment percentage. 

# In[16]:


#Sorting by enrollment percentage to get new rankings
df_asian_percent = df_asian.sort_values("Asian enrollment percentage",ascending=False)
df_asian_percent.index = range(1,len(df_asian_percent)+1)

plt.figure(figsize=(15,12))
barlist = plt.barh(range(len(df_asian_percent["Institution name"][:35])),df_asian_percent["Asian enrollment percentage"][:35],alpha=0.7)
barlist[34].set_color('r')
plt.xlabel("Asian enrollment percentage",size=14)
plt.ylabel("University",size=14)
plt.yticks(range(len(df_asian_percent["Institution name"][:35])),df_asian_percent["Institution name"],size=12)
plt.xticks(size=12)
plt.xlim(10,50)
plt.title("Bar plot of Asian enrollment percentage of the top 35 universities",size=15)
plt.show()


# In[17]:


#Sorting by enrollment percentage to get new rankings
df_asian_enroll = df_asian.sort_values("Asian students enrollment",ascending=False)
df_asian_enroll.index = range(1,len(df_asian_enroll)+1)

plt.figure(figsize=(15,12))
barlist = plt.barh(range(len(df_asian_enroll["Institution name"][:35])),df_asian_enroll["Asian students enrollment"][:35],alpha=0.7)
#barlist[34].set_color('r')
plt.xlabel("Total Asian enrollment",size=14)
plt.ylabel("University",size=14)
plt.yticks(range(len(df_asian_enroll["Institution name"][:35])),df_asian_enroll["Institution name"],size=12)
plt.xticks(size=12)
plt.xlim(2000,11000)
plt.title("Bar plot of total Asian enrollment of the top 35 universities",size=15)
plt.show()


# In[18]:


df_asian_enroll.loc[:,["Institution name","Asian students enrollment","Combined SAT score"]].head(10)


# In[19]:


#Retrieving the information for just Harvard
harvard = df_asian_enroll[df_asian_enroll.apply(lambda x: "Harvard" in x[0],axis=1)]
print (harvard)


# In[20]:


#Calculating Harvard's total Asian enrollment percentile
print ("Harvard's Asian enrollment percentage percentile: ",st.percentileofscore(df_asian["Asian students enrollment"],float(harvard["Asian students enrollment"])))


# In[21]:


df_asian_percent.head(35)


# In[22]:


print (df_asian_percent.sort_values("Asian enrollment percentage",ascending=False).head()["Institution name"])
print (df_asian_percent.sort_values("Combined SAT score",ascending=False).head()["Institution name"])
print (df_asian_percent.sort_values("Asian students enrollment",ascending=False).head()["Institution name"])


# #### California Institute of Technology leads the pack with a staggering 43.29% Asian enrollment followed by University of the Sciences at 35.8%. Harvard, as we determined earlier, sits at number 35 with a score of 19.42%. This can also be seen on the barplot with Harvard marked in *Red*.
# 
# #### Now that we have a slightly better understanding of the rankings, let's compare these universities using the three main variables in our analysis - total Asian enrollment, Asian enrollment percentage and combined SAT score. 
# 
# #### For a fun experiment we can see how the rankings of the top colleges change when we switch between the three variables stated above and then compare these changes with that of Harvard. In order to do this we take the initial data (ordered by SAT scores) and then determine how the rankings for each university change when sorted by enrollment percentage and total enrollment respectively. 
# 
# ## Creating datasets for each type of ranking
# 
# #### Here we create the three datasets each one sorted by a different variable:
# - SAT score
# - Total Asian enrollment
# - Asian enrollment percentage
# 
# #### Each sorted dataset is then indexed starting from 1 and then sorted again based on college names. Once that is done, we get 3 lists of ranks which can be referenced by the college names. 

# ### Ordered by SAT score

# In[23]:


#Sorting based on college name
df_asian_percent = df_asian_percent.sort_values("Combined SAT score",ascending=False)
df_asian_percent.index = range(1,len(df_asian_percent.index)+1)
df_asian_percent = df_asian_percent.sort_values("Institution name")
sat_rank = df_asian_percent.index
df_asian_percent.head()


# ### Ordered by Asian enrollment percentage

# In[24]:


df_asian_percent = df_asian_percent.sort_values("Asian enrollment percentage",ascending=False)
df_asian_percent.index = range(1,len(df_asian_percent.index)+1)
df_asian_percent = df_asian_percent.sort_values("Institution name")
percent_rank = df_asian_percent.index
df_asian_percent.head()


# ### Ordered by total Asian enrollment

# In[25]:


#df_asian_enrollment = df_asian.sort_values("Asian students enrollment",ascending=False)
#df_asian_enrollment.index = range(1,len(df_asian_percent.index)+1)
#df_asian_enrollment = df_asian_enrollment.sort_values("Institution name")
#enrollment_rank = df_asian_enrollment.index
df_asian_percent = df_asian_percent.sort_values("Asian students enrollment",ascending=False)
df_asian_percent.index = range(1,len(df_asian_percent.index)+1)
df_asian_percent = df_asian_percent.sort_values("Institution name")
enrollment_rank = df_asian_percent.index
df_asian_percent.head()


# #### The three rank lists can now be put together into one dataset and referenced by the college name which can be included as the index

# In[26]:


df_rank = pd.DataFrame()
df_rank["SAT rank"] = sat_rank
df_rank["Enrollment rank"] = enrollment_rank
df_rank["Percent rank"] = percent_rank
df_rank.index = df_asian_percent["Institution name"]
df_rank.head()


# #### Let's verify the rankings before proceeding with the analysis

# In[27]:


print (df_rank.sort_values("SAT rank").head())
print (df_rank.sort_values("Enrollment rank").head())
print (df_rank.sort_values("Percent rank").head())


# #### Looks like we are good to go. Let's sort the dataset by SAT score to make that a reference.

# In[28]:


df_rank = df_rank.sort_values("SAT rank")
df_rank.head()


# #### One way to analyze the rankings is to find the correlation between them. We can separate the analysis into 2 groups: 
# - SAT vs enrollment
# - SAT vs percent

# ### SAT vs Enrollment
# 
# #### First we see if there is any correlation between the two sets of rankings. So we are basically trying to find out if a higher ranked college would have a higher enrollment. 

# In[29]:


print ("Correlation :",np.corrcoef(df_rank["SAT rank"],df_rank["Enrollment rank"])[0][1])

plt.figure(figsize=(10,8))
plt.scatter(df_rank["SAT rank"],df_rank["Enrollment rank"],alpha=0.8)
plt.xlabel("SAT score ranks")
plt.ylabel("Enrollment ranks")
plt.title("Enrollment rank vs SAT rank")
plt.show()


# #### The correlation score of 0.239 is extremely low and the graph shows that there is no association between the two variables. 

# In[30]:


df_rank["SAT_enrollment"] = df_rank["SAT rank"] - df_rank["Enrollment rank"]
df_rank.head()


# In[31]:


plt.figure(figsize=(15,12))
barlist = plt.barh(range(len(df_rank))[:30],df_rank["SAT_enrollment"][:30],color='red',alpha=0.7)
barlist[4].set_color('g')
plt.xlabel("Change in rankings from SAT score to total Asian enrollment",size=13)
plt.ylabel("Universities",size=13)
plt.title("Change in rankings of top 30 universities by SAT score when ranked by total Asian enrollment",size=15)
plt.yticks(range(len(df_rank))[:30],list(df_rank.index[:30]),size=11)
plt.xticks(size=12)
plt.show()


# #### Finally, we have the graph we were looking for. Analysing just the top 30 schools (SAT score), we can immediately see that all of the schools have a drop in ranking when ordered by total Asian enrollment. From the graph, it appears that the biggest negative values exist for the lesser renowned schools with Webb Institute seeing the most change followed by Franklin W Olin College of Engineering, Harvey Mudd College and Haverford College. These colleges either have a relatively small Asian enrollment or their SAT scores are really high. 
# 
# #### Harvard, our point of interest, actually sees a relatively small shift of around -50 spots. Harvard can be identified by the *Green* bar. 
# 
# 

# #### In order to get the exact numbers, we can sort the dataset based on the change in rankings. 

# In[32]:


#Sorting the dataset by "SAT_enrollment"
df_rank_enrollment_change = df_rank.sort_values("SAT_enrollment")
df_rank_enrollment_change.head(30)


# #### Now we can clearly see the numbers which were not absolutely clear from the graph. From the data you can see that for the top 2 (Webb and Franklin), the SAT rankings are really good (within top 10) and their Asian enrollment rankings are some of the worst at 248 and 235 respectively. 

# ### SAT vs Percentage

# In[33]:


print ("Correlation :",np.corrcoef(df_rank["SAT rank"],df_rank["Percent rank"])[0][1])

plt.figure(figsize=(10,8))
plt.scatter(df_rank["SAT rank"],df_rank["Percent rank"],alpha=0.8)
plt.xlabel("SAT score ranks",size=12)
plt.ylabel("Percent ranks",size=12)
plt.title("Percent rank vs SAT rank",size=15)
plt.show()


# #### These two variables have a relatively better association with a 0.516 correlation score. Also, the graph shows that there is a __slight positive__ trend between the two variables but it's weak and not exactly linear. Therefore, it can't be said that there is any tangible association between the two variables after all

# In[34]:


df_rank["SAT_percentage"] = df_rank["SAT rank"] - df_rank["Percent rank"]
df_rank.head()


# In[35]:


plt.figure(figsize=(15,12))
barlist = plt.barh(range(len(df_rank))[:30],df_rank["SAT_percentage"][:30],color='red',alpha=0.7)
barlist[4].set_color('g')
plt.xlabel("Change in rankings from SAT score to Asian enrollment percentage",size=13)
plt.ylabel("Universities",size=13)
plt.title("Change in rankings of top 30 universities by SAT score when ranked by Asian enrollment percentage",size=15)
plt.yticks(range(len(df_rank))[:30],list(df_rank.index[:30]),size=11)
plt.xticks(size=12)
plt.show()


# In[36]:


#Sorting the dataset by "SAT_enrollment"
df_rank_enrollment_change = df_rank.sort_values("SAT_percentage")
df_rank_enrollment_change.head(30)


# #### Once again, out of the top 30 schools by SAT score, most of them have a drop in rankings when compared to Asian enrollment percentage while only 3 universities have a positive change - Stanford, Wellesley and Carnegie Mellon but only by a few points which tells us that their Asian percentage ranks are comparable to their respective SAT score rankings. Amazingly, California Institute of Technology sees no movement in rankings which is a little hard to see on the graph as there's no bar present for it. 
# 
# #### The highest drop is around -70 which is Franklin W Olin College of Engineering. Interestingly, both Frankling and Webb Institute re-appear as two of the higher drops in this dataset as well. In this case, however, it's mostly due to their high SAT rankings as opposed to a lower ranking in Asian enrollment percentage. 
# 
# #### From the data above, Jewish Theological Seminary of America and Washingt and Lee University see the most negative changes followed by Notre Dame and Colgate University. For the top 2, their drop in rankings has mostly to do with their relatively low Asian percentage at 258 and 208 respectively. 
# 
# #### Harvard is marked in Green once again and shows a change of just -30 points. One way to truly get a measure of Harvard's Asian enrollment is by comparing it with other prestigious schools. We do this in two ways:
# - Comparing Harvard with other colleges with the lowest admission rates
# - Comparing Harvard with other Ivy League schools

# ## Lowest admission rates analysis

# #### To get a meaningful dataset, the original dataset is filtered to contain only colleges that have an admission rate and asian enrollment percentage of greater than 0. Then, the top 30 schools based on admission rate are picked for analysis

# In[37]:


df_adm = df[(df["ADM_RATE"]>0.0)&(df["UGDS_ASIAN"]>0.0)] #Removing colleges with admission rates equal to 0

df_adm_256 = df_adm.sort_values("ADM_RATE")[["INSTNM","ADM_RATE","UGDS_ASIAN","UGDS"]][:256]
df_adm_256.index = range(1,len(df_adm_256)+1)
df_adm_256.columns = ["Institution name","Admission rate","Asian enrollment percentage","Total enrollment"]
df_adm_256["Admission rate"] = df_adm_256["Admission rate"]*100
df_adm_256["Asian enrollment percentage"] = df_adm_256["Asian enrollment percentage"]*100

df_adm = df_adm.sort_values("ADM_RATE")[["INSTNM","ADM_RATE","UGDS_ASIAN","UGDS"]][:30]
df_adm.index = range(1,len(df_adm)+1)
df_adm.columns = ["Institution name","Admission rate","Asian enrollment percentage","Total enrollment"]
df_adm["Admission rate"] = df_adm["Admission rate"]*100
df_adm["Asian enrollment percentage"] = df_adm["Asian enrollment percentage"]*100
df_adm.loc[:,["Institution name","Admission rate","Asian enrollment percentage"]].head(10)


# #### As expected, you have the usual suspects such as Stanford, Harvard and Princeton at the top of the table but you also see other lesser known college such as Juilliard School and Ponoma College make the top 10 while Curtis Institute of Music tops the list with an admission rate of 3.3%. 

# In[38]:


df_adm = df_adm.sort_values("Asian enrollment percentage",ascending=False)
df_adm.index = range(1,len(df_adm)+1)
df_adm.loc[:,["Institution name","Admission rate","Asian enrollment percentage"]].head(10)


# #### The dataset is then sorted by Asian enrollment percentage to find out where Harvard sits when it comes to colleges with the lowest admission rates. 

# In[39]:


#Sorting by enrollment percentage to get new rankings
plt.figure(figsize=(15,12))
barlist = plt.barh(range(len(df_adm["Institution name"])),df_adm["Asian enrollment percentage"],alpha=0.7)
barlist[9].set_color('r')
plt.xlabel("Asian enrollment percentage",size=14)
plt.ylabel("University",size=14)
plt.yticks(range(len(df_adm["Institution name"])),df_adm["Institution name"],size=12)
plt.xticks(size=12)
plt.xlim(0,45)
plt.title("Bar plot of Asian enrollment percentage of the top 30 universities",size=15)
plt.show()


# #### From the graph above, Harvard (shown in red) sits at number 10 when it comes to Asian enrollment percentage among the top 30 schools with the lowest admission rates. 

# In[107]:


#Calculating Harvard's Asian enrollment percentage percentile among 30 lowest admission rate schools
print ("Harvard's Asian enrollment percentage percentile: ",st.percentileofscore(df_adm["Asian enrollment percentage"],float(harvard["Asian enrollment percentage"])))
print ("Harvard's Asian enrollment percentage percentile: ",st.percentileofscore(df_adm_256["Asian enrollment percentage"],float(harvard["Asian enrollment percentage"])))
print ("Asian enrollment percent distribution mean: ",np.mean(df_adm_256["Asian enrollment percentage"]))
print ("Asian enrollment percent distribution median: ",np.percentile(df_adm_256["Asian enrollment percentage"],50))

plt.figure(figsize=(14,10))
plt.hist(df_adm_256["Asian enrollment percentage"],alpha=0.7)
plt.axvline(19.42,color='r',label="Harvard's Asian enrollment percentage",linestyle="--",linewidth=3,alpha=0.5)
plt.axvline(np.mean(df_adm_256["Asian enrollment percentage"]),color='g',label='Asian enrollment percent mean',linestyle="--",linewidth=3,alpha=0.5)
plt.axvline(np.percentile(df_adm_256["Asian enrollment percentage"],50),color='orange',label='Asian enrollment percent median',linestyle="--",linewidth=3,alpha=0.5)
plt.title("Asian enrollment percentage distribution of the 256 schools with lowest admission rates",size=14)
plt.xlabel("Percent")
plt.ylabel("Frequency")
plt.legend()
plt.show()


# #### Above, we have plotted a histogram of the Asian enrollment percentage for the lowest 256 colleges. We picked 256 since the analysis with the SAT scores was done with the same number. As expected from a right skewed distribution, the mean is higher than the median. Another thingto note is that Harvard falls in the 87.11 percentile of the above distribution. 

# ## Ivy League analysis

# Here are the eight Ivy League colleges: 
# - Brown University
# - Harvard
# - University of Pennsylvania
# - Yale
# - Princeton
# - Dartmouth
# - Columbia
# - Cornell
# 
# The first thing to do is to extract these colleges from the larger dataset. Let's try to get them using the generic names for these colleges. 

# In[41]:


ivy_league = ["Harvard","Cornell","Dartmouth","Columbia","University of Pennsylvania","Princeton","Brown University","Yale"]
df_reduced = df.loc[:,['INSTNM','UGDS','UGDS_ASIAN','ADM_RATE']]
df_ivy = pd.DataFrame(columns=df_reduced.columns)
for i in ivy_league:
    college = df_reduced[df_reduced.apply(lambda x: i in x['INSTNM'],axis=1)]
    df_ivy = df_ivy.append(pd.DataFrame(college))
    
df_ivy.columns = ["Institution name","Total enrollment","Asian enrollment percentage","Admission rate"]
df_ivy


# #### That wasn't a good idea as we got 102 colleges that share the same generic names. Let's see if we can sort them by *ADM_RATE* to see if we can find the names we are looking for. The reasoning behind this is that the Ivy League colleges are expected to have relatively lower admission rates. Let's find out if this theory holds true. 

# In[42]:


df_ivy = df_ivy.sort_values("Admission rate")
df_ivy.head(20)


# #### That actually worked! Now we can filter out the first 10 entries from this sorted dataframe to get our Ivy League dataset. 

# In[43]:


df_ivy = df_ivy.iloc[:8,:]
df_ivy.index = range(1,len(df_ivy)+1)
df_ivy["Asian enrollment percentage"] = df_ivy["Asian enrollment percentage"]*100
df_ivy.loc[:,["Institution name","Asian enrollment percentage"]]


# #### We can immediately see that Harvard in fact has the lowest admission rate out of the 8 prestigious universities at an astonishing 5.16%. 
# 
# #### Now we can sort the dataset by the Asian enrollment percentage and also visualize the results

# In[44]:


df_ivy.iloc[2,0] = "Columbia"
df_ivy = df_ivy.sort_values("Asian enrollment percentage",ascending=False)
df_ivy.index = range(1,len(df_ivy)+1)
df_ivy.loc[:,["Institution name","Asian enrollment percentage"]]


# #### Harvard sit's at number 3 when it comes to the enrollment percentage of Asian students. I have to say that I was not expecting that. Let's see what the total Asian enrollment looks like. 

# In[45]:


#Sorting by enrollment percentage to get new rankings
plt.figure(figsize=(8,6))
barlist = plt.bar(range(len(df_ivy["Institution name"])),df_ivy["Asian enrollment percentage"],alpha=0.7)
barlist[2].set_color('r')
plt.ylabel("Asian enrollment percentage",size=14)
plt.xlabel("University",size=14)
plt.xticks(range(len(df_ivy["Institution name"])),df_ivy["Institution name"],size=12,rotation=90)
plt.yticks(size=12)
plt.title("Histogram of Asian enrollment percentage of the top 30 universities",size=15)
plt.show()


# In[46]:


df_ivy["Asian enrollment"] = df_ivy["Asian enrollment percentage"]*df_ivy["Total enrollment"]
df_ivy = df_ivy.sort_values("Asian enrollment",ascending=False)
df_ivy.index = range(1,len(df_ivy)+1)
df_ivy


# #### The rankings change a little when it comes to total Asian enrollment but Harvard still sits in third spot. 

# ## Statistical inference

# #### Now that we have an idea regarding the distribution of universities based on different factors such as the admission rate, SAT score and Asian enrollment, let's do some statistical analysis to find out whether Harvard's Asian enrollment is comparable to other schools of similar stature and credentials
# 
# #### To do this, first we need to define the population using which we can do our test. A good population to choose from would be universities whose combined SAT scores are above 1200. The metric that we would be using is the Asian students enrollment percentage or the *UGDS_ASIAN* variable. 

# In[47]:


df_sat.shape


# #### Since our population size is a little small at 256, let's increase our population to include a larger number of colleges, so we relax the combined SAT criteria to 700 instead of 1200. 

# In[48]:


#Determining the combined math and reading score for each university
df_sat2 = df[["UNITID","INSTNM","SATVRMID","SATMTMID","UGDS","UGDS_ASIAN"]]
df_sat2["Asian enrollment"] = df_sat2["UGDS"]*df_sat2["UGDS_ASIAN"]
df_sat2["SAT_combined"] = df_sat2["SATVRMID"]+df_sat2["SATMTMID"]

#Sorting universities in descending order based on the combined SAT scores
df_sat2 = df_sat2.sort_values("SAT_combined",ascending=False)
df_sat2.index = range(1,len(df_sat2)+1)
df_sat2.index.name = "Rank"
df_sat2 = df_sat2[df_sat2["Asian enrollment"]>700]
df_sat2.columns = ["ID","Institution name","SAT reading median score","SAT math median score","Total undergraduate enrollment","Asian enrollment percentage","Total Asian enrollment","Combined SAT score"]
df_sat2["Asian enrollment percentage"] = df_sat2["Asian enrollment percentage"]*100
df_sat2.shape


# In[49]:


df_sat2.head()


# #### Our population is now larger with 310 observations. The reason we need a larger population is to have a decent sample size to satisfy the independence condition for hypothesis testing. For the independence condition, not only do we have to use random sampling in selecting the data for the test but also the number of samples has to be less than 10% of the population. Since 10% of 310 is 31, that is a decent sample size to use even if the distribution of the data is not normal. 

# In[50]:


df_sat2.describe()


# #### From the summary stats above, we can see that the mean SAT score of the population is 1217.41 and the mean Asian enrollment percentage is 6.92% which is actually much lower than that of Harvard's which is 19.42% as we discovered earlier. 

# In[105]:


print ("Asian enrollment percent distribution mean: ",np.mean(df_sat2["Asian enrollment percentage"]))
print ("Asian enrollment percent distribution median: ",np.percentile(df_sat2["Asian enrollment percentage"],50))

plt.figure(figsize=(14,10))
plt.hist(df_sat2["Asian enrollment percentage"],alpha=0.7)
plt.axvline(19.42,color='r',label="Harvard's Asian enrollment percentage")
plt.axvline(np.mean(df_sat2["Asian enrollment percentage"]),color='g',label='Asian enrollment mean')
plt.axvline(np.percentile(df_sat2["Asian enrollment percentage"],50),color='orange',label='Asian enrollment percent median')
plt.title("Asian enrollment percentage distribution of the 310 schools with highest combined SAT scores",size=14)
plt.xlabel("Percent")
plt.ylabel("Frequency")
plt.legend()
plt.show()


# In[106]:


plt.figure(figsize=(14,10))
df_sat2_sample = df_sat2.sample(n=30,random_state=15,replace=False)
pdf = st.norm.pdf(df_sat2_sample["Asian enrollment percentage"])
plt.hist(df_sat2_sample["Asian enrollment percentage"],range=(0,35),bins=7,alpha=0.7)
plt.axvline(19.42,color='r',label="Harvard's Asian enrollment percentage")
plt.axvline(np.mean(df_sat2_sample["Asian enrollment percentage"]),color='g',label='Sample mean')
plt.axvline(np.percentile(df_sat2_sample["Asian enrollment percentage"],50),color='orange',label='Asian enrollment percent median')
plt.title("Asian enrollment percentage distribution of 30 samples",size=14)
plt.xlabel("Percent")
plt.ylabel("Frequency")
plt.legend()
plt.show()


# In[53]:


plt.scatter(df_sat2_sample["Asian enrollment percentage"],pdf,color='red',alpha=0.8)
plt.show()


# In[54]:


df_sat2_sample.describe()


# ### Confidence Interval

# In[100]:


#Total dataset
ci_total = st.norm.interval(0.95, loc=np.mean(df_sat2["Asian enrollment percentage"]), scale=np.std(df_sat2["Asian enrollment percentage"]))

#Sampled data
ci_sample = st.norm.interval(0.95, loc=np.mean(df_sat2_sample["Asian enrollment percentage"]), scale=np.std(df_sat2_sample["Asian enrollment percentage"]))

print ("Total dataset CI: ",ci_total)
print ("Sample dataset CI: ",ci_sample)


# ### Visualizing confidence intervals

# #### Total dataset

# In[102]:


plt.figure(figsize=(14,10))
plt.hist(df_sat2["Asian enrollment percentage"],alpha=0.7)
plt.axvline(19.42,color='r',label="Harvard's Asian enrollment percentage",linestyle="--",linewidth=3,alpha=0.5)
plt.axvline(np.mean(df_sat2["Asian enrollment percentage"]),color='g',label='Asian enrollment mean',linestyle="--",linewidth=3,alpha=0.5)
plt.axvline(np.percentile(df_sat2["Asian enrollment percentage"],50),color='orange',label='Asian enrollment percent median',linestyle="--",linewidth=3,alpha=0.5)
plt.axvline(ci_total[0],color='black',label='95% CI lower limit',linestyle="-",linewidth=3,alpha=0.8)
plt.axvline(ci_total[1],color='black',label='95% CI higher limit',linestyle="-",linewidth=3,alpha=0.8)
plt.title("Asian enrollment percentage distribution of of the 310 schools with highest combined SAT scores",size=14)
plt.xlabel("Percent")
plt.ylabel("Frequency")
plt.legend()
plt.show()


# #### Sampled set

# In[103]:


plt.figure(figsize=(14,10))
plt.hist(df_sat2_sample["Asian enrollment percentage"],range=(0,35),bins=7,alpha=0.7)
plt.axvline(19.42,color='r',label="Harvard's Asian enrollment percentage",linestyle="--",linewidth=3,alpha=0.5)
plt.axvline(np.mean(df_sat2_sample["Asian enrollment percentage"]),color='g',label='Sample mean',linestyle="--",linewidth=3,alpha=0.5)
plt.axvline(np.percentile(df_sat2_sample["Asian enrollment percentage"],50),color='orange',label='Asian enrollment percent median',linestyle="--",linewidth=3,alpha=0.5)
plt.axvline(ci_sample[0],color='black',label='CI lower limit',linestyle="-",linewidth=3,alpha=0.8)
plt.axvline(ci_sample[1],color='black',label='CI higher limit',linestyle="-",linewidth=3,alpha=0.8)
plt.title("Asian enrollment percentage distribution of 30 samples",size=14)
plt.xlabel("Percent")
plt.ylabel("Frequency")
plt.legend()
plt.show()


# ### Hypothesis testing

# #### Null hypothesis (H0): population mean = Harvard's score (0.1942)
# #### Alternate hypothesis (HA): population mean < Harvard's score

# In[99]:


t,p = st.ttest_1samp(df_sat2["Asian enrollment percentage"],19.42)
pvalue = st.t.cdf(t,len(df_sat2["Asian enrollment percentage"])-1)
print ("Right sided t-statistic: ",t)
print ("Right sided p-value: ",pvalue)


# In[96]:


t,p = st.ttest_1samp(df_sat2_sample["Asian enrollment percentage"],19.42)
pvalue = st.t.cdf(t,len(df_sat2_sample["Asian enrollment percentage"])-1)
print ("Right sided t-statistic: ",t)
print ("Right sided p-value: ",pvalue)


# ### Visualizing p-value

# In[60]:


plt.figure(figsize=(14,12))
plt.hist(df_sat2["Asian enrollment percentage"],alpha=0.7)
plt.axvline(0.1942,color='r',label="Harvard's Asian enrollment percentage")
plt.axvline(np.mean(df_sat2["Asian enrollment percentage"]),color='g',label='Asian enrollment mean')
plt.title("Asian enrollment percentage distribution of 30 samples",size=14)
plt.xlabel("Percent")
plt.ylabel("Frequency")
plt.legend()
plt.show()


# ### Reasons for analyzing the entire population
# - Small population size (310 observations)
# - Uncommon characteristics between observations: Each college has unique attributes
# 
# #### Note: when sampling the entire population (non-probability sampling), *statistical generalizations* cannot be made but *analytical generalizations* can be made. This basically means that statistical inference cannot be generalized but researach or insight from the data can be gained. 
