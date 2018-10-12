from __future__ import print_function

import argparse
import json
import pprint
import requests
import sys
import urllib
import sqlite3
import matplotlib.pyplot as plt
from collections import Counter
import math_stats_functions as st

conn = sqlite3.connect('Chinese.sqlite')
cur = conn.cursor()
conn.text_factory = str

cur.execute('''SELECT id, Name, Rating, Price, Reviews from Chinese''')

id = []
rev = []
rating = []
prices = []

for line in cur:
	id.append(line[0])
	rev.append(line[4])
	rating.append(line[2])


	if line[3] <> None: 
		prices.append(len(line[3]))
	else:
		prices.append(0)


#fig, ax = plt.subplots(1,2)
	
count = Counter(rating)

#ax[0].xlim(-0.5,5.5)
#ax[0].ylim(-100,3800)

#ax[0].bar([x-0.2 for x in count.keys()],count.values(),0.4)
plt.subplot(2,3,1)
plt.xlim(-0.5,5.5)
#plt.ylim(0,200)
plt.bar([x-0.2 for x in count.keys()],count.values(),0.4)
plt.grid()
plt.title("Pizza restaurant ratings histogram")
plt.xlabel("Rating")
plt.ylabel("Number of restaurants")
	
#ax[1].xlim(-10,480)
#ax[1].ylim(-100,3800)
#ax[1].scatter(id,rev)
plt.subplot(2,3,2)
plt.xlim(-10,898)
plt.ylim(-10, 3800)
plt.scatter(id,rev)
plt.grid()
plt.title("Number of reviews")
plt.xlabel("Restaurant ID")
plt.ylabel("Number of reviews")

plt.subplot(2,3,3)
plt.bar([x-0.4 for x in Counter(prices).keys()],Counter(prices).values(),0.8)
plt.xlabel("Price Level")
plt.ylabel("Number of restaurants")
plt.title("Price level distribution")

plt.subplot(2,3,4)
plt.scatter(rating,rev)
plt.xlabel("Rating")
plt.ylabel("Number of reviews")

plt.subplot(2,3,5)
plt.scatter(prices,rev)
plt.xlabel("Price")
plt.ylabel("Number of reviews")

#10 Highest number of reviews
cur.execute('''SELECT id, Name, Rating, Price, Reviews from Chinese ORDER BY Reviews''')

name = []
rev_ord = []
id_ord = [] 

for line in cur:
	rev_ord.append(line[4])
	name.append(line[1])
	id_ord.append(line[0])
	
plt.subplot(2,3,6)
plt.xticks(range(10),id_ord[-10:])
for i in range(-10,0,1):
	print (name[i])
	print (range(10)[i])
	print (rev_ord[i])
	try:
		plt.annotate(name[i],xy=(range(10)[i],rev_ord[i]))
	except:
		plt.annotate("Name error",xy=(range(10)[i],rev_ord[i]))
plt.scatter(range(10),rev_ord[-10:])

'''
plt.figure()
for i in range(-10,0,1):
	plt.annotate(name[i],xy=(range(10)[i],rev_ord[i]))
plt.scatter(range(10),rev_ord[-10:])
'''
#Creating a normal distribution for the reviews
mu = st.mean(prices)
sigma = st.standard_deviation(prices)
xs = range(int(min(prices)),int(max(prices)))
ys = [st.normal_pdf(x,mu,sigma) for x in xs]
print (mu)
print (sigma)

plt.figure()
#st.make_hist(0.5,1000,10000)
plt.plot(xs,ys)

plt.show()

cur.close()