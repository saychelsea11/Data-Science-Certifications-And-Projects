from __future__ import division
from functools import partial
import random
import collections
from collections import Counter
import matplotlib.pyplot as plt
import math
import csv
import bs4 
from bs4 import BeautifulSoup
import requests
import re
from time import sleep
import json
import dateutil.parser 
from dateutil.parser import parse
from twython import Twython
from twython import TwythonStreamer
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import xml.etree.ElementTree as ET

CONSUMER_KEY = "ZrUFf8TWyffjP1fCe0El4rwMF"
CONSUMER_SECRET = "TdHNkp9FbmpmfV34r2Ac86EFr06bGMvoI7QbdJjMbXPTtC7Khx"
ACCESS_TOKEN = "841828278486896640-eIDwwlSKAStWzDecWkT8F8Fwucf3zds"
ACCESS_TOKEN_SECRET = "LWbmuWmU2sYrvGxXQ0MoJoMy2qXFDINtAI4dFjWFqFumS"

"""Linear Algebra Functions"""

"""Vectors"""

def vector_add(v,w):
	return [v_i + w_i for v_i, w_i in zip(v,w)]
	
def vector_subtract(v,w):
	return [v_i - w_i for v_i, w_i in zip(v,w)]
	
def vector_sum(vectors):
	result = vectors[0]
	for vector in vectors[1:]:
		result = vector_add(result, vector)
		return result
		
def scalar_multiply(c,v):
	return [c*v_i for v_i in v]
	
def vector_means(vectors):
	n = len(vectors)
	return scalar_multiply(1/n, vector_sum(vectors))
	
def dot(v,w):
	return sum(v_i*w_i for v_i, w_i in zip(v,w))
	
def sum_of_squares(v):
	return dot(v,v)

def magnitude(v):
	return math.sqrt(sum_of_squares(v))
	
def squared_distance(v,w):
	return sum_of_squares(vector_subtract(v,w))
	
"""def distance(v,w):
	return math.sqrt(squared_distance(v,w))"""
	
def distance(v,w):
	return magnitude(vector_subtract(v,w))
	
"""Matrices"""

def shape(A):
	num_rows = len(A)
	num_cols = len(A[0]) if A else 0
	return num_rows, num_cols
	
def get_row(A,i):
	return A[i]
	
def get_column(A,j):
	return [A_i[j] for A_i in A]
	
def make_matrix(num_rows, num_cols, entry_fn):
	return [[entry_fn(i,j) for j in range(num_cols)] for i in range(num_rows)]
	
def is_diagonal(i,j):
	return 1 if i==j else 0
	
identity_matrix = make_matrix(5,5,is_diagonal)



"""Statistics"""


"""Central Tendencies"""

def mean(x):
	return sum(x)/len(x)
	
def median(v):
	n = len(v)
	sorted_v = sorted(v)
	midpoint = n // 2
	
	if n%2 == 1:
		return sorted_v[midpoint]
	else:
		lo = midpoint - 1
		hi = midpoint 
		return (sorted_v[lo] + sorted_v[hi])/2
		
def quantile(x,p):
	p_index = int(p*len(x))
	return sorted(x)[p_index]
	
def mode(x):
	counts = Counter(x)
	max_count = max(counts.values())
	return [x_i for x_i, count in counts.iteritems() if count==max_count]

"""Dispersion"""	

def data_range(x):
	return max(x) - min(x)
	
def de_mean(x):
	x_bar = mean(x)
	return [x_i - x_bar for x_i in x]
	
def variance(x):
	n = len(x)
	deviations = de_mean(x)
	return sum_of_squares(deviations)/(n-1)
	
def standard_deviation(x):
	return math.sqrt(variance(x))
	
def interquartile_range(x):
	return quantile(x,0.75) - quantile(x,0.25)
	
"""Correlation"""

def covariance(x,y):
	n = len(x)
	return dot(de_mean(x), de_mean(y))/(n-1)
	
def correlation(x,y):
	stdev_x = standard_deviation(x)
	stdev_y = standard_deviation(y)
	if stdev_x > 0 and stdev_y > 0:
		return covariance(x,y)/stdev_x / stdev_y
	else:
		return 0
		
def uniform_pdf(x):
	return 1 if x >= 0 and x < 1 else 0
	
def uniform_cdf(x):
	if x < 0:
		return 0
	elif x < 1:
		return x
	else:
		return 1
		
def normal_pdf(x, mu=0, sigma=1):
	sqrt_two_pi = math.sqrt(2 * math.pi)
	return (math.exp(-(x-mu)**2/2/sigma**2)/(sqrt_two_pi*sigma))
	
#xs = [x/10 for x in range (-50,50)]
#plt.plot(xs,[normal_pdf(x,sigma=1) for x in xs], '-', label = 'mu=0,sigma=1')

def normal_cdf(x,mu=0,sigma=1):
	return(1+math.erf((x-mu)/math.sqrt(2)/sigma))/2
	
#xs = [ x / 10.0 for x in range ( - 50 , 50 )] plt . plot ( xs ,[ normal_cdf ( x , sigma = 1 ) for x in xs ], '-'

def inverse_normal_cdf ( p , mu = 0 , sigma = 1 , tolerance = 0.00001 ): 
#find approximate inverse using binary search  if not standard, compute standard and rescale 
	if mu != 0 or sigma != 1 : 
		return mu + sigma * inverse_normal_cdf ( p , tolerance = tolerance ) 
	low_z , low_p = - 10.0 , 0 # normal_cdf(-10) is (very close to) 0 
	hi_z , hi_p = 10.0 , 1 # normal_cdf(10) is (very close to) 1 
	while hi_z - low_z > tolerance :
		mid_z = ( low_z + hi_z ) / 2 # consider the midpoint 
		mid_p = normal_cdf ( mid_z ) # and the cdf's value there 
		if mid_p < p : # midpoint is still too low, search above it 
			low_z , low_p = mid_z , mid_p 
		elif mid_p > p : # midpoint is still too high, search below it 
			hi_z , hi_p = mid_z , mid_p 
		else : break 
	return mid_z
	
def bernoulli_trial(p):
	return 1 if random.random() < p else 0
	
def binomial(n,p):
	return sum(bernoulli_trial(p) for _ in range(n))
	
def make_hist ( p , n , num_points ): 
	data = [ binomial ( n , p ) for _ in range ( num_points )] # use a bar chart to show the actual binomial samples 
	histogram = Counter ( data ) 
	plt . bar ([ x - 0.4 for x in histogram . keys ()], [ v / num_points for v in histogram . values ()],
	0.8 , color = '0.75' ) 
	mu = p * n 
	sigma = math . sqrt ( n * p * ( 1 - p )) # use a line chart to show the normal approximation 
	print "mean: ", mu
	print "std dev: ", sigma
	xs = range ( min ( data ), max ( data ) + 1 ) 
	ys = [ normal_cdf ( i + 0.5 , mu , sigma ) - normal_cdf ( i - 0.5 , mu , sigma ) for i in xs ] 
	#print xs
	#print ys
	plt . plot ( xs , ys ) 
	plt . title ( "Binomial Distribution vs normal distribution")
	plt.show()
	
def normal_approximation_to_binomial(n,p):
	"""finds mu and sigma corresponding to a Binomial(n,p)"""
	mu = p*n
	sigma = math.sqrt(p*(1-p)*n)
	return mu,sigma
	
#the normal cdf is the probability the variable is below a threshold
normal_probability_below = normal_cdf

#It's above the threshold if it's not below the threshold 
def normal_probability_above(lo,mu=0,sigma=1):
	return 1 - normal_cdf(lo,mu,sigma)
	
#It's between if it's less than hi but not less than lo
def normal_probability_between(lo,hi,mu=0,sigma=1):
	return normal_cdf(hi,mu,sigma) - normal_cdf(lo,mu,sigma)

#It's outside if it's not between 
def normal_probability_outside(lo,hi,mu=0,sigma=1):
	return 1 - normal_probability_between(lo,hi,mu,sigma)
	
def normal_upper_bound(probability, mu=0, sigma = 1):
	"""returns the z for which P(Z <= z) = probability"""
	return inverse_normal_cdf(probability,mu,sigma)
	
def normal_lower_bound(probability, mu=0, sigma = 1):
	"""returns the z for which P(Z <= z) = probability"""
	return inverse_normal_cdf(1 - probability,mu,sigma)
	
def normal_two_sided_bounds(probability,mu=0,sigma=1):
	"""returns the symmetric (about the mean) bounds that contain
	the specified probability"""
	tail_probability = (1-probability)/2
	
	#upper bound should have tail_probability above it	
	upper_bound = normal_lower_bound(tail_probability,mu,sigma)
	
	#lower bound should have tail_probabiity below it
	lower_bound = normal_upper_bound(tail_probability,mu,sigma)
	
	return lower_bound, upper_bound
	
def two_sided_p_value(x,mu=0,sigma=1):
	if x>mu:
		#If x is greater than the mean, the tail is what's greater than x-mu
		return 2*normal_probability_above(x,mu,sigma)
	else:
		#if x is less than the mean, the tail is what's less than x-mu
		return 2*normal_probability_below(x,mu,sigma)
		
#upper_p_value = normal_probability_above
#lower_p_value = normal_probability_below

def B(alpha,beta):
	"""a normalizing constant so that the total probability is 1"""
	return math.gamma(alpha)*math.gamma(beta)/math.gamma(alpha+beta)
		
def beta_pdf(x,alpha,beta):
	if x < 0 or x > 1:
		return 0
	return x**(alpha-1)*(1-x)**(beta-1)/B(alpha,beta)
	
"""Gradient Descent"""

def difference_quotient(f,x,h):
	return (f(x+h) - f(x))/h
	
#Actual derivative and estimate comparison example
def square(x):
	return x*x
	
def derivative(x):
	return 2*x 
	
def random_poly_func(v):
	x,y = v	
	return (x*x) + (x*y*y)
	
def dev_x(x,y):
	return (2*x) + (y*y)
	
def dev_y(x,y):
	return 2*x*y
	

#Gradient estimate for single variable functions
"""	
derivative_estimate = partial(difference_quotient, square, h=0.00001)

#plot to show they're basically the same
x = range(-10,10)
plt.title("Actual Derivative vs Estimates")
plt.plot(x,map(derivative,x),'rx',label='Actual')
#red x 
plt.plot(x,map(derivative_estimate,x), 'b+', label='Estimate')
#blue + 
plt.legend(loc=9)
plt.show()
"""

def partial_difference_quotient(f,v,i,h):
	"""compute the ith partial difference quotient of f at v"""
	#add h to just the ith element of v
	w = [v_j + (h if j == i else 0) for j, v_j in enumerate(v)]  
	return (f(w) - f(v)) / h 
	
def estimate_gradient(f,v,h=0.00001):
	return [partial_difference_quotient(f,v,i,h)
	for i,_ in enumerate(v)]
	
#Gradient estimate for multiple variable functions

#Single point gradient estimate routine
"""
x = 3
y = 3
v = [x,y]
grad = estimate_gradient(random_poly_func,v,h=0.00001)
print grad
"""

#Multiple points gradient estimate technique
"""
x = range(-10,10)
y = range(20,40)
result = []

values_x = map(dev_x,x,y)
values_y = map(dev_y,x,y)

for i,_ in enumerate(x):
	v = [x[i],y[i]]
	grad = estimate_gradient(random_poly_func,v,h=0.00001)
	result.append(grad)
	
result_x = [k[0] for k in result]
result_y = [k[1] for k in result]

plt.plot(values_x,'o')
plt.plot(result_x,'-')
plt.plot(values_y,':')
plt.plot(result_y,'x')
plt.show()
"""

def step(v,direction,step_size):
	"""move step_size in the direction from v"""
	return [v_i + step_size * direction_i
			for v_i, direction_i in zip(v,direction)]
			
def sum_of_squares_gradient(v):
	return [2 * v_i for v_i in v]
	
#pick a random starting point
v = [random.randint(-10,10) for i in range(3)]

tolerance = 0.001

while True:
	gradient = sum_of_squares_gradient(v) #compute the gradient at v
	next_v = step(v,gradient,-0.01) #take a negative gradient step
	if distance(next_v,v) < tolerance: #stop if we're converging
		break
	v = next_v #continue if we are not
	
def safe(f):
	"""return a new function that's the same as f, except 
	that it outputs infinity whenever f produces an error"""
	
	def safe_f(*args, **kwargs):
		try:
			return f(*args, **kwargs)
		except:
			return float('inf') #this means "infinity" in Python
	return safe_f

#Minimizing a function
def minimize_batch(target_fn, gradient_fn,theta_0,tolerance=0.000001):
	"""use gradient descent to find theta that minimizes target function"""
	
	step_sizes = [100,10,1,0.1,0.01, 0.001,0.0001,0.00001]
	theta = theta_0 							#theta set to initial value
	target_fn = safe(target_fn)   				#safe version of target_fn
	value = target_fn(theta)					#value we're minimizing
	
	while True:
		gradient = gradient_fn(theta)
		next_thetas = [step(theta,gradient, -step_size)
						for step_size in step_sizes]
		#choose the one that minimizes the error function
		next_theta = min(next_thetas, key = target_fn)
		next_value = target_fn(next_theta)
		
		#stop if we're "converging"
		if abs(value - next_value) < tolerance: 
			return theta
		else: 
			theta, value = next_theta, next_value
			
#Maximizing a function
def negate(f):
	"""return a function that for any input x returns -f(x)"""
	return lambda *args, **kwargs: -f(*args,**kwargs)
	
def negate_all(f):
	"""the same when f returns a list of numbers"""
	return lambda *args, **kwargs: [-y for y in f (*args, **kwargs)]
	
def maximize_batch(target_fn, gradient_fn, theta_0,tolerance=0.000001):
	return minimize_batch(negate(target_fn),negate_all(gradient_fn),theta_0,tolerance)
	
"""Stochastic Gradient Descent"""
def in_random_order(data):
	"""generator that returns the elements of data in random order"""
	indexes = [i for i,_ in enumerate(data)] #create a list of indexes 
	random.shuffle(indexes) #shuffle them
	for i in indexes:		#return the data in that order
		yield data[i]
		
def minimize_stochastic(target_fn,gradient_fn,x,y,theta_0,
						alpha_0 = 0.01):
	data = zip(x,y)
	theta = theta_0  		#initial guess
	alpha = alpha_0			#initial step size
	
	min_theta,min_value = None, float("inf") #the minimum so far
	iterations_with_no_improvement = 0
	
	#if we ever go 100 iterations with no improvement, stop 
	while iterations_with_no_improvement < 100:
		value = sum(target_fn(x_i,y_i,theta) for x_i,y_i in data)
		
		if value < min_value:
			#if we've found a new minimum, remember it
			#and go back to the original step size
			min_theta, min_value = theta, value
			iterations_with_no_improvement = 0
			alpha = alpha_0 
		else:
			#otherwise we're not improving, so try shrinking the step size
			iterations_with_no_improvement += 1	
			alpha *= 0.9
			
			#and take a gradient step for each of the data points 
			for x_i,y_i in in_random_order(data):
				gradient_i = gradient_fn(x_i,y_i,theta)
				theta = vector_subtract(theta, scalar_multiply(alpha,gradient_i))
				
	return min_theta	
	
	def maximize_stochastic(target_fn, gradient_fn,x,y,theta_0,alpha_0 = 0.01):
		return minimize_stochastic(negate(target_fn),negate_all(gradient_fn),
									x,y,theta_0,alpha_0)
									
"""Getting Data"""
									
"""#egrep.py
import sys,re

#sys.argv is the list of command-line arguments
#sys.argv[0] is the name of the program itself
#sys.argv[1] will be the regex specified at the command line

regex = "foot"

#for every line passed into the script
for line in sys.stdin:
	#if it matches the regex, write it to stdout
	if re.search(regex, line):
		sys.stdout.write(line)
		
#line_count.py
import sys

count = 0 
for line in sys.stdin:
	count += 1
	
#print goes to sys.stdout
print count

#most common words
import sys 
from collection import Counter 

#pass in number of words as first argument 
try:
	num_words = int(sys.argv[1])
except:
	print "usage: most_common_words.py num_words"
	sys.exit(1) #non-zero exit code indicates error 

counter = Counter(word.lower()        #lower case words 
				  for line in sys.stdin 
				  for word in line.strip().split() #split on spaces
				  if word) 				#skip empty words 
				  
for word, count in counter.most_common(num_words):
	sys.stdout.write(str(count))
	sys.stdout.write("\t")
	sys.stdout.write(word)
	sys.stdout.write("\n")
"""

"""Reading files"""

def get_domain(email_address):
	"""split on '@' and return the last piece"""
	return email_address.lower().split("@")[-1]
	
"""with open('email_addresses.txt','r') as f:
	domain_counts = Counter(get_domain(line.strip())
							for line in f 
							if "@" in line)"""
							

"""Example: O'Reilly Data Books"""
#you don't have to split the url like this unless it needs to fit in a book
url = "http://shop.oreilly.com/category/browse-subjects/data.do?sortby=publicationDate&page=1"
soup = BeautifulSoup(requests.get(url).text,'html5lib')

#Example relevant HTML for one book in the source code
"""<td class="thumbtext">
	<div class="thumbcontainer">
		<div class="thumbdiv">
			<a href="/product/</a<>img src="..."/>9781118903407.do">
			</a>
		</div>
	</div>
	<div class="widthchange">
		<div class="thumbheader">
			<a href="/product/9781118903407.do">Getting a Big Data Job
For Dummies</a>
		</div>
		<div class="AuthorName">By Jason Williamson</div>
		<span class="directorydate"> December 2014 </span>
		<div <divstyilde==""1c4l6e3a5r0:"b>oth;">
			<div id="146350">
				<span class="pricelabel">
									Ebook:
									
									<span
class="pr</iscpea"n>>&nbsp;$29.99</span>
				</span>
			</div>
		</div>
	</div>
</td> """

def is_video(td):
	"""It's a video if it has exactly one pricelabel, and if 
	the stripped text inside that pricelabel starts with 'Video'"""
	pricelabels = td('span','pricelabel')
#	return (len(pricelabels) ==1 and
#			pricelabels[0].text.strip().startswith("Video"))
	return (pricelabels[0].text.strip().startswith("Video"))
			
def book_info(td):
	"""given a BeautifulSoup <td> Tag representing a book, 
	extract the book's details and return a dict"""
	
	title = td.find("div","thumbheader").a.text 
	by_author = td.find('div','AuthorName').text
	authors = [x.strip() for x in re.sub("^By ", "", by_author)
	.split(",")]
	isbn_link = td.find("div","thumbheader").a.get("href")
	isbn = re.match("/product/(.*)\.do",isbn_link).groups()[0]
	#isbn = isbn_link.split("/")[2].split(".")[0]
	date = td.find("span","directorydate").text.strip()
	
	return {
		"title" : title, 
		"authors" : authors, 
		"isbn" : isbn, 
		"date" : date
	}
	
#Scraping exercise - number of books on each page
"""base_url = "http://shop.oreilly.com/category/browse-subjects/" + \
"data.do?sortby=publicationDate&page="
books = []
NUM_PAGES = 31 # at the time of writing, probably more by now
for page_num in range(1, NUM_PAGES + 1):
print "souping page", page_num, ",", len(books), " found so
far"
url = base_url + str(page_num)
soup = BeautifulSoup(requests.get(url).text, 'html5lib')
for td in soup('td', 'thumbtext'):
if not is_video(td):
books.append(book_info(td))
# now be a good citizen and respect the robots.txt
sleep(30)
"""
							
def get_year(book):
	"""book["date"] looks like 'November 2014' so we need to
	split on the space and then take the second piece"""
	return int(book["date"].split()[1])
	
#Number of books published each year
"""
#2014 is the last completet year of data (when I ran this)
year_counts = Counter(get_year(book) for book in books
							if get_year(book) <= 2014)
years = sorted(year_counts)
book_counts = [year_counts[year] for year in years]
plt.plot(years,books_counts)
plt.show()							
"""
							
"""Twitter Search API"""

"""
twitter = Twython(CONSUMER_KEY, CONSUMER_SECRET)

# search for tweets containing the phrase "data science"
for status in twitter.search(q='"data science"')["statuses"]:
	user = status["user"]["screen_name"].encode('utf-8')
	text = status["text"].encode('utf-8')
	print user, ":", text
	print
"""
	
"""Twitter Streaming API"""

#Appending data to a global variable is pretty poor form 
#but it makes the example much simpler
tweets = []

class MyStreamer(TwythonStreamer):
	"""our own subclass of TwythonStreamer that specifies
	how to interact with the stream"""
	def on_success(self, data):
		"""what do we do when twitter sends us data?
		here data will be a Python dict representing a tweet"""

		# only want to collect English-language tweets
		if data['lang'] == 'en':
			tweets.append(data)
			print "received tweet #", len(tweets)
		
		# stop when we've collected enough
		if len(tweets) >= 50:
			self.disconnect()

	def on_error(self, status_code, data):
		print status_code, data
		self.disconnect()


stream = MyStreamer(CONSUMER_KEY, CONSUMER_SECRET,
					ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

# starts consuming public statuses that contain the keyword 'data'
#stream.statuses.filter(track='data')
# if instead we wanted to start consuming a sample of *all* public statuses
# stream.statuses.sample()

"""Exploring Data"""

#Histogram creating algorithm for a random set of data
def bucketize(point,bucket_size):
	"""floor the point to the next lower multiple of bucket size"""
	return bucket_size * math.floor(point/bucket_size)
	
def make_histogram(points, bucket_size):
	"""buckets the points and counts how many in each bucket"""
	return Counter(bucketize(point, bucket_size) for point in points)

def plot_histogram(points, bucket_size, title=""):
	histogram = make_histogram(points, bucket_size)
	plt.bar(histogram.keys(), histogram.values(),width=bucket_size)
	plt.title(title)
	plt.show()
	
def random_normal():
	"""returns a random draw from a standard normal distribution"""
	return inverse_normal_cdf(random.random())
	
def correlation_matrix(data):
	"""returns the num_columns x num_columns matrix whose (i,j)th 
	entry is the correlation between columns i and j of data"""
	
	_, num_columns = shape(data)
	
	def matrix_entry(i,j):
		return correlation(get_column(data,i), get_column(data,j))
	
	return make_matrix(num_columns, num_columns, matrix_entry)
	
#Correlation matrix graph example
"""
data = [[2,3,5],[5,3,6],[7,6,5]]
_, num_columns = shape(data)
fig, ax = plt.subplots(num_columns,num_columns)

for i in range(num_columns):
	for j in range(num_columns):
	
		#scatter column_j on the x-axis vs column_i on the y-axis 
		if i!=j: ax[i][j].scatter(get_column(data,j),get_column(data,i))
		
		# unless i == j, in which case show the series name
		else: ax[i][j].annotate("series " + str(i), (0.5, 0.5),
									xycoords="axes fraction",
									ha="center", va="center")
		# then hide axis labels except left and bottom charts
		if i < num_columns - 1: ax[i][j].xaxis. set_visible(False)
		if j > 0: ax[i][j]. yaxis. set_visible(False)
	
# fix the bottom right and top left axis labels, which are wrong
#because
# their charts only have text in them
#ax[-1][ -1]. set_xlim(ax[0][ -1].get_xlim())
#ax[0][0].set_ylim (ax|[0][1].get_ylim())

plt.show()
"""

#Data cleaning and munging
def try_or_none(f):
	"""wraps f to return None if f raises an exception
	assumes f takes only one input"""
	def f_or_none(x):
		try: return f(x)
		except: return None 
		
	return f_or_none 

def parse_row(input_row, parsers):
	"""given a list of parsers (some of which may be None) 
	apply the appropriate one to each element of the input_row"""
	
	return [try_or_none(parser)(value) if parser is not None else value
			for value, parser in zip(input_row, parsers)]

def parse_rows_with(reader, parsers):
	"""wrap a reader to apply the parsers to each of its rows"""
	for row in reader:
		yield parse_row(row,parsers)
		
#Parsers for csv.DictReader
def try_parse_field(field_name, value, parser_dict):
	"""try to parse value using the appropriate function from
	parser_dict"""
	parser = parser_dict.get(field_name) #None if no such entry
	if parser is not None:
		return try_or_none(parser)(value)
	else:
		return value
		
#Manipulating data
def picker(field_name): 
	"""returns a function that picks a field out of a dict"""
	return lambda row: row[field_name]
	
def pluck(field_name, rows):
	"""turns a list of dicts into the list of field_name values"""
	return map(picker(field_name),rows)
	
	
def parse_dict(input_dict, parser_dict):
	return {field_name: try_parse_field(field_name, value,
	parser_dict)
			for field_name, value in input_dict.iteritems()}
			
"""Manipulating Data"""

"""Ex. Highest AAPL price"""
"""
max_aapl_price = max(row["price"]
				for row in data
				if row["price"] == "AAPL")
"""
				
"""Ex. Highest price for each stock"""
"""
# group rows by symbol
by_symbol = defaultdict(list)
for row in data:
	by_symbol[row["symbol"]].append(row)
# use a dict comprehension to find the max for each symbol
max_price_by_symbol = { symbol : max(row["closing_price"]
						for row in grouped_rows)
					for symbol, grouped_rows in by_symbol.iteritems()}
"""
	
def group_by(grouper, rows, value_transform=None):
	#key is output of grouper, value is list of rows
	grouped = defaultdict(list)
	for row in rows:
		grouped[grouper(row)].append(row)
		
	if value_transform is None:
		return grouped
	else:
		return {key: value_transform(rows)
				for key, rows in grouped.iteritems()}
				
"""Ex. Max price by symbol using 'group_by' function"""
"""
max_price_by_symbol = group_by(picker("symbol"),data, lambda rows: max(pluck("closing_price", rows)))
"""
def percent_price_change(yesterday,today):
	return today["price"]/yesterday["price"]-1
	
def day_over_day_changes(grouped_rows):
	#sort the rows by date 
	ordered = sorted(grouped_rows, key=picker("date"))
	
	#zip with an offset to get pairs of consecutive days
	return [{"stock": today["stock"],
			"date":today["date"],
			"change":percent_price_change(yesterday,today)}
			for yesterday,today in zip(ordered,ordered[1:])]
			
#to combine percent changes, we add 1 to each, multiply them, and 
#subtract 1
#To combine 10% and 20% => (1+10%)*(1+20%) - 1 = -12%
def combine_pct_changes(pct_change1, pct_change2):
	return (1+pct_change1)*(1+pct_change2) - 1
	
def overall_change(changes):
	return reduce(combine_pct_changes, pluck("change",changes))
	
"""Rescaling"""
def scale(data_matrix):
	"""returns the means and standard deviations of each column"""
	num_rows, num_cols = shape(data_matrix)
	means = [mean(get_column(data_matrix,j)) 
			for j in range(num_cols)]
	stdevs = [standard_deviation(get_column(data_matrix,j))
			for j in range(num_cols)]
	return means, stdevs
	
def rescale(data_matrix):
	"""rescales the input data so that each column has mean 0
	and standard deviation 1 leaves alone columns with no 
	deviation"""
	means, stdevs = scale(data_matrix)
	
	def rescaled(i,j):
		if stdevs[j] > 0:
			return (data_matrix[i][j]-means[j])/stdevs[j]
		else:
			return data_matrix[i][j]
			
	num_rows, num_cols = shape(data_matrix)
	return make_matrix(num_rows,num_cols, rescaled)
	
"""Principal Component Analysis"""
def de_mean_matrix(A):
	"""returns the result of subtracting from every value in A
	the mean value of its column. The resulting matrix has mean 0
	in every column"""
	nr, nc = shape(A)
	column_means, _ = scale(A)
	return make_matrix(nr,nc, lambda i,j: A[i][j] - column_means[j])
	
def direction(w):
	mag = magnitude(w)
	return [w_i/mag for w_i in w]
	
def directional_variance_i(x_i,w):
	"""the variance of the row x_i in the direction determined by w"""
	return dot(x_i,direction(w))**2
	
def directional_variance(X,w):
	"""the variance of the data in the direction determined w"""
	return sum(directional_variance_i(x_i,w)
			for x_i in X)
			
def directional_variance_gradient_i(x_i,w):
	"""the contribution of row x_i to the gradient of the 
	direction-w variance"""
	projection_length = dot(x_i,direction(w))
	return [2*projection_length*x_ij for x_ij in x_i]
	
def directional_variance_gradient(X,w):
	return vector_sum([directional_variance_gradient_i(x_i,w)
					for x_i in X])
					
def first_principal_component(X):
	guess = [1 for _ in X[0]]
	unscaled_maximizer = maximize_batch(
			partial(directional_variance,X), #is now a function of w
			partial(directional_variance_gradient,X), #function of w
			guess)
	return direction(unscaled_maximizer)
	
# here there is no "y", so we just pass in a vector of Nones
# and functions that ignore that input
def first_principal_component_sgd(X): #for stochastic gradient descent
	guess = [1 for _ in X[0]]
	unscaled_maximizer = maximize_stochastic(
		lambda x, _, w: directional_variance_i(x, w),
		lambda x, _, w: directional_variance_gradient_i(x, w),
		X,
		[None for _ in X], # the fake "y"
		guess)
	return direction(unscaled_maximizer)
	
def project(v,w):
	"""return the projection of v onto the direction w"""
	projection_length = dot(v,w)
	return scalar_multiply(projection_length,w)
	
def remove_projection_from_vector(v,w):
	"""projects v onto w and subtracts the result from v"""
	return vector_subtract(v,project(v,w))
	
def remove_projection(X,w):
	"""for each row of X 
	projects the row onto w, subtracts the result from the row"""
	return [remove_projection_from_vector(x_i,w) for x_i in X]
	
def principal_component_analysis(X,num_components):
	components = []
	for _ in range(num_components):
		component = first_principal_component(X)
		component.append(component)
		X = remove_projection(X,component)
		
	return components
	
def transform_vector(v,components):
	return [dot(v,w) for w in components]
	
def transform(X,components):
	return [transform_vector(x_i,components) for x_i in X]
	
"""PCA example"""
"""
df pd.DataFrame(data=np.random.normal(0,1(20,10)))
pca = PCA(n_components=5)
pca.fit(df)
pca.components_ #gives the components which are the directions
				#of the max variance in the data
pca.fit_transform(df) #Gives the dimensionally reduced data 
"""
	
def split_data(data, prob):
	"""split data into fractions [prob, 1-prob]"""
	results = [],[]
	for row in data:
		results[0 if random.random() < prob else 1].append(row)
	return results
	
def train_test_split(x,y,test_pct):
	data = zip(x,y) 		#pair corresponding values
	train,test = split_data(data,1-test_pct) #split the data set of pairs
	x_train, y_train = zip(*train) 	#magical un-zip trick
	x_test, y_test = zip(*test)
	return x_train, x_test, y_train, y_test
	
"""k-Nearest Neighbours"""
	
def raw_majority_vote(labels):
	votes = Counter(labels)
	winner, _ = votes.most_common(1)[0]
	return winner
	
def majority_vote(labels):
	"""assumes that labels are ordered from nearest to farthest"""
	vote_counts = Counts(labels)
	winner, winner_count = vote_counts.most_common(1)[0]
	num_winners = len([count for count in vote_counts.values()
					if count == winner.count])
					
	if num_winners == 1:
		return winner			#unique winner so return it
	else:
		return majority_vote(labels[:-1]) #try again without the farthest
		
def knn_classify(k,labeled_points, new_points):
	"""each labeled point should be a pair (point, label)"""
	#order the labeled points from nearest to farthest 
	by_distance = sorted(labeled_points, key=lambda (point,_):
							distance(point,new_point))
							
	#find the labels for the k closest
	k_nearest_labels = [label for _, label in by_distance[:k]]
	
	#and let them vote
	return majority_vote(k_nearest_labels)
	
#Plotting favourite programming languages and corresponding cities

#Sample data
"""
cities = [([-122.3 , 47.53], "Python"), # Seattle
			([ -96.85, 32.85], "Java"), # Austin
			([ -89.33, 43.13], "R"), # Madison
			# ... and so on
			]
			
#key is language, value is pair (longitudes, latitudes)
plots = { "Java" : ([], []), "Python" : ([], []), "R" : ([], []) }

# we want each language to have a different marker and color
markers = { "Java" : "o", "Python" : "s", "R" : "^" }
colors = { "Java" : "r", "Python" : "b", "R" : "g" }

for (longitude, latitude), language in cities:
	plots[language][0].append(longitude)
	plots[language][1].append(latitude)
	
#create a scatter series for each language
for language, (x,y) in plots.iteritems():
	plt.scatter(x,y,color=colors[language],marker=markers[language],
				label=language, zorder=10)
				
#plot_state_borders(plt)		#pretend we have a function that does this

plt.legend(loc=0)			#let matplotlib choose the location
plt.axis([-130,-60,20,55])  #set the axes

plt.title("Favorite Programming Languages")
plt.show()
"""

#Routine to predict each cities preferred language using its 
#neighbours other than itself

#try several different values for k	
"""
for k in [1, 3, 5, 7]:
	num_correct = 0
	
	for city in cities:
		location, actual_language = city
		other_cities = [other_city
						for other_city in cities
						if other_city != city]

		predicted_language = knn_classify(k, other_cities,
										location)
		if predicted_language == actual_language:
			num_correct += 1
	print k, "neighbor[s]:", num_correct, "correct out of", len(cities)
"""

#Classifying each region according to the nearest neighbour scheme
"""
plots = { "Java" : ([], []), "Python" : ([], []), "R" : ([], []) }

k = 1 # or 3, or 5, or ...

for longitude in range(-130, -60):
	for latitude in range(20, 55):
		predicted_language = knn_classify(k, cities, [longitude,
										latitude])
		plots[predicted_language][0].append(longitude)
		plots[predicted_language][1].append(latitude)
"""

#generating random points
def random_point(dim):
	return [random.random() for _ in range(dim)]
	
def random_distances(dim,num_pairs):
	return [distance(random_point(dim), random_point(dim))
			for _ in range(num_pairs)]
			
#Compute 10,000 distances in 1 to 100 dimensions
#Then compute the average distance between points and the minimum
#distance between points in each dimension
"""
dimensions = range(1,101)

avg_distances = []
min_distances = []

random.seed(0)
for dim in dimensions:
	distances = random_distances(dim,10000) #10,000 random pairs
	avg_distances.append(mean(distances))   #track the average
	min_distances.append(min(distances))    #track the minimum
	
min_avg_ratio = [min_dist/avg_dist for min_dist, avg_dist in
				zip(min_distances,avg_distances)]
"""



		
			
			
						
						
	
	
	
	
	
