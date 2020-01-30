from __future__ import print_function

import argparse
import json
import pprint
import requests
import sys
import urllib
import sqlite3


# This client code can run on Python 2.x or 3.x.  Your imports can be
# simpler if you only need one of those.
try:
    # For Python 3.0 and later
    from urllib.error import HTTPError
    from urllib.parse import quote
    from urllib.parse import urlencode
except ImportError:
    # Fall back to Python 2's urllib2 and urllib
    from urllib2 import HTTPError
    from urllib import quote
    from urllib import urlencode

CLIENT_ID = 'eJEcCIzJ5bqc0QpRGHSVCw'
CLIENT_SECRET = 'EaWwzqMVy2QsGKijCsGhRykDIVQnXiY9GnGFQ1g4SwZKCU510mwxvUnNoSSk442r'

# API constants, you shouldn't have to change these.
API_HOST = 'https://api.yelp.com'
SEARCH_PATH = '/v3/businesses/search'
BUSINESS_PATH = '/v3/businesses/'  # Business ID will come after slash.
TOKEN_PATH = '/oauth2/token'
GRANT_TYPE = 'client_credentials'

# Defaults for our simple example.
DEFAULT_TERM = 'dinner'
DEFAULT_LOCATION = 'San Francisco, CA'
SEARCH_LIMIT = 10

data = urlencode({
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'grant_type': GRANT_TYPE,
    })
	
headers = {
        'content-type': 'application/x-www-form-urlencoded',
    }

oauth = 'https://api.yelp.com/oauth2/token'
#url = 'https://api.yelp.com/v2/search?term=food&location=San+Francisco'
url = 'https://api.yelp.com/v2/search'

#response = requests.request('POST', oauth, data, headers)
response = requests.post(oauth, data, headers)
#print (response.json())
bearer_token = response.json()['access_token']
#print (bearer_token)
headers_bearer = {
        'Authorization': 'Bearer %s' % bearer_token,
    }
'''
url_params = {
        'term': 'hookah',
        'location': 'Greensboro,+NC',
        'limit': 5
    }
	'''

bus = []	
temp = [] 

city = raw_input('Please enter the search city: ')

num = raw_input('Please enter the number of search results to retrieve: ')
num = int(num)

if num > 50:
	rem = num%50
	loops = num//50
	loop_end = num - rem
	#print (loop_end)
	
	for i in range(50,loop_end+50,50):
		if i == 50:
			url = API_HOST + SEARCH_PATH + '?term=Pizza&location=' + city + '&limit=' + str(i)
			response = requests.get(url,headers=headers_bearer)
			json_res = response.json()
			bus = [x for x in json_res['businesses']]
			#print (len(bus))
		else: 
			#print (i)
			url = API_HOST + SEARCH_PATH + '?term=Pizza&location=' + city + '&limit=50&offset=' + str(i-50)
			response = requests.get(url,headers=headers_bearer)
			json_res = response.json()
			bus.extend(json_res['businesses'])
			#print (i)
else:
	url = API_HOST + SEARCH_PATH + '?term=Pizza&location=' + city + '&limit=' + str(num)
	response = requests.get(url,headers=headers_bearer)
	json_res = response.json()
	bus = [x for x in json_res['businesses']]
	
print (i)


url = API_HOST + SEARCH_PATH + '?term=Pizza&location=' + city + '&limit=' + str(rem) + '&offset=' + str(i)
response = requests.get(url,headers=headers_bearer)
json_res = response.json()
bus.extend(json_res['businesses'])


#response = requests.get('https://api.yelp.com/v3/businesses/search?term=Pizza&location=San Francisco, CA&limit=2',headers=headers_bearer)
#response = requests.get(url,headers=headers_bearer)
	
#response = requests.request('GET',url,headers=headers_bearer,params=url_params)
#response = requests.get('https://api.yelp.com/v3/autocomplete?text=del&latitude=37.786882&longitude=-122.399972',headers=headers_bearer)

#json_res = response.json()

#pprint.pprint (json_res['businesses'])


#region = json_res.get('region')
#businesses = json_res.get('businesses')
#print ((json_res.get('region'))[0])
#print (region)
#pprint.pprint(json_res)

conn = sqlite3.connect('pizza.sqlite')
cur = conn.cursor()
conn.text_factory = str

cur.execute('''DROP TABLE IF EXISTS Pizza ''')

cur.execute('''CREATE TABLE IF NOT EXISTS Pizza 
    (id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, Name TEXT, Rating FLOAT, 
     Price INTEGER, Category TEXT, Reviews TEXT)''')

#print (len(businesses))


for i in range(len(bus)):
	#pprint.pprint (businesses[i])
	
	#print (businesses[i]['id'])

	#try:
		#print (businesses[i]['price'])
	#except:
		#print ("Price not provided")
	
	
	try:
		cur.execute('''INSERT INTO Pizza (Name, Rating, Price, Category, Reviews) 
		VALUES (?,?,?,?,?)''',(bus[i]['name'], bus[i]['rating'], bus[i]['price'], bus[i]['categories'][0]['alias'], bus[i]['review_count']))
	except: 
		cur.execute('''INSERT INTO Pizza (Name, Rating, Price, Category, Reviews) 
		VALUES (?,?,?,?,?)''',(bus[i]['name'], bus[i]['rating'], None, bus[i]['categories'][0]['alias'], bus[i]['review_count']))
conn.commit()
cur.close()

    
#pprint.pprint (businesses[1])
"""
id = businesses[0]['id']
#id = id.replace('-',' ')

business_url = API_HOST + BUSINESS_PATH + id
#print (business_url)

response = requests.get(business_url,headers=headers_bearer)
#pprint.pprint(response.json())
#print ((response.json())['id'])
business = response.json()
print (business['id'])
#print (business['price'])
#print (len(business['price']))
"""




