
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 4 - Document Similarity & Topic Modelling

# ## Part 1 - Document Similarity
# 
# For the first part of this assignment, you will complete the functions `doc_to_synsets` and `similarity_score` which will be used by `document_path_similarity` to find the path similarity between two documents.
# 
# The following functions are provided:
# * **`convert_tag:`** converts the tag given by `nltk.pos_tag` to a tag used by `wordnet.synsets`. You will need to use this function in `doc_to_synsets`.
# * **`document_path_similarity:`** computes the symmetrical path similarity between two documents by finding the synsets in each document using `doc_to_synsets`, then computing similarities using `similarity_score`.
# 
# You will need to finish writing the following functions:
# * **`doc_to_synsets:`** returns a list of synsets in document. This function should first tokenize and part of speech tag the document using `nltk.word_tokenize` and `nltk.pos_tag`. Then it should find each tokens corresponding synset using `wn.synsets(token, wordnet_tag)`. The first synset match should be used. If there is no match, that token is skipped.
# * **`similarity_score:`** returns the normalized similarity score of a list of synsets (s1) onto a second list of synsets (s2). For each synset in s1, find the synset in s2 with the largest similarity value. Sum all of the largest similarity values together and normalize this value by dividing it by the number of largest similarity values found. Be careful with data types, which should be floats. Missing values should be ignored.
# 
# Once `doc_to_synsets` and `similarity_score` have been completed, submit to the autograder which will run `test_document_path_similarity` to test that these functions are running correctly. 
# 
# *Do not modify the functions `convert_tag`, `document_path_similarity`, and `test_document_path_similarity`.*

# In[11]:

import numpy as np
import nltk
from nltk.corpus import wordnet as wn
import pandas as pd


def convert_tag(tag):
    """Convert the tag given by nltk.pos_tag to the tag used by wordnet.synsets"""
    
    tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
    try:
        return tag_dict[tag[0]] #Maps the first letter of the input pos tag to 'tag_dict'
    except KeyError:
        return None #Return nothing if the tag does not exist in this set


def doc_to_synsets(doc):
    """
    Returns a list of synsets in document.

    Tokenizes and tags the words in the document doc.
    Then finds the first synset for each word/tag combination.
    If a synset is not found for that combination it is skipped.

    Args:
        doc: string to be converted

    Returns:
        list of synsets

    Example:
        doc_to_synsets('Fish are nvqjp friends.')
        Out: [Synset('fish.n.01'), Synset('be.v.01'), Synset('friend.n.01')]
    """
    
    doc_tok = nltk.word_tokenize(doc)
    doc_pos = nltk.pos_tag(doc_tok)
    doc_pos = [y for x,y in doc_pos]
    tag_conv = list(map(convert_tag,doc_pos))
    #print (doc_tok,len(doc_tok))
    #print (tag_conv,len(doc_tok))
    syn_list = []
    count = 0
    
    for i,j in zip(doc_tok,tag_conv):
        try:
            syn_list.append(wn.synsets(i,j)[0])
        except:
            continue

    #print (syn_list)
    #print ("")
    return syn_list


def similarity_score(s1, s2):
    """
    Calculate the normalized similarity score of s1 onto s2

    For each synset in s1, finds the synset in s2 with the largest similarity value.
    Sum of all of the largest similarity values and normalize this value by dividing it by the
    number of largest similarity values found.

    Args:
        s1, s2: list of synsets from doc_to_synsets

    Returns:
        normalized similarity score of s1 onto s2

    Example:
        synsets1 = doc_to_synsets('I like cats')
        synsets2 = doc_to_synsets('I like dogs')
        similarity_score(synsets1, synsets2)
        Out: 0.73333333333333339
    """
    max = 0
    large_sims = []
    count = 0

    for i in s1:
        for j in s2:
            try:
                sim = i.path_similarity(j)
                #print (sim)
                if sim > max:
                    max = sim
                    #print (sim)
            except:
                continue
        #print ("")
        if max!=0:
            large_sims.append(max)
            count = count + 1
            max = 0
        else:
            pass
        
    #print (count)
    #print ("large sims",large_sims)
    #print (sum(large_sims)/count)
    return sum(large_sims)/count


def document_path_similarity(doc1, doc2):
    """Finds the symmetrical similarity between doc1 and doc2"""

    synsets1 = doc_to_synsets(doc1)
    synsets2 = doc_to_synsets(doc2)
    
    #print ("")
    #print (synsets1)
    #print (synsets2)
    #print ("")
    #print (similarity_score(synsets1, synsets2))
    #print (similarity_score(synsets2, synsets1))
    #print ("")

    return (similarity_score(synsets1, synsets2) + similarity_score(synsets2, synsets1)) / 2


# ### test_document_path_similarity
# 
# Use this function to check if doc_to_synsets and similarity_score are correct.
# 
# *This function should return the similarity score as a float.*

# In[12]:

def test_document_path_similarity():
    doc1 = 'This is a function to test document_path_similarity.'
    doc2 = 'Use this function to see if your code in doc_to_synsets     and similarity_score is correct!'
    #doc1 = 'I like cats'
    #doc2 = 'I like dogs'
    #doc1 = 'Ali saw the man with the telescope.'
    #doc2 = 'The man with the telescope was seen by Ali.'
    return document_path_similarity(doc1, doc2)


# <br>
# ___
# `paraphrases` is a DataFrame which contains the following columns: `Quality`, `D1`, and `D2`.
# 
# `Quality` is an indicator variable which indicates if the two documents `D1` and `D2` are paraphrases of one another (1 for paraphrase, 0 for not paraphrase).

# In[4]:

# Use this dataframe for questions most_similar_docs and label_accuracy
paraphrases = pd.read_csv('paraphrases.csv')
paraphrases.head()


# ___
# 
# ### most_similar_docs
# 
# Using `document_path_similarity`, find the pair of documents in paraphrases which has the maximum similarity score.
# 
# *This function should return a tuple `(D1, D2, similarity_score)`*

# In[11]:

def most_similar_docs():
    
    paraphrases["similarity"] = paraphrases.apply(lambda x: document_path_similarity(x[1],x[2]),axis=1)
    sorted_paraphrases = paraphrases.sort_values(by="similarity",ascending=False)
    
    return sorted_paraphrases.iloc[0,1],sorted_paraphrases.iloc[0,2],sorted_paraphrases.iloc[0,3]


# ### label_accuracy
# 
# Provide labels for the twenty pairs of documents by computing the similarity for each pair using `document_path_similarity`. Let the classifier rule be that if the score is greater than 0.75, label is paraphrase (1), else label is not paraphrase (0). Report accuracy of the classifier using scikit-learn's accuracy_score.
# 
# *This function should return a float.*

# In[19]:

def label_accuracy():
    from sklearn.metrics import accuracy_score

    paraphrases["similarity"] = paraphrases.apply(lambda x: document_path_similarity(x[1],x[2]),axis=1)
    paraphrases["label"] = (paraphrases["similarity"]>0.75).map(int)
    #print (paraphrases)
    score = accuracy_score(paraphrases["Quality"],paraphrases["label"])
    return score


# ## Part 2 - Topic Modelling
# 
# For the second part of this assignment, you will use Gensim's LDA (Latent Dirichlet Allocation) model to model topics in `newsgroup_data`. You will first need to finish the code in the cell below by using gensim.models.ldamodel.LdaModel constructor to estimate LDA model parameters on the corpus, and save to the variable `ldamodel`. Extract 10 topics using `corpus` and `id_map`, and with `passes=25` and `random_state=34`.

# In[2]:

import pickle
import gensim
from sklearn.feature_extraction.text import CountVectorizer

# Load the list of documents
with open('newsgroups', 'rb') as f:
    newsgroup_data = pickle.load(f)

# Use CountVectorizor to find three letter tokens, remove stop_words, 
# remove tokens that don't appear in at least 20 documents,
# remove tokens that appear in more than 20% of the documents
vect = CountVectorizer(min_df=20, max_df=0.2, stop_words='english', 
                       token_pattern='(?u)\\b\\w\\w\\w+\\b')
# Fit and transform
X = vect.fit_transform(newsgroup_data)

# Convert sparse matrix to gensim corpus.
corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)

# Mapping from word IDs to words (To be used in LdaModel's id2word parameter)
id_map = dict((v, k) for k, v in vect.vocabulary_.items())
#print (vect.vocabulary_)
#print ("")
#print (id_map)


# In[4]:

# Use the gensim.models.ldamodel.LdaModel constructor to estimate 
# LDA model parameters on the corpus, and save to the variable `ldamodel`

# Your code here:
ldamodel = gensim.models.ldamodel.LdaModel(corpus=corpus,num_topics=10,id2word=id_map,passes=25, random_state=34)
#print (ldamodel.print_topics())


# ### lda_topics
# 
# Using `ldamodel`, find a list of the 10 topics and the most significant 10 words in each topic. This should be structured as a list of 10 tuples where each tuple takes on the form:
# 
# `(9, '0.068*"space" + 0.036*"nasa" + 0.021*"science" + 0.020*"edu" + 0.019*"data" + 0.017*"shuttle" + 0.015*"launch" + 0.015*"available" + 0.014*"center" + 0.014*"sci"')`
# 
# for example.
# 
# *This function should return a list of tuples.*

# In[ ]:

def lda_topics():
    
    return ldamodel.print_topics()


# ### topic_distribution
# 
# For the new document `new_doc`, find the topic distribution. Remember to use vect.transform on the the new doc, and Sparse2Corpus to convert the sparse matrix to gensim corpus.
# 
# *This function should return a list of tuples, where each tuple is `(#topic, probability)`*

# In[6]:

new_doc = ["\n\nIt's my understanding that the freezing will start to occur because of the\ngrowing distance of Pluto and Charon from the Sun, due to it's\nelliptical orbit. It is not due to shadowing effects. \n\n\nPluto can shadow Charon, and vice-versa.\n\nGeorge Krumins\n-- "]


# In[8]:

def topic_distribution():
    
    # Fit and transform
    X2 = vect.transform(new_doc)

    # Convert sparse matrix to gensim corpus.
    corpus2 = gensim.matutils.Sparse2Corpus(X2, documents_columns=False)
    
    # Mapping from word IDs to words (To be used in LdaModel's id2word parameter)
    id_map_2 = dict((v, k) for k, v in vect.vocabulary_.items())
    
    ldamodel = gensim.models.ldamodel.LdaModel(corpus=corpus2,num_topics=10,id2word=id_map_2,passes=25, random_state=34)
    return list(ldamodel.get_document_topics(corpus2)) [0]

topic_distribution()


# ### topic_names
# 
# From the list of the following given topics, assign topic names to the topics you found. If none of these names best matches the topics you found, create a new 1-3 word "title" for the topic.
# 
# Topics: Health, Science, Automobiles, Politics, Government, Travel, Computers & IT, Sports, Business, Society & Lifestyle, Religion, Education.
# 
# *This function should return a list of 10 strings.*

# In[ ]:

def topic_names():
    
    topics = ["Health","Science","Automobiles","Politics","Government","Travel","Computers","Business","Education","Religion"]
    
    return topics

