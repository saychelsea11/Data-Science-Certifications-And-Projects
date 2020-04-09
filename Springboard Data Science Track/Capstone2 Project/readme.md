# Identifying keywords and topics from questions on Stack Exchange sites

## Contents

- **Data**: Consists of 2 files:

1. The first one is a CSV file which contains a table for comparing LDA coherence scores across model parameters. 
2. The second file is a text file which contains a link to the dataset. Since the complete dataset is over 400MB, it could not be added to Github as the limit is 100MB. 
- **Notebooks**: There are a total of 6 Python Jupyter Notebooks in the following order: 

1. *stack_exchange_text_data_cleaning_wrangling.ipynb* - Performs intial cleaning by removing HTML tags from the text as well as removing newline characters. 
2. *word_tokenization_stopword_removal_tokenization.ipynb* -  
3. *stack_exchange_EDA_and_POS_tagging.ipynb* - 
4. *topic_modeling_lda.ipynb* - 
5. *prediction_and_evluation.ipynb*	-
6. *lda_word2vec_gensim.ipynb* - 

- **Reports**: 

## Goal

- To identify keywords. topics and summaries  from user questions on Stack Exchange.
- Use the results to optimize sorting of user queries, providing quicker and more relevant answers.

## Approach 

- Use topic modeling to identify keywords related to questions using Latent Dirichlet Allocation (LDA).
- Analyze the text data using Natural Language Processing (NLP) techniques to find insights.


