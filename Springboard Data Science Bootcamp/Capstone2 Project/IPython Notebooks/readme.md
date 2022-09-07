# IPython notebooks used in this project, listed in sequential order: 

1. **stack_exchange_text_data_wrangling_initial_cleaning.ipynb** - Performs intial cleaning by removing HTML tags from the text as well as removing newline characters. 
2. **word_tokenization_stopword_removal_tokenization.ipynb** - Tokenizes the text data, changes text to lowercase and performs stopwords removal in an iterative manner. Also performs lemmatization which is a key feature engineering step for text data. 
3. **stack_exchange_EDA_and_POS_tagging.ipynb** - Explores the cleaned data, looks at word frequencies and conducts part of speech (POS) tagging. After POS analysis, only nouns are kept in the dataset while other tags are removed. 
4. **topic_modeling_lda.ipynb** - Notebook where several topic models are created and evaluated. Initial baseline LDA model is created using Python's *gensim* library. Model is then optimized by exploring parameters such as **number of topics, alpha** and **beta** and evaluated using various visualizations and coherence scores.
5. **prediction_and_evluation.ipynb**	- Final model is used to predict topics for train and test (unseen) documents and the results evaluated using tables and visualizations. 
6. **lda_word2vec_gensim.ipynb** - Additional notebook which explores *word2vec* which finds word similarities within the corpus by looking at the contexts in which particular words occur in the text. 
