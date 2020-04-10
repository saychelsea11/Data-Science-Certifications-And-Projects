# Jupyter notebook descriptions

1. **Cleaning_and_wrangling_Seattle_collision_data.ipynb**: This is the first notebook in chronological order. It takes in the initial dataset and goes through several cleaning steps. Some variables are ommitted right at the beginning as they didn't provide much insight for analysis or modeling purposes. Following that, the columns are renamed and missing values are handled following by some outlier analysis. 

2. **binary_class_conversion_handling_unknown_category.ipynb**: Contains some more handling of missing values as well as feature engineering for weather related variables as well as the severity target variable. 

3. **adding_neighborhood_speed_and_road_variables.ipynb**: Separate notebook for acquiring additional variables using API transactions. Used the Tomtom API to extract the average free flow speed variable. Used the HERE API to acquire the road length and road congestion features. 

4. **exploratory_data_analysis.ipynb**: Possibly the largest file in this directory, this notebook contains extensive exploratory data analysis for the remaining variables. Different types of analysis is conducted such as geographical analysis using location coordinates, analysis for weather variables, analysis for road variables as well as timeseries analysis. 

5. **Statistical_analysis.ipynb**: Contains statistical inference for both categorical and numerical variables with regards to the severity target variable. For comparing 2 categorical variables, chi-squared method is used to perform hypothesis testing and prove statistical significance of variables. For a categorical and numberical variable, permutation replicates as well as the t-test methods are used for hypothesis testing. 

6. **modeling_predicting_severe_collisions.ipynb**: Notebook used to create the baseline logistic regression model and perform evaluation and optimization in an iterative process. The final model is created using LightGBM. 
