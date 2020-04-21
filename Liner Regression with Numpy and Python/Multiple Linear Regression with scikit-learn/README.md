## Objectives:
In this topic, I am going to focus on the given learning objectives:
-  I will build and evaluate multiple linear regression models using Python. I will use [scikit-learn](https://scikit-learn.org/) to calculate the regression, while using [pandas](https://pandas.pydata.org/) for data management and [seaborn](https://seaborn.pydata.org/) for plotting. The data for this project consists of the very popular [Advertising](https://www.kaggle.com/ishaanv/ISLR-Auto#Advertising.csv) dataset to predict sales revenue based on advertising spending through media such as TV, radio, and newspaper.

By the end of this project, I will be able to:

- Build univariate and multivariate linear regression models using scikit-learn
- Perform Exploratory Data Analysis (EDA) and data visualization with seaborn
- Evaluate model fit and accuracy using numerical measures such as R² and RMSE
- Model interaction effects in regression using basic feature engineering techniques

## Structure:
# Task 1: Introduction and Overview
I will introduce the model I will be building as well the [Advertising](https://www.kaggle.com/ishaanv/ISLR-Auto#Advertising.csv) dataset for this project.

# Task 2: Load the Data
In this task, I will load the very popular Advertising dataset about various costs incurred on advertising by different media such as through TV, radio, newspaper, and the sales for a particular product. Next, I will briefly explore the data to get some basic information about what I am going to be working with.

# Task 3: Relationship between Features and Target
It is good practice to first visualize the data before proceeding with analysis and model building. In this task, I will apply seaborn to create scatter plots of each of the three features and the target. This will allow to make a qualitative observations about the linear or non-linear relationships between the features and the target.

# Task 4: Multiple Linear Regression Model
I will extend the simple linear regression model to include multiple features. My approach will give each predictor a separate slope coefficient in a single model. This way, I can avoid the drawbacks of fitting a separate simple linear model to each predictor. In this task, I used scikit-learn's LinearRegression( ) estimator to calculate the multiple regression coefficient estimates when TV, radio, and newspaper advertising budgets are used to predict sales revenue. Lastly, I will compare and contrast the coefficient estimates from multiple regression to those from simple linear regression.

# Task 5: Feature Selection
I will do all the predictors help to explain the target, or is only a subset of the predictors useful? I will address exactly this question in this task. I will use feature selection to determine which predictors are associated with the response, so as to fit a single model involving only those features. I will use R², the most common numerical measure of model fit and understand its limitations.

# Task 6: Model Evaluation Using Train/Test Split and Model Metrics
Assessing model accuracy is very similar to that of simple linear regression. My first step will be to split the data into a training set and a testing set using the train_test_split( ) helper function from sklearn.metrics. Next, I will create two separate models, one of which uses all predictors, while the other excludes newspaper. I fitted the training set to the estimator and make predictions on the testing set. Model fit and the accuracy of the predictions will be evaluated using R² and RMSE. Visual assessment of my models will involve comparing the residual behaviors and the prediction errors using [Yellowbrick](https://www.scikit-yb.org/en/latest/). Yellowbrick is an open source, pure Python project that extends the scikit-learn API with visual analysis and diagnostic tools. It is commonly used inside of a Jupyter Notebook alongside pandas data frames.

# Task 7: Interaction Effect (Synergy) in Regression Analysis
From my previous analysis of the residuals, I concluded that I need to incorporate interaction terms due to the non-additive relationship between the features and target. A simple method to extend my model to allow for interaction effects is to include a third feature by taking the product of the other two features in my model. This feature will have its separate slope coefficient which can be interpreted as the increase in the effectiveness of radio advertising for a one unit increase in TV advertising or vice versa.
