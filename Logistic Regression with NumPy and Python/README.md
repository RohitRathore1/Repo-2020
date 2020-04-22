## Logistic regression with NumPy and Python
Logistic Regression is simple and easy but one of the widely used binary classification algorithm on the field of machine learning. It is easy to execute, and it works well in many situations. Like other machine learning algorithms, the knowledge of statistic, linear algebra and calculus is needed to understand this algorithm. In this topic, I will see how to implement the Logistic regression algorithm from scratch in Python using NumPy only. Coding Logistic Regression is not so difficult but its a bit tricky. Let's see.

## Objectives:
In this topic, I am going to focus on three learning objectives:
- Implement the gradient descent algorithm from scratch.
- Perform logistic regression with NumPy and Python.
- Create data visualizations with Matplotlib and Seaborn
By the end of this topic, I will be able to build logistic regression models from scratch using NumPy and Python, without the use of machine learning frameworks such as scikit-learn and statsmodels.

## Structure:
# Task 1: Introduction and Project Overview
- Introduction to the data set and the problem overview.
- See a demo of the final product I will build by the end of this topic.

# Task 2: Load the Data and Import Libraries
- Load the dataset using [pandas](https://pandas.pydata.org/).
- Import essential modules and helper functions from [NumPy](https://numpy.org/) and [Matplotlib](https://matplotlib.org/).
- Explore the pandas dataframe using the head() and info() functions.

# Task 3: Visualize the Data
- Before starting on any task, it is often useful to understand the data by visualizing it.
- For this dataset, I will use a scatter plot using [Seaborn](https://seaborn.pydata.org/) to visualize the data, since it has only two variables: scores for test 1, scores for test 2.

# Task 4: Define the Logistic Sigmoid Function 𝜎(𝑧)
- I can interpret the output of the logistic sigmoid function as a probability, since this function outputs in the range 0 to 1 for any input.
- I can threshold the function at 50% to make my classification.
- If the output is greater than or equal to 0.5, I will classify it as passed, and less than 0.5 as failed.
- The maximal uncertainty, I will easily see if I plug in 0 as the input. So when the model is most uncertain it tells me the data point has a 50% probability of being in either of the classes.
- I'm going to be using this function to make my predictions based on the input.

# Task 5: Compute the Cost Function 𝐽(𝜃) and Gradient
- Now that I have defined the logistic sigmoid, I can go ahead and define the objective function for logistic regression.
- The mathematics of how I arrived at the result is beyond the scope of this topic. But I highly recommend you do some reading on your own time.
- I can use the standard tool from convex optimization, the simplest of which is gradient descent to minimize the cost function.

# Task 6: Cost and Gradient at Initialization
- Before doing gradient descent, never forget to do feature scaling for a multivariate problem.
- Initialize the cost and gradient before any optimization steps.

# Task 7: Implement Gradient Descent from scratch in Python
- Recall that the parameters of my model are the 𝜃_j values.
- These are the values I will adjust to minimize the cost J(𝜃).
- One way to do this is to use the batch gradient descent algorithm.
- In batch gradient descent, each iteration performs the following update.
- With each step of gradient descent, the parameters 𝜃_j come closer to the optimal values that will achieve the lowest cost J(𝜃).
- Since I already have a function for computing the gradient previously, let’s not repeat the calculation and add on an alpha term here to update Θ.
- Let’s now actually run gradient descent using my data and run it for 200 iterations.
- The alpha parameter controls how big or small of a step I take in the direction of steepest slope. Set it too small, and my model may take a very long time to converge or never converge. Set it too large and my model may overshoot and never find the minimum.

# Task 8: Plotting the Convergence of 𝐽(𝜃)
- Let’s plot how the cost function varies with the number of iterations.
- When I ran gradient descent previously, it returns the history of J(𝜃) values in a vector “costs”.
- I will now plot the J values against the number of iterations.

# Task 9: Plotting the Decision Boundary
- Let's over the scatterplot from Task 3 with the learned logistic regression decision boundary.

# Task 10: Predictions Using the Optimized 𝜃 Values
- In this final task, let’s use my final values for 𝜃 to make predictions.
