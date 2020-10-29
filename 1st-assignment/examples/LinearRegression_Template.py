# =============================================================================
# HOMEWORK 1 - Supervised learning
# LINEAR REGRESSION ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================



# From 'sklearn' library, we need to import:
# 'datasets', for loading our data
# 'metrics', for measuring scores
# 'linear_model', which includes the LinearRegression() method
# From 'scipy' library, we need to import:
# 'stats', which includes the spearmanr() and pearsonr() methods for computing correlation
# Additionally, we need to import 
# 'pyplot' from package 'matplotlib' for our visualization purposes
# 'numpy', which implementse a wide variety of operations
# =============================================================================

# IMPORT NECESSARY LIBRARIES HERE
from sklearn import datasets, metrics, linear_model, model_selection
from scipy import stats
from matplotlib import pyplot
import numpy
# =============================================================================




# Load diabetes data from 'datasets' class
# =============================================================================

# ADD COMMAND TO LOAD DATA HERE
diabetes = datasets.load_diabetes()

# =============================================================================



# Get samples from the data, and keep only the features that you wish.
# =============================================================================

# Load just 1 feature for simplicity and visualization purposes...
# X: features
# Y: target value (prediction target)
X = diabetes.data[:, numpy.newaxis, 2]  # Here we load the bmi feature (?)
y = diabetes.target

# =============================================================================


# Create linear regression model. All models behave differently, according to
# their own, model-specific parameter values. In our case, however, the linear
# regression model does not have any substancial parameters to tune. Refer
# to the documentation of this technique for more information.
# =============================================================================


# ADD COMMAND TO CREATE LINEAR REGRESSION MODEL HERE
linearRegressionModel = linear_model.LinearRegression()


# =============================================================================



# Split the dataset that we have into two subsets. We will use
# the first subset for the training (fitting) phase, and the second for the evaluation phase.
# By default, the train set is 75% of the whole dataset, while the test set makes up for the rest 25%.
# This proportion can be changed using the 'test_size' or 'train_size' parameter.
# Alsao, passing an (arbitrary) value to the parameter 'random_state' "freezes" the splitting procedure 
# so that each run of the script always produces the same results (highly recommended).
# Apart from the train_test_function, this parameter is present in many routines and should be
# used whenever possible.
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=0)



# Let's train our model.
# =============================================================================

# ADD COMMAND TO TRAIN YOUR MODEL HERE
linearRegressionModel.fit(x_train, y_train)
# =============================================================================




# Ok, now let's predict the output for the test input set
# =============================================================================

# ADD COMMAND TO MAKE A PREDICTION HERE
y_predicted = linearRegressionModel.predict(x_test)

# =============================================================================



# Time to measure scores. We will compare predicted output (resulting from input x_test)
# with the true output (i.e. y_test).
# You can call 'pearsonr()' or 'spearmanr()' methods for computing correlation,
# 'mean_squared_error()' for computing MSE,
# 'r2_score()' for computing r^2 coefficient.
# =============================================================================

# ADD COMMANDS TO EVALUATE YOUR MODEL HERE (AND PRINT ON CONSOLE)
correlation = stats.spearmanr(y_test, y_predicted)
mean_squared_error = metrics.mean_squared_error(y_test, y_predicted)
r2_score = metrics.r2_score(y_test, y_predicted)
print(f'Correlation: {correlation.correlation:.2f}')
print(f'Mean Squared Error: {mean_squared_error:.2f}')
print(f'R2 Score: {r2_score:.2f}')

# =============================================================================




# Plot results in a 2D plot (scatter() plot, line plot())
# =============================================================================

# ADD COMMANDS FOR VISUALIZING DATA (SCATTER PLOT) AND REGRESSION MODEL
pyplot.scatter(x_test, y_test,  color='red')
pyplot.plot(x_test, y_predicted, color='black')

pyplot.xticks(())
pyplot.yticks(())

# Display 'ticks' in x-axis and y-axis
pyplot.xticks()
pyplot.yticks()

# Show plot
pyplot.show()

# =============================================================================
