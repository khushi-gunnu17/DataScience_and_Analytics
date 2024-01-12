# from sklearn.linear_model import LinearRegression
# import numpy as np

# # Example data
# x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
# y = np.array([2, 4, 5, 4, 5])

# # Create a linear regression model
# model = LinearRegression()

# # Fit the model to the data
# model.fit(x, y)

# # Get the slope and intercept
# slope = model.coef_[0]
# intercept = model.intercept_

# # Display results
# print(f"Linear Regression Equation: y = {slope}x + {intercept}")




# ---------------------------------- + --------------------------------------------------------

# OR


def linear_regression(x, y):
    n = len(x)

    # Calculate mean of x and y
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    # Calculate slope and intercept
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denominator = sum((x[i] - mean_x) ** 2 for i in range(n))

    # Avoid division by zero
    if denominator != 0:
        slope = numerator / denominator
        intercept = mean_y - slope * mean_x
    else:
        # Set slope and intercept to undefined values
        slope = intercept = 0.0

    return slope, intercept

# Example data
x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

# Perform linear regression
slope, intercept = linear_regression(x, y)

# Display results
print(f"Linear Regression Equation: y = {slope}x + {intercept}")
