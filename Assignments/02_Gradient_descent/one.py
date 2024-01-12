import numpy as np
from sklearn.preprocessing import StandardScaler

train_data = np.genfromtxt("train.csv", delimiter=",")
test_data = np.genfromtxt("test.csv", delimiter=",")

train_data.shape
test_data.shape

X = train_data[:, :13]
Y = train_data[:, 13]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

train_data = np.column_stack((X_scaled, Y))

X.shape
Y.shape

def scale_test_data(test_data, scaler):
    return scaler.transform(test_data)

scaled_X = scale_test_data(X, scaler)
scaled_test_data = scale_test_data(test_data, scaler)

# This function finds the new gradient at each step for multiple features
def step_gradient(points, learning_rate, m, c):
    m_slope = np.zeros_like(m)
    c_slope = 0
    M = len(points)

    for i in range(M):
        x = points[i, :-1]  # Exclude the last column which is the target variable
        y = points[i, -1]   # Target variable

        m_slope += (-2/M) * (y - np.dot(m, x) - c) * x
        c_slope += (-2/M) * (y - np.dot(m, x) - c)

    new_m = m - learning_rate * m_slope
    new_c = c - learning_rate * c_slope
    return new_m, new_c

# The Gradient Descent Function for multiple features
def gd(points, learning_rate, num_iterations):
    m = np.zeros(points.shape[1] - 1)  # Initialize weights for each feature
    c = 0
    for i in range(num_iterations):
        m, c = step_gradient(points, learning_rate, m, c)
        print(i, "Cost : ", cost(points, m, c))
    return m, c

# This function finds the new cost after each optimization for multiple features
def cost(points, m, c):
    total_cost = 0
    M = len(points)

    for i in range(M):
        x = points[i, :-1]  # Exclude the last column which is the target variable
        y = points[i, -1]   # Target variable

        total_cost += (1/M) * ((y - np.dot(m, x) - c)**2)
    return total_cost

def run():
    data = scaled_X
    learning_rate = 0.01
    num_iterations = 1000
    m, c = gd(data, learning_rate, num_iterations)
    print("Weights (m):", m)
    print("Intercept (c):", c)
    return m, c

run()

def predict(data, m, c):
    predictions = []
    for i in range(len(data)):
        x_test = data[i, :-1]  
        y_pred = np.dot(m, x_test) + c
        predictions.append(y_pred)
    return np.array(predictions)

m = [ 0.11934211,  0.02198045,  0.09153026, -0.05528189,  0.16497196,
        -0.40897173,  0.30528339,  0.06771261,  0.00333266, -0.01564392,
         0.02671131, -0.13386579]
c = -4.091345318091297e-17

predictions = predict(scaled_X, m, c)
print(predictions)


def run():
    data = scaled_test_data
    learning_rate = 0.01
    num_iterations = 1000
    m, c = gd(data, learning_rate, num_iterations)
    print("Weights (m):", m)
    print("Intercept (c):", c)
    return m, c

run()


def hyperparameter_tuning(train_data, test_data, learning_rates, num_iterations):
    best_m = None
    best_c = None
    best_cost = float('inf')

    for lr in learning_rates:
        for num_iter in num_iterations:
            m, c = gd(train_data, lr, num_iter)
            current_cost = cost(train_data, m, c)

            print("Learning Rate:", lr, "Num Iterations:", num_iter)
            print("Weights (m):", m)
            print("Intercept (c):", c)
            print("Training Cost:", current_cost)

            # Evaluate on test data
            test_cost = cost(test_data, m, c)
            print("Testing Cost:", test_cost)

            if test_cost < best_cost:
                best_cost = test_cost
                best_m = m
                best_c = c

    return best_m, best_c

# Example of trying different combinations
learning_rates_to_try = [0.01, 0.1, 0.5]
num_iterations_to_try = [500, 1000, 1500]

best_m, best_c = hyperparameter_tuning(scaled_X, scaled_test_data, learning_rates_to_try, num_iterations_to_try)

# Example of predicting with the best combination
predictions = predict(scaled_test_data, best_m, best_c)
print(predictions)


