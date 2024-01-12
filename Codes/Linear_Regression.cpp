#include <iostream>
#include <vector>

using namespace std;

// Function to perform linear regression
void linearRegression(const vector<double>& x, const vector<double>& y, double& slope, double& intercept) {
    int n = x.size();

    // Calculate mean of x and y
    double mean_x = 0.0, mean_y = 0.0;
    for (int i = 0; i < n; ++i) {
        mean_x += x[i];
        mean_y += y[i];
    }
    mean_x /= n;
    mean_y /= n;

    // Calculate slope and intercept
    double numerator = 0.0, denominator = 0.0;
    for (int i = 0; i < n; ++i) {
        numerator += (x[i] - mean_x) * (y[i] - mean_y);
        denominator += (x[i] - mean_x) * (x[i] - mean_x);
    }

    // Avoid division by zero
    if (denominator != 0) {
        slope = numerator / denominator;
        intercept = mean_y - slope * mean_x;
    } else {
        // Set slope and intercept to undefined values
        slope = intercept = 0.0;
    }
}

int main() {
    // Example data
    vector<double> x = {1, 2, 3, 4, 5};
    vector<double> y = {2, 4, 5, 4, 5};

    // Variables to store slope and intercept
    double slope, intercept;

    // Perform linear regression
    linearRegression(x, y, slope, intercept);

    // Display results
    cout << "Linear Regression Equation: y = " << slope << "x + " << intercept << endl;

    return 0;
}
