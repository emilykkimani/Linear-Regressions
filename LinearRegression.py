import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = r"C:\Users\Administrator\Downloads\Nairobi Office Price Ex.csv"
df = pd.read_csv(file_path)
print(df.head())

# Select relevant columns
df = df[['SIZE', 'PRICE']]

# Define Mean Squared Error (MSE) function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Define Gradient Descent function
def gradient_descent(x, y, m, c, learning_rate):
    N = len(y)
    y_pred = m * x + c
    dm = (-2 / N) * np.sum(x * (y - y_pred))  # Derivative w.r.t. m
    dc = (-2 / N) * np.sum(y - y_pred)        # Derivative w.r.t. c
    m -= learning_rate * dm                    # Update m
    c -= learning_rate * dc                    # Update c
    return m, c

# Initialize parameters
m, c = np.random.rand(), np.random.rand()  # Random initial slope and y-intercept
learning_rate = 0.01
epochs = 10

# Prepare data for training
x = df['SIZE'].values
y = df['PRICE'].values
errors = []

# Training loop for 10 epochs
for epoch in range(epochs):
    y_pred = m * x + c
    error = mean_squared_error(y, y_pred)
    errors.append(error)
    m, c = gradient_descent(x, y, m, c, learning_rate)
    print(f"Epoch {epoch + 1}, Mean Squared Error: {error:.4f}")

# Plotting the line of best fit
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, m * x + c, color='red', label='Line of Best Fit')
plt.xlabel('Office Size (sq. ft.)')
plt.ylabel('Office Price')
plt.title('Office Size vs. Price')
plt.legend()
plt.show()

# Make a prediction for an office size of 100 sq. ft.
size_to_predict = 100
predicted_price = m * size_to_predict + c
print(f"Predicted office price for size {size_to_predict} sq. ft.: {predicted_price:.2f}")
