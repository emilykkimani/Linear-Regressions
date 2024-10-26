import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = r"C:\Users\Administrator\Downloads\Nairobi Office Price Ex.csv"
df = pd.read_csv(file_path)

# Select relevant columns
df = df[['SIZE', 'PRICE']]

# Standardize the data to improve gradient descent convergence
x = (df['SIZE'].values - df['SIZE'].mean()) / df['SIZE'].std()
y = (df['PRICE'].values - df['PRICE'].mean()) / df['PRICE'].std()

# Define Mean Squared Error (MSE) function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Define Gradient Descent function
def gradient_descent(x, y, m, c, learning_rate):
    N = len(y)
    y_pred = m * x + c
    dm = (-2 / N) * np.sum(x * (y - y_pred))  # Derivative w.r.t. m
    dc = (-2 / N) * np.sum(y - y_pred)        # Derivative w.r.t. c
    m -= learning_rate * dm                   # Update m
    c -= learning_rate * dc                   # Update c
    return m, c

# Initialize parameters (setting intercept c to 0)
m, c = np.random.rand(), 0  # Random initial slope, intercept set to 0
learning_rate = 0.003       # Adjusted learning rate for smoother updates
epochs = 10

# Training loop for 10 epochs
for epoch in range(epochs):
    y_pred = m * x + c
    error = mean_squared_error(y, y_pred)
    print(f"Epoch {epoch + 1}, Mean Squared Error: {error:.4f}")
    m, c = gradient_descent(x, y, m, c, learning_rate)

# Convert standardized line of best fit back to original scale for accurate plotting
m_original = m * (df['PRICE'].std() / df['SIZE'].std())
c_original = df['PRICE'].mean() + (c * df['PRICE'].std()) - m_original * df['SIZE'].mean()

# Plotting the line of best fit
plt.scatter(df['SIZE'], df['PRICE'], color='blue', label='Data points')
plt.plot(df['SIZE'], m_original * df['SIZE'] + c_original, color='red', label='Line of Best Fit')
plt.xlabel('Office Size (sq. ft.)')
plt.ylabel('Office Price')
plt.title('Office Size vs. Price')
plt.legend()
plt.show()

# Predict office price for size 100 sq. ft. using original scale
size_to_predict = 100
predicted_price = m_original * size_to_predict + c_original
print(f"Predicted office price for size {size_to_predict} sq. ft.: {predicted_price:.2f}")
