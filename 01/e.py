import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read data
Data = pd.read_csv("data.csv")
x_values = Data["YearsExperience"].values
y_values = Data["Salary"].values

def cost_function(x, y, m, c):
    """Calculate the mean squared error cost function"""
    n = len(x)
    total_err = 0.0
    for i in range(n):
        y_cal = (m * x[i] + c)
        total_err += (y[i] - y_cal) ** 2  # Accumulate the error
    return total_err / n

def gradient_descent(x, y, m_current=0, c_current=0, epochs=100, learning_rate=0.01):
    """Perform gradient descent to optimize m and c"""
    n = len(x)
    m = m_current
    c = c_current
    cost_history = []
    
    for _ in range(epochs):
        m_gradient = 0
        c_gradient = 0
        
        for i in range(n):
            y_pred = m * x[i] + c
            err = y_pred - y[i]
            m_gradient += (2/n) * (err * x[i])
            c_gradient += (2/n) * err
            
        m = m - learning_rate * m_gradient
        c = c - learning_rate * c_gradient
        current_cost = cost_function(x, y, m, c)
        cost_history.append(current_cost)
        
    return m, c, cost_history

initial_m = 0
initial_c = 0
# After running gradient descent (your existing code)
result_m, result_c, cost_history = gradient_descent(x_values, y_values, initial_m, initial_c, epochs=1000)

# Generate predicted values for the line
x_line = np.linspace(min(x_values), max(x_values), 100)  # 100 points for smooth line
y_line = result_m * x_line + result_c  # y = mx + c

# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(x_values, y_values, color='blue', label='Actual Data')  # Original data points
plt.plot(x_line, y_line, color='red', label='Prediction Line')     # Regression line
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression Fit")
plt.legend()
plt.grid(True)
plt.savefig("regression_plot.png")  # Save the plot
plt.show()  # Display the plot
for i in range(4):
  # TODO: write code...
  x=float(input("inter the expersience"))
  print(result_m*x+result_c)