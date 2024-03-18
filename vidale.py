import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(0)
n_obs = 100
advertising = np.random.uniform(100, 1000, n_obs)  # Random advertising expenditure values
sales = 50 + 0.1 * advertising + np.random.normal(0, 50, n_obs)  # Generate sales revenue with noise

# Define the Vidale-Wolfe model
def vidale_wolfe_model(params, x):
    # params[0] is the intercept, params[1] is the slope
    return params[0] + params[1] * x

# Define the loss function (Mean Squared Error)
def loss_function(params, x, y):
    y_pred = vidale_wolfe_model(params, x)
    return np.mean((y_pred - y) ** 2)

# Initialize model parameters and hyperparameters
params = np.random.randn(2)  # Random initial values for intercept and slope
learning_rate = 0.0001
epochs = 1000

# Perform gradient descent optimization
for epoch in range(epochs):
    # Compute gradients
    gradients = np.zeros(2)
    for i in range(n_obs):
        gradients[0] += 2 * (vidale_wolfe_model(params, advertising[i]) - sales[i])
        gradients[1] += 2 * (vidale_wolfe_model(params, advertising[i]) - sales[i]) * advertising[i]
    gradients /= n_obs

    # Update parameters
    params -= learning_rate * gradients

    # Print loss for every 100 epochs
    if epoch % 100 == 0:
        loss = loss_function(params, advertising, sales)
        print(f'Epoch {epoch}: Loss = {loss}')

# Print final parameters
intercept, slope = params
print(f'Intercept: {intercept}, Slope: {slope}')

# Plot the data and the regression line
plt.figure(figsize=(10, 6))
plt.scatter(advertising, sales, label='Data points')
plt.plot(advertising, vidale_wolfe_model(params, advertising), color='red', label='Regression line')
plt.xlabel('Advertising Expenditure')
plt.ylabel('Sales Revenue')
plt.title('Vidale-Wolfe Advertising Model')
plt.legend()
plt.grid(True)
plt.show()
