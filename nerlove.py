import numpy as np
import pandas as pd
import statsmodels.api as sm

# Generate sample data
np.random.seed(0)
n_obs = 100
price = np.random.uniform(5, 15, n_obs)  # Random price values
output = 10 + 0.5 * price + np.random.normal(0, 2, n_obs)  # Generate output with noise

# Create a DataFrame for the data
data = pd.DataFrame({'Price': price, 'Output': output})

# Add a lagged variable for Output
data['Output_Lag1'] = data['Output'].shift(1)

# Define the independent and dependent variables
X = data[['Output_Lag1', 'Price']]
y = data['Output']

# Add a constant term to the independent variables
X = sm.add_constant(X)

# Fit the Nerlove-Arrow model
model = sm.OLS(y, X)
results = model.fit()

# Print the model summary
print(results.summary())
