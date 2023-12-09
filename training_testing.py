# -*- coding: utf-8 -*-
"""Training-Testing.ipynb
"""

# Train the model using the fit() method
model.fit(
    X_train,  # Training data
    y_train,  # Training labels
    epochs=50,  # Number of training epochs
    batch_size=32,  # Number of samples per gradient update during training
    validation_data=(X_validate, y_validate)  # Validation data for evaluating the model
)

# Print a summary of the model architecture and parameters
model.summary()

"""**The given code performs predictions using a trained model (model) on different datasets (X_test, X_train, X_validate). It then applies an inverse transformation to the predicted values using a transformer (t_transformer) to obtain the original scale of the target variable. The inverse transformed predictions are stored in variables with the suffix _inv. The code imports necessary functions from the math and sklearn.metrics modules for calculating metrics such as mean squared error.**"""

from math import sqrt
from sklearn.metrics import mean_squared_error

# Perform predictions on the test data
y_pred = model.predict(X_test)

# Inverse transform the target variable
y_test_inv = t_transformer.inverse_transform(y_test.reshape((len(y_test), 1)))
y_pred_inv = t_transformer.inverse_transform(y_pred)

# calculate RMSE
rmse = sqrt(mean_squared_error(y_test_inv, y_pred_inv))
print('Test RMSE: %.3f' % rmse)

# Calculate NRMSE
actual_test = y_test_inv
Nrmse_test = rmse / (actual_test.max() - actual_test.min())

# Calculate MAPE
from sklearn.metrics import mean_absolute_percentage_error
mean_absolute_percentage_error(y_test_inv, y_pred_inv)

# Calculate MAE
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test_inv, y_pred_inv)

"""**The given code uses the matplotlib.pyplot library to create a line plot comparing the predicted values (y_pred_inv) and the actual values (y_test_inv).**"""

The given code uses the matplotlib.pyplot library to create a line plot comparing the predicted values (y_pred_inv) and the actual values (y_test_inv).

import matplotlib.pyplot as plt

# Set the figure size
plt.figure(figsize=(15, 6))

# Plot the predicted values
plt.plot(y_pred_inv, label="Prediction")

# Plot the actual values
plt.plot(y_test_inv, label="Actual")

# Display the legend
plt.legend()

# Set the y-axis label
plt.ylabel('Load')

# Set the x-axis label
plt.xlabel('time step')

# Set the plot title
plt.title('Load forecasting')

# Display the plot
plt.show()

"""**The given code uses the matplotlib.pyplot library to create a line plot that zooms in on a specific range of data.**"""

import matplotlib.pyplot as plt

# Set the figure size
plt.figure(figsize=(15, 6))

# Plot a subset of the actual values
plt.plot(y_test_inv[3900:4200])

# Plot a subset of the predicted values
plt.plot(y_pred_inv[3900:4200])

# Display the plot
plt.show()
