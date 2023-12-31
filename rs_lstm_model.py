# -*- coding: utf-8 -*-
"""RS-LSTM Model.ipynb
**The given code defines a function build_model that constructs a neural network model using the *Keras* library. The model consists of multiple *LSTM* layers followed by a dense layer. The hyperparameters of the *LSTM* layers are defined using the *Keras Tuner library*, which allows for automatic hyperparameter tuning. The model is compiled with the Adam optimizer and the mean squared error loss function.**
"""

import pandas as pd
import math
import keras
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

import keras_tuner
from keras_tuner import RandomSearch
from keras_tuner import HyperModel
from keras_tuner import HyperParameters

import tensorflow as tf

# Define a function to build the model with hyperparameters
def build_model(hp):
    model = keras.Sequential()

    # Add the first LSTM layer with tunable number of units
    model.add(layers.LSTM(units=hp.Int('units_1',
                                        min_value=70,
                                        max_value=92,
                                        step=2),
                          input_shape=(X_train.shape[1], X_train.shape[2]),
                          return_sequences=True))

    # Add the second LSTM layer with tunable number of units
    model.add(layers.LSTM(units=hp.Int('units_2',
                                        min_value=320,
                                        max_value=362,
                                        step=4),
                          return_sequences=True))

    # Add the third LSTM layer with tunable number of units
    model.add(layers.LSTM(units=hp.Int('units_3',
                                        min_value=276,
                                        max_value=312,
                                        step=4),
                          return_sequences=True))

    # Add the last LSTM layer with tunable number of units
    model.add(layers.LSTM(units=hp.Int('Last_units',
                                        min_value=212,
                                        max_value=252,
                                        step=4)))

    # Add a dropout layer (optional)
    #if hp.Boolean("dropout"):
    #    model.add(layers.Dropout(rate=0.25))

    # Add a dense layer with 1 unit
    model.add(layers.Dense(1))

    # Compile the model with Adam optimizer and mean squared error loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='mean_squared_error',
        metrics=['mse'])

    return model



"""**The given code initializes a RandomSearch object from the Keras Tuner library. This object is used for hyperparameter search and optimization. It takes the build_model function as an argument, which is the function responsible for constructing the neural network model. The objective of the search is to minimize the validation mean squared error (val_mse). The max_trials parameter specifies the maximum number of hyperparameter combinations to try, and executions_per_trial specifies the number of times to train and evaluate each model configuration. The overwrite parameter determines whether to overwrite previously saved results of the search.**"""

# Initialize a RandomSearch object for hyperparameter search
tuner = RandomSearch(
    build_model,  # The function responsible for constructing the model
    objective='val_mse',  # The objective to minimize (validation mean squared error)
    max_trials=3,  # Maximum number of hyperparameter combinations to try
    executions_per_trial=1,  # Number of times to train and evaluate each model configuration
    overwrite=True  # Whether to overwrite previously saved results
)

tuner.search_space_summary()

from tensorflow.keras.callbacks import EarlyStopping

# Create an instance of EarlyStopping callback
custom_early_stopping = EarlyStopping(
    monitor='val_loss',  # Quantity to monitor for early stopping (validation loss)
    patience=3,  # Number of epochs with no improvement after which training will be stopped
    min_delta=0.0001,  # Minimum change in the monitored quantity to qualify as improvement
    mode='auto'  # Direction of improvement ('auto' determines it automatically based on the monitored quantity)
)




"""**The given code performs a hyperparameter search using the tuner object. It calls the search() method on the tuner and provides the necessary arguments for the search.**"""

# Perform hyperparameter search using the tuner
tuner.search(
    x=X_train,  # Training data
    y=y_train,  # Training labels
    epochs=200,  # Number of training epochs for each model configuration
    batch_size=32,  # Number of samples per gradient update during training
    validation_data=(X_validate, y_validate),  # Validation data for evaluating the models
    callbacks=[custom_early_stopping]  # List of callback functions to be called during training
)




# Display a summary of the hyperparameter search results
tuner.results_summary()




# Retrieve the best model found during the hyperparameter search
model = tuner.get_best_models()[0]
