# -*- coding: utf-8 -*-
"""num_of_layers.ipynb

"""

def build_model(hp):
    model = keras.Sequential()

    # Define the input shape based on X_train
    tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2]))

    # Iterate over the number of layers defined by the hyperparameter 'num_layers'
    for i in range(hp.Int('num_layers', 0, 10)):
        # Add an LSTM layer with variable units based on the hyperparameter 'units_i'
        model.add(layers.LSTM(units=hp.Int('units_' + str(i),
                                            min_value=32,
                                            max_value=612,
                                            step=32),
                              return_sequences=True))

    # Add the final LSTM layer with units specified by the hyperparameter 'units_2'
    model.add(layers.LSTM(units=hp.Int('units_2',
                                       min_value=32,
                                       max_value=612,
                                       step=32)))

    # Add a dropout layer if the hyperparameter 'dropout' is set to True
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=0.25))

    # Add a dense layer with 1 unit
    model.add(layers.Dense(1))

    # Compile the model with the Adam optimizer, mean squared error loss, and mean squared error metric
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='mean_squared_error',
        metrics=['mse'])

    # Return the constructed model
    return model
