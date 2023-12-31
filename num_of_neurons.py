# -*- coding: utf-8 -*-
"""num_of_neurons.ipynb
"""

def build_model(hp):
    model = keras.Sequential()

    # Add the first LSTM layer with units specified by the hyperparameter 'units_1'
    model.add(layers.LSTM(units=hp.Int('units_1',
                                       min_value=32,
                                       max_value=612,
                                       step=32),
                          input_shape=(X_train.shape[1], X_train.shape[2]),
                          return_sequences=True))

    # Add the second LSTM layer with units specified by the hyperparameter 'units_2'
    model.add(layers.LSTM(units=hp.Int('units_2',
                                       min_value=32,
                                       max_value=612,
                                       step=32),
                          return_sequences=True))

    # Add the third LSTM layer with units specified by the hyperparameter 'units_3'
    model.add(layers.LSTM(units=hp.Int('units_3',
                                       min_value=32,
                                       max_value=612,
                                       step=32),
                          return_sequences=True))

    # Add the last LSTM layer with units specified by the hyperparameter 'Last_units'
    model.add(layers.LSTM(units=hp.Int('Last_units',
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
