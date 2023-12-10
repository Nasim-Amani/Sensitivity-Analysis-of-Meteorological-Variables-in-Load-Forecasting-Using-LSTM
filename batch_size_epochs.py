# -*- coding: utf-8 -*-
"""batch_size/epochs.ipynb
"""

import keras_tuner

class RandomSearch(keras_tuner.tuners.BayesianOptimization):

  def run_trial(self, trial, *args, **kwargs):

    # Set the batch_size hyperparameter from the trial
    kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 32, 256, step=32)

    # Set the epochs hyperparameter from the trial
    kwargs['epochs'] = trial.hyperparameters.Int('epochs', 50, 200)

    # Run the trial using the parent class's run_trial method
    return super(RandomSearch, self).run_trial(trial, *args, **kwargs)
