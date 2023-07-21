"""
This module defines the following routines used by the 'train' step of the regression recipe:

- ``estimator_fn``: Defines the customizable estimator type and parameters that are used
  during training to produce a model recipe.
"""
from typing import Dict, Any

def estimator_fn(estimator_params: Dict[str, Any] = None):
    """
    Returns an *unfitted* estimator that defines ``fit()`` and ``predict()`` methods.
    The estimator's input and output signatures should be compatible with scikit-learn
    estimators.
    """
    #
    # FIXME::OPTIONAL: return a scikit-learn-compatible regression estimator with fine-tuned
    #                  hyperparameters.

    if estimator_params is None:
        estimator_params = {}

    from sklearn.linear_model import SGDRegressor

    return SGDRegressor(random_state=42, **estimator_params)

    #from sklearn.ensemble import RandomForestRegressor

    #return  RandomForestRegressor(random_state=10, oob_score = True, **estimator_params)

def estimator_rf(estimator_params: Dict[str, Any] = {}):
    """
    Returns an *unfitted* estimator that defines ``fit()`` and ``predict()`` methods.
    The estimator's input and output signatures should be compatible with scikit-learn
    estimators.
    """
    #
    # FIXME::OPTIONAL: return a scikit-learn-compatible regression estimator with fine-tuned
    #                  hyperparameters.

    from sklearn.ensemble import RandomForestRegressor

    return  RandomForestRegressor(random_state=10, **estimator_params)

def estimator_xgb(estimator_params: Dict[str, Any] = {}):
    """
    Returns an *unfitted* estimator that defines ``fit()`` and ``predict()`` methods.
    The estimator's input and output signatures should be compatible with scikit-learn
    estimators.
    """
    #
    # FIXME::OPTIONAL: return a scikit-learn-compatible regression estimator with fine-tuned
    #                  hyperparameters.

    from sklearn.ensemble import RandomForestRegressor

    return  RandomForestRegressor(random_state=10, **estimator_params)

def my_early_stop_fn(*args):
    from hyperopt.early_stop import no_progress_loss

    return no_progress_loss(10)(*args)
