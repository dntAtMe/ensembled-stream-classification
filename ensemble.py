# Author: Kacper Pieniążek, PWr (236606)

from skmultiflow.meta import AccuracyWeightedEnsembleClassifier
from skmultiflow.meta import BatchIncrementalClassifier
from skmultiflow.meta import DynamicWeightedMajorityClassifier

def build_premade_ensemble(type: str, n_estimators: int, base_estimator: object):
    """
    Build a premade ensemble of models.

    Parameters
    ----------
    type : str
        The type of ensemble to build.
    n_estimators : int
        The number of estimators to use in the ensemble.
    base_estimator : object
        The base estimator to use in the ensemble.

    Returns
    -------
    ensemble : object
        The created ensemble
    """
    if type == 'awe':
        return build_accuracy_weighted_ensemble(n_estimators, base_estimator)
    elif type == 'batch':
        return build_batch_incremental_ensemble(n_estimators, base_estimator)
    elif type == 'dwm':
        return build_dynamic_weighted_majority_ensemble(n_estimators = n_estimators, base_estimator = base_estimator)
    else:
        raise ValueError(f'Invalid ensemble type: {type}')
    
def build_accuracy_weighted_ensemble(n_estimators: int, base_estimator: object):
    """
    Build an accuracy weighted ensemble of models.

    Parameters
    ----------
    models : list
        The list of models to use in the ensemble.

    Returns
    -------
    ensemble : object
        The created ensemble
    """
    AccuracyWeightedEnsembleClassifier(n_estimators = n_estimators, base_estimator = base_estimator)

def build_batch_incremental_ensemble(n_estimators: int, base_estimator: object):
    """
    Build a batch incremental ensemble of models.

    Parameters
    ----------
    models : list
        The list of models to use in the ensemble.

    Returns
    -------
    ensemble : object
        The created ensemble
    """
    BatchIncrementalClassifier(n_estimators = n_estimators, base_estimator = base_estimator)

def build_dynamic_weighted_majority_ensemble(n_estimators: int, base_estimator: object):
    """
    Build a dynamic weighted majority ensemble of models.

    Parameters
    ----------
    models : list
        The list of models to use in the ensemble.

    Returns
    -------
    ensemble : object
        The created ensemble
    """
    DynamicWeightedMajorityClassifier(n_estimators = n_estimators, base_estimator = base_estimator)

def build_custom_ensemble(type: str, estimators: list):
    """
    Build a custom ensemble of models.

    Parameters
    ----------
    type : str
        The type of ensemble to build.
    estimators : list
        The list of models to use in the ensemble.

    Returns
    -------
    ensemble : object
        The created ensemble
    """    

    if type == 'simple':
        return 
    elif type == 'aec':
        return build_adaptive_ensemble_classifier(estimators)
    else:
        raise ValueError(f'Invalid ensemble type: {type}')
    
def build_simple_ensemble(estimators: list):
    """
    Build a simple ensemble of models.

    Parameters
    ----------
    estimators : list
        The list of models to use in the ensemble.

    Returns
    -------
    ensemble : object
        The created ensemble
    """
    pass
    
def build_adaptive_ensemble_classifier(estimators: list):
    """
    Build an adaptive ensemble classifier.

    Parameters
    ----------
    estimators : list
        The list of models to use in the ensemble.

    Returns
    -------
    ensemble : object
        The created ensemble
    """
    pass