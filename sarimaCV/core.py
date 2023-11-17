import os
from joblib import Parallel, delayed
from .utils import *


def cross_validation(data, p_values, d_values, q_values, P_values, D_values, Q_values, s, n_folds=5, recursive=False, forecast_size=3, n_cores=-1, max_runtime=None):
    '''Perform SARIMA cross-validation on a time series

    Parameters
    ----------
    data : array_like
        The time series to be cross-validated
    p_values, d_values, q_values, P_values, D_values, Q_values : list
        Lists of integers for the AR(I)MA and seasonal AR(I)MA parameters
    s : int
        The seasonal period
    n_folds : int, optional
        The number of folds to be used in the cross-validation process
    recursive : bool, optional
        Whether to use the recursive forecasting method or not
    forecast_size : int, optional
        The number of steps to forecast
    n_cores : int, optional
        The number of cores to use in the cross-validation process
    max_runtime : int, optional
        The maximum runtime in seconds. If exceeded, the process will be cancelled.
    '''
    if n_cores == -1:
        n_cores = os.cpu_count() or 4

    initial_train_size = len(data) - n_folds * forecast_size
    param_combinations = [
        (p, d, q, P, D, Q)
        for p in p_values
        for d in d_values
        for q in q_values
        for P in P_values
        for D in D_values
        for Q in Q_values
    ]

    sample_size = min(3, len(param_combinations))
    if max_runtime:
        average_time = estimate_time_per_combination(
            data, sample_size, param_combinations, n_folds, recursive, forecast_size, initial_train_size, s)
    max_combinations = int(
        max_runtime / average_time) if max_runtime else len(param_combinations)
    selected_combinations = param_combinations[:max_combinations]
    results = Parallel(n_jobs=n_cores)(
        delayed(evaluate_params)(
            data, *comb, s=s, n_folds=n_folds, recursive=recursive, forecast_size=forecast_size, initial_train_size=initial_train_size
        ) for comb in selected_combinations
    )

    results_dict = {params: rmse for params, rmse in results}

    return format_results_to_table(results_dict, s)
