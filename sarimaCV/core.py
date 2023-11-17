import os
from joblib import Parallel, delayed
from .utils import *


def cross_validation(data, max_p=3, max_d=1, max_q=3, max_P=1, max_D=1, max_Q=1, s=12, n_folds=5, recursive=False, forecast_size=3, n_cores=-1, max_runtime=None):
    '''Perform SARIMA cross-validation on a time series

    Parameters
    ----------
    data : array_like
        The time series to be cross-validated
    max_p : int, optional
        The maximum value of the AR parameter
    max_d : int, optional
        The maximum value of the I parameter
    max_q : int, optional
        The maximum value of the MA parameter
    max_P : int, optional
        The maximum value of the seasonal AR parameter
    max_D : int, optional
        The maximum value of the seasonal I parameter
    max_Q : int, optional
        The maximum value of the seasonal MA parameter
    s : int, optional
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
        for p in range(max_p + 1)
        for d in range(max_d + 1)
        for q in range(max_q + 1)
        for P in range(max_P + 1)
        for D in range(max_D + 1)
        for Q in range(max_Q + 1)
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

    return format_results_to_table(results_dict)
