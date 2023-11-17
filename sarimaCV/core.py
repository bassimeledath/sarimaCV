import numpy as np
import pandas as pd
import warnings
import os
import time
import concurrent.futures
from math import sqrt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning


def eval_fold(train, test, p, d, q, P, D, Q, s, recursive=False):
    '''Evaluate a single fold of the cross-validation process'''
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        warnings.simplefilter('ignore', ConvergenceWarning)

        if recursive:
            # Recursive forecasting method
            forecasts = []
            for _ in range(len(test)):
                model = SARIMAX(train, order=(p, d, q),
                                seasonal_order=(P, D, Q, s))
                forecast = model.fit(disp=0).forecast(steps=1)
                forecasts.append(forecast[0])
                train = np.append(train, forecast[0])
        else:
            # Standard forecasting method
            model = SARIMAX(train, order=(p, d, q),
                            seasonal_order=(P, D, Q, s))
            forecasts = model.fit(disp=0).forecast(steps=len(test))

        se = (test - np.array(forecasts))**2
        return np.sum(se)


def evaluate_params(data, p, d, q, P, D, Q, s, n_folds, recursive, forecast_size, initial_train_size):
    '''Evaluate a single parameter combination'''
    total_rmse = 0
    for fold in range(n_folds):
        train_end = initial_train_size + fold * forecast_size
        test_end = train_end + forecast_size
        train_data = data[:train_end]
        test_data = data[train_end:test_end]
        se = eval_fold(train_data, test_data, p, d, q, P, D, Q, s, recursive)
        total_rmse += sqrt(se / forecast_size) / n_folds

    return (p, d, q, P, D, Q), total_rmse


def format_results_to_table(results_dict):
    '''Format results to a pandas DataFrame'''
    df = pd.DataFrame.from_dict(
        results_dict, orient='index', columns=['AVG RMSE'])
    df = df.reset_index()
    df.columns = ['Order', 'AVG RMSE']
    df = df.sort_values(by='AVG RMSE', ascending=True)
    return df


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
    results_dict = {}
    initial_train_size = len(data) - n_folds * forecast_size
    start_time = time.time()

    param_combinations = [(p, d, q, P, D, Q) for p in range(max_p + 1) for d in range(max_d + 1)
                          for q in range(max_q + 1) for P in range(max_P + 1)
                          for D in range(max_D + 1) for Q in range(max_Q + 1)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_cores) as executor:
        future_to_params = {executor.submit(evaluate_params, data, p, d, q, P, D, Q, s, n_folds, recursive, forecast_size, initial_train_size): (p, d, q, P, D, Q)
                            for p, d, q, P, D, Q in param_combinations}

        try:
            for future in concurrent.futures.as_completed(future_to_params):
                if max_runtime and (time.time() - start_time) > max_runtime:
                    raise TimeoutError(
                        "Max runtime exceeded. Cancelling remaining tasks can take up to 1-2 mins.")

                params = future_to_params[future]
                _, rmse_value = future.result()
                results_dict[params] = rmse_value
        except TimeoutError as e:
            print(e)
            for future in future_to_params:
                future.cancel()
            executor.shutdown(wait=True)
            return format_results_to_table(results_dict)

    return format_results_to_table(results_dict)
