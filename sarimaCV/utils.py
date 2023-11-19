import numpy as np
import pandas as pd
import warnings
from math import sqrt
import time
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

        sse = np.sum((test - np.array(forecasts))**2)
        return sse


def evaluate_params(data, p, d, q, P, D, Q, s, n_folds, recursive, forecast_size, initial_train_size):
    '''Evaluate a single parameter combination'''
    avg_rmse = 0
    for fold in range(n_folds):
        train_end = initial_train_size + fold * forecast_size
        test_end = train_end + forecast_size
        train_data = data[:train_end]
        test_data = data[train_end:test_end]
        sse = eval_fold(train_data, test_data, p, d, q, P, D, Q, s, recursive)
        avg_rmse += sqrt(sse / forecast_size) / n_folds

    return (p, d, q, P, D, Q), avg_rmse


def estimate_time_per_combination(data, sample_size, param_combinations, n_folds, recursive, forecast_size, initial_train_size, s):
    '''Estimate the average time per parameter combination'''
    total_time = 0
    for params in param_combinations[:sample_size]:
        start_time = time.time()
        evaluate_params(data, *params, s=s, n_folds=n_folds, recursive=recursive,
                        forecast_size=forecast_size, initial_train_size=initial_train_size)
        total_time += time.time() - start_time
    average_time = total_time / sample_size
    return average_time


def format_results_to_table(results_dict, s):
    '''Format results to a pandas DataFrame with a specific string format for orders'''
    formatted_rows = []
    for (p, d, q, P, D, Q), rmse in results_dict.items():
        order_str = f"SARIMA[({p}, {d}, {q}), ({P}, {D}, {Q}, {s})]"
        formatted_rows.append([order_str, rmse])
    df = pd.DataFrame(formatted_rows, columns=['Order', 'RMSE'])
    df = df.sort_values(by='RMSE', ascending=True)
    return df
