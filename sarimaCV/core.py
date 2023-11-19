import os
from collections import namedtuple
from functools import wraps
from joblib import Parallel, delayed
from tqdm import tqdm
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

def parallelize(n_folds, pred_len):
    def decorator(func: callable):
        @wraps(func)
        def wrapper(**kwargs):
            required = ['data', 'p', 'd', 'q']
            if not all([k in kwargs for k in required]):
                raise ValueError('Missing required keyword argument(s)')

            seasonal_params = ['P', 'D', 'Q', 's']
            if any([k in kwargs for k in seasonal_params]):
                if not all([k in kwargs for k in seasonal_params]):
                    raise ValueError('Missing required keyword argument(s)')


            if 's' in kwargs:
                comb = namedtuple('comb', ['p', 'd', 'q', 'P', 'D', 'Q'])
                param_combinations = [
                    comb(p, d, q, P, D, Q)
                    for p in kwargs.get('p', [])
                    for d in kwargs.get('d', [])
                    for q in kwargs.get('q', [])
                    for P in kwargs.get('P', [])
                    for D in kwargs.get('D', [])
                    for Q in kwargs.get('Q', [])
                ]
            else:
                comb = namedtuple('comb', ['p', 'd', 'q'])
                param_combinations = [
                    comb(p, d, q)
                    for p in kwargs.get('p', [])
                    for d in kwargs.get('d', [])
                    for q in kwargs.get('q', [])
                ]
            param_combinations = [(comb, k) for comb in param_combinations for k in range(n_folds)]

            data = kwargs['data']
            tt_data = list()
            initial_train_size = len(data) - n_folds * pred_len
            for fold in range(n_folds):
                train_end = initial_train_size + fold * pred_len
                test_end = train_end + pred_len
                train_data = data[:train_end]
                test_data = data[train_end:test_end]
                tt_data.append((list(train_data), list(test_data)))


            misc_args = {k: v for k, v in kwargs.items() if k not in required and k not in seasonal_params}
            if 's' in kwargs:
                misc_args['s'] = kwargs['s']

            results = Parallel(n_jobs=-1)(
                delayed(func)(
                    data=tt_data[k], **comb._asdict(), **misc_args
                ) for (comb, k) in tqdm(param_combinations)
            )

            results = [{'ARIMA':comb, 'k':k, 'Error':res} for (comb, k), res in zip(param_combinations, results)]
            df = pd.DataFrame(results)
            df = df.groupby('ARIMA').mean()
            df = df.drop(columns=['k'])
            df = df.sort_values('Error')

            return df
        return wrapper
    return decorator
