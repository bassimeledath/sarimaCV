import os
import time
import concurrent.futures
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
            executor.shutdown(wait=False)
            return format_results_to_table(results_dict)

    return format_results_to_table(results_dict)
