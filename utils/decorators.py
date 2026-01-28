import time
from typing import Any, Callable, Dict, Tuple


def timed_metric(func: Callable[[], dict]) -> Callable[[], dict]:
    """
    Decorator to time the execution of a metric evaluation function. The returned function add a 'runtime' key
    to the dictionary returned by the decorated function, indicating the time taken to execute it.

    Args:
        func (callable): The metric evaluation function to be decorated. Should return a dict of metrics.

    Returns:
        callable: The decorated function that returns a dict of metrics with an added 'runtime' key.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        metrics = func(*args, **kwargs)
        runtime = time.time() - start_time

        assert isinstance(metrics, dict), "The decorated function must return a dictionary."
        if "runtime" in metrics:
            Warning("The decorated function's return dictionary already has a 'runtime' key. It will be overwritten.")

        metrics.update({"runtime": runtime})
        return metrics
    return wrapper