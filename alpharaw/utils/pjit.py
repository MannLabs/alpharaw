"""Thread and progress callback utilities.

Original implementation in AlphaTims.
"""

from __future__ import annotations

import multiprocessing
from typing import Callable

MAX_THREADS = multiprocessing.cpu_count()
PROGRESS_CALLBACK = True


def set_threads(threads: int, *, set_global: bool = True) -> int:
    """Parse and set the (global) number of threads.

    Parameters
    ----------
    threads : int
        The number of threads.
        If larger than available cores, it is trimmed to the available maximum.
        If 0, it is set to the maximum cores available.
        If negative, it indicates how many cores NOT to use.
    set_global : bool
        If False, the number of threads is only parsed to a valid value.
        If True, the number of threads is saved as a global variable.
        Default is True.

    Returns
    -------
    : int
        The number of threads.

    """
    max_cpu_count = multiprocessing.cpu_count()
    if threads > max_cpu_count:
        threads = max_cpu_count
    else:
        while threads <= 0:
            threads += max_cpu_count
    if set_global:
        global MAX_THREADS  # noqa: PLW0603
        MAX_THREADS = threads
    return threads


def threadpool(
    _func: Callable | None = None,
    *,
    thread_count: int | None = None,
    include_progress_callback: bool = True,
    return_results: bool = False,
) -> None:
    """A decorator that parallelizes a function with threads and callback.

    The original function should accept a single element as its first argument.
    If the caller function provides an iterable as first argument,
    the function is applied to each element of this iterable in parallel.

    Parameters
    ----------
    _func : callable, None
        The function to decorate.
    thread_count : int, None
        The number of threads to use.
        This is always parsed with alphatims.utils.set_threads.
        Not possible as positional arguments,
        it always needs to be an explicit keyword argument.
        Default is None.
    include_progress_callback : bool
        If True, the default progress callback will be used as callback.
        (See "progress_callback" function.)
        If False, no callback is added.
        See `set_progress_callback` for callback styles.
        Default is True.
    return_results : bool
        If True, it returns the results in the same order as the iterable.
        This can be much slower than not returning results. Iti is better to
        store them in a buffer results array instead
        (be carefull to avoid race conditions).
        If the iterable is not an iterable but a single index, a result is
        always returned.
        Default is False.

    Returns
    -------
    : function
        A parallelized decorated function.

    """
    import functools
    import multiprocessing.pool

    def parallel_func_inner(func: Callable) -> Callable:
        def wrapper(iterable, *args, **kwargs):  # noqa: ANN001, ANN202
            def starfunc(iterable):  # noqa: ANN001, ANN202
                return func(iterable, *args, **kwargs)

            try:
                iter(iterable)
            except TypeError:
                return func(iterable, *args, **kwargs)
            if thread_count is None:
                current_thread_count = MAX_THREADS
            else:
                current_thread_count = set_threads(thread_count, set_global=False)
            with multiprocessing.pool.ThreadPool(current_thread_count) as pool:
                if return_results:
                    results = []
                    for result in progress_callback(
                        pool.imap(starfunc, iterable),
                        total=len(iterable),
                        include_progress_callback=include_progress_callback,
                    ):
                        results.append(result)  # noqa: PERF402
                    return results
                for _ in progress_callback(
                    pool.imap_unordered(starfunc, iterable),
                    total=len(iterable),
                    include_progress_callback=include_progress_callback,
                ):
                    pass
                return None

        return functools.wraps(func)(wrapper)

    if _func is None:
        return parallel_func_inner
    return parallel_func_inner(_func)


def conditional_njit(*, use_numba: bool = True, **kwargs) -> Callable:  # noqa: D417
    """A conditional decorator that applies @numba.njit() when use_numba=True, otherwise returns the original function unchanged.

    Args:
        use_numba (bool): If True, applies numba.njit(). If False, returns original function.

    Returns:
        Decorated function or original function

    """

    def decorator(func: Callable) -> Callable:
        if use_numba:
            import numba

            return numba.njit(**kwargs)(func)
        return func

    return decorator


def pjit(  # noqa: ANN201, D417, C901
    _func: Callable | None = None,
    *,
    thread_count: int | None = None,
    include_progress_callback: bool = True,
    use_numba: bool = True,
    **kwargs,
):
    """A decorator that parallelizes the numba.njit decorator with threads.

    The first argument of the decorated function must be an iterable.
    A range-object will be most performant as iterable.
    The original function must accept a single element of this iterable
    as its first argument.
    Important note: the type of the first argument will change to iterable for the wrapped function!

    The original function cannot return values, instead it should store
    results in e.g. one if its input arrays that acts as a buffer array.
    The original function needs to be numba.njit compatible.
    Numba argument "nogil" is always set to True.

    Parameters
    ----------
    _func : callable, None
        The function to decorate. Default is None.
    thread_count : int, None
        The number of threads to use. This is always parsed with alphatims.utils.set_threads.
        Default is None.
    include_progress_callback : bool
        If True, the default progress callback will be used as callback. (See "progress_callback" function and
        `set_progress_callback` for callback styles.)
        If False, no callback is added.
        Default is True.
    use_numba : bool
        If True, the function is compiled with numba.njit.
        If False, the function is not compiled (this is handy for debugging and unit testing).
        Default is True.

    Returns
    -------
    Callable:
        A thread-parallelized numba.njit decorated function.

    """
    import functools
    import threading

    import numba
    import numpy as np

    def _handle_progress_callback(iterable, progress_counter) -> None:  # noqa: ANN001
        import time

        granularity = 1000 if len(iterable) > 10**6 else len(iterable)
        progress_bar = 0
        progress_count = np.sum(progress_counter)
        for _ in progress_callback(
            range(granularity), include_progress_callback=include_progress_callback
        ):
            while progress_bar >= progress_count:
                time.sleep(0.01)  # this will be done by the main thread
                progress_count = granularity * np.sum(progress_counter) / len(iterable)
            progress_bar += 1  # noqa: SIM113

    def _parallel_compiled_func_inner(func: Callable) -> Callable:  # noqa: C901
        wrapped_func = numba.njit(nogil=True, **kwargs)(func) if use_numba else func

        @conditional_njit(use_numba=use_numba, nogil=True)
        def wrapped_func_parallel(  # noqa: PLR0913
            iterable,  # noqa: ANN001
            thread_id: int,
            progress_counter: int,
            start: int,
            stop: int,
            step: int,
            *args,
        ) -> None:
            if len(iterable) == 0:
                for i in range(start, stop, step):
                    wrapped_func(
                        i, *args
                    )  # here, the first argument of the wrapped function is a single index
                    progress_counter[thread_id] += 1
            else:
                for i in iterable:
                    wrapped_func(i, *args)
                    progress_counter[thread_id] += 1

        def wrapper(iterable, *args) -> None:  # noqa: ANN001
            """A wrapper function that parallelizes the numba.njit function.

            The first argument of the wrapped function is seperately stored as `iterable` and its elements are
            subsequently passed to the original function.
            """
            if thread_count is None:
                current_thread_count = MAX_THREADS
            else:
                current_thread_count = set_threads(thread_count, set_global=False)

            threads = []
            progress_counter = np.zeros(current_thread_count, dtype=np.int64)

            for thread_id in range(current_thread_count):
                thread_local_iterable = iterable[thread_id::current_thread_count]
                if isinstance(
                    thread_local_iterable, range
                ):  # TODO: does the speedup mentioned in the docstring still apply?
                    start = thread_local_iterable.start
                    stop = thread_local_iterable.stop
                    step = thread_local_iterable.step
                    thread_local_iterable = np.array([], dtype=np.int64)
                else:
                    start = -1
                    stop = -1
                    step = -1

                thread = threading.Thread(
                    target=wrapped_func_parallel,
                    args=(
                        thread_local_iterable,
                        thread_id,
                        progress_counter,
                        start,
                        stop,
                        step,
                        *args,
                    ),
                    daemon=True,
                )
                thread.start()
                threads.append(thread)

            if include_progress_callback:
                _handle_progress_callback(iterable, progress_counter)

            for thread in threads:
                thread.join()
                del thread

        return functools.wraps(func)(wrapper)

    if _func is None:
        return _parallel_compiled_func_inner
    return _parallel_compiled_func_inner(_func)


def progress_callback(  # noqa: ANN201
    iterable,  # noqa: ANN001
    *,
    include_progress_callback: bool = True,
    total: int = -1,
):
    """A generator that adds progress callback to an iterable.

    Parameters
    ----------
    iterable
        An iterable.
    include_progress_callback : bool
        If True, the default progress callback will be used as callback.
        If False, no callback is added.
        See `set_progress_callback` for callback styles.
        Default is True.
    total : int
        The length of the iterable.
        If -1, this will be read as len(iterable), if __len__ is implemented.
        Default is -1.

    Returns
    -------
    : iterable
        A generator over the iterable with added callback.

    """
    global PROGRESS_CALLBACK  # noqa: PLW0602
    current_progress_callback = PROGRESS_CALLBACK if include_progress_callback else None
    if total == -1:
        total = len(iterable)
    if current_progress_callback is None:
        for element in iterable:
            yield element
    elif isinstance(current_progress_callback, bool) and current_progress_callback:
        import tqdm

        with tqdm.tqdm(total=total) as progress_bar:
            for element in iterable:
                yield element
                progress_bar.update()
    else:
        try:
            current_progress_callback.max = total
            current_progress_callback.value = 0
        except AttributeError:
            raise ValueError("Not a valid progress callback") from None
        steps = current_progress_callback.max / 1000
        progress = 0
        for element in iterable:
            progress += 1  # noqa: SIM113
            if progress % steps < 1:
                current_progress_callback.value = progress
            yield element
        current_progress_callback.value = total


def set_progress_callback(progress_callback) -> None:  # noqa: ANN001
    """Set the global progress callback.

    Parameters
    ----------
    progress_callback :
        The new global progress callback.
        Options are:

            - None, no progress callback will be used
            - True, a textual progress callback (tqdm) will be enabled
            - Any object that supports a `max` and `value` variable.

    """
    global PROGRESS_CALLBACK  # noqa: PLW0603
    PROGRESS_CALLBACK = progress_callback
