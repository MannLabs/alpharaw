def performance_function(
    _func: callable = None,
    *,
    worker_count: int = None,
    compilation_mode: str = None,
    **decorator_kwargs,
) -> callable:
    """A decorator to compile a given function and allow multithreading over an multiple indices.

    NOTE This should only be used on functions that are compilable.
    Functions that need to be decorated need to have an `index` argument as first argument.
    If an iterable is provided to the decorated function,
    the original (compiled) function will be applied to all elements of this iterable.
    The most efficient way to provide iterables are with ranges, but numpy arrays work as well.
    Functions can not return values,
    results should be stored in buffer arrays inside thge function instead.

    Args:
        worker_count (int): The number of workers to use for multithreading.
            If None, the global MAX_WORKER_COUNT is used at runtime.
            Default is None.
        compilation_mode (str): The compilation mode to use. Will be forwarded to the `compile_function` decorator.
        **decorator_kwargs: Keyword arguments that will be passed to numba.jit or cuda.jit compilation decorators.

    Returns:
        callable: A decorated function that is compiled and parallelized.

    """
    if worker_count is not None:
        worker_count = set_worker_count(worker_count, set_global=False)
    if compilation_mode is None:
        if DYNAMIC_COMPILATION_ENABLED:
            compilation_mode = "dynamic"
        else:
            compilation_mode = COMPILATION_MODE
    elif COMPILATION_MODE.startswith("python"):
        compilation_mode = "python"
    else:
        is_valid_compilation_mode(compilation_mode)
    def _decorated_function(func):
        if compilation_mode != "dynamic":
            compiled_function = compile_function(
                func,
                compilation_mode=compilation_mode,
                **decorator_kwargs
            )
        def _parallel_python(
            compiled_function,
            iterable,
            start,
            stop,
            step,
            *func_args
        ):
            if start != -1:
                for index in range(start, stop, step):
                    compiled_function(index, *func_args)
            else:
                for index in iterable:
                    compiled_function(index, *func_args)
        _parallel_numba = numba.njit(nogil=True)(_parallel_python)
    
        def _performance_function(iterable, *func_args):
            if compilation_mode == "dynamic":
                selected_compilation_mode = COMPILATION_MODE
                _compiled_function = compile_function(
                    func,
                    compilation_mode=selected_compilation_mode,
                    **decorator_kwargs
                )
            else:
                _compiled_function = compiled_function
                selected_compilation_mode = compilation_mode
            try:
                iter(iterable)
            except TypeError:
                iterable = np.array([iterable])
            if worker_count is None:
                selected_worker_count = MAX_WORKER_COUNT
            else:
                selected_worker_count = worker_count
            if selected_compilation_mode == "cuda":
                _parallel_cuda(_compiled_function, iterable, *func_args)
            else:
                if "python" in selected_compilation_mode:
                    parallel_function = _parallel_python
                elif "numba" in selected_compilation_mode:
                    parallel_function = _parallel_numba
                else:
                    raise NotImplementedError(
                        f"Compilation mode {selected_compilation_mode} is not valid. "
                        "This error should not be possible, something is seriously wrong!!!"
                    )
                if (selected_compilation_mode in ["python", "numba"]) or (selected_worker_count == 1):
                    iterable_is_range = isinstance(iterable, range)
                    x = np.empty(0, dtype=np.int64) if iterable_is_range else iterable
                    parallel_function(
                        _compiled_function,
                        np.empty(0, dtype=np.int64) if iterable_is_range else iterable,
                        iterable.start if iterable_is_range else -1,
                        iterable.stop if iterable_is_range else -1,
                        iterable.step if iterable_is_range else -1,
                        *func_args
                    )
                else:
                    workers = []
                    for worker_id in range(selected_worker_count):
                        local_iterable = iterable[worker_id::selected_worker_count]
                        iterable_is_range = isinstance(local_iterable, range)
                        worker = threading.Thread(
                            target=parallel_function,
                            args=(
                                _compiled_function,
                                np.empty(0, dtype=np.int64) if iterable_is_range else local_iterable,
                                local_iterable.start if iterable_is_range else -1,
                                local_iterable.stop if iterable_is_range else -1,
                                local_iterable.step if iterable_is_range else -1,
                                *func_args
                            )
                        )
                        worker.start()
                        workers.append(worker)
                    for worker in workers:
                        worker.join()
                        del worker
        return functools.wraps(func)(_performance_function)
    if _func is None:
        return _decorated_function
    else:
        return _decorated_function(_func)