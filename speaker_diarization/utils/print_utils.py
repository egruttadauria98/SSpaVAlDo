"""
Utility functions for printing.
"""

def print_args_decorator(func):
    """
    Decorator to print the arguments of a function before calling it.
    Use it for functions that start the experiment, so all configs can be seen in the logs.
    
    Ex:
    @print_args_decorator
    def func(*args, **kwargs):
        ...
    -> "Calling func(arg1, arg2, kwarg1=kwarg1_val, kwarg2=kwarg2_val)"
    """
    def wrapper(*args, **kwargs):
        arg_list = [repr(a) for a in args]  # Convert positional arguments to their string representation
        kwarg_list = [f"{k}={v!r}" for k, v in kwargs.items()]  # Convert keyword arguments to their string representation
        
        all_args = ', '.join(arg_list + kwarg_list)
        
        print(f"Calling {func.__name__}({all_args})")
        
        return func(*args, **kwargs)
    return wrapper