import functools
from time import time

def timing(f):
    @functools.wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"TIMING: function: {f.__name__}, time: {te - ts}")
        return result

    return wrap
