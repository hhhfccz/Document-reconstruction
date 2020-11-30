# ——*——coding:utf-8——*——
# author: hhhfccz(胡珈魁) time:2020/11/30
import time
from functools import wraps


def decorator_used_time(f):
    @wraps(f)
    def wrapped_function(*args, **kwargs):
        start = time.perf_counter()
        ret = f(*args, **kwargs)
        print('Used time: {:.2f} s'.format(time.perf_counter() - start))
        return ret
    return wrapped_function
