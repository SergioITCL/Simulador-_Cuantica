from time import time
def time_calculation(function):
    def wrapper(*args,string: str = "Time", time_returned: bool = False, **kwargs):
        initial_time = time()
        result = function(*args, **kwargs)
        resultant_time = time() - initial_time
        print(f"{string}", resultant_time)
        if time_returned:
            return result, resultant_time
        return result
    return wrapper


def suma(a, b):
    return a + b

b = time_calculation(suma(9,5))
print(b)