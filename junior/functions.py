import time


def greet(name):
    return f"Hello, {name}!"

say_hi = greet   # ← просто присваиваем функцию переменной
print(say_hi("Alice"))

def apply_twice(func, x):
    return func(func(x))

def add_five(n):
    return n + 5

result = apply_twice(add_five, 10)
print(result)


def outer():
    x = 10
    def inner():
        print(f"x = {x}")
    inner()

outer()


def make_multiplier(n):
    def multiply(x):
        return x * n
    return multiply

times3 = make_multiplier(3)
times5 = make_multiplier(5)

print(times3(10))  # 30
print(times5(10))  # 50

print(times3.__closure__[0].cell_contents)

def make_loss_function(scale):
    def loss(y_true, y_pred):
        return scale * (y_true - y_pred) ** 2
    return loss

loss_fn = make_loss_function(0.5)
print(loss_fn(10, 8))


def logger(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result}")
        return result
    return wrapper

def add(a, b):
    return a + b

add = logger(add)

add(3, 4)

def timeit(func):
    def time_wrapper():
        start = time.time()
        result = func()
        print(result)
        print(f"slow_func took {(time.time()-start)}")
        return result
    return time_wrapper




def slow_func():
    time.sleep(1)
    return "done"

timeit = timeit(slow_func)


slow_func()
