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