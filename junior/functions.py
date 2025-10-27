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