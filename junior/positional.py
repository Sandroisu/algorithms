def add_all(*args):
    return sum(args)

print(add_all(1, 2, 3))  # 6


def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key} = {value}")

print_info(name="Alice", age=30)


def example(a, b=2, *args, **kwargs):
    print(f"a={a}, b={b}, args={args}, kwargs={kwargs}")

example(10, 20, 30, 40, name="Bob", active=True)