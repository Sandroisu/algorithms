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


def greet(name, greeting="Hello"):
    print(f"{greeting}, {name}!")


args = ("Alice",)
kwargs = {"greeting": "Hi"}

greet(*args, **kwargs)


def f(*args, **kwargs):
    print(args)
    print(kwargs)


f(1, 2, a=3, b=4)


def create_model(layers, **kwargs):
    print("Создаём модель с параметрами:")
    print(f"Слои: {layers}")
    print(f"Другие параметры: {kwargs}")


create_model(
    layers=[64, 32, 10],
    activation="relu",
    dropout=0.2,
    learning_rate=0.001
)


def train_model(data, **params):
    print(f"Training on {data} with parameters: {params}")

train_model("dataset.csv", learning_rate=0.01, batch_size=32, optimizer="adam")