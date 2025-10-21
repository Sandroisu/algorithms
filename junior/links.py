def add_one(numbers):
    numbers.append(1)


a = [0]

add_one(a)
print(a)


def reassign_list(numbers):
    numbers = [4, 5, 6]
    print(numbers)


b = [0]

reassign_list(b)
print(b)  # всё ещё [0]


def mutate(a_list):
    a_list += [4, 5]


def reassign(a_list):
    a_list = a_list + [6, 7]
    print(a_list)


x = [1, 2, 3]
mutate(x)
print("After mutate:", x)

reassign(x)
print("After reassign:", x)