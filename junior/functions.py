import time
import heapq


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
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.6f} seconds")
        return result
    return wrapper



@timeit
def slow_func():
    time.sleep(1)
    return "done"

slow_func()

def top_k_largest(nums, k):
    if k <= 0:
        return []
    if k >= len(nums):
        return sorted(nums, reverse=True)

    heap = nums[:k]
    heapq.heapify(heap)

    for x in nums[k:]:
        if x > heap[0]:
            heapq.heapreplace(heap, x)

    return sorted(heap, reverse=True)

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.map = {}
        self.head = [None, None, None, None]
        self.tail = [None, None, None, None]
        self.head[3] = self.tail
        self.tail[2] = self.head

    def _remove(self, node):
        prev_node = node[2]
        next_node = node[3]
        prev_node[3] = next_node
        next_node[2] = prev_node

    def _add_to_front(self, node):
        first = self.head[3]
        node[2] = self.head
        node[3] = first
        self.head[3] = node
        first[2] = node

    def get(self, key):
        node = self.map.get(key)
        if node is None:
            return -1
        self._remove(node)
        self._add_to_front(node)
        return node[1]

    def put(self, key, value):
        node = self.map.get(key)
        if node is not None:
            node[1] = value
            self._remove(node)
            self._add_to_front(node)
            return

        if len(self.map) == self.capacity:
            lru = self.tail[2]
            self._remove(lru)
            del self.map[lru[0]]

        new_node = [key, value, None, None]
        self.map[key] = new_node
        self._add_to_front(new_node)