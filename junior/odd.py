from datetime import datetime

odds = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19,
        21, 23, 25, 27, 29, 31, 33, 35, 37, 39,
        41, 43, 45, 47, 49, 51, 53, 55, 57, 59]

right_this_minute = datetime.today().minute

if right_this_minute in odds:
    print("This minute seems a little odd.")
else:
    print("Not an odd minute")


def is_palindrome(text):
    cleaned = text.replace(" ", "").lower()
    reversed_text = cleaned[::-1]
    if cleaned == reversed_text:
        print("Palindrome")
    else:
        print("Not a palindrome")


user_input = input("Enter text: ")
is_palindrome(user_input)

import random


def guess_number():
    number = random.randint(1, 20)
    attempts = 0

    while True:
        user_input = input("Guess a number from 1 to 20: ")
        guess = int(user_input)
        attempts += 1

        if guess == number:
            print("Correct")
            break
        elif guess < number:
            print("Too small")
        else:
            print("Too big")

    print("Attempts:", attempts)


guess_number()


def word_stats(text):
    cleaned = text.lower()
    words = cleaned.split()
    stats = {}

    for w in words:
        if w in stats:
            stats[w] += 1
        else:
            stats[w] = 1

    for word, count in stats.items():
        print(word, count)


user_input = input("Enter text: ")
word_stats(user_input)


def is_perfect_number(n):
    total = 0
    for i in range(1, n):
        if n % i == 0:
            total += i

    if total == n:
        print("Perfect number")
    else:
        print("Not perfect")


user_input = input("Enter a number: ")
number = int(user_input)
is_perfect_number(number)


def validate_password(password):
    has_upper = False
    has_lower = False
    has_digit = False
    has_length = len(password) >= 8

    for ch in password:
        if ch.isupper():
            has_upper = True
        elif ch.islower():
            has_lower = True
        elif ch.isdigit():
            has_digit = True

    if has_upper and has_lower and has_digit and has_length:
        print("Valid")
    else:
        print("Invalid")


user_input = input("Enter password: ")
validate_password(user_input)


def sum_of_digits(n):
    text = str(n)
    total = 0

    for ch in text:
        total += int(ch)

    print(total)


user_input = input("Enter a number: ")
number = int(user_input)
sum_of_digits(number)


def second_largest(numbers):
    if len(numbers) < 2:
        print("Not enough numbers")
        return

    first = max(numbers)
    second = None

    for n in numbers:
        if n != first:
            if second is None or n > second:
                second = n

    if second is None:
        print("All numbers are equal")
    else:
        print(second)


user_input = input("Enter numbers separated by spaces: ")
parts = user_input.split()
nums = [int(x) for x in parts]

second_largest(nums)


def unique_in_order(items):
    seen = []
    for x in items:
        if x not in seen:
            seen.append(x)
    print(seen)


user_input = input("Enter items separated by spaces: ")
parts = user_input.split()
unique_in_order(parts)


def validate_email(email):
    if "@" not in email:
        print("Invalid")
        return

    parts = email.split("@")
    if len(parts) != 2:
        print("Invalid")
        return

    name = parts[0]
    domain = parts[1]

    if name == "" or domain == "":
        print("Invalid")
        return

    if "." not in domain:
        print("Invalid")
        return

    print("Valid")


user_input = input("Enter email: ")
validate_email(user_input)


def second_max(nums):
    if len(nums) < 2:
        return None

    max1 = nums[0]
    max2 = None

    for x in nums[1:]:
        if x > max1:
            max2 = max1
            max1 = x
        elif max2 is None or x > max2:
            max2 = x

    return max2

def first_unique_char(s):
    counts = {}

    for ch in s:
        if ch in counts:
            counts[ch] += 1
        else:
            counts[ch] = 1

    for ch in s:
        if counts[ch] == 1:
            return ch

    return None


def two_sum_pairs(nums, target):
    seen = set()
    pairs = []

    for x in nums:
        need = target - x
        if need in seen:
            pairs.append((need, x))
        seen.add(x)

    return pairs

def longest_unique_substring(s):
    last_pos = {}
    left = 0
    best = 0

    for right, ch in enumerate(s):
        if ch in last_pos and last_pos[ch] >= left:
            left = last_pos[ch] + 1

        last_pos[ch] = right
        best = max(best, right - left + 1)

    return best

def is_prime(n):
    if n < 2:
        return False

    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1

    return True

def is_valid_brackets(s):
    stack = []
    pairs = {
        ')': '(',
        ']': '[',
        '}': '{'
    }

    for ch in s:
        if ch in pairs.values():
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False

    return len(stack) == 0


def binary_search(nums, target):
    left = 0
    right = len(nums) - 1

    while left <= right:
        mid = (left + right) // 2

        if nums[mid] == target:
            return mid
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1