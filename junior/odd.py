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

