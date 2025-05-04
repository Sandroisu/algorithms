def binary_search(list_of_numbers, item):
    lowest = 0
    highest = len(list_of_numbers) - 1
    while lowest <= highest:
        middle = (lowest + highest)//2
        guess = list_of_numbers[middle]
        if guess == item:
            return middle
        if guess > item:
            highest = middle - 1
        else:
            lowest = middle + 1
    return None


my_list = [1, 3, 5, 7, 9]

print(binary_search(my_list, 7))
print(binary_search(my_list, -1))
