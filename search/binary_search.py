def binary_search(list_of_numbers, item):
    low = 0
    high = len(list_of_numbers) - 1
    while low <= high:
        mid = (low + high)//2
        print("now mid is ", mid)
        guess = list_of_numbers[mid]
        if guess == item:
            return mid
        if guess > item:
            high = mid - 1
        else:
            low = mid + 1
    return None


my_list = [1, 3, 5, 7, 9]

print(binary_search(my_list, 7))
print(binary_search(my_list, -1))
