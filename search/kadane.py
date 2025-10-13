from typing import List, Tuple

def max_subarray_sum(nums: List[int]) -> int:
    """
    Возвращает максимальную сумму непрерывного подмассива.
    Алгоритм Кадане. Время: O(n), память: O(1).

    Идея:
    - идём слева направо, поддерживаем лучшую сумму,
      заканчивающуюся на текущем элементе (current).
    - если current < 0, то лучше начать заново с текущего элемента.
    """
    if not nums:
        raise ValueError("nums must be non-empty")

    best = current = nums[0]
    for x in nums[1:]:
        # либо продолжаем текущий подмассив, либо начинаем новый с x
        current = max(x, current + x)
        best = max(best, current)
    return best


def max_subarray_sum_with_indices(nums: List[int]) -> Tuple[int, int, int]:
    """
    Расширенная версия: возвращает (max_sum, left, right),
    где [left, right] — границы подмассива с максимальной суммой.

    Пример:
    nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    -> (6, 3, 6)  # подмассив [4, -1, 2, 1]
    """
    if not nums:
        raise ValueError("nums must be non-empty")

    best = current = nums[0]
    best_l = best_r = 0
    start = 0

    for i in range(1, len(nums)):
        x = nums[i]

        # Если «тащить хвост» даёт хуже, чем начать с x — начинаем заново
        if x > current + x:
            current = x
            start = i
        else:
            current = current + x

        # Обновляем глобально лучшую сумму и её границы
        if current > best:
            best = current
            best_l = start
            best_r = i

    return best, best_l, best_r


# --- маленькая проверка ---
if __name__ == "__main__":
    arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    print(max_subarray_sum(arr))                # 6
    print(max_subarray_sum_with_indices(arr))   # (6, 3, 6)
