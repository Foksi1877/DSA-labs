#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ЛР 4. MergeSort, Counting, Radix и HeapSort: оценка производительности — стартовая заготовка.

Заготовка — это каркас, а не решение: заполните все TODO самостоятельно
в соответствии с КИМ-04 и правилами использования генеративного ИИ
(docs/ai-verification.md). QuickSort перенесите из собственной ЛР 3.

Запуск: python lab04-mergesort-linear-heapsort-starter.py --variant N
"""
from __future__ import annotations

import argparse
import random
import statistics
import time
import math
import sys
import matplotlib.pyplot as plt

sys.setrecursionlimit(20000)


def merge(left: list, right: list, key) -> list:
    """Стабильное слияние двух отсортированных списков в новый список.

    Стабильность: при равенстве ключей элемент из left идёт раньше.
    Ожидаемая сложность: O(n), где n - суммарная длина списков.
    """
    result = []
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        if key(left[i]) <= key(right[j]):
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def merge_sort(a: list, key=lambda x: x) -> list:
    """MergeSort: возвращает новый отсортированный список.
    Ожидаемая сложность: O(n log n) во всех случаях."""
    if len(a) <= 1:
        return list(a)
    mid = len(a) // 2
    left = merge_sort(a[:mid], key)
    right = merge_sort(a[mid:], key)
    return merge(left, right, key)


def counting_sort(a: list[int], key_max: int) -> list[int]:
    """Устойчивая сортировка подсчётом: ключи — целые из диапазона [0, key_max].

    Схема: массив счётчиков -> префиксные суммы -> расстановка с конца.
    Ожидаемые время и память: O(n + k), где k = key_max + 1.
    """
    if not a:
        return []
    k = key_max + 1
    count = [0] * k
    for x in a:
        count[x] += 1

    for i in range(1, k):
        count[i] += count[i - 1]

    result = [0] * len(a)
    for i in range(len(a) - 1, -1, -1):
        x = a[i]
        result[count[x] - 1] = x
        count[x] -= 1
    return result


def radix_sort_lsd(a: list[int]) -> list[int]:
    """RadixSort LSD по десятичным разрядам, неотрицательные целые.

    Поверх устойчивой поразрядной counting-сортировки: от младшего разряда
    к старшему. Ожидаемая сложность: O(d * n), где d - число разрядов.
    """
    if not a:
        return []
    max_val = max(a)
    exp = 1
    res = list(a)
    while max_val // exp > 0:
        count = [0] * 10
        for x in res:
            digit = (x // exp) % 10
            count[digit] += 1
        for i in range(1, 10):
            count[i] += count[i - 1]

        out = [0] * len(res)
        for i in range(len(res) - 1, -1, -1):
            digit = (res[i] // exp) % 10
            out[count[digit] - 1] = res[i]
            count[digit] -= 1
        res = out
        exp *= 10
    return res


def sift_down(a: list[int], i: int, n: int) -> None:
    """Просейка вниз элемента i в куче a[0:n] (max-heap), на месте."""
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and a[left] > a[largest]:
        largest = left
    if right < n and a[right] > a[largest]:
        largest = right
    if largest != i:
        a[i], a[largest] = a[largest], a[i]
        sift_down(a, largest, n)


def heap_sort(a: list[int]) -> None:
    """HeapSort in-place: сортирует список a на месте, без вспомогательного массива.

    Схема: построение max-кучи просейками вниз (снизу вверх, O(n)),
    затем обмен вершины с последним элементом и просейка. Сложность: O(n log n).
    """
    n = len(a)
    for i in range(n // 2 - 1, -1, -1):
        sift_down(a, i, n)
    for i in range(n - 1, 0, -1):
        a[0], a[i] = a[i], a[0]
        sift_down(a, 0, i)


def quick_sort(a: list[int]) -> list[int]:
    """QuickSort с рандомизацией опорного элемента — перенести из своей ЛР 3."""
    if len(a) <= 1:
        return list(a)
    res = list(a)

    def _qs(arr, lo, hi):
        if lo >= hi:
            return
        # Оптимизация для маленьких массивов (защита от глубокой рекурсии)
        if hi - lo < 16:
            for i in range(lo + 1, hi + 1):
                key = arr[i]
                j = i - 1
                while j >= lo and arr[j] > key:
                    arr[j + 1] = arr[j]
                    j -= 1
                arr[j + 1] = key
            return

        pivot_idx = random.randint(lo, hi)
        pivot = arr[pivot_idx]
        arr[pivot_idx], arr[hi] = arr[hi], arr[pivot_idx]
        i = lo
        for j in range(lo, hi):
            if arr[j] <= pivot:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
        arr[i], arr[hi] = arr[hi], arr[i]
        _qs(arr, lo, i - 1)
        _qs(arr, i + 1, hi)

    _qs(res, 0, len(res) - 1)
    return res


def self_check() -> None:
    """Граничные и типовые случаи, стабильность, сверка с эталоном sorted()."""
    assert merge_sort([]) == []
    assert merge_sort([7]) == [7]
    assert merge_sort([5, 5, 5]) == [5, 5, 5]
    assert counting_sort([], key_max=10) == []
    assert radix_sort_lsd([0]) == [0]
    single = [42]
    heap_sort(single)
    assert single == [42]

    rng = random.Random(0)
    pairs = [(rng.randint(0, 5), i) for i in range(100)]
    assert merge_sort(pairs, key=lambda p: p[0]) == sorted(pairs, key=lambda p: p[0])

    for _ in range(200):
        a = [rng.randint(0, 999) for _ in range(rng.randint(0, 60))]
        expected = sorted(a)
        assert merge_sort(a) == expected
        assert counting_sort(a, key_max=999) == expected
        assert radix_sort_lsd(a) == expected
        buf = list(a)
        heap_sort(buf)
        assert buf == expected
        assert sorted(buf) == expected
    print("self_check: OK")


SIZES = [1_000, 3_000, 10_000, 30_000, 100_000]
REPEATS = 5
SMALL_KEY_MAX = 100
WIDE_KEY_MAX = 10 ** 9


def make_datasets(rng: random.Random, n: int) -> dict[str, list[int]]:
    """Массивы разных распределений и диапазонов ключей для размера n."""
    almost = list(range(n))
    for _ in range(max(1, n // 100)):
        i, j = rng.randrange(n), rng.randrange(n)
        almost[i], almost[j] = almost[j], almost[i]
    return {
        "random_wide": [rng.randint(0, WIDE_KEY_MAX) for _ in range(n)],
        "random_small": [rng.randint(0, SMALL_KEY_MAX) for _ in range(n)],
        "sorted": list(range(n)),
        "reversed": list(range(n, 0, -1)),
        "almost_sorted": almost,
    }


def bench(fn, data: list[int]) -> float:
    """Медиана времени fn на копии data по REPEATS запускам, с прогревом."""
    fn(list(data))
    times = []
    for _ in range(REPEATS):
        buf = list(data)
        t0 = time.perf_counter()
        fn(buf)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def run_benchmarks(seed: int) -> None:
    rng = random.Random(seed)
    algorithms = [
        ("merge_sort", merge_sort),
        ("counting_sort", lambda a: counting_sort(a, key_max=max(a, default=0))),
        ("radix_sort_lsd", radix_sort_lsd),
        ("heap_sort", heap_sort),
        ("quick_sort (ЛР 3)", quick_sort),
        ("sorted() (эталон)", sorted),
    ]

    results = {}

    for n in SIZES:
        datasets = make_datasets(rng, n)
        print(f"\nn = {n}:")
        for dist_name, data in datasets.items():
            print(f"  распределение {dist_name}:")
            results[(n, dist_name)] = {}
            for algo_name, fn in algorithms:
                if algo_name == "counting_sort" and dist_name == "random_wide":
                    print(f"    {algo_name:>18}: пропуск (диапазон ключей)")
                    continue
                t = bench(fn, data)
                print(f"    {algo_name:>18}: t = {t:.6f} c")
                results[(n, dist_name)][algo_name] = t

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    dists = ["random_wide", "random_small", "sorted", "reversed", "almost_sorted"]

    for ax, dist in zip(axes, dists):
        for algo_name, _ in algorithms:
            if algo_name == "counting_sort" and dist == "random_wide":
                continue
            xs, ys = [], []
            for n in SIZES:
                t = results[(n, dist)].get(algo_name)
                if t is not None:
                    xs.append(math.log10(n))
                    ys.append(math.log10(t))
            if xs:
                ax.plot(xs, ys, marker='o', label=algo_name)
        ax.set_title(dist)
        ax.set_xlabel("log10(n)")
        ax.grid(True)

    axes[0].set_ylabel("log10(t), с")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig("lab04_sorting.png", dpi=150)
    print("\nГрафик сохранён: lab04_sorting.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", type=int, required=True, help="номер варианта")
    args = ap.parse_args()
    seed = 30 + args.variant
    random.seed(seed)
    self_check()
    run_benchmarks(seed)


if __name__ == "__main__":
    main()