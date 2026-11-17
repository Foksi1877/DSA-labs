#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ЛР 6. Линейный, бинарный и интерполяционный поиск — стартовая заготовка.

Заготовка — это каркас, а не решение: заполните все TODO самостоятельно
в соответствии с КИМ-06 и правилами использования генеративного ИИ
(docs/ai-verification.md).

Каждая функция поиска работает с отсортированным по неубыванию массивом
и возвращает пару (индекс найденного элемента или -1, число сравнений ключей).

Запуск: python lab06-search-starter.py --variant N
"""
from __future__ import annotations

import argparse
import bisect
import random
import statistics
import time
import math
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Алгоритмы поиска (реализуются вручную, без in/index/bisect в основной части)
# ---------------------------------------------------------------------------


def linear_search(a: list[int], key: int) -> tuple[int, int]:
    """Линейный поиск первого вхождения key.

    Возвращает (индекс первого вхождения или -1, число сравнений ключей).
    Ожидаемая сложность: O(n) в худшем случае, O(1) в лучшем.
    """
    comparisons = 0
    for i in range(len(a)):
        comparisons += 1
        if a[i] == key:
            return i, comparisons
    return -1, comparisons


def binary_search(a: list[int], key: int) -> tuple[int, int]:
    """Бинарный поиск key в отсортированном массиве.

    Возвращает (индекс любого вхождения или -1, число сравнений ключей).
    Аккуратно поддерживайте инвариант обеих границ lo и hi: искомый элемент,
    если он есть, всегда лежит в текущем отрезке; цикл обязан завершаться.
    Ожидаемая сложность: O(log n).
    """
    lo = 0
    hi = len(a) - 1
    comparisons = 0

    while lo <= hi:
        mid = (lo + hi) // 2
        comparisons += 1

        if a[mid] == key:
            return mid, comparisons
        elif a[mid] < key:
            lo = mid + 1
        else:
            hi = mid - 1

    return -1, comparisons


def interpolation_search(a: list[int], key: int) -> tuple[int, int]:
    """Интерполяционный поиск key в отсортированном массиве.

    Ожидаемая сложность: O(log log n) на равномерных данных, O(n) в худшем.
    """
    lo = 0
    hi = len(a) - 1
    comparisons = 0

    while lo <= hi and a[lo] <= key <= a[hi]:
        comparisons += 1

        # Если все элементы одинаковые — защита от деления на ноль
        if a[hi] == a[lo]:
            if a[lo] == key:
                return lo, comparisons
            return -1, comparisons

        # Вычисляем позицию зонда
        mid = lo + (key - a[lo]) * (hi - lo) // (a[hi] - a[lo])

        # Проверяем, что не вышли за границы из-за погрешности вычислений
        if mid < lo: mid = lo
        if mid > hi: mid = hi

        if a[mid] == key:
            return mid, comparisons
        elif a[mid] < key:
            lo = mid + 1
        else:
            hi = mid - 1

    # Если цикл не сработал, но массив не пуст, возвращаем -1 с 1 сравнением
    if len(a) > 0:
        return -1, max(comparisons, 1)
    return -1, 0

# ---------------------------------------------------------------------------
# 2. Данные по варианту: два распределения, присутствующие/отсутствующие ключи
# ---------------------------------------------------------------------------


def make_dataset(n: int, distribution: str, rng: random.Random) -> list[int]:
    """Отсортированный массив n ключей: 'uniform' — равномерное распределение,
    'exponential' — неравномерное (экспоненциальное, значения сгущаются у нуля)."""
    if distribution == "uniform":
        return sorted(rng.randint(0, 10 * n) for _ in range(n))
    if distribution == "exponential":
        return sorted(int(rng.expovariate(1.0 / 1000.0)) for _ in range(n))
    raise ValueError(f"неизвестное распределение: {distribution!r}")


def pick_keys(a: list[int], rng: random.Random, count: int,
              present: bool) -> list[int]:
    """count ключей для серии поисков: присутствующие берутся из массива,
    отсутствующие подбираются вне множества его значений."""
    if present:
        return [a[rng.randrange(len(a))] for _ in range(count)]
    have = set(a)
    hi = (a[-1] if a else 0) + 2
    keys: list[int] = []
    while len(keys) < count:
        k = rng.randint(-hi, hi)
        if k not in have:
            keys.append(k)
    return keys


# ---------------------------------------------------------------------------
# 3. Репрезентативные тесты и инварианты (шаги 1–3 методики верификации)
# ---------------------------------------------------------------------------

ALGORITHMS = [
    ("linear", linear_search),
    ("binary", binary_search),
    ("interpolation", interpolation_search),
]


def self_check() -> None:
    """Граничные случаи + сверка с эталоном (оператор in, list.index, bisect)."""
    for name, fn in ALGORITHMS:
        assert fn([], 1) == (-1, 0), f"{name}: пустой массив"
        assert fn([7], 7)[0] == 0, f"{name}: один элемент, ключ найден"
        assert fn([7], 3)[0] == -1, f"{name}: один элемент, ключ отсутствует"
        a = [1, 3, 3, 5, 8, 13, 21]
        assert fn(a, 1)[0] == 0, f"{name}: первый элемент"
        assert fn(a, 21)[0] == len(a) - 1, f"{name}: последний элемент"
        assert fn(a, 4)[0] == -1, f"{name}: отсутствующий ключ внутри диапазона"
        assert fn(a, 100)[0] == -1, f"{name}: ключ вне диапазона"

    rng = random.Random(0)
    for _ in range(300):
        a = sorted(rng.randint(0, 60) for _ in range(rng.randint(1, 40)))
        key = rng.randint(-5, 65)
        j = bisect.bisect_left(a, key)
        present = j < len(a) and a[j] == key
        assert present == (key in a)
        for name, fn in ALGORITHMS:
            idx, cmp_count = fn(a, key)
            assert cmp_count >= 1, f"{name}: счётчик сравнений не ведётся"
            if present:
                assert idx != -1 and a[idx] == key, f"{name}: ключ не найден"
            else:
                assert idx == -1, f"{name}: найден отсутствующий ключ"
        if present:
            assert linear_search(a, key)[0] == a.index(key)
    print("self_check: OK")


# ---------------------------------------------------------------------------
# 4. Бенчмарк: сравнения и время от n (методика — docs/reproducibility.md)
# ---------------------------------------------------------------------------

SIZES = [1_000, 3_000, 10_000, 30_000, 100_000]
REPEATS = 5
N_KEYS = 200


def bench(fn, a: list[int], keys: list[int]) -> float:
    """Медиана времени одной серии поисков по REPEATS запускам, с прогревом."""
    for k in keys:
        fn(a, k)
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        for k in keys:
            fn(a, k)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def mean_comparisons(fn, a: list[int], keys: list[int]) -> float:
    """Среднее число сравнений ключей на один поиск."""
    return statistics.mean(fn(a, k)[1] for k in keys)


def run_benchmarks(seed: int) -> None:
    rng = random.Random(seed)

    # Словари для сохранения результатов для графиков
    results_comparisons = {}
    results_time = {}

    for distribution in ("uniform", "exponential"):
        print(f"\nРаспределение: {distribution}")
        results_comparisons[distribution] = {name: [] for name, _ in ALGORITHMS}
        results_time[distribution] = {name: [] for name, _ in ALGORITHMS}

        for n in SIZES:
            a = make_dataset(n, distribution, rng)
            for present in (True, False):
                keys = pick_keys(a, rng, N_KEYS, present)
                label = "есть" if present else "нет"
                for name, fn in ALGORITHMS:
                    if name == "linear" and n > 30_000:
                        continue

                    comp = mean_comparisons(fn, a, keys)
                    t = bench(fn, a, keys)

                    print(f"  n={n:>7}  ключ {label:<4}  {name:<13}"
                          f"  сравнений={comp:>10.1f}"
                          f"  t={t:.6f} c")

                    if present:
                        results_comparisons[distribution][name].append((n, comp))
                        results_time[distribution][name].append((n, t))

    for distribution in ("uniform", "exponential"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # График сравнений
        for name, _ in ALGORITHMS:
            points = results_comparisons[distribution][name]
            if points:
                xs = [math.log10(n) for n, _ in points]
                ys = [math.log10(c) if c > 0 else 0 for _, c in points]
                ax1.plot(xs, ys, marker='o', label=name)
        ax1.set_title(f"Сравнения (log-log): {distribution}")
        ax1.set_xlabel("log10(n)")
        ax1.set_ylabel("log10(сравнений)")
        ax1.grid(True)
        ax1.legend()

        # График времени
        for name, _ in ALGORITHMS:
            points = results_time[distribution][name]
            if points:
                xs = [math.log10(n) for n, _ in points]
                ys = [math.log10(t) for _, t in points]
                ax2.plot(xs, ys, marker='o', label=name)
        ax2.set_title(f"Время (log-log): {distribution}")
        ax2.set_xlabel("log10(n)")
        ax2.set_ylabel("log10(время), с")
        ax2.grid(True)
        ax2.legend()

        fig.tight_layout()
        fig.savefig(f"lab06_search_{distribution}.png", dpi=150)
        print(f"\nГрафик сохранён: lab06_search_{distribution}.png")


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