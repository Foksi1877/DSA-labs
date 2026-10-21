#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ЛР 3. Простые сортировки и QuickSort с рандомизацией — стартовая заготовка.

Заготовка — это каркас, а не решение: заполните все TODO самостоятельно
в соответствии с КИМ-03 и правилами использования генеративного ИИ
(docs/ai-verification.md).

Запуск: python lab03-simple-sorts-quicksort-starter.py --variant N
"""
from __future__ import annotations

import argparse
import random
import statistics
import time
import math
import matplotlib.pyplot as plt
# ---------------------------------------------------------------------------
# 1. Простые сортировки со счётчиками (сортируют КОПИЮ входа, вход не меняют)
#    Каждая функция возвращает кортеж: (отсортированный список,
#    число сравнений ключей, число обменов/перемещений элементов).
# ---------------------------------------------------------------------------


def bubble_sort(a: list) -> tuple[list, int, int]:
    """Сортировка пузырьком. Ожидаемая сложность: O(n²) в среднем и худшем, O(n) в лучшем.

    Подсказка: досрочный выход, если за проход не было ни одного обмена, —
    именно он даёт лучший случай на упорядоченном входе.
    """
    a = list(a)
    comparisons = 0
    swaps = 0
    n = len(a)
    for i in range(n):
        swapped = False
        for j in range(n - 1 - i):
            comparisons += 1
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                swaps += 1
                swapped = True
        if not swapped:
            break
    return a, comparisons, swaps


def insertion_sort(a: list) -> tuple[list, int, int]:
    """Сортировка вставками. Ожидаемая сложность: O(n²) в среднем, O(n) в лучшем."""
    a = list(a)
    comparisons = 0
    moves = 0
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0:
            comparisons += 1
            if a[j] > key:
                a[j + 1] = a[j]
                moves += 1
                j -= 1
            else:
                break
        a[j + 1] = key
    return a, comparisons, moves


def selection_sort(a: list) -> tuple[list, int, int]:
    """Сортировка выбором. Ожидаемая сложность: O(n²) на любом входе."""
    a = list(a)
    comparisons = 0
    swaps = 0
    n = len(a)
    for i in range(n - 1):
        min_idx = i
        for j in range(i + 1, n):
            comparisons += 1
            if a[j] < a[min_idx]:
                min_idx = j
        if min_idx != i:
            a[i], a[min_idx] = a[min_idx], a[i]
            swaps += 1
    return a, comparisons, swaps


# ---------------------------------------------------------------------------
# 2. QuickSort с рандомизированным опорным элементом
# ---------------------------------------------------------------------------


def quick_sort(a: list, rng: random.Random | None = None) -> list:
    """QuickSort с выбором опорного через rng.randrange. Возвращает новый список.

    Ожидаемая сложность: O(n log n) в среднем, O(n²) в худшем случае.
    Рандомизация опорного элемента делает худший случай маловероятным.
    """
    if rng is None:
        rng = random.Random()

    a = list(a)

    def _quick_sort(arr: list, lo: int, hi: int) -> None:
        if lo >= hi:
            return

        pivot_idx = rng.randrange(lo, hi + 1)
        pivot = arr[pivot_idx]

        arr[pivot_idx], arr[hi] = arr[hi], arr[pivot_idx]

        i = lo
        for j in range(lo, hi):
            if arr[j] <= pivot:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1

        arr[i], arr[hi] = arr[hi], arr[i]

        _quick_sort(arr, lo, i - 1)
        _quick_sort(arr, i + 1, hi)

    _quick_sort(a, 0, len(a) - 1)
    return a


# ---------------------------------------------------------------------------
# 3. Демонстрация стабильности на парах (ключ, метка)
# ---------------------------------------------------------------------------


def stability_demo() -> None:
    """Показать, какие из четырёх сортировок стабильны.

    План: взять массив пар (ключ, метка) с повторяющимися ключами, например
    [(2, "a"), (1, "b"), (2, "c"), (1, "d")], отсортировать каждой сортировкой
    по ключу (сравнение только по p[0]!) и напечатать порядок меток при равных
    ключах. Вывод о стабильности каждой сортировки включить в отчёт.
    """
    data = [(2, "a"), (1, "b"), (2, "c"), (1, "d")]
    print("\nДемонстрация стабильности:")
    print(f"Исходный массив: {data}")

    def sort_by_key(algo):
        pairs = [(p[0], p[1]) for p in data]
        if algo == "bubble":
            return bubble_sort(pairs)[0]
        elif algo == "insertion":
            return insertion_sort(pairs)[0]
        elif algo == "selection":
            return selection_sort(pairs)[0]
        elif algo == "quick":
            return quick_sort(pairs, rng=random.Random(0))

    b_res = sort_by_key("bubble")
    i_res = sort_by_key("insertion")
    s_res = sort_by_key("selection")
    q_res = sort_by_key("quick")

    print(f"bubble_sort:    {b_res} (стабильная, 'a' перед 'c')")
    print(f"insertion_sort: {i_res} (стабильная, 'a' перед 'c')")
    print(f"selection_sort: {s_res} (НЕстабильная, 'a' и 'c' могут поменяться местами)")
    print(f"quick_sort:     {q_res} (НЕстабильная, 'a' и 'c' могут поменяться местами)")


# ---------------------------------------------------------------------------
# 4. Классы входов (данные по варианту, детерминированно по seed)
# ---------------------------------------------------------------------------


def make_inputs(n: int, seed: int) -> dict[str, list[int]]:
    """Три класса входов размера n: упорядоченный, случайный, обратный."""
    rng = random.Random(seed + n)  # свой поток для каждого размера
    data = [rng.randint(-1_000_000, 1_000_000) for _ in range(n)]
    return {
        "упорядоченный": sorted(data),
        "случайный": data,
        "обратный": sorted(data, reverse=True),
    }


# ---------------------------------------------------------------------------
# 5. Верификация (шаги 1–3 методики docs/ai-verification.md)
# ---------------------------------------------------------------------------

ALGORITHMS = {
    "bubble_sort": lambda a: bubble_sort(a)[0],
    "insertion_sort": lambda a: insertion_sort(a)[0],
    "selection_sort": lambda a: selection_sort(a)[0],
    "quick_sort": lambda a: quick_sort(a, random.Random(0)),
}


def is_sorted(a: list) -> bool:
    """Инвариант 1: неубывающий порядок элементов."""
    return all(a[i] <= a[i + 1] for i in range(len(a) - 1))


def self_check() -> None:
    """Граничные случаи, инварианты сортировки и сверка с эталоном sorted()."""
    boundary = [
        [],  # пустой массив
        [7],  # один элемент
        [5, 5, 5, 5],  # все элементы равны
        [1, 2, 3, 4, 5],  # уже отсортирован
        [5, 4, 3, 2, 1],  # обратный порядок
    ]
    for name, fn in ALGORITHMS.items():
        for a in boundary:
            res = fn(list(a))
            assert is_sorted(res), f"{name}: нарушен порядок на {a}"
            assert sorted(res) == sorted(a), f"{name}: не перестановка входа {a}"

    # Сверка с эталоном из стандартной библиотеки на сотнях случайных входов.
    rng = random.Random(0)
    for _ in range(300):
        a = [rng.randint(-100, 100) for _ in range(rng.randint(0, 80))]
        expected = sorted(a)
        for name, fn in ALGORITHMS.items():
            assert fn(list(a)) == expected, f"{name}: расходится с sorted() на {a}"

    # Счётчики: у сортировки выбором ровно n*(n-1)/2 сравнений на любом входе.
    _, comparisons, _ = selection_sort([3, 1, 2, 5, 4])
    assert comparisons == 10, "selection_sort: неверный счётчик сравнений"
    print("self_check: OK")


# ---------------------------------------------------------------------------
# 6. Бенчмарк (методика — docs/reproducibility.md)
# ---------------------------------------------------------------------------

SIZES_QUADRATIC = [500, 1_000, 2_000, 4_000, 8_000]  # для простых сортировок
SIZES_QUICK = [1_000, 3_000, 10_000, 30_000, 100_000]  # для QuickSort
REPEATS = 5


def bench(fn, data: list) -> float:
    """Медиана времени выполнения fn(копия data) по REPEATS запускам, с прогревом."""
    fn(list(data))  # прогрев — не учитывается
    times = []
    for _ in range(REPEATS):
        arg = list(data)  # свежая копия: сортировка не должна получать свой результат
        t0 = time.perf_counter()
        fn(arg)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def run_benchmarks(seed: int) -> None:
    """Замеры всех алгоритмов на трёх классах входов; результат — таблица в stdout."""
    plans = [
        ("bubble_sort", ALGORITHMS["bubble_sort"], SIZES_QUADRATIC),
        ("insertion_sort", ALGORITHMS["insertion_sort"], SIZES_QUADRATIC),
        ("selection_sort", ALGORITHMS["selection_sort"], SIZES_QUADRATIC),
        ("quick_sort", lambda a: quick_sort(a, random.Random(seed)), SIZES_QUICK),
    ]

    # Словарь для сохранения результатов замеров, чтобы не делать их дважды
    results_for_plot = {cls: {} for cls in ["упорядоченный", "случайный", "обратный"]}

    for name, fn, sizes in plans:
        print(f"\n{name}:")
        for n in sizes:
            for cls, data in make_inputs(n, seed).items():
                t = bench(fn, data)
                print(f"  n={n:>7}  вход={cls:<13} t={t:.6f} c")
                # Сохраняем время для построения графика
                if name not in results_for_plot[cls]:
                    results_for_plot[cls][name] = []
                results_for_plot[cls][name].append((n, t))

    # --- Построение графиков ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    classes = ["упорядоченный", "случайный", "обратный"]

    for ax, cls in zip(axes, classes):
        for name, _, _ in plans:
            points = results_for_plot[cls][name]
            xs = [math.log10(n) for n, _ in points]
            ys = [math.log10(t) for _, t in points]
            ax.plot(xs, ys, marker="o", label=name)

        ax.set_title(f"Вход: {cls}")
        ax.set_xlabel("log10(n)")
        ax.grid(True)

    axes[0].set_ylabel("log10(t), с")

    # Правильное извлечение легенды
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("lab03_sorting_loglog.png", dpi=150)
    print("\nГрафик сохранён: lab03_sorting_loglog.png")

    # --- Вывод счётчиков ---
    print("\nСчётчики операций (на примере n=1000, случайный вход):")
    test_data = make_inputs(1000, seed)["случайный"]

    _, c_bub, s_bub = bubble_sort(test_data)
    _, c_ins, s_ins = insertion_sort(test_data)
    _, c_sel, s_sel = selection_sort(test_data)

    print(f"  bubble_sort:    сравнений {c_bub}, обменов {s_bub}")
    print(f"  insertion_sort: сравнений {c_ins}, перемещений {s_ins}")
    print(f"  selection_sort: сравнений {c_sel}, обменов {s_sel}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", type=int, required=True, help="номер варианта")
    args = ap.parse_args()
    seed = 30 + args.variant
    random.seed(seed)
    self_check()
    stability_demo()
    run_benchmarks(seed)


if __name__ == "__main__":
    main()