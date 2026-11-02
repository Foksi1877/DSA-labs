#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ЛР 5. Бинарное дерево поиска: обходы, вставка и удаление — стартовая заготовка.

Заготовка — это каркас, а не решение: заполните все TODO самостоятельно
в соответствии с КИМ-05 и правилами использования генеративного ИИ
(docs/ai-verification.md). Класс BST понадобится повторно в ДЗ 2
(Cases/kim-02-bst-height-experiment.md) — держите его чистым и переносимым.

Запуск: python lab05-bst-starter.py --variant N
"""
from __future__ import annotations

import argparse
import random
import statistics
import time
from collections import deque

# ---------------------------------------------------------------------------
# 1. Бинарное дерево поиска (реализуется вручную, без сторонних библиотек)
# ---------------------------------------------------------------------------


class Node:
    """Узел BST: ключ и ссылки на левое и правое поддеревья."""

    __slots__ = ("key", "left", "right")

    def __init__(self, key: int) -> None:
        self.key = key
        self.left: Node | None = None
        self.right: Node | None = None


class BST:
    """Бинарное дерево поиска без дубликатов ключей.

    Инвариант: для каждого узла все ключи левого поддерева меньше key,
    все ключи правого — больше. Ожидаемая сложность операций: TODO
    (выразить через высоту h и обосновать в отчёте).
    """

    def __init__(self) -> None:
        self.root: Node | None = None

    def insert(self, key: int) -> None:
        """Вставка ключа; повторная вставка существующего ключа игнорируется.
        Ожидаемая сложность: O(h), где h - высота дерева.
        В среднем (на случайных данных) h ≈ log(n), поэтому сложность O(log n).
        В худшем случае (на упорядоченных данных) дерево вырождается, и сложность становится O(n).
        """
        if self.root is None:
            self.root = Node(key)
            return

        current = self.root
        while True:
            if key < current.key:
                if current.left is None:
                    current.left = Node(key)
                    return
                current = current.left
            elif key > current.key:
                if current.right is None:
                    current.right = Node(key)
                    return
                current = current.right
            else:
                return

    def search(self, key: int) -> bool:
        """Возвращает True, если ключ есть в дереве.
        Ожидаемая сложность: O(h), где h — высота дерева.
        В среднем O(log n), в худшем O(n)."""
        current = self.root
        while current is not None:
            if key == current.key:
                return True
            elif key < current.key:
                current = current.left
            else:
                current = current.right
        return False

    def delete(self, key: int) -> None:
        """Удаление ключа. Обработать три случая:
        1) лист — просто отцепить от родителя;
        2) один ребёнок — заменить узел его ребёнком;
        3) два ребёнка — заменить ключ преемником (минимум правого
           поддерева) и удалить преемника из правого поддерева.
        Отсутствующий ключ игнорируется.
        Сложность: O(h), где h - высота дерева (в среднем O(log n)).
        """
        def _delete(node,key):
            if node is None:
                return None
            if key < node.key:
                node.left = _delete(node.left, key)
            elif key > node.key:
                node.right = _delete(node.right, key)
            else:
                #Нашли узел, который нужно удалить
                #Случай 1 и 2: нет ребенка или только один ребенок
                if node.left is None:
                    return node.right
                elif node.right is None:
                    return node.left
                #Случай 3: два ребенка
                else:
                    #Ищем минимум в правом поддереве (преемника)
                    temp = node.right
                    while temp.left is not None:
                        temp = temp.left
                    #Заменяем ключ текущего узла на ключ преемника
                    node.key = temp.key
                    #Удаляем преемника из правого поддерева
                    node.right = _delete(node.right, temp.key)
            return node

        self.root = _delete(self.root, key)

    # -- Обходы --------------------------------------------------------------

    def in_order(self) -> list[int]:
        """Симметричный обход (лево — узел — право): рекурсивно.
         Ожидаемая сложность: O(n), так как мы посещаем каждый узел ровно один раз.
         """
        result = []
        def _in_order(node):
            if node:
                _in_order(node.left)
                result.append(node.key)
                _in_order(node.right)

        _in_order(self.root)
        return result

    def pre_order(self) -> list[int]:
        """Прямой обход (узел — лево — право): рекурсивно.
        Ожидаемая сложность: O(n).
        """
        result = []
        def _pre_order(node):
            if node:
                result.append(node.key)
                _pre_order(node.left)
                _pre_order(node.right)
        _pre_order(self.root)
        return result

    def post_order(self) -> list[int]:
        """Обратный обход (лево — право — узел): рекурсивно.
        Ожидаемая сложность: O(n).
        """
        result = []
        def _post_order(node):
            if node:
                _post_order(node.left)
                _post_order(node.right)
                result.append(node.key)
        _post_order(self.root)
        return result


    def level_order(self) -> list[int]:
        """Обход в ширину по уровням: итеративно через очередь (deque).
        Ожидаемая сложность: O(n).
        """
        if not self.root:
            return []
        result = []
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            result.append(node.key)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        return result

    # -- Характеристики дерева ----------------------------------------------

    def height(self) -> int:
        """Высота дерева: число рёбер на самом длинном пути от корня.

        Пустое дерево имеет высоту -1, дерево из одного узла — 0.
        Ожидаемая сложность: O(n), так как нужно обойти все узлы.
        """
        def _height(node):
            if not node:
                return -1
            return 1 + max(_height(node.left), _height(node.right))
        return _height(self.root)

    def size(self) -> int:
        """Число узлов в дереве.
        Ожидаемая сложность: O(n).
        """
        def _size(node):
            if not node:
                return 0
            return 1 + _size(node.left) + _size(node.right)
        return _size(self.root)


# ---------------------------------------------------------------------------
# 2. Верификация: инвариант BST и сверка с эталоном (docs/ai-verification.md)
# ---------------------------------------------------------------------------


def self_check() -> None:
    """Граничные случаи + сверка операций с эталоном set/sorted."""
    # Пустое дерево
    t = BST()
    assert t.search(1) is False
    assert t.in_order() == []
    assert t.height() == -1 and t.size() == 0
    t.delete(1)  # удаление из пустого дерева не должно падать

    # Один узел
    t.insert(5)
    assert t.search(5) and not t.search(4)
    assert t.height() == 0 and t.size() == 1

    # Три случая удаления на маленьком дереве: 5, 2, 8, 1, 3
    for k in (2, 8, 1, 3):
        t.insert(k)
    t.delete(1)                       # лист
    assert t.in_order() == [2, 3, 5, 8]
    t.delete(2)                       # узел с одним ребёнком
    assert t.in_order() == [3, 5, 8]
    t.delete(5)                       # корень с двумя детьми (через преемника)
    assert t.in_order() == [3, 8]

    # Обходы на известном дереве 4-2-6-1-3-5-7 (вставка по уровням)
    t2 = BST()
    for k in (4, 2, 6, 1, 3, 5, 7):
        t2.insert(k)
    assert t2.in_order() == [1, 2, 3, 4, 5, 6, 7]
    assert t2.pre_order() == [4, 2, 1, 3, 6, 5, 7]
    assert t2.post_order() == [1, 3, 2, 5, 7, 6, 4]
    assert t2.level_order() == [4, 2, 6, 1, 3, 5, 7]
    assert t2.height() == 2 and t2.size() == 7

    # Случайные последовательности команд: сверка с эталоном из stdlib
    rng = random.Random(0)
    for _ in range(100):
        t3 = BST()
        etalon: set[int] = set()
        for _ in range(rng.randint(1, 80)):
            k = rng.randint(-30, 30)
            if rng.random() < 0.6:
                t3.insert(k)
                etalon.add(k)
            else:
                t3.delete(k)
                etalon.discard(k)
            # Инвариант BST: in-order даёт отсортированную последовательность
            assert t3.in_order() == sorted(etalon)
        for k in range(-30, 31):
            assert t3.search(k) == (k in etalon)
        assert t3.size() == len(etalon)
    print("self_check: OK")


# ---------------------------------------------------------------------------
# 3. Бенчмарк: поиск в BST против поиска в списке (docs/reproducibility.md)
# ---------------------------------------------------------------------------

SIZES = [1_000, 3_000, 10_000, 30_000, 100_000]
REPEATS = 5
QUERIES = 1_000  # число поисков на один замер


def bench(fn) -> float:
    """Медиана времени выполнения fn() по REPEATS запускам, с прогревом."""
    fn()  # прогрев — не учитывается
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def run_benchmarks(seed: int) -> None:
    """Сравнение времени QUERIES поисков в BST и в списке (оператор in)."""
    rng = random.Random(seed)
    print(f"{'n':>8} {'h(BST)':>7} {'t_bst, c':>10} {'t_list, c':>10}")

    results_n = []
    results_bst = []
    results_list = []

    for n in SIZES:
        keys = rng.sample(range(10 * n), n)  # случайные ключи без повторов
        queries = [rng.randint(0, 10 * n) for _ in range(QUERIES)]
        tree = BST()
        for k in keys:
            tree.insert(k)

        def search_bst() -> None:
            for q in queries:
                tree.search(q)

        def search_list() -> None:
            for q in queries:
                q in keys  # линейный поиск для сравнения

        t_bst = bench(search_bst)
        t_list = bench(search_list)

        results_n.append(n)
        results_bst.append(t_bst)
        results_list.append(t_list)

        print(f"{n:>8} {tree.height():>7} {t_bst:>10.6f} {t_list:>10.6f}")

    # --- Построение графика ---
    import matplotlib.pyplot as plt
    import math

    xs = [math.log10(n) for n in results_n]
    ys_bst = [math.log10(t) for t in results_bst]
    ys_list = [math.log10(t) for t in results_list]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys_bst, marker="o", label="Поиск в BST")
    ax.plot(xs, ys_list, marker="s", label="Поиск в list")

    ax.set_xlabel("log10(n)")
    ax.set_ylabel("log10(время), с")
    ax.set_title("Сравнение поиска в BST и в списке")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig("lab05_bst_search.png", dpi=150)
    print("\nГрафик сохранён: lab05_bst_search.png")

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", type=int, required=True, help="номер варианта")
    args = ap.parse_args()
    seed = 30 + args.variant  # seed по варианту — см. docs/reproducibility.md
    random.seed(seed)
    self_check()
    run_benchmarks(seed)


if __name__ == "__main__":
    main()
