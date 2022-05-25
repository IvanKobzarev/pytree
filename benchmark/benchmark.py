import sys
from abc import ABC
from timeit import Timer

import jax._src.tree_util as jax_pytree

import pytree
import torch
import torch.utils._pytree as torch_pytree
from prettytable import PrettyTable


class BenchBase(ABC):
    def run(self, lib, x):
        getattr(self, lib)(x)


class Flatten(BenchBase):
    def __init__(self):
        pass

    def bench_args(self):
        return [
            ("empty_list", []),
            ("single_list", [torch.randn(5, 5)]),
            ("5 flat list", [torch.randn(5, 5) for _ in range(5, 5)]),
            ("100 flat list", [torch.randn(5, 5) for _ in range(100)]),
            ("dict with 5 list entry", {i: [torch.randn(5, 5)] for i in range(5)}),
        ]

    def pytree(self, x):
        pytree.tree_flatten(x)

    def torch_pytree(self, x):
        torch_pytree.tree_flatten(x)

    def jax_pytree(self, x):
        jax_pytree.tree_flatten(x)

    def __str__(self):
        return self.__class__.__name__


METRICS_NAMES = ["avg(us)", "min(us)", "max(us)"]
TIMEUNIT = "us"
TIMEUNIT_IN_SEC = 1e6
BENCH_REPEAT = 100


def time_str(secs):
    return "{0:.2f}".format(secs * TIMEUNIT_IN_SEC)


def metrics(population):
    avg = 0 if len(population) == 0 else sum(population) / len(population)
    return [time_str(m) for m in [avg, min(population), max(population)]]


def main() -> int:
    libs = [
        "torch_pytree",
        "jax_pytree",
        "pytree",
    ]

    benchmarks = [Flatten()]

    results = {}
    for lib in libs:
        results[lib] = {}
        for benchmark in benchmarks:
            results[lib][str(benchmark)] = {}
            args = benchmark.bench_args()
            for arg_name, arg in args:
                timer = Timer(lambda: benchmark.run(lib, arg))
                run_times = timer.repeat(number=1, repeat=BENCH_REPEAT)
                results[lib][str(benchmark)][arg_name] = run_times

    table = PrettyTable()
    table.field_names = ["Library", "Benchmark", "Input", *METRICS_NAMES]
    for benchmark in benchmarks:
        for arg_name, _ in benchmark.bench_args():
            for lib in libs:
                table.add_row(
                    [
                        lib,
                        str(benchmark),
                        arg_name,
                        *metrics(results[lib][str(benchmark)][arg_name]),
                    ]
                )

    print(table)

    return 0


if __name__ == "__main__":
    sys.exit(main())
