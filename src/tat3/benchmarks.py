import time
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm

from tat3.utils import sorted_csvs_from_folder, Table
from tat3.evaluation import TrecLogger

# Typing imports
from typing import Any, Literal
from tat3.pipelines import Pipeline
from collections.abc import Generator

# # # # # # # # # # # # # # # # # # # # # #
# # # # Base classes for benchmarks # # # #
# # # # # # # # # # # # # # # # # # # # # #


class Benchmark(ABC):
    def __init__(self) -> None:
        super().__init__()

        self._tables_folder: Path
        self.output_folder: Path

    @abstractmethod
    def queries(self) -> Generator[Table, None, None]:
        raise NotImplementedError

    @abstractmethod
    def dump_config(self) -> dict:
        raise NotImplementedError

    def __call__(self, task: Pipeline, debug: bool = False) -> None:
        logger = BenchmarkLogger(self, task)
        if debug:
            logger.debug()

        queries = list(self.queries())
        for query in tqdm(queries):
            start = time.time()
            result = task(query)
            stop = time.time()
            logger.log(query.meta_data["queryId"], result, start - stop)

        logger.keep()


class BenchmarkLogger(object):
    def __init__(self, benchmark: Benchmark, task: Pipeline):
        super().__init__()
        self._start_time = time.time()

        self._benchmark = benchmark
        self._task = task

        self._run_folder = benchmark.output_folder / time.strftime("%Y-%m-%d_%H-%M-%S")
        self._run_folder.mkdir(exist_ok=True, parents=True)

        with open(self._run_folder / "benchmark.yaml", "w") as f:
            yaml.dump(self._benchmark.dump_config(), f)

        with open(self._run_folder / "task.yaml", "w") as f:
            yaml.dump(self._task.dump_config(), f)

        self._trec_logger = TrecLogger(self._run_folder / "trec.log")

    def keep(self) -> None:
        (self._run_folder / "keep.flag").touch()

    def debug(self) -> None:
        (self._run_folder / "debug.flag").touch()

    def log(self, query_id: str, results: list[str], time: float) -> None:
        self._trec_logger.log(query_id, results)
        with open(self._run_folder / "times.log", "a") as f:
            f.write(f"{query_id}\t{time}\n")


# # # # # # # # # # # # # # # # # # # # # # #
# # # # Subject suggestion benchmark  # # # #
# # # # # # # # # # # # # # # # # # # # # # #


class SubjectSuggestionBenchmark(Benchmark):

    def __init__(
            self,
            folder: str = "data/benchmark",
            seeds: Literal[1, 2, 3, 4, 5] = 1,
            ) -> None:
        super().__init__()
        self._folder = Path(folder)
        self.output_folder = Path("data/output")

        self.seeds = seeds

    def queries(self) -> Generator[Table, None, None]:
        for path in sorted_csvs_from_folder(self._folder):
            query = Table.from_path(path)
            query.meta_data["seeds"] = self.seeds
            query.meta_data["ground_truth"] = query.table.iloc[self.seeds:, 0].tolist()
            query.table = query.table.iloc[:self.seeds]
            yield query

    def dump_config(self) -> dict:
        return {
            "class": self.__class__.__name__,
            "seeds": self.seeds,
            "folder": self._folder.as_posix(),
            "meta": {
                "output_folder": self.output_folder.as_posix(),
            }
        }

