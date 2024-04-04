from pathlib import Path
import pandas as pd
from collections import Counter
import subprocess
import yaml
import os
from tqdm import tqdm


class TrecLogger(object):
    def __init__(self, log_path: str | Path) -> None:
        super().__init__()

        self._log_path = Path(log_path)
        if self._log_path.exists():
            raise FileExistsError(f"Log file {self._log_path} already exists.")

    def log(self, table_id: str, results: list[str]) -> None:
        flushed_values = set()
        with open(self._log_path, "a") as f:
            if len(results) == 0:
                results.append("[[:no_result:]]")
            for i, result in enumerate(results):
                sim = 2 ** (-i)
                result = TrecLogger.escape(result)
                if result in flushed_values:
                    continue
                flushed_values.add(result)
                f.write(
                    f"{table_id}\tQ0\t{result}\t{i+1}\t{sim}\t{self._log_path.stem}\n"
                )

    @staticmethod
    def escape(s: str):
        if s == "":
            return "[[:empty:]]"
        s = s.replace(" ", "[[:space:]]")
        s = s.replace("#", "[[:pound:]]")
        s = s.replace("\t", "[[:tab:]]")
        return s
    
    @staticmethod
    def unescape(s: str) -> str:
        s = s.replace("[[:space:]]", " ")
        s = s.replace("[[:pound:]]", "#")
        s = s.replace("[[:tab:]]", "\t")
        return s


def output_completion(
    completion: str, path: str | Path
) -> None:
    path = Path(path)
    with open(path, "a") as f:
        f.write("\n####################\n")
        f.write(completion)
        f.write("\n####################\n")

def get_sorted_run_folders(path: str | Path) -> list[Path]:
    path = Path(path)
    folders = []
    for candidate in sorted(path.iterdir(), key=lambda x: os.path.getctime(x)):
        if not candidate.is_dir():
            continue
        if not (candidate / "trec.log").exists():
            continue
        folders.append(candidate)

    return folders
    


class QRelFile(object):
    def __init__(self, path: str | Path) -> None:
        super().__init__()

        self.path = Path(path)
        if not self.path.exists():
            raise FileExistsError(f"QRel file {self.path} does not exist.")
        
        self._content = pd.read_csv(self.path, sep="\t", header=None)
        self._content.columns = ["query_id", "0", "doc_id", "relevance"]
    
    def get_gt(self, query_id: str) -> list[str]:
        return self._content[self._content.query_id == query_id].sort_values("relevance", ascending=False).doc_id.tolist()
    
    def get_query_ids(self) -> list[str]:
        return self._content.query_id.unique().tolist()

class ResultsFile(object):
    def __init__(self, path: str | Path) -> None:
        super().__init__()

        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Results file {self.path} does not exist.")

        self._content = pd.read_csv(self.path, sep="\t", header=None)
        self._content.columns = ["query_id", "0", "doc_id", "rank", "score", "run_id"]

    def get_results(self, query_id: str) -> list[str]:
        return self._content[self._content.query_id == query_id].sort_values("score", ascending=False).doc_id.tolist()

class TrecResult(object):
    def __init__(self, results_path: str | Path, qrel_path: str | Path) -> None:
        super().__init__()

        self._results_path = Path(results_path)
        self._qrel_path = Path(qrel_path)
        
        self._content = self._run_trec_eval()

    def _run_trec_eval(self):
        command = f"/home/schnell/MasterThesis/data/trec_eval/trec_eval -q -m success.1,5 -m P.5 {self._qrel_path} {self._results_path}"
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        if process.returncode != 0:
            print(output.decode())
        
        trec_result = pd.DataFrame(
            [line.split() for line in output.decode().split("\n") if line != ""],
            columns=["metric", "query", "score"],
        )

        trec_result["score"] = trec_result["score"].astype(float)

        return trec_result
    
    def metrics_all(self) -> dict[str, float]:
        rows_all = self._content[self._content["query"] == "all"]

        return {metric: score for metric, score in zip(rows_all["metric"], rows_all["score"])}

    def covered_queries(self) -> dict[str, list[str]]:
        covered_queries = {}
        metrics = self._content["metric"].unique()
        for metric in metrics:
            covered_queries[metric] = self._content[self._content["metric"] == metric & self._content["score"] > 0]["query"].tolist()

        return covered_queries
    
    def num_of_queries(self) -> int:
        return len(self._content["query"].unique()) - 1 # -1 for "all" query
    
    def __str__(self) -> str:
        return (
            f"queries={self.num_of_queries():4d}, " +
            ", ".join([f"{metric}={score:.1%}" for metric, score in sorted(self.metrics_all().items(), key=lambda x: x[0][0], reverse=True)])
        )


class SubjectSuggestionRun(object):
    data_folder = Path("/home/schnell/MasterThesis/data")

    def __init__(self, path: str | Path) -> None:
        super().__init__()

        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Run folder {self.path} does not exist.")
        
        self.benchmark = yaml.safe_load((self.path / "benchmark.yaml").read_text())
        self.task = yaml.safe_load((self.path / "task.yaml").read_text())

        self.completed = (self.path / "keep.flag").exists()
        self.debug = (self.path / "debug.flag").exists()
        self.validation = "validation" in self.benchmark["folder"]

        self.results = ResultsFile(self.path / "trec.log")
        self.qrel = QRelFile(self._infer_qrel_path())
        self.trec_result = TrecResult(self.results.path, self.qrel.path)

    def name(self) -> str:
        return self.path.name
    
    def get_query(self, query_id: str) -> tuple[list[str], list[str]]:
        return (
            list(map(TrecLogger.unescape, self.results.get_results(query_id))),
            list(map(TrecLogger.unescape, self.qrel.get_gt(query_id))),
        )
    
    def get_query_ids(self) -> list[str]:
        return self.qrel.get_query_ids()


    
    def _infer_qrel_path(self) -> Path:
        ground_truth_folder_name = "ground_truth_benchmark" if not self.validation else "ground_truth_validation"
        seeds = self.benchmark["seeds"]
        return SubjectSuggestionRun.data_folder / f"{ground_truth_folder_name}/seeds_{seeds}.qrel"
        
    
    def get_baseline_results(self) -> list[tuple[str, TrecResult]]:
        seeds = self.benchmark["seeds"]
        baseline_gt_path = SubjectSuggestionRun.data_folder / f"baselines/ground_truth/gt_e{seeds}.txt"
        baseline_output_path = SubjectSuggestionRun.data_folder / f"baselines/output"
        baseline_results = []
        for path in tqdm(sorted(baseline_output_path.glob(f"BL3_co*{seeds}.txt"), key=lambda x: os.path.getsize(x), reverse=True), leave=False):
            baseline_trec_result = TrecResult(path, baseline_gt_path)
            baseline_results.append((path.name, baseline_trec_result))

        return baseline_results
    
    def __str__(self) -> str:
        return (
            f"name={self.path.name: >19}, seeds={self.benchmark['seeds']}, " +
            str(self.trec_result) +
            (" (debug)" if self.debug else "") +
            (" (validation)" if self.validation else "")) #  + ", temperature=" + str(self.task["predictor"]["decoding_params"]["temperature"]) + ", top_p=" + (str(self.task["predictor"]["decoding_params"]["top_p"]) if "top_p" in self.task["predictor"]["decoding_params"] else "1")



