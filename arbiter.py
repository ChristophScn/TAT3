from pathlib import Path
import pandas as pd
from collections import defaultdict


output_dir = Path('../data/output')
runs_dir = Path('../data/output')

runs = sorted([run for run in runs_dir.iterdir() if run.is_dir()], key=lambda x: x.name)

ensemble_runs = []

for i, run in enumerate(runs):
    if not (run / "trec.log").exists():
        continue
    print(f"Run {i+1}: {run.name}")

while inp := input("Run number: "):
    if inp == "all":
        ensemble_runs = runs[:]
        break
    run = runs[int(inp)-1]
    ensemble_runs.append(run)
    print(f"Added {run.name} to ensemble")

benchmarks = []
for run in ensemble_runs:
    benchmarks.append((run / "benchmark.yaml").read_text())

ensemble_runs = [run for run, benchmark in zip(ensemble_runs, benchmarks) if benchmark == benchmarks[0]]

if len(ensemble_runs) != len(benchmarks):
    print("Some runs have different benchmark.yaml files and are not included in the ensemble")

ensembled_results = defaultdict(lambda: defaultdict(list))
for run in ensemble_runs:
    df = pd.read_csv(run / "trec.log", sep="\t", header=None, names=["qid", "Q0", "docid", "rank", "score", "tag"])
    for _, row in df.iterrows():
        ensembled_results[row["qid"]][row["docid"]] += [row["score"]]

ensemble_dir = output_dir / "ensemble"
ensemble_dir.mkdir(exist_ok=True)
with open(ensemble_dir / "trec.log", "w") as f:
    for qid, doc_scores in ensembled_results.items():
        for i, (docid, score) in enumerate(sorted(doc_scores.items(), key=lambda x: (len(x[1]), sum(x[1])), reverse=True)):
            score = 2**-i
            f.write(f"{qid}\tQ0\t{docid}\t{i+1}\t{score}\tensemble\n")
    
# Copy benchmark.yaml
(ensemble_dir / "benchmark.yaml").write_text(benchmarks[0])
with open(ensemble_dir / "task.yaml", "w") as f:
    f.write(f"""
ensemble: [ {", ".join([str(run.name) for run in ensemble_runs])} ]
""")
