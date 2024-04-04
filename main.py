from tat3.utils import load_object_from_yaml
from tat3.benchmarks import SubjectSuggestionBenchmark

import sys
import os


def main():
    debug = False
    if any(arg in ["-d", "--debug", "debug"] for arg in sys.argv):
        debug = True
    

    # Start SubjectSuggestionBenchmark specific code
    all_ = None
    if any(arg in ["-a", "--all", "all"] for arg in sys.argv):
        all_ = True
    
    seeds = None
    if any(arg in ["1", "2", "3", "4", "5"] for arg in sys.argv):
        seeds = [int(arg) for arg in sys.argv if arg in ["1", "2", "3", "4", "5"]]
    if all_:
        seeds = [1, 2, 3, 4, 5]

    validation_ = None
    if any(arg in ["-v", "--validation", "validation"] for arg in sys.argv):
        validation_ = True
        if seeds is None:
            seeds = [1, 2, 3, 4, 5]
    
    benchmarks = None
    if seeds is not None:
        folder = "data/validation" if validation_ else "data/benchmark"
        benchmarks = [SubjectSuggestionBenchmark(folder=folder, seeds=seed) for seed in seeds] # type: ignore
    # End SubjectSuggestionBenchmark specific code


    if benchmarks is None:
        obj = load_object_from_yaml("config/benchmark.yaml")
        benchmarks = obj if isinstance(obj, list) else [obj]
    
    obj = load_object_from_yaml("config/pipeline.yaml")
    pipelines = obj if isinstance(obj, list) else [obj]
    
    
    for benchmark in benchmarks:
        for pipeline in pipelines:
            print(f"Benchmark: {benchmark.__class__.__name__}")
            print(f"Pipeline: {pipeline.__class__.__name__}")
            for key, value in pipeline.dump_config().items():
                if isinstance(value, dict):
                    print(f"  {key}: {value['class']}")
            
            benchmark(pipeline, debug=debug)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    main()
