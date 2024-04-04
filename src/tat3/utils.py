import pandas as pd
import yaml
import xml.etree.ElementTree
import json
from io import StringIO
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing import Any

@dataclass
class Table(object):
    table_description: str
    table: pd.DataFrame
    meta_data: dict

    @classmethod
    def from_path(cls, path: Path):
        query_id__table_id = path.stem
        with open(path.parent / f"{query_id__table_id}_meta.json") as f:
            table_meta = json.load(f)
        table = pd.read_csv(path, dtype=str)

        table_descriptions = [
            table_meta["pgTitle"],
            table_meta["secondTitle"],
            table_meta["caption"],
        ]
        table_descriptions = [
            description.strip()
            for description in table_descriptions
            if description.strip()
        ]
        table_descriptions = list(dict.fromkeys(table_descriptions))
        table_description = " - ".join(table_descriptions)

        return cls(table_description, table, table_meta)

class PipelineBlock(ABC):
    @abstractmethod
    def __call__(self, *args) -> Any:
        pass

    @abstractmethod
    def dump_config(self) -> dict[str, Any]: # type: ignore
        pass


def sorted_csvs_from_folder(path: str | Path) -> list[Path]:
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Path {path} is not a directory.")

    return sorted(path.rglob("*.csv"))

def xml_prompt_to_messages(prompt: str) -> list[dict[str, str]]:
    prompt_as_xml = xml.etree.ElementTree.fromstring(prompt)

    messages = []

    for message in prompt_as_xml:
        messages.append({"role": message.tag, "content": message.text})

    return messages


def load_object_from_yaml(path: str | Path) -> Any:

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Path {path} is not a file.")

    with open(path) as f:
        object_dict = yaml.load(f, Loader=yaml.FullLoader)

    return load_object_from_dict(object_dict)

from tat3.benchmarks import *
from tat3.pipelines import *
from tat3.predictors import *
from tat3.prosers import *
from tat3.extractors import *
from tat3.tutors import *

def load_object_from_dict(object_dict: dict | list) -> Any:
    if isinstance(object_dict, list):
        return [load_object_from_dict(x) for x in object_dict]
    
    if not isinstance(object_dict, dict):
        return object_dict
    
    if "class" not in object_dict:
        return object_dict
    
    if "meta" in object_dict:
        del object_dict["meta"]

    class_name = object_dict["class"]
    class_ = globals()[class_name]
    object_dict.pop("class")

    for kwarg, value in object_dict.items():
        if isinstance(value, dict):
            object_dict[kwarg] = load_object_from_dict(value)

    return class_(**object_dict)
    
