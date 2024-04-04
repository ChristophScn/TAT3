from abc import abstractmethod
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape
import json
from jinja2 import Environment, FileSystemLoader

from tat3.utils import PipelineBlock

from typing import Any
from tat3.utils import Table

class Proser(PipelineBlock):
    @abstractmethod
    def __call__(self, query: Table, examples: list[Table]) -> str:
        raise NotImplementedError


class Jinja2Proser(Proser):

    def __init__(self, template_path: str | Path, **kwargs: Any) -> None:
        super().__init__()
        self._template_path = Path(template_path)

        self.env = Environment(
            loader=FileSystemLoader(self._template_path.parent),
            keep_trailing_newline=True,
        )

        self._kwargs = kwargs

        self._template = self.env.get_template(self._template_path.name)
        with open(self._template_path, "r") as f:
            self._raw_template = f.read()

    def __call__(self, query: Table, examples: list[Table]) -> str:
        return self._template.render(query=query, examples=examples, xml_escape=xml_escape, enumerate=enumerate, json=json, **self._kwargs)

    def dump_config(self) -> dict:
        return {
            "class": self.__class__.__name__,
            "template_path": self._template_path.as_posix(),
            "meta": {
                "raw_template": self._raw_template,
            },
            **{
                key: (value.dump_config() if hasattr(value, "dump_config") else value)
                for key, value in self._kwargs.items()
            },
        }
