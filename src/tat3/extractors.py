from abc import ABC, abstractmethod
import regex
from ast import literal_eval
from typing import Any

from tat3.utils import PipelineBlock


class Extractor(PipelineBlock):
    @abstractmethod
    def __call__(self, text: str) -> list[str]:
        raise NotImplementedError

    def dump_config(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
        }


class FirstLineExtractor(Extractor):
    def __call__(self, text: str) -> list[str]:
        return [text.split("\n")[0]]


class FirstCSVTokenExtractor(Extractor):
    def __call__(self, text: str) -> list[str]:
        text = text.strip()
        if text.startswith('"'):
            tokens = regex.findall(r'^"([^"]*)"', text)
            return [tokens[0] if len(tokens) > 0 else ""]
        else:
            return [text.split(",")[0]]
        
class JSONListExtractor(Extractor):
    def __init__(self, lead="") -> None:
        super().__init__()

        self._lead = lead


    def __call__(self, text: str) -> list[str]:
        text = self._lead + text
        text = text.strip()
        if len(text) == 0:
            return []
        if not text.startswith("["):
            return [text]
        
        candidate = None
        try:
            candidate = literal_eval(text)
        except (SyntaxError, ValueError):
            for i in range(1, len(text)):
                try:
                    parsed_list = literal_eval(text[:-i] + "]")
                    if len(parsed_list) > 0:
                        candidate = parsed_list
                        break
                except (SyntaxError, ValueError):
                    pass
        
        return [text] if candidate is None else list(map(str, candidate))
    
    def dump_config(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "lead": self._lead,
        }


class DebugExtractor(Extractor):
    def __init__(self, extractor: Extractor, output: bool = True) -> None:
        super().__init__()
        self._extractor = extractor
        self._output = output

    def __call__(self, text: str) -> list[str]:
        if not self._output:
            return self._extractor(text)
        
        print("------- Start DebugExtractor -------")
        print(text)
        extracted = self._extractor(text)
        print("------------------------------------")
        print(extracted)
        print("-------  End DebugExtractor  -------")
        return extracted
    
    def dump_config(self) -> dict:
        return {
            "class": self.__class__.__name__,
            "extractor": self._extractor.dump_config(),
            "output": self._output,
        }
