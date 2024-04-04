from abc import ABC, abstractmethod

from tat3.utils import Table
from tat3.predictors import Predictor
from tat3.tutors import Tutor
from tat3.prosers import Proser
from tat3.extractors import Extractor

from typing import Any


class Pipeline(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, query: Table) -> Any:
        raise NotImplementedError

    @abstractmethod
    def dump_config(self) -> dict:
        raise NotImplementedError


class SubjectSuggestionPipeline(Pipeline):
    def __init__(
            self,
            tutor: Tutor,
            proser: Proser,
            predictor: Predictor,
            extractor: Extractor
    ) -> None:
        super().__init__()

        self._tutor = tutor
        self._proser = proser
        self._predictor = predictor
        self._extractor = extractor

    def __call__(self, query: Table) -> list[str]:
        preprocessed_query = self._preprocess(query)
        examples = self._tutor(preprocessed_query)
        prompt = self._proser(preprocessed_query, examples)
        free_text_answer = self._predictor(prompt)
        raw_suggestions = self._extractor(free_text_answer)
        suggestions = self._postprocess(raw_suggestions)
        return suggestions
    
    def _preprocess(self, query: Table) -> Table:
        return query
    
    def _postprocess(self, suggestions: list[str]) -> list[str]:
        return suggestions

    def dump_config(self) -> dict:
        return {
            "class": self.__class__.__name__,
            "tutor": self._tutor.dump_config(),
            "proser": self._proser.dump_config(),
            "predictor": self._predictor.dump_config(),
            "extractor": self._extractor.dump_config(),
        }

