from abc import abstractmethod
from dataclasses import replace
import random
from pathlib import Path
import pickle
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import faiss

from tat3.utils import sorted_csvs_from_folder, Table, PipelineBlock

from typing import Literal

class Tutor(PipelineBlock):
    @abstractmethod
    def __call__(self, query: Table) -> list[Table]:
        raise NotImplementedError


class ZeroShotTutor(Tutor):
    def __call__(self, query: Table) -> list[Table]:
        return []
    
    def dump_config(self):
        return {
            "class": self.__class__.__name__,
        }

class RandomTutor(Tutor):
    def __init__(self, examples_folder: str, amount: int, seed: int = 42) -> None:
        super().__init__()
        self._seed = seed
        self._random = random.Random(seed)
        self._examples_folder = Path(examples_folder)
        self._amount = amount

        self._candidates = sorted_csvs_from_folder(examples_folder)

    def __call__(self, query: Table) -> list[Table]:
        return [
            Table.from_path(path)
            for path in self._random.sample(self._candidates, self._amount)
        ]
    
    def dump_config(self):
        return {
            "class": self.__class__.__name__,
            "examples_folder": self._examples_folder.as_posix(),
            "amount": self._amount,
            "seed": self._seed
        }

class FixedRandomTutor(Tutor):
    def __init__(self, examples_folder: str, amount: int | None = None, seed: int | None = None) -> None:
        super().__init__()
        self._seed = seed if seed is not None else random.randint(0, 100000)
        self._random = random.Random(seed)
        self._examples_folder = Path(examples_folder)
        self._og_amount = amount
        self._amount = self._random.randint(1, 5) if amount is None else amount

        self._candidates = sorted_csvs_from_folder(examples_folder)
        self._sample = self._random.sample(self._candidates, self._amount)

    def __call__(self, query: Table) -> list[Table]:
        return [
            Table.from_path(path)
            for path in self._sample
        ]
    
    def dump_config(self):
        return {
            "class": self.__class__.__name__,
            "examples_folder": self._examples_folder.as_posix(),
            "seed": self._seed,
            "amount": self._og_amount,
            "meta": {
                "sample": [path.as_posix() for path in self._sample],
                "amount": self._amount,
            },
        }



class KNNTutor(Tutor):

    def __init__(self, examples_folder: str, amount: int, order: Literal['highest first', 'lowest first', 'random'] = 'lowest first') -> None:
        super().__init__()

        
        self._examples_folder = Path(examples_folder)
        self._index_path = Path(examples_folder) / 'index.faiss'
        self._amount = amount
        self._order = order
        self._random = random.Random(42)

        self._index = faiss.read_index(self._index_path.as_posix())
        self._id_to_path = pickle.load(open(self._examples_folder / 'id_to_path.pkl', 'rb'))

        self._model_name = "sentence-transformers/all-mpnet-base-v2"
        # self._device = torch.device("cpu")
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name).to(self._device)

    def get_embeddings(self, sentences):
        encoded_input = self._tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self._device)

        with torch.no_grad():
            model_output = self._model(**encoded_input)

        attention_mask = encoded_input['attention_mask']
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float() # type: ignore
        sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings.to('cpu').numpy()
    
    

    def __call__(self, query: Table) -> list[Table]:
        joined_col = ", ".join(query.table.iloc[:, 0].tolist())
        embeddings = self.get_embeddings([joined_col])
        _, indices = self._index.search(embeddings, self._amount)
        queries = [
            Table.from_path(self._index_path.parent / self._id_to_path[idx])
            for idx in indices[0]
        ]

        match self._order:
            case 'highest first':
                return queries
            case 'lowest first':
                return queries[::-1]
            case 'random':
                return self._random.sample(queries, len(queries))
            case _:
                raise ValueError(f"Unknown order {self._order}")
    
    def dump_config(self):
        return {
            "class": self.__class__.__name__,
            "examples_folder": self._examples_folder.as_posix(),
            "amount": self._amount,
            "order": self._order,
            "meta": {
                "model_name": self._model_name,
            },
        }
    
class SampleFromOtherTutor(Tutor):

    def __init__(self, amount: int, tutor: Tutor) -> None:
        self._amount = amount
        self._tutor = tutor

        self._random = random.Random(42)

    def __call__(self, query: Table) -> list[Table]:
        pool = self._tutor(query)
        return self._random.sample(pool, self._amount)
    
    def dump_config(self):
        return {
            "class": self.__class__.__name__,
            "amount": self._amount,
            "tutor": self._tutor.dump_config(),
        }
    

class PickFromOtherTutor(Tutor):

    def __init__(self, predictor: 'Predictor', amount: int, tutor: Tutor, tries: int, link_id: str) -> None: # type: ignore
        super().__init__()
        self._predictor = predictor
        self._amount = amount
        self._tutor = tutor
        self._tries = tries
        self._link_id = link_id

        self._random = random.Random(42)

    def __call__(self, query: Table) -> list[Table]:
        pool = self._tutor(query)

        # TODO: Adapt for cell filling if it works
        reduced_query = replace(query, table=query.table.iloc[:-1, :])
        target = query.table.iloc[-1, 0]

        for _ in range(self._tries):
            examples = self._random.sample(pool, self._amount)
            with open(self._link_id, 'wb') as f:
                pickle.dump(examples, f)

            trial_results = self._predictor(reduced_query)

            Path(self._link_id).unlink()

            print(target, trial_results)
            print(len(trial_results) > 0 and trial_results[0] in query.meta_data["ground_truth"], query.meta_data["ground_truth"])

            if target in trial_results:
                return examples
        
        print("Could not find a match")
        return pool[:self._amount]
    
    def dump_config(self):
        return {
            "class": self.__class__.__name__,
            "amount": self._amount,
            "tries": self._tries,
            "link_id": self._link_id,
            "tutor": self._tutor.dump_config(),
            "predictor": self._predictor.dump_config(),
        }
    
class PickFromOtherTutorLink(Tutor):
    def __init__(self, link_id: str) -> None:
        super().__init__()

        self._link_id = link_id

    def __call__(self, query: Table) -> list[Table]:
        with open(self._link_id, 'rb') as f:
            return pickle.load(f)
        
    def dump_config(self):
        return {
            "class": self.__class__.__name__,
            "link_id": self._link_id,
        }
