from abc import ABC, abstractmethod
import logging
from collections import Counter

import openai
from huggingface_hub import InferenceClient
import huggingface_hub.inference._text_generation
import vllm
import transformers
import torch
import tiktoken
import requests

from tat3.utils import PipelineBlock, Table, xml_prompt_to_messages

# Typing imports
from typing import Any


class Predictor(PipelineBlock):
    @abstractmethod
    def __call__(self, prompt: str) -> str:
        raise NotImplementedError

class DummyPredictor(Predictor):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, prompt: str) -> str:
        return "Dummy prediction"
    
    def dump_config(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__
        }

class DebugPredictor(Predictor):
    def __init__(self, predictor: Predictor, output: bool = True) -> None:
        super().__init__()

        self._predictor = predictor
        self._output = output

    def __call__(self, prompt: str) -> str:
        if not self._output:
            return self._predictor(prompt)

        print("------- Start DebugPredictor -------")
        print(prompt)
        print("------------------------------------")
        input("Press Enter to continue...")
        free_text_answer = self._predictor(prompt)
        print("------------------------------------")
        print(free_text_answer)
        print("-------- End DebugPredictor --------")
        input("Press Enter to continue...")
        return free_text_answer
    
    def dump_config(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "predictor": self._predictor.dump_config()
        }

class TextGenerationInferencePredictor(Predictor):
    def __init__(self, decoding_params: dict):
        super().__init__()
        self._client = InferenceClient(decoding_params["model"], headers={"X-Api-Key": "123"})

        self._decoding_params = decoding_params

    def __call__(self, prompt: str) -> str:
        decoding_params = self._decoding_params.copy()
        try:
            if "truncate" in self._decoding_params:
                del decoding_params["truncate"]
            import random
            seed = random.randint(0, 1000)
            completion = self._client.text_generation(prompt=prompt, seed=seed, **decoding_params)
        except huggingface_hub.inference._text_generation.ValidationError as e:
            if "truncate" in self._decoding_params:
                logging.warn("Prompt too long, truncating to fit the model.")
                completion = self._client.text_generation(prompt=prompt, seed=3, **self._decoding_params)
            else:
                raise e

        return completion

    def dump_config(self) -> dict[str, Any]:
        if self._decoding_params["model"].startswith("http"):
            model = requests.get(
                self._decoding_params["model"] + "/info",
                headers={"X-Api-Key": "123"}
            ).json()["model_id"]
        else:
            model = self._decoding_params["model"]
        
        return {
            "class": self.__class__.__name__,
            "decoding_params": self._decoding_params,
            "meta": {
                "model": model
            }
        }

class OpenAIPredictor(Predictor):
    max_context_lengths = {
        "gpt-3.5-turbo-instruct": 4096,
        "gpt-3.5-turbo-0125": 16384,
        "gpt-4-0125-preview": 128000,
    }

    def __init__(self, decoding_params: dict):
        super().__init__()
        self._client = openai.OpenAI(
            api_key="not public"
        )
        self._decoding_params = decoding_params

        self._encoding = tiktoken.encoding_for_model(self._decoding_params["model"])

    def __call__(self, prompt: str) -> str:
        if self._decoding_params["model"] == "gpt-3.5-turbo-instruct":
            return self._completion(prompt)
        else:
            return self._chat(prompt)
    
    def _chat(self, prompt: str) -> str:
        messages = xml_prompt_to_messages(prompt)

        while self._num_tokens_from_messages(messages, self._decoding_params["model"]) > self.max_context_lengths[self._decoding_params["model"]]:
            # Always pop user and assistant messages together
            message_to_pop = 1 if messages[0]["role"] == "system" else 0
            messages.pop(message_to_pop)
            messages.pop(message_to_pop)

        response = self._client.chat.completions.create(
            messages=messages, **self._decoding_params # type: ignore
        ) # type: ignore

        assistant_message = response.choices[0].message.content

        return assistant_message
    
    def _completion(self, prompt: str) -> str:
        tokenized_prompt = self._encoding.encode(prompt)
        max_context_length = self.max_context_lengths[self._decoding_params["model"]]
        
        # If the prompt is too long, truncate it to fit the max context length
        # snipping off the beginning of the prompt
        if len(tokenized_prompt) + self._decoding_params["max_tokens"] > max_context_length:
            tokenized_prompt = tokenized_prompt[
                -max_context_length + self._decoding_params["max_tokens"] :
            ]
            prompt = self._encoding.decode(tokenized_prompt)
            logging.warn("Truncated prompt to fit the max context length.")

        response = self._client.completions.create(prompt=prompt, **self._decoding_params)

        completion = response.choices[0].text

        return completion
    
    def _num_tokens_from_messages(self, messages: list[dict[str, str]], model: str) -> int:
        tokens_per_message = 3
        tokens_per_name = 1
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(self._encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens
    
    def dump_config(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "decoding_params": self._decoding_params,
            "meta": {
                "max_context_length": self.max_context_lengths[self._decoding_params["model"]]
            }
        }


class HFTransformersPredictor(Predictor):
    def __init__(self, model: str, decoding_params: dict):
        super().__init__()
        self._model = model
        self._decoding_params = decoding_params

        self._transformers_pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            device_map="auto",
        )

        self._tokenizer = self._transformers_pipeline.tokenizer

    def __call__(self, prompt: str) -> str:
        # Use tokenizer to truncate the prompt if it's too long
        # encoded = self._tokenizer(prompt, return_tensors="pt") # type: ignore
        # if encoded["input_ids"].shape[1] > 100:
        #     prompt = self._tokenizer.decode(encoded["input_ids"][0][-130:], skip_special_tokens=True) # type: ignore
        #     print(prompt)


        try:
            return self._transformers_pipeline(prompt, return_full_text=False, **self._decoding_params)[0]["generated_text"] # type: ignore
        except Exception as e:
            logging.error(f"Error in HFTransformersPredictor: {e}")
            return self._transformers_pipeline(prompt, return_full_text=False, truncation=True, **self._decoding_params)[0]["generated_text"] # type: ignore
    
    def dump_config(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "model": self._model,
            "decoding_params": self._decoding_params
        }

class VLLMPredictor(Predictor):
    MODEL_MAX_LENGTHS = {
        "google/gemma-7b-it": 4096,
        "mistralai/Mixtral-8x7B-Instruct-v0.1": 32768,
        "meta-llama/Llama-2-70b-hf": 200,
    }
    def __init__(self, model: str, decoding_params: dict, gpus: int):
        super().__init__()
        self._model = model
        self._decoding_params = decoding_params
        self._gpus = gpus

        self._vllm_decoding_params = vllm.SamplingParams(**decoding_params)
        self._llm = vllm.LLM(model, tensor_parallel_size=gpus)
        self._tokenizer = self._llm.get_tokenizer()

    def __call__(self, prompt: str) -> str:
        max_tokens = self._vllm_decoding_params.max_tokens
        max_tokens = 0 if max_tokens is None else max_tokens
        truncate_to = self.MODEL_MAX_LENGTHS[self._model] - max_tokens
        prompt_token_ids = self._tokenizer.encode(prompt, return_tensors="pt")

        if len(prompt_token_ids) > truncate_to:
            logging.warn("Prompt too long, truncating to fit the model.")
            prompt_token_ids = prompt_token_ids[-truncate_to:]

        return self._llm.generate(
            prompt_token_ids=prompt_token_ids.tolist(), # type: ignore
            use_tqdm=False,
            )[0].outputs[0].text

    
    def dump_config(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "model": self._model,
            "decoding_params": self._decoding_params,
            "gpus": self._gpus,
        }


