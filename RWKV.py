import os
import pathlib
import tokenizers
import time
import sampling
import rwkv_cpp_model
import rwkv_cpp_shared_library

from functools import partial
from typing import Any, Dict, List, Mapping, Optional, Set

from pydantic import Extra, Field, root_validator

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens


class RWKV(LLM):
    tokens_per_generation: int = 100
    temperature: float = 0.8
    top_p: float = 0.5
    tokenizer_path = '/home/ubuntu/rwkv.cpp/rwkv/20B_tokenizer.json'
    tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
    model_path = '/home/ubuntu/model-Q8_0'
    thread_count = 8
    gpu_layers_count = 32
    model = None


    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @staticmethod
    def _model_param_names() -> Set[str]:
        return {
            "tokens_per_generation",
            "temperature",
            "top_p",
            "tokenizer_path",
            "model_path",
            "thread_count",
            "gpu_layers_count"
        }

    def _default_params(self) -> Dict[str, Any]:
        return {
            "tokens_per_generation": self.tokens_per_generation,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "tokenizer_path": self.tokenizer_path,
            "model_path": self.model_path,
            "thread_count": self.thread_count,
            "gpu_layers_count": self.gpu_layers_count
        }

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in the environment."""

        library = rwkv_cpp_shared_library.load_rwkv_shared_library()
        cls.model = rwkv_cpp_model.RWKVModel(
                                library,
                                cls.model_path,
                                thread_count=cls.thread_count,
                                gpu_layers_count=cls.gpu_layers_count)
        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model_path,
            **self._default_params(),
            **{
                k: v for k, v in self.__dict__.items() if k in self._model_param_names()
            },
        }

    @property
    def _llm_type(self) -> str:
        """Return the type of llm."""
        return "RWKV"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        text_callback = None
        if run_manager:
            text_callback = partial(run_manager.on_llm_new_token, verbose=self.verbose)
        text = ""

        prompt_tokens = self.tokenizer.encode(prompt).ids
        prompt_token_count = len(prompt_tokens)
        print(f'{prompt_token_count} tokens in prompt')

        init_logits, init_state = None, None

        for token in prompt_tokens:
            init_logits, init_state = self.model.eval(token, init_state, init_state, init_logits)

        start = time.time()

        logits, state = init_logits.clone(), init_state.clone()

        for i in range(self.tokens_per_generation):
            token = sampling.sample_logits(logits, self.temperature, self.top_p)
            decoded_token = self.tokenizer.decode([token])
            if text_callback:
                text_callback(decoded_token)
            text += decoded_token
            if stop is not None:
                text = enforce_stop_tokens(text, stop)
            logits, state = self.model.eval(token, state, state, logits)

        delay = time.time() - start
        print('\n\nTook %.3f sec, %d ms per token' % (delay, delay / self.tokens_per_generation * 1000))
        return text
