import torch

from functools import partial
from typing import Any, Dict, List, Mapping, Optional, Set

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from pydantic import Extra, root_validator
from transformers import GPTJForCausalLM, AutoTokenizer


class Llama(LLM):
    tokenizer_name: str = 'EleutherAI/gpt-j-6B'
    model_name: str = 'EleutherAI/gpt-j-6B'
    device_map: str = "auto"
    load_in_8bit: bool = True

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    @staticmethod
    def _model_param_names() -> Set[str]:
        return {
            "tokenizer_name",
            "model_name",
            "device_map",
            "load_in_8bit"
        }

    def _default_params(self) -> Dict[str, Any]:
        return {
            "tokenizer_name": self.tokenizer_name,
            "model_name": self.model_name,
            "device_map": self.device_map,
            "load_in_8bit": self.load_in_8bit
        }

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        values["tokenizer"] = AutoTokenizer.from_pretrained(values["tokenizer_name"])
        values["model"] = GPTJForCausalLM.from_pretrained(
            values["model_name"],
            device_map=values["device_map"],
            offload_folder="offload",
            load_in_8bit=values["load_in_8bit"]
        )

        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            **self._default_params(),
            **{
                k: v for k, v in self.__dict__.items() if k in self._model_param_names()
            },
        }

    @property
    def _llm_type(self) -> str:
        """Return the type of llm."""
        return "Llama"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        text_callback = None
        if run_manager:
            text_callback = partial(run_manager.on_llm_new_token, verbose=self.verbose)

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        outputs = self.model.model.generate(
                                input_ids,
                                do_sample=True,
                                temperature=0.9,
                                max_length=100,
                            )
        text = self.tokenizer.decode(outputs[0])
        if text_callback:
            text_callback(text)
        return text
