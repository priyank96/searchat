from functools import partial
from pydantic import Extra, Field, root_validator
from typing import Any, Dict, List, Mapping, Optional, Set

from torch import bfloat16
from transformers import pipeline, AutoModelForCausalLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM


class AutoLM(LLM):
    tokenizer_name: str = "databricks/dolly-v2-3b"
    model_name: str = "databricks/dolly-v2-3b"
    load_in_8bit: bool = True
    device_map: str = "auto"

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    @staticmethod
    def _model_param_names() -> Set[str]:
        return {
            "tokenizer_name",
            "model_name",
            "load_in_8bit",
            "device_map"
        }

    def _default_params(self) -> Dict[str, Any]:
        return {
            "tokenizer_name": self.tokenizer_name,
            "model_name": self.model_name,
            "load_in_8bit": self.load_in_8bit,
            "device_map": self.device_map
        }

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        values["model"] = pipeline(
            model=values["model_name"],
            device_map=values["device_map"],
            torch_dtype=bfloat16,
            trust_remote_code=True,
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
        return "AutoLM"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        text_callback = None
        if run_manager:
            text_callback = partial(run_manager.on_llm_new_token, verbose=self.verbose)

        text = self.model(prompt)[0]["generated_text"]
        if text_callback:
            text_callback(text)
        return text
