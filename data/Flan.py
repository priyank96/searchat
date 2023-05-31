from functools import partial
from pydantic import Extra, Field, root_validator
from typing import Any, Dict, List, Mapping, Optional, Set

from transformers import T5Tokenizer, T5ForConditionalGeneration

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM


class Flan(LLM):
    tokenizer_name: str = "google/flan-t5-base"
    model_name: str = "google/flan-t5-base"
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
        values["tokenizer"] = T5Tokenizer.from_pretrained(values["tokenizer_name"])
        values["model"] = T5ForConditionalGeneration.from_pretrained(
            values["model_name"],
            device_map=values["device_map"],
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
        return "Flan-T5"

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
        outputs = self.model.generate(input_ids)
        text = self.tokenizer.decode(outputs[0])
        if text_callback:
            text_callback(text)
        return text
