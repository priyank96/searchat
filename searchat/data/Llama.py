import torch

from functools import partial
from typing import Any, Dict, List, Mapping, Optional, Set

from langchain.llms.base import LLM
from pydantic import Extra, root_validator
from langchain.llms import HuggingFacePipeline
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
from langchain.callbacks.manager import CallbackManagerForLLMRun

class Llama(LLM):
    tokenizer_name: str = 'openlm-research/open_llama_3b'
    model_name: str = 'openlm-research/open_llama_3b'
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
        tokenizer = LlamaTokenizer.from_pretrained(values["tokenizer_name"])
        model = LlamaForCausalLM.from_pretrained(
            values["model_name"],
            torch_dtype=torch.float16,
            device_map=values["device_map"],
            offload_folder="offload",
            load_in_8bit=values["load_in_8bit"]
        )
        pipe = pipeline(
            "text-generation",
            model=tokenizer,
            tokenizer=model,
            max_length=256,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.2
        )
        values["model_pipeline"] = HuggingFacePipeline(pipeline=pipe)
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
        print(prompt)
        text_callback = None
        if run_manager:
            text_callback = partial(run_manager.on_llm_new_token, verbose=self.verbose)

        text = self.model_pipeline(prompt)
        if text_callback:
            text_callback(text)
        return text
