from abc import ABC, abstractmethod
from ..config import get_required_env

class LLMUnavailableError(RuntimeError):
    pass

class LLMProvider(ABC):
    @property
    @abstractmethod
    def model_id(self) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError
    
class OpenAILLMProvider(LLMProvider):
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        try:
            from openai import OpenAI
            self._model = model or get_required_env('LLM_MODEL')
            self._api_key = api_key or get_required_env('OPENAI_API_KEY')
            self._client = OpenAI(api_key = self._api_key)
            
        except Exception as e:
            raise LLMUnavailableError(f'Could not initialize OpenAI client: {e}')
        
    @property
    def model_id(self) -> str:
        return self._model
    
    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError('Prompt cannot be empty')
        
        try:
            response = self._client.responses.create(model = self._model, input = prompt)
            
        except Exception as e:
            raise LLMUnavailableError(f'Language model generation failed: {e}')
        return response.output_text