import os
from typing import Any, Dict, Optional
from langchain_ollama import ChatOllama
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

load_dotenv()

class BaseAgent:
    """Base class for agents with shared Ollama configuration and retry logic."""
    
    def __init__(self, model_name: str, temperature: float = 0, format: Optional[str] = None):
        self.model_name = model_name
        self.temperature = temperature
        self.format = format
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
    def get_llm(self):
        """Returns a configured ChatOllama instance."""
        return ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
            base_url=self.base_url,
            format=self.format,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def ainvoke_llm(self, messages: list, tags: list = None, **kwargs) -> Any:
        """Invokes the LLM asynchronously with retry logic."""
        llm = self.get_llm()
        config = {"tags": tags} if tags else {}
        return await llm.ainvoke(messages, config=config, **kwargs)
