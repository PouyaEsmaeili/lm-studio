from pydantic_model.http_client import HTTPClient
from pydantic_model.data_models import (
    ModelData,
    ModelsResponse,
    Message,
    ChatCompletionsRequest,
    ChatCompletionsResponse,
    TextCompletionsRequest,
    TextCompletionsResponse
)
from typing import List, Optional

class LMStudioClient:
    def __init__(self, base_url="http://127.0.0.1:1234"):
        """
        Initialize the LM Studio client.
        
        Args:
            base_url (str): The base URL of the LM Studio server. Defaults to localhost:1234
        """
        self.base_url = base_url
        self.client = HTTPClient(base_url=base_url)
    
    def get_models(self) -> ModelsResponse:
        """
        Get all available models from the LM Studio server.
        
        Returns:
            ModelsResponse: Response containing the list of models or error information
        """
        response = self.client.get("api/v0/models")
        return ModelsResponse(**response)
    
    def get_model(self, model_id: str) -> ModelData:
        """
        Get details of a specific model by its ID.
        
        Args:
            model_id (str): The ID of the model to fetch
            
        Returns:
            ModelData: Details of the requested model
        """
        response = self.client.get(f"api/v0/models/{model_id}")
        return ModelData(**response)
    
    def create_chat_completions(
        self,
        model: str,
        messages: List[Message],
        temperature: Optional[float] = 0.7,
        max_tokens: Optional[int] = 100,
        stream: Optional[bool] = False
    ) -> ChatCompletionsResponse:
        """
        Create a chat completions request.
        
        Args:
            model (str): The model ID to use
            messages (List[Message]): List of messages in the conversation
            temperature (float, optional): Sampling temperature. Defaults to 0.7
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 100
            stream (bool, optional): Whether to stream the response. Defaults to False
            
        Returns:
            ChatCompletionsResponse: The chat completions response
        """
        request = ChatCompletionsRequest(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream
        )
        response = self.client.post("api/v0/chat/completions", json=request.model_dump())
        return ChatCompletionsResponse(**response)
    
    def create_text_completions(
        self,
        model: str,
        prompt: str,
        temperature: Optional[float] = 0.7,
        max_tokens: Optional[int] = 100,
        stream: Optional[bool] = False,
        stop: Optional[str] = None
    ) -> TextCompletionsResponse:
        """
        Create a text completions request.
        
        Args:
            model (str): The model ID to use
            prompt (str): The prompt to complete
            temperature (float, optional): Sampling temperature. Defaults to 0.7
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 100
            stream (bool, optional): Whether to stream the response. Defaults to False
            stop (str, optional): Stop sequence to end generation. Defaults to None
            
        Returns:
            TextCompletionsResponse: The text completions response
        """
        request = TextCompletionsRequest(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            stop=stop
        )
        response = self.client.post("api/v0/completions", json=request.model_dump())
        return TextCompletionsResponse(**response)
