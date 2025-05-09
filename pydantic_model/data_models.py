from pydantic import BaseModel
from typing import List, Literal, Optional, Any


class ModelData(BaseModel):
    id: str
    object: Literal["model"]
    type: Literal["llm", "embeddings"]
    publisher: str
    arch: str
    compatibility_type: str
    quantization: str
    state: str
    max_context_length: int


class ModelsResponse(BaseModel):
    data: List[ModelData]
    object: Literal["list"]


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionsRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float]
    max_tokens: Optional[int]
    stream: Optional[bool]


class ChatMessage(BaseModel):
    role: Literal["assistant"]
    content: str


class ChatCompletionsChoice(BaseModel):
    index: int
    logprobs: Optional[Any]
    finish_reason: str
    message: ChatMessage


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Stats(BaseModel):
    tokens_per_second: float
    time_to_first_token: float
    generation_time: float
    stop_reason: str


class ModelInfo(BaseModel):
    arch: str
    quant: str
    format: str
    context_length: int


class Runtime(BaseModel):
    name: str
    version: str
    supported_formats: List[str]


class ChatCompletionsResponse(BaseModel):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: List[ChatCompletionsChoice]
    usage: Usage
    stats: Stats
    model_info: ModelInfo
    runtime: Runtime


class TextCompletionsRequest(BaseModel):
    model: str
    prompt: str
    temperature: Optional[float]
    max_tokens: Optional[int]
    stream: Optional[bool]
    stop: Optional[str]


class TextCompletionsChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[Any]
    finish_reason: str


class TextCompletionsResponse(BaseModel):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: List[TextCompletionsChoice]
    usage: Usage
    stats: Stats
    model_info: ModelInfo
    runtime: Runtime
