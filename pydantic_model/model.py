from dataclasses import dataclass, field
from typing import Optional

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
)
from pydantic_ai.models import (
    Model,
    ModelRequestParameters,
    check_allow_model_requests,
)
from pydantic_ai.providers import Provider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import Usage
from pydantic_model.lm_studio_client import LMStudioClient


class LMStudioProvider(Provider[LMStudioClient]):
    """Provider for LM Studio API."""

    def __init__(self) -> None:
        self._client = LMStudioClient()

    @property
    def name(self) -> str:
        return "lm-studio"

    @property
    def base_url(self) -> str:
        return str(self._client.base_url)

    @property
    def client(self) -> LMStudioClient:
        return self._client


@dataclass(init=False)
class LMStudio(Model):
    """A model that uses the LM Studio API."""

    client: LMStudioClient = field(repr=False)
    _model_name: str = field(repr=False)
    _system: str = field(default="LM Studio", repr=False)

    def __init__(self, model_name: str, *, provider: LMStudioProvider) -> None:
        self._model_name = model_name
        self.client = provider.client

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def system(self) -> str:
        return self._system

    def _get_max_tokens(self, model_settings: Optional[ModelSettings]) -> Optional[int]:
        value = model_settings.get("max_tokens") if model_settings else None
        return int(value) if isinstance(value, int) else None

    def _get_temperature(
        self, model_settings: Optional[ModelSettings]
    ) -> Optional[float]:
        value = model_settings.get("temperature") if model_settings else None
        return float(value) if isinstance(value, float) else None

    def _get_stop(self, model_settings: Optional[ModelSettings]) -> Optional[str]:
        value = model_settings.get("stop_sequences") if model_settings else None
        return str(value[0]) if isinstance(value, list) and value else None

    async def request(
        self,
        messages: list[ModelRequest],
        model_settings: Optional[ModelSettings],
        model_request_parameters: Optional[ModelRequestParameters],
    ) -> tuple[ModelResponse, Usage]:
        check_allow_model_requests()

        request_body = {
            "model": self._model_name,
            **(
                {"max_tokens": max_tokens}
                if (max_tokens := self._get_max_tokens(model_settings))
                else {}
            ),
            **(
                {"temperature": temperature}
                if (temperature := self._get_temperature(model_settings))
                else {}
            ),
            # **({"stop": stop} if (stop := self._get_stop(model_settings)) else {}),
        }

        extracted_messages = []
        for message in messages:
            if message.instructions:
                extracted_messages.append(
                    {"role": "system", "content": message.instructions}
                )
            for part in message.parts:
                if part.part_kind == "user-prompt" and isinstance(part.content, str):
                    extracted_messages.append({"role": "user", "content": part.content})
                if part.part_kind == "system-prompt" and isinstance(part.content, str):
                    extracted_messages.append(
                        {"role": "system", "content": part.content}
                    )

        request_body["messages"] = extracted_messages
        response = self.client.create_chat_completions(**request_body)

        usage = Usage(
            request_tokens=response.usage.prompt_tokens,
            response_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )

        text_parts = [
            TextPart(content=choice.message.content) for choice in response.choices
        ]
        return ModelResponse(text_parts, model_name=self.model_name), usage


# Example usage
if __name__ == "__main__":
    provider = LMStudioProvider()
    model = LMStudio(model_name="phi-4", provider=provider)
    system_prompt = (
        "You are a friendly and patient English tutor helping learners improve their language skills. "
        "You explain grammar in simple terms and provide gentle corrections without overwhelming the student."
    )
    instructions = (
        "Always encourage the student, correct their mistakes kindly, and suggest better ways to phrase sentences. "
        "Focus on improving grammar, vocabulary, and conversational fluency."
    )
    user_prompt = (
        "I writed a email to my boss but I'm not sure if it's correct. Can you check this:\n"
        '"Dear Mr. John, I am writing to inform you I has finished the report yesterday. Please let me know if you have any question. Thank you."'
    )

    agent = Agent(
        instructions=instructions,
        model=model,
        system_prompt=system_prompt,
    )

    llm_settings: ModelSettings = {
        "max_tokens": 384,
        "temperature": 0.6,
        "stop_sequences": ["\n\n", "END"],
    }

    result = agent.run_sync(
        user_prompt=user_prompt,
        model_settings=llm_settings,
    )

    print(result)
