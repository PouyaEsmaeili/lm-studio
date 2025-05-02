from pydantic_ai.models import Model, ModelRequestParameters, check_allow_model_requests
from dataclasses import dataclass, field
from pydantic_ai.providers import Provider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ModelRequest, ModelResponsePart, TextPart
from pydantic_ai.usage import Usage
from pydantic_model.lm_studio_client import LMStudioClient

from pydantic_ai import Agent, DocumentUrl





llm_config = {
    # "contextOverflowPolicy": "stop",         # or "truncate", "warn", etc.
    "maxTokens": 512,
    "minPSampling": 0.05,
    # "promptTemplate": "{system}\n{instruction}\n{input}",  # Or use actual prompt template if needed  -> IMPORTANT
    "repeatPenalty": 1.1,
    # "stopStrings": ["<|endoftext|>", "</s>"],
    # "structured": True,  -> IMPORTANT
    "temperature": 0.7,
    # "toolCallStopStrings": ["<|toolend|>"],
    # "rawTools": [],  # or list of tool schemas
    "topKSampling": 40,
    "topPSampling": 0.95,

    # Flattened llama-specific settings
    "cpuThreads": 8,  # Adjust to your CPU

    # # Flattened reasoning settings
    # "reasoningParsing": "auto",  # or "none", "strict", etc.
    #
    # # Flattened speculative decoding settings
    # "draftModel": "gpt-3.5-turbo",  # Or your local draft model
    # "speculativeDecodingMinDraftLengthToConsider": 10,
    # "speculativeDecodingMinContinueDraftingProbability": 0.8,
    # "speculativeDecodingNumDraftTokensExact": 20
}


class LMStudioProvider(Provider[LMStudioClient]):
    """Provider for LM Studio API."""

    @property
    def name(self) -> str:
        return 'lm-studio'

    @property
    def base_url(self) -> str:
        return str(self._client.base_url)

    @property
    def client(self) -> LMStudioClient:
        return self._client

    def __init__(
            self,
    ):
        self._client = LMStudioClient()


@dataclass(init=False)
class LMStudio(Model):
    """A model that uses the LM Studio API.

    Internally, this uses the [lmstudio-python](https://github.com/lmstudio-ai/lmstudio-python)
    """
    client: LMStudioClient = field(repr=False)

    _model_name: str = field(repr=False)
    _system: str = field(default='LM Studio', repr=False)


    def __init__(
            self,
            model_name: str,
            *,
            provider: LMStudioProvider,
    ):
        self._model_name = model_name
        self.client = provider.client

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def model_name(self) -> str:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The system / model provider."""
        return self._system

    async def request(
        self,
        messages: list[ModelRequest],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters | None,
    ):
        check_allow_model_requests()

        print(messages)
        print(model_settings)
        print(model_request_parameters)

        # extract user prompt
        message = messages[0]  # Why it is a list?!
        for part in message.parts:
            if part.part_kind == 'user-prompt':
                user_prompt = part.content
                if type(user_prompt) == list: # 'document-url'
                    print("NOT SUPPORTED")
            if part.part_kind == 'system-prompt':
                system_prompt = part.content

        # extract instruction
        instructions = message.instructions

        # extract settings
        if model_settings:
            max_tokens = model_settings.get('max_tokens')  # maxTokens
            temperature = model_settings.get('temperature')  # temperature
            top_p = model_settings.get('top_p')  # topPSampling
            stop_sequences = model_settings.get('stop_sequences')  # stopStrings
            presence_penalty = model_settings.get('presence_penalty')  # repeatPenalty
            frequency_penalty = model_settings.get('frequency_penalty')  # repeatPenalty

            config = {}
            if max_tokens:
                config['maxTokens'] = max_tokens
            if temperature:
                config['temperature'] = temperature
            if top_p:
                config['topKSampling'] = top_p
            if stop_sequences:
                config['stopSequences'] = stop_sequences
            if presence_penalty:
                config['presencePenalty'] = presence_penalty
            if frequency_penalty:
                config['frequencyPenalty'] = frequency_penalty

        print("X0")
        if model_request_parameters:
            print("X1")
            output_tools = model_request_parameters.output_tools  # Why list?!
            output_tool = output_tools[0]
            parameters_json_schema = output_tool.parameters_json_schema
            print("X2")



        x = self._model.respond("hi my name is pouya. Who are you?", config=config)

        items = []
        items.append(TextPart(content="blabla"))
        return ModelResponse(items, model_name="name"), Usage()



provider = LMStudioProvider()
obj = LMStudio(model_name="phi-4", provider=provider)


agent = Agent(
    instructions="You are a concise assistant that helps with math problems.",
    model=obj,
    system_prompt="Use the customer's name while replying to them.",
)



model_settings = ModelSettings(temperature=0.5, max_tokens=100)
usage = Usage()


result = agent.run_sync(
    user_prompt=  'What is the main content of this document?',
    model_settings=model_settings,
)

print(result)

