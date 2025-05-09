## Integration of LMStudio as a Server Using Pydantic AI

Given the widespread usage and popularity of LM Studio in developing local applications based on LLMs, 
I decided to enable the use of the **LM Studio as a Local LLM API Server** by integrating it with the Pydantic AI agent development framework.
LM Studio supports a wide range of language models. Therefore, I aimed to cover the most essential functionalities that are commonly used across all supported models.
Although an official Python SDK for LMStudio has already been developed, it was intentionally not used in this project. 
This decision led to faster and simpler implementation, while also reducing external dependencies.

### Example usage:

```python

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

```

You can run the example by:

```shell
python -m pydantic_model.example
```

### Setup environment

```shell
uv sync
```

```shell
source .venv/bin/activate

```