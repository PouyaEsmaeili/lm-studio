from ctypes import pythonapi

## Integration of LMStudio as a Server Using Pydantic AI

Given the widespread usage and popularity of LM Studio in developing local applications based on LLMs, 
I decided to enable the use of the LMStudio as a server feature by integrating it with the Pydantic AI agent development framework.
LM Studio supports a wide range of language models. Therefore, I aimed to cover the most essential functionalities that are commonly used across all supported models.
Although an official Python SDK for LMStudio has already been developed, it was intentionally not used in this project. 
This decision led to faster and simpler implementation, while also reducing external dependencies.

## How to run?

```python
python -m pydantic_model.model
```