from pydantic_model.model import LMStudioProvider, LMStudio
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings


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
