from typing import Optional
from openai import OpenAI

def call_llm_api(
    user_prompt: str,
    system_prompt: str,
    api_base: Optional[str],
    api_key: Optional[str],
    model_name: str,
    max_tokens: int,
    temperature: float,
):
    client = OpenAI(api_key=api_key, base_url=api_base)
    # try:
    #     models = client.models.list()
    #     dynamic_model_id = models.data[0].id if models.data else model_name
    #     print(dynamic_model_id)
    # except Exception:
    #     dynamic_model_id = model_name
        
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        extra_body={"max_completion_tokens": max_tokens},
    )
    response_content = response.choices[0].message.content
    messages.append(
        {"role": "assistant", "content": response_content}
    )

    # usage = response.usage
    # with open("token_usage.log", "a") as f:
    #     f.write(
    #         f"prompt_tokens={usage.prompt_tokens}, "
    #         f"completion_tokens={usage.completion_tokens}, "
    #         f"total_tokens={usage.total_tokens}\n"
    #     )

    return messages
