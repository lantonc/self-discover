# import openai
from openai import AzureOpenAI

import os

model_dict = {
    "GPT3": {
        "name": os.environ.get('MODEL_GPT3'),
        "key": os.environ.get('OPENAI_KEY_3'),
        "base": os.environ.get('ENDPOINT_GPT3'),
        "deployment": "pocbian",
        "tokens": 4096
    },
    "GPT3_16K": {
        "name": os.environ.get('MODEL_GPT3_16K'),
        "key": os.environ.get('OPENAI_KEY_3_16K'),
        "base": os.environ.get('ENDPOINT_GPT3_16K'),
        "deployment": "pocmigracion",
        "tokens": 16377
    },
    "GPT4": {
        "name": os.environ.get('MODEL_GPT4'),
        "key": os.environ.get('OPENAI_KEY_4'),
        "base": os.environ.get('ENDPOINT_GPT4'),
        "deployment": "pocmigracion4",
        "tokens": 32768
    }
}

def invoke(prompt: str, api_key: str = None, temp: float = 0.8, max_context: int = 2000, model_name: str = None, print_in_out=False):
    if print_in_out:
        print("PROMPT "+"="*50 )
        print(prompt)
        print("END PROMPT "+"="*50)
    client = AzureOpenAI(
                api_key=model_dict[model_name]["key"],
                api_version="2023-05-15",
                azure_endpoint=model_dict[model_name]["base"],
            )
    response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI chatbot who pays very close attention to instructions from the user - especially any instructions on how to format your response."
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ],
            model=model_dict[model_name]["deployment"],
            temperature=temp,
            max_tokens=max_context,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
    if print_in_out:
        print("RESPONSE "+"="*50)
        print(response.choices[0].message.content)
        print("END RESPONSE "+"="*50)
    return response.choices[0].message.content