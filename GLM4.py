from zhipuai import ZhipuAI
import os
from typing import List, Optional, Literal, Union
import dataclasses
import time
# ZHIPU_API_KEY = "" # your zhipu api key
ZHIPU_API_KEY  = "Your API Key"
zhipuClient = ZhipuAI(api_key = ZHIPU_API_KEY)
model="glm-4"
def glm_chat(
    model: str,
    messages,
    max_tokens: int = 8192,
    temperature:  float = 0.2,
    num_comps=1
) -> Union[List[str], str]:
    # print(model, max_tokens, temperature, top_p, num_comps)
    # print(f"Sending messages to GLM: {[message.content for message in messages]}")
    response = zhipuClient.chat.completions.create(
        model=model,
        messages=[dataclasses.asdict(message) for message in messages],
        max_tokens=max_tokens,
        temperature=temperature
    )

    # set time to wait x seconds for limited requests per minute
    time.sleep(1)
    if num_comps == 1:
        return response.choices[0].message.content  # type: ignore

    return [choice.message.content for choice in response.choices]  # type: ignore