import time
from typing import Optional
import litellm
from ayamytk.test.bench.models import MessageList, SamplerBase

class LitellmSampler(SamplerBase):
    """
    Sample from OpenRouter API using LiteLLM
    """

    def __init__(
        self,
        model: str,
        system_message: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _pack_message(self, role, content):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> str:
        trial = 0
        while True:
            try:
                # If system message exists, add it to the first message
                messages = message_list.copy()
                if self.system_message:
                    # Add system message if provided
                    messages = [{"role": "system", "content": self.system_message}] + messages

                response = litellm.completion(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                return response.choices[0].message.content
            except Exception as e:
                exception_backoff = 2**trial  # exponential back off
                print(
                    f"Exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
                # unknown error shall throw exception
