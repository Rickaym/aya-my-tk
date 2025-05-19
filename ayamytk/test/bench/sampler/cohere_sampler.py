import time
import cohere.client_v2 as cohere
from typing import Optional
from ayamytk.test.bench.models import MessageList, SamplerBase, SamplerResponse


class CohereSampler(SamplerBase):
    """
    Sample from Cohere API using cohere library
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
        self.client = cohere.ClientV2()

    def _pack_message(self, role, content):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        trial = 0
        while True:
            try:
                messages = message_list.copy()
                if self.system_message:
                    # Add system message if provided
                    messages = [
                        {"role": "system", "content": self.system_message}
                    ] + messages

                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )

                return SamplerResponse(
                    response_text=response.message.content[0].text,
                    actual_queried_message_list=message_list,
                    response_metadata={},
                )
            except Exception as e:
                exception_backoff = 2**trial  # exponential back off
                print(
                    f"Exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
