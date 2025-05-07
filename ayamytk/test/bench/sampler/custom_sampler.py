import time
from typing import Callable
from ayamytk.test.bench.models import MessageList, SamplerBase


class CustomSampler(SamplerBase):
    """
    Custom call sample.
    """

    def __init__(self, chat: Callable[[MessageList], str]):
        self.chat = chat

    def _pack_message(self, role, content):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> str:
        trial = 0
        while True:
            try:
                response = self.chat(message_list)
                return response
            except Exception as e:
                exception_backoff = 2**trial  # exponential back off
                print(
                    f"Exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
                # unknown error shall throw exception
