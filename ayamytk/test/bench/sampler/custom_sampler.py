import time
from typing import Callable
from traceback import print_exc
from ayamytk.test.bench.models import MessageList, SamplerBase, SamplerResponse


class CustomSampler(SamplerBase):
    """
    Custom call sample.
    """

    def __init__(self, chat: Callable[[MessageList], str]):
        self.chat = chat

    def _pack_message(self, role, content):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        trial = 0
        while True:
            try:
                response = self.chat(message_list)
                return SamplerResponse(
                    response_text=response,
                    actual_queried_message_list=message_list,
                    response_metadata={},
                )
            except Exception as e:
                exception_backoff = 2**trial  # exponential back off
                print(
                    f"Exception so wait and retry {trial} after {exception_backoff} sec",
                )
                print_exc()
                time.sleep(exception_backoff)
                trial += 1
                # unknown error shall throw exception
