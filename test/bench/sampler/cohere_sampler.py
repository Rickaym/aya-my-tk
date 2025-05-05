import time
from typing import Optional
import cohere

from models import MessageList, SamplerBase

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
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = cohere.Client()

    def _pack_message(self, role, content):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> str:
        trial = 0
        while True:
            try:
                # Convert message list to Cohere format
                messages = message_list.copy()
                chat_history = []

                # Add system message if provided
                preamble = self.system_message if self.system_message else None

                # Convert messages to Cohere format
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    if role == "user":
                        chat_history.append({"role": "USER", "message": content})
                    elif role == "assistant":
                        chat_history.append({"role": "CHATBOT", "message": content})

                # Get the last user message
                user_message = None
                for msg in reversed(messages):
                    if msg["role"] == "user":
                        user_message = msg["content"]
                        break

                if not user_message:
                    user_message = messages[-1]["content"]

                response = self.client.chat(
                    model=self.model,
                    message=user_message,
                    chat_history=chat_history[:-1] if chat_history else None,
                    preamble=preamble,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )

                return response.text
            except cohere.CohereAPIError as e:
                if "rate_limit" in str(e).lower():
                    exception_backoff = 2**trial  # exponential back off
                    print(
                        f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                        e,
                    )
                    time.sleep(exception_backoff)
                    trial += 1
                else:
                    # unknown error shall throw exception
                    raise