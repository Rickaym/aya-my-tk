from ayamytk.test.bench.sampler.chat_completion_sampler import ChatCompletionSampler

OpenRouterSampler = lambda model: ChatCompletionSampler(
    model=model,
    base_url="https://openrouter.ai/api/v1",
    api_key_name="OPENROUTER_API_KEY",
)
