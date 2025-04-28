import json
from litellm import completion
from model_testing import litellm_models, cohere_models, co

print("Loading benchmark results from benchmark_results.json...")
with open("benchmark_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)
print(f"Loaded data with {len(data)} question types")

for q_type, questions in data.items():
    print(f"\nProcessing question type: {q_type} with {len(questions)} questions")
    for idx, q in enumerate(questions):
        for model, response in q["model_responses"].items():
            if not response.startswith("Error:"):
                continue

            print(f"  Fixing error for {q_type}, question {idx+1}, model {model}")
            print(f"  Original error: {response}")

            formatted_prompt = q["formatted_prompt"]
            if model in litellm_models:
                print(f"  Retrying with LiteLLM model: {model}")
                response = completion(
                    model=model.replace(":free", ""),
                    messages=[{"role": "user", "content": formatted_prompt}],
                )
                data[q_type][idx]["model_responses"][model] = response.choices[
                    0
                ].message.content
                print(f"  Successfully retrieved new response")
            elif model in cohere_models:
                print(f"  Retrying with Cohere model: {model}")
                response = co.chat(
                    model=model,
                    message=formatted_prompt,
                    # Consider adding parameters like max_tokens, temperature if needed
                    # max_tokens=200,
                    # temperature=0.3
                )
                model_response_content = response.text
                data[q_type][idx]["model_responses"][model] = model_response_content
                print(f"  Successfully retrieved new response")

print("\nSaving fixed results to benchmark_results_fixed.json...")
with open("benchmark_results_fixed.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)
print("Done! Fixed results saved successfully.")
