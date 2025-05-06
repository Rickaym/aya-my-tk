import json
import time
from typing import List, Dict, Any
from litellm import completion


SYSTEM_INSTRUCTION = """You are an expert Myanmar-English bilingual translator specializing in English to Myanmar translations.

# Process:
1. **Review** the English source text.
2. **Analyze** possible translation quality bottlenecks.
3. **Provide** a fully correct translation.

# Key Focus Areas:

## Translation Accuracy
- Verify that core meaning is preserved completely
- Identify mistranslations of key terms
- Check for omitted or added information
- Use natural Myanmar expressions instead of literal translations
- Preserve the tone and register of the original text

## Grammar and Syntax
- Correct word order appropriate to Myanmar grammar
- Proper use of particles (အား, ကို, ၏, မှာ, etc.)
- Appropriate verb forms and tenses
- Accurate sentence structure

## Technical Content Preservation
- DO NOT translate content inside code blocks (```...```)
- DO NOT translate HTML tags, markdown syntax, or other markup
- DO NOT translate programming variables, function names, or commands
- DO NOT leave any nontechnical common English word untranslated unless it is impossible
- Preserve all technical syntax exactly as in the original

# Output Format:
```
Source: [English text]
Analysis: [Brief quality assessment]
Translation: [Current Myanmar translation]
```"""


def format_prompt(template: str, question_data: Dict[str, Any]) -> str:
    return template.format(**question_data)


def setup():
    with open("en.json", "r") as f:
        en_data = json.load(f)
    with open("my.json", "r", encoding="utf-8") as f:
        my_data = json.load(f)

    my_map = {i["id"]: i for i in my_data}
    for i in en_data:
        my_map[i["id"]]["en_output"] = i["output"]

    with open("en_my.json", "w", encoding="utf-8") as f:
        json.dump(list(my_map.values()), f, indent=2, ensure_ascii=False)


def save(results: Dict[str, Any], results_filename = "translate_results.json"):
    try:
        with open(results_filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results successfully saved to {results_filename}")
    except Exception as e:
        print(f"Error saving results to JSON file '{results_filename}': {e}")


def main():
    # --- Main processing loop ---
    print("\n--- Starting Translation Run ---")
    start_time = time.time()
    results = {}
    model = "openrouter/google/gemini-2.5-flash-preview"

    with open("en_my.json", "r", encoding="utf-8") as f:
        data_list = json.load(f)

    benchmark_name = "bactrian-x"

    print(f"\n--- Running Translation: {benchmark_name} ---")
    benchmark_results: List[Dict[str, Any]] = []

    if not data_list:
        print(f"Warning: No data found for benchmark '{benchmark_name}'. Skipping.")
        return

    for idx, row in enumerate(data_list):
        print(f"  Processing {benchmark_name} Item {idx+1}/{len(data_list)}...")
        formatted_prompt = f"Translate the following English text to Myanmar: {row['en_output']}"

        question_results: Dict[str, Any] = {
            "question_index": row.get("id"),
            "original_output": row.get("en_output"),
            "original_answer": row.get("output"),  # Store original answer for later evaluation
            "formatted_prompt": formatted_prompt,
            "model_responses": {},
        }
        while True:
            try:
                # print(f"    Querying {model} (LiteLLM)...")
                response = completion(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_INSTRUCTION},
                        {"role": "user", "content": formatted_prompt},
                    ],
                    # Consider adding parameters like max_tokens, temperature if needed
                    # max_tokens=200,
                    # temperature=0.3
                )
                model_response_content = response.choices[0].message.content
                question_results["model_responses"][model] = model_response_content
                # print(f"      Response received.")
            except Exception as e:
                error_message = f"Error: {type(e).__name__} - {e}"
                print(f"    Error querying {model} (LiteLLM): {error_message}")
                question_results["model_responses"][model] = error_message

                # Check if it's a rate limit error
                if "rate limit" in str(e).lower() or "ratelimit" in str(e).lower():
                    print(f"Rate limit detected. Pausing for 5 minutes before continuing...")
                    time.sleep(300)  # 5 minutes in seconds
                    continue
            else:
                break
            finally:
                save(results)
                time.sleep(0.5)  # Small delay to help avoid rate limiting

        benchmark_results.append(question_results)
        print(f"  Finished Question {idx+1}/{len(data_list)}.")

        results[benchmark_name] = benchmark_results
        print(f"--- Benchmark {benchmark_name} Complete ---")

    # --- Output results ---
    end_time = time.time()
    print(
        f"\n--- All Benchmarks Complete (Total Time: {end_time - start_time:.2f} seconds) ---"
    )

    # Example: Saving results to a JSON file
    save(results)
    # Optional: Print a summary or first few results
    # print("\nSample Results:")
    # for benchmark_name, benchmark_data in results.items():
    #     if benchmark_data:
    #         print(f"\n{benchmark_name} (First Result):")
    #         print(json.dumps(benchmark_data[0], indent=2, ensure_ascii=False))
    #     else:
    #         print(f"\n{benchmark_name}: No results generated.")


if __name__ == "__main__":
    # setup()
    main()
