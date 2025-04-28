import cohere
from typing import TypedDict, List, Dict, Any
from litellm import completion
from utils import load_csv_as_dicts
from templates import (
    TF_PROMPT,
    MULTIPLE_CHOICE_PROMPT,
    SHORT_QNA as SHORT_QNA_TEMPLATE,
    FIB_PROMPT,
)
import litellm
import logging
import time
import json
import os
from gradio_client import Client

litellm.set_verbose = False
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


litellm_models = [
    # "gpt-4o",
    # "gemini/gemini-2.0-flash",
    # "openrouter/google/gemma-3-12b-it:free",
]


cohere_models = [
    # "c4ai-aya-expanse-8b",
    # "c4ai-aya-expanse-32b",
    # "command-r7b-12-2024",
    # "command-r-08-2024"
]

gradio_client_models = [
    "sail/Sailor2-20B-Chat",
    "sail/Sailor-14B-Chat"
]


class Question(TypedDict):
    Chapter: str
    Question: str
    Answer: str
    Title: str


class ChoiceBased(Question):
    Choices: str


# Load data with error handling
try:
    TRUE_OR_FALSE: List[ChoiceBased] = load_csv_as_dicts("book/tof_qna.csv")[:1]
    SHORT_QNA_DATA: List[Question] = load_csv_as_dicts("book/short_qna.csv")[:1]
    MULTIPLE_CHOICE: List[ChoiceBased] = load_csv_as_dicts(
        "book/multiple_choice_qna.csv"
    )[:1]
    FILL_IN_BLANK: List[Question] = load_csv_as_dicts("book/fib.csv")[:1]
    print("Benchmark data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading CSV data: {e}.")
    print(
        "Please ensure the 'book' directory exists and contains the necessary CSV files:"
    )
    print("- tof_qna.csv")
    print("- short_qna.csv")
    print("- multiple_choice_qna.csv")
    print("- fib.csv")
    exit(1)
except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")
    exit(1)


# Initialize Cohere client (ensure API key is configured via environment variable COHERE_API_KEY)
try:
    # Ensure COHERE_API_KEY is set in your environment variables
    if not os.getenv("COHERE_API_KEY"):
        raise ValueError("COHERE_API_KEY environment variable not set.")
    co = cohere.Client()
    print("Cohere client initialized.")
except Exception as e:
    print(f"Error initializing Cohere client: {e}.")
    exit(1)

# Define benchmark configurations
benchmarks = {
    "TRUE_OR_FALSE": {"data": TRUE_OR_FALSE, "prompt_template": TF_PROMPT},
    "SHORT_QNA": {"data": SHORT_QNA_DATA, "prompt_template": SHORT_QNA_TEMPLATE},
    "MULTIPLE_CHOICE": {
        "data": MULTIPLE_CHOICE,
        "prompt_template": MULTIPLE_CHOICE_PROMPT,
    },
    "FILL_IN_BLANK": {"data": FILL_IN_BLANK, "prompt_template": FIB_PROMPT},
}

results: Dict[str, List[Dict[str, Any]]] = {}


# --- Helper function to format prompts ---
def format_prompt(template: str, question_data: Dict[str, Any]) -> str:
    """Formats the prompt string using question data."""
    format_kwargs = {
        "question": question_data.get("Question", ""),
        "choices": question_data.get("Choices", ""),  # Needed for TF and MC
    }
    try:
        # Only include 'choices' if the template actually uses it
        if "{choices}" in template:
            return template.format(
                question=format_kwargs["question"], choices=format_kwargs["choices"]
            )
        else:
            return template.format(question=format_kwargs["question"])
    except KeyError as e:
        print(
            f"Warning: Formatting error for template. Missing key {e} for question: {question_data.get('Question')}"
        )
        # Fallback to just the question text if formatting fails
        return question_data.get("Question", "Missing Question Text")
    except Exception as e:
        print(f"Warning: An unexpected error occurred during prompt formatting: {e}")
        return question_data.get("Question", "Missing Question Text")


def main():
    # --- Main processing loop ---
    print("\n--- Starting Benchmark Runs ---")
    start_time = time.time()
    gradio_models = {}

    for benchmark_name, config in benchmarks.items():
        print(f"\n--- Running Benchmark: {benchmark_name} ---")
        data_list = config["data"]
        prompt_template = config["prompt_template"]
        benchmark_results: List[Dict[str, Any]] = []

        if not data_list:
            print(f"Warning: No data found for benchmark '{benchmark_name}'. Skipping.")
            continue

        for idx, question_data in enumerate(data_list):
            print(f"  Processing {benchmark_name} Question {idx+1}/{len(data_list)}...")
            formatted_prompt = format_prompt(prompt_template, question_data)
            question_results: Dict[str, Any] = {
                "question_index": idx,
                "original_question": question_data.get("Question"),
                "original_answer": question_data.get(
                    "Answer"
                ),  # Store original answer for later evaluation
                "formatted_prompt": formatted_prompt,
                "model_responses": {},
            }

            # Run LiteLLM models
            for model in litellm_models:
                try:
                    # print(f"    Querying {model} (LiteLLM)...")
                    response = completion(
                        model=model,
                        messages=[{"role": "user", "content": formatted_prompt}],
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
                finally:
                    time.sleep(0.5)  # Small delay to help avoid rate limiting

            # Run Cohere models
            for model in cohere_models:
                try:
                    # print(f"    Querying {model} (Cohere)...")
                    response = co.chat(
                        model=model,
                        message=formatted_prompt,
                        # Consider adding parameters like max_tokens, temperature if needed
                        # max_tokens=200,
                        # temperature=0.3
                    )
                    model_response_content = response.text
                    question_results["model_responses"][model] = model_response_content
                    # print(f"      Response received.")
                except Exception as e:
                    error_message = f"Error: {type(e).__name__} - {e}"
                    print(f"    Error querying {model} (Cohere): {error_message}")
                    question_results["model_responses"][model] = error_message
                finally:
                    time.sleep(0.5)  # Small delay

            for model in gradio_client_models:
                if model not in gradio_models:
                    gradio_models[model] = Client(model)
                response = gradio_models[model].predict(formatted_prompt, api_name="/chat")
                question_results["model_responses"][model] = response

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
    results_filename = "benchmark_results.json"
    try:
        with open(results_filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results successfully saved to {results_filename}")
    except Exception as e:
        print(f"Error saving results to JSON file '{results_filename}': {e}")

    # Optional: Print a summary or first few results
    # print("\nSample Results:")
    # for benchmark_name, benchmark_data in results.items():
    #     if benchmark_data:
    #         print(f"\n{benchmark_name} (First Result):")
    #         print(json.dumps(benchmark_data[0], indent=2, ensure_ascii=False))
    #     else:
    #         print(f"\n{benchmark_name}: No results generated.")

if __name__ == "__main__":
    main()
