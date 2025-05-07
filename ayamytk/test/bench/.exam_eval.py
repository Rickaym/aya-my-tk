import cohere
from typing import TypedDict, List, Dict, Any
from litellm import completion
from utils import load_csv_as_dicts
import litellm
import logging
import time
import json
import csv
import os
from gradio_client import Client

litellm.set_verbose = False
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


TF_PROMPT = """အောက်ပါတို့ကို မှားလျှင် (မှား)၊ မှန်လျှင် (မှန်) ဟု ဖြေဆိုပါ

{question}
"""

MULTIPLE_CHOICE_PROMPT = """အောက်ပါတို့မှ အဖြေမှန်ကို ရွေးပါ

{question}

{choices}
"""

SHORT_QNA_TEMPLATE = """အောက်ပါတို့ကို ဖြေဆိုပါ

{question}
"""


FIB_PROMPT = """အောက်ပါတို့ကို ကွက်လပ်ဖြည့်ပါ

{question}
"""

litellm_models = [
    # "gpt-4o",
    # "gemini/gemini-2.0-flash",
    # "openrouter/google/gemma-3-12b-it",
    # "openrouter/google/gemma-3-27b-it",
    # "openrouter/deepseek/deepseek-chat"
    "openrouter/anthropic/claude-3.7-sonnet"
]


cohere_models = [
    # "c4ai-aya-expanse-8b",
    # "c4ai-aya-expanse-32b",
    # "command-r7b-12-2024",
    # "command-r-08-2024"
]

gradio_client_models = [
    # "sail/Sailor2-20B-Chat",
    # "sail/Sailor-14B-Chat"
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
    TRUE_OR_FALSE: List[ChoiceBased] = load_csv_as_dicts("G12-Exam/tof_qna.csv")
    SHORT_QNA_DATA: List[Question] = load_csv_as_dicts("G12-Exam/short_qna.csv")
    MULTIPLE_CHOICE: List[ChoiceBased] = load_csv_as_dicts(
        "G12-Exam/multiple_choice_qna.csv"
    )
    FILL_IN_BLANK: List[Question] = load_csv_as_dicts("G12-Exam/fib.csv")
    print("Benchmark data loaded successfully.")
    print(f"TRUE_OR_FALSE: {len(TRUE_OR_FALSE)} questions")
    print(f"SHORT_QNA_DATA: {len(SHORT_QNA_DATA)} questions")
    print(f"MULTIPLE_CHOICE: {len(MULTIPLE_CHOICE)} questions")
    print(f"FILL_IN_BLANK: {len(FILL_IN_BLANK)} questions")

    # Process MULTIPLE_CHOICE to split Choices into separate columns
    ANSWER_MAP = {
        "(က)": "A",
        "(ခ)": "B",
        "(ဂ)": "C",
    }
    NEW_MULTIPLE_CHOICE = []
    for item in MULTIPLE_CHOICE:
        if not item["Answer"]:
            continue

        try:
            choices = item["Choices"].strip().split('\n')

            # Create a new dictionary with the correct TypedDict fields plus the choice options
            processed_item = {
                "Chapter": item["Chapter"].strip(),
                "Title": item["Title"].strip(),
                "Question": item["Question"].strip().replace("\n", " "),
                "Answer": ANSWER_MAP[item["Answer"]],
            }

            # Add choice options as separate entries
            processed_item["A"] = choices[0].strip().replace("(က) ", "").strip()
            processed_item["B"] = choices[1].strip().replace("(ခ) ", "").strip()
            processed_item["C"] = choices[2].strip().replace("(ဂ) ", "").strip()

            # Update the item with the processed version
            NEW_MULTIPLE_CHOICE.append(processed_item)
        except:
            print(f"Error processing MULTIPLE_CHOICE: {item}")
            continue
    # Save processed MULTIPLE_CHOICE data back to CSV
    if NEW_MULTIPLE_CHOICE:
        with open("G12-Exam/multiple_choice_processed.csv", "w", newline="", encoding="utf-8") as f:
            fieldnames = ["Chapter",  "Title", "Question", "Answer", "A", "B", "C"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for item in NEW_MULTIPLE_CHOICE:
                writer.writerow(item)
        print("Processed MULTIPLE_CHOICE data saved to multiple_choice_processed.csv")

    # Print a sample of processed MULTIPLE_CHOICE data
    if MULTIPLE_CHOICE:
        print(f"MULTIPLE_CHOICE (processed): {MULTIPLE_CHOICE[0]}")
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
        "question": question_data.get("Question", "").strip(),
        "choices": question_data.get("Choices", "").strip(),  # Needed for TF and MC
    }
    try:
        # Only include 'choices' if the template actually uses it
        if "{choices}" in template:
            return template.format(
                question=format_kwargs["question"], choices=format_kwargs["choices"]
            ).strip()
        else:
            return template.format(question=format_kwargs["question"]).strip()
    except KeyError as e:
        print(
            f"Warning: Formatting error for template. Missing key {e} for question: {question_data.get('Question')}"
        )
        # Fallback to just the question text if formatting fails
        return question_data.get("Question", "Missing Question Text")
    except Exception as e:
        print(f"Warning: An unexpected error occurred during prompt formatting: {e}")
        return question_data.get("Question", "Missing Question Text")


def exam_fix():
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

                try:
                    response = gradio_models[model].predict(
                        formatted_prompt, api_name="/chat"
                    )
                    question_results["model_responses"][model] = response
                except Exception as e:
                    error_message = f"Error: {type(e).__name__} - {e}"
                    print(f"    Error querying {model} (Gradio): {error_message}")
                    question_results["model_responses"][model] = error_message

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


# if __name__ == "__main__":
#     main()
