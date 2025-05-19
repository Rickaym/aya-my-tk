import json
import os
import argparse
import pandas as pd

import sys

sys.path.append(os.path.abspath("."))

from ayamytk.test.bench import common
from ayamytk.test.bench.mmlu_eval import MMLUEval
from ayamytk.test.bench.exam_eval import ExamEval
from ayamytk.test.bench.sampler.chat_completion_sampler import ChatCompletionSampler
from ayamytk.test.bench.sampler.cohere_sampler import CohereSampler


evals_default = "mmlu,exam"
models_default = "all"
language_default = "MYA"

OpenRouterSampler = lambda model: ChatCompletionSampler(
    model=model,
    base_url="https://openrouter.ai/api/v1",
    api_key_name="OPENROUTER_API_KEY",
)

MODELS = {
    "deepseek-chat": OpenRouterSampler(model="deepseek/deepseek-chat"),
    "gemini-2.0-flash": OpenRouterSampler(
        model="google/gemini-2.0-flash-001"
    ),
    "gemma-3-4b-it": OpenRouterSampler(model="google/gemma-3-4b-it"),
    "gemma-3-12b-it": OpenRouterSampler(model="google/gemma-3-12b-it"),
    "gemma-3-27b-it": OpenRouterSampler(model="google/gemma-3-27b-it"),
    "aya-8b": CohereSampler(model="c4ai-aya-expanse-8b"),
    "aya-32b": CohereSampler(model="c4ai-aya-expanse-32b"),
    "command-r7b": CohereSampler(model="command-r7b-12-2024"),
    "command-r": CohereSampler(model="command-r-08-2024"),
    "command-a": CohereSampler(model="command-a-03-2025"),
    "gpt-4o": ChatCompletionSampler(model="gpt-4o"),
    "claude-3.7-sonnet": OpenRouterSampler(
        model="anthropic/claude-3.7-sonnet"
    ),
    "claude-3-haiku": OpenRouterSampler(model="anthropic/claude-3-haiku"),
    "qwen-2.5-7b": OpenRouterSampler(model="qwen/qwen-2.5-7b-instruct"),
    "qwen-2.5-72b": OpenRouterSampler(model="qwen/qwen-2.5-72b-instruct"),
}


def main():
    parser = argparse.ArgumentParser(
        description="Run sampling and evaluations using different samplers and evaluations."
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List available models"
    )
    parser.add_argument(
        "--model", "-m", type=str, help="Select a model by name", default=models_default
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--examples", type=int, help="Number of examples to use (overrides default)"
    )
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        help="Language to use (overrides default)",
        default=language_default,
    )
    parser.add_argument(
        "--evals",
        "-e",
        type=str,
        help="Comma-separated list of evaluations to run (e.g., 'mmlu,exam')",
        default=evals_default,
    )

    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for model_name in MODELS.keys():
            print(f" - {model_name}")
        return

    if args.model:
        if args.model == "all":
            models = MODELS
        elif args.model not in MODELS:
            print(f"Error: Model '{args.model}' not found.")
            return
        else:
            models = {args.model: MODELS[args.model]}
    else:
        models = MODELS

    # grading_sampler = OpenRouterSampler(model="google/gemini-2.0-flash")
    # equality_checker = OpenRouterSampler(model="google/gemini-2.0-flash")
    # ^^^ used for fuzzy matching, just for math
    run(
        examples=args.examples,
        debug=args.debug,
        evals=args.evals,
        samplers=models,
        language=args.language,
    )


def run(
    sampler=None,
    examples=None,
    debug=False,
    evals=evals_default,
    samplers=MODELS,
    language=language_default,
):
    def get_evals(eval_name, debug_mode):
        num_examples = examples if examples is not None else (5 if debug_mode else None)
        # Set num_examples = None to reproduce full evals
        match eval_name:
            case "mmlu":
                return MMLUEval(
                    num_examples=1 if debug_mode else num_examples, language=language
                )
            case "exam":
                return ExamEval(
                    grader_model=OpenRouterSampler(
                        model="google/gemini-2.0-flash-001"
                    ),
                    num_examples=1 if debug_mode else num_examples,
                    language=language,
                )
            case _:
                raise Exception(f"Unrecognized eval type: {eval_name}")

    # Define available evaluations
    available_evals = [
        "mmlu",
        "exam",
    ]

    if sampler:
        samplers = {sampler.__name__: sampler}

    # Determine which evals to run
    if evals:
        requested_evals = evals.split(",")
        # Check if all requested evals are valid
        for eval_name in requested_evals:
            if eval_name not in available_evals:
                print(f"Warning: Unrecognized evaluation '{eval_name}'. Skipping.")
        # Filter to only include valid requested evals
        eval_names = [e for e in requested_evals if e in available_evals]
        if not eval_names:
            print("No valid evaluations specified. Exiting.")
            return
    else:
        # Default to all available evals if none specified
        eval_names = available_evals

    evals = {eval_name: get_evals(eval_name, debug) for eval_name in eval_names}

    print(f"Running evaluations: {', '.join(eval_names)}")
    print(evals)
    debug_suffix = "_DEBUG" if debug else ""
    print(debug_suffix)
    mergekey2resultpath = {}
    for model_name, sampler in samplers.items():
        for eval_name, eval_obj in evals.items():
            result = eval_obj(sampler)
            # ^^^ how to use a sampler
            file_stem = f"./tmp/{eval_name}_{model_name}"
            os.makedirs(os.path.dirname(file_stem), exist_ok=True)
            report_filename = f"{file_stem}{debug_suffix}.html"

            print(f"Writing report to {report_filename}")
            with open(report_filename, "w", encoding="utf-8") as fh:
                fh.write(common.make_report(result))

            # Handle the case where result.metrics might be None
            if result.metrics is not None:
                metrics = result.metrics | {"score": result.score}
            else:
                metrics = {"score": result.score}

            print(metrics)
            result_filename = f"{file_stem}{debug_suffix}.json"
            with open(result_filename, "w", encoding="utf-8") as f:
                f.write(json.dumps(metrics, indent=2))
            print(f"Writing results to {result_filename}")
            mergekey2resultpath[f"{file_stem}"] = result_filename

    merge_metrics = []
    for eval_model_name, result_filename in mergekey2resultpath.items():
        try:
            result = json.load(open(result_filename, "r+"))
        except Exception as e:
            print(e, result_filename)
            continue
        result = result.get("f1_score", result.get("score", None))
        eval_name = eval_model_name[: eval_model_name.find("_")]
        model_name = eval_model_name[eval_model_name.find("_") + 1 :]
        merge_metrics.append(
            {"eval_name": eval_name, "model_name": model_name, "metric": result}
        )
    merge_metrics_df = pd.DataFrame(merge_metrics).pivot(
        index=["model_name"], columns="eval_name"
    )
    print("\nAll results: ")
    print(merge_metrics_df.to_markdown())
    return merge_metrics


if __name__ == "__main__":
    main()
