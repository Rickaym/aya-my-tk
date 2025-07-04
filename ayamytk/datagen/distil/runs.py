#!/usr/bin/env python3
import os
import pandas as pd
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Callable, Literal
from tqdm import tqdm

sys.path.append(os.path.abspath("."))

from ayamytk.test.bench.models import MessageList, SamplerBase, SamplerResponse
from ayamytk.datagen.distil.formatters import simple_formatter


def process_single_row(
    row_data: tuple,
    sampler: SamplerBase,
    output_column: str,
    formatter_func: Callable[[pd.Series], str],
) -> tuple:
    """Process a single row with the given sampler"""
    idx, row = row_data

    # Skip if output already exists
    if pd.notna(row.get(output_column)) and str(row.get(output_column)).strip():
        return idx, row[output_column], None, True  # idx, output, error, skipped

    try:
        # Use the formatter function to create the user message
        user_message_content = formatter_func(row)

        if not user_message_content:
            return idx, "", "Formatter function returned empty content", False

        # Create message list for the sampler
        message_list: MessageList = [{"role": "user", "content": user_message_content}]

        # Call the sampler
        response: SamplerResponse = sampler(message_list)
        output_text = response.response_text.strip()

        return idx, output_text, None, False

    except Exception as e:
        return idx, "", str(e), False


def run_distillation(
    input_file: str,
    sampler: SamplerBase,
    output_file: str = "inplace",
    output_column: str = "output",
    formatter_func: Callable[[pd.Series], str] = simple_formatter,
    max_workers: int = 4,
    save_frequency: int = 10,
    debug: bool = False,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Run knowledge distillation on a CSV file using the specified sampler

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        sampler: Sampler instance to use for generation
        output_column: Name of the output column in CSV
        formatter_func: Function to format row data into user message content
        max_workers: Number of parallel workers
        save_frequency: Save progress after this many completions
        debug: Whether to run in debug mode (limits rows processed)

    Returns:
        Dictionary with statistics about the run
    """
    print(f"Loading data from {input_file}")
    df = pd.read_csv(input_file, encoding="utf-8")

    if debug:
        print("DEBUG MODE: Processing only first 5 rows")
        df = df.head(5)

    if output_file == "inplace":
        output_file = input_file

    total_rows = len(df)
    print(f"Total rows to process: {total_rows}")

    # Ensure output column exists
    if output_column not in df.columns:
        df[output_column] = ""

    if overwrite:
        df[output_column] = ""

    # Identify rows that need processing (i.e., output is missing or blank)
    mask_to_process = df[output_column].isna() | (df[output_column].astype(str).str.strip() == "")
    rows_to_process = df[mask_to_process]
    print(f"Rows needing generation: {len(rows_to_process)}")

    if len(rows_to_process) == 0:
        print("All rows already have outputs. Nothing to do.")
        return {
            "total_rows": total_rows,
            "processed": 0,
            "errors": 0,
            "skipped": total_rows,
        }

    # Only process rows that need generation
    row_data = [(idx, df.loc[idx]) for idx in rows_to_process.index]

    stats = {
        "total_rows": total_rows,
        "processed": 0,
        "errors": 0,
        "skipped": total_rows - len(row_data),
        "error_details": [],
    }

    print(f"Starting parallel processing with {max_workers} workers...")
    print(f"Using formatter: {formatter_func.__name__}")

    # Create progress bar
    pbar = tqdm(
        total=len(row_data),
        desc="Processing rows",
        unit="rows",
        dynamic_ncols=True
    )
    pbar.set_postfix_str(f"P:0 E:0 S:{stats['skipped']}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit only tasks for rows that need processing
        future_to_idx = {
            executor.submit(
                process_single_row, data, sampler, output_column, formatter_func
            ): data[0]
            for data in row_data
        }

        # Process completed tasks
        for future in as_completed(future_to_idx):
            try:
                idx, output_text, error, skipped = future.result()

                if error:
                    stats["errors"] += 1
                    stats["error_details"].append(f"Row {idx}: {error}")
                    tqdm.write(f"Error processing row {idx}: {error}")
                else:
                    df.at[idx, output_column] = output_text
                    stats["processed"] += 1

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix_str(f"P:{stats['processed']} E:{stats['errors']} S:{stats['skipped']}")

                # Save progress periodically
                if (stats["processed"] + stats["errors"]) % save_frequency == 0:
                    tqdm.write(f"Saving progress...")
                    df.to_csv(output_file, index=False, encoding="utf-8")

            except Exception as e:
                idx = future_to_idx[future]
                stats["errors"] += 1
                error_msg = f"Unexpected error: {str(e)}"
                stats["error_details"].append(f"Row {idx}: {error_msg}")
                tqdm.write(f"Unexpected error processing row {idx}: {e}")

                # Update progress bar even for unexpected errors
                pbar.update(1)
                pbar.set_postfix_str(f"P:{stats['processed']} E:{stats['errors']} S:{stats['skipped']}")

    # Close progress bar
    pbar.close()

    # Final save
    print(f"Saving final results to {output_file}")
    df.to_csv(output_file, index=False, encoding="utf-8")

    # Print summary
    print("\n" + "=" * 50)
    print("DISTILLATION SUMMARY")
    print("=" * 50)
    print(f"Total rows: {stats['total_rows']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Skipped (already had output): {stats['skipped']}")
    print(f"Errors: {stats['errors']}")

    if stats["error_details"]:
        print("\nError details:")
        for error in stats["error_details"][:10]:  # Show first 10 errors
            print(f"  {error}")
        if len(stats["error_details"]) > 10:
            print(f"  ... and {len(stats['error_details']) - 10} more errors")

    return stats