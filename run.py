from datasets import Dataset
from googletrans import Translator
import pandas as pd
import asyncio
import tqdm


async def translate_to_mya(text):
    async with Translator() as translator:
        result = await translator.translate(text, src="en", dest="my")
        return result.text


async def tag_dataset():
    ds = Dataset.from_file("data/alpaca_cleaned/train/data-00000-of-00001.arrow")

    # Convert dataset to pandas DataFrame
    df = pd.DataFrame(ds)

    # Save the dataset as CSV
    csv_path = "dataset.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(
        f"Dataset saved to {csv_path} with {len(df)} rows and {len(df.columns)} columns"
    )

    # try:
    #     # Load translations CSV
    #     translations_df = pd.read_csv('translations.csv', encoding='utf-8')
    #     print(f"Successfully loaded translations.csv with {len(translations_df)} rows")

    #     # Convert dataset to list of dictionaries for easier processing
    #     dataset_rows = []
    #     for idx, row in enumerate(tqdm.tqdm(ds, desc="Processing dataset")):
    #         if idx >= len(translations_df):
    #             break
    #         # Get all fields from the dataset
    #         row_dict = {f"{key}-en": value for key, value in row.items()}
    #         for k, v in translations_df.loc[idx].items():
    #             row_dict[k] = v

    #         dataset_rows.append(row_dict)

    #     # Convert to DataFrame and save
    #     final_df = pd.DataFrame(dataset_rows)
    #     # Save the updated DataFrame
    #     final_df.to_csv('translations.csv', index=False, encoding='utf-8')
    #     print(f"Updated translations.csv with {len(final_df)} rows and {len(final_df.columns)} columns")

    # except Exception as e:
    #     print(f"Error processing translations.csv: {e}")


async def translate_dataset():
    # Load the dataset
    ds = Dataset.from_file("data/alpaca_cleaned/train/data-00000-of-00001.arrow")
    burmese_texts = []

    try:
        existing_df = pd.read_csv("translations.csv", encoding="utf-8")
        start_idx = len(existing_df)
        burmese_texts = existing_df.to_dict("records")
        print(f"Continuing from row {start_idx}")
    except FileNotFoundError:
        start_idx = 0
        print("Starting new translation file")

    # Process each row, starting from where we left off
    for idx, row in enumerate(tqdm.tqdm(ds, total=len(ds))):
        if idx < start_idx:
            continue

        record = {}
        for key, value in row.items():
            burmese_text = await translate_to_mya(value)
            record[f"{key}-en"] = value
            record[key] = burmese_text

        # Add a small delay to avoid rate limiting
        await asyncio.sleep(0.5)

        burmese_texts.append(record)

        # Save progress after each translation
        df = pd.DataFrame(burmese_texts)
        df.to_csv("translations.csv", index=False)


def combine_flores_datasets():
    # Read both datasets
    eng_df = pd.read_csv("flores_plus_eng.csv")
    mya_df = pd.read_csv("flores_plus_mya.csv")

    # Create a new DataFrame with entries matched by index
    combined_df = pd.DataFrame(
        {
            "english_text": eng_df["text"],
            "burmese_text": mya_df["text"],
            "url": eng_df["url"],  # Taking URL from English dataset
            "domain": eng_df["domain"],
            "topic": eng_df["topic"],
        }
    )

    # Save to CSV
    combined_df.to_csv("flores_combined.csv", index=False, encoding="utf-8")
    print(f"Combined dataset saved with {len(combined_df)} pairs")
    print(f"Original English dataset: {len(eng_df)} entries")
    print(f"Original Burmese dataset: {len(mya_df)} entries")


def split_dataset():
    # Read the dataset
    df = pd.read_csv("dataset.csv")

    # Calculate the size of each split
    total_rows = len(df)
    split_size = total_rows // 6

    # Create 6 splits
    for i in range(6):
        start_idx = i * split_size
        # For the last split, include any remaining rows
        end_idx = start_idx + split_size if i < 5 else total_rows

        # Get the split
        split_df = df.iloc[start_idx:end_idx]

        # Save to file
        output_file = f"dataset_part_{i+1}.csv"
        split_df.to_csv(output_file, index=False, encoding="utf-8")
        print(f"Part {i+1}: {len(split_df)} rows saved to {output_file}")

    print(f"\nTotal rows in original dataset: {total_rows}")
    print(f"Approximate rows per split: {split_size}")


def csvfy():
    import json
    with open("bnc.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    csv_rows = []
    for type, entries in data.items():
        for entry in entries:
            csv_rows.append(
                {"type": type, "question": entry["formatted_prompt"], "answer": entry["original_answer"], **entry["model_responses"]}
            )

    # Create DataFrame from the rows
    df = pd.DataFrame(csv_rows)

    # Save to CSV file
    output_file = "bnc_responses.csv"
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Saved {len(df)} rows to {output_file}")


# Run the translation
if __name__ == "__main__":
    # asyncio.run(translate_dataset())
    # asyncio.run(tag_dataset())
    # combine_flores_datasets()
    # split_dataset()
    csvfy()
